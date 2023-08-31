/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "rvv/rvv.h"
#include "rvv_mathfun_fp32.h"

static inline float fast_exp32(float y)
{
    union {
        float d;
        uint32_t x;
    } data = {y};

    data.x = (12102203 * y + 1064866804);

    return data.d;
}

int shl_rvv_softmax_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_softmax_params *params)
{
    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(input);
    }
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int axis = params->axis;
    // FlatSize() = outer_size * inner_size * cnt;
    int64_t outer_size = 1;
    for (int i = 0; i < axis; i++) {
        outer_size *= input->dim[i];
    }

    int64_t inner_size = 1;
    for (int i = axis + 1; i < input->dim_count; i++) {
        inner_size *= input->dim[i];
    }

    int cnt = input->dim[axis];
    int pack2n = csrr_vlenb() / sizeof(float) * 2;
    float *exp_buffer = (float *)shl_mem_alloc(inner_size * cnt * sizeof(float));
    for (int i = 0; i < outer_size; i++) {
        for (int k = 0; k < inner_size; k++) {
            float acc_exp = 0.0f;
            float max = -FLT_MAX;
            // Find max element value which we'll use to ensure numerical stability
            // taking advantage of the following equality:
            // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))

            float *ptr = input_data + k;
            size_t vl = vsetvl_e32m2(pack2n);
            vfloat32m2_t _max = vfmv_v_f_f32m2(max, vl);
            int j = 0;
            for (; j + pack2n <= cnt; j += pack2n) {
                vfloat32m2_t _input_data = vlse32_v_f32m2(ptr, inner_size * sizeof(float), vl);
                _max = vfmax_vv_f32m2(_input_data, _max, vl);
                ptr += vl * inner_size;
            }
            vfloat32m1_t _min_f = vfmv_v_f_f32m1(max, 4);
            vfloat32m1_t _max0 = vfredmax_vs_f32m2_f32m1(vundefined_f32m1(), _max, _min_f, vl);
            max = vfmv_f_s_f32m1_f32(_max0);
            for (; j < cnt; j++) {
                max = fmax(max, *(input_data + j * inner_size + k));
            }

            ptr = input_data + k;
            for (int j = 0; j < cnt; j++) {
                float tmp = fast_exp32(*(input_data + j * inner_size + k) - max);
                exp_buffer[j * inner_size + k] = tmp;
                acc_exp += tmp;
            }

            ptr = exp_buffer + k;
            float *ptr2 = output_data + k;
            int n = cnt;
            while (n > 0) {
                size_t vl = vsetvl_e32m2(n);
                vfloat32m2_t _exp = vlse32_v_f32m2(ptr, inner_size * sizeof(float), vl);
                vfloat32m2_t _output_data = vfdiv_vf_f32m2(_exp, acc_exp, vl);
                vsse32_v_f32m2(ptr2, inner_size * sizeof(float), _output_data, vl);

                ptr += vl * inner_size;
                ptr2 += vl * inner_size;
                n -= vl;
            }
        }
        input_data += inner_size * cnt;
        output_data += inner_size * cnt;
    }
    shl_mem_free(exp_buffer);
    return CSINN_TRUE;
}
