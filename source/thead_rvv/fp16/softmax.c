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
#include "rvv_mathfun_fp16.h"

static inline __fp16 fast_exp16(__fp16 y)
{
    union {
        __fp16 d;
        uint16_t x;
    } data = {y};

    data.x = (uint16_t)(y * 1477 + (15321));

    return data.d;
}

int shl_rvv_softmax_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_softmax_params *params)
{
    /* TODO: fp16 quantize */
    if (fabs(input->qinfo->scale - 1) > FLT_EPSILON ||
        fabs(output->qinfo->scale - 1) > FLT_EPSILON) {
        shl_debug_error("unsupport fp16 quantization of softmax op\n");
        return CSINN_FALSE;
    }
    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(input);
    }

    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;

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
    int pack2n = csrr_vlenb() / sizeof(__fp16) * 2;
    __fp16 *exp_buffer = (__fp16 *)shl_mem_alloc(inner_size * cnt * sizeof(__fp16));
    for (int i = 0; i < outer_size; i++) {
        for (int k = 0; k < inner_size; k++) {
            __fp16 acc_exp = 0.0f;
            __fp16 max = -65504;
            // Find max element value which we'll use to ensure numerical stability
            // taking advantage of the following equality:
            // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))

            __fp16 *ptr = input_data + k;
            size_t vl = vsetvl_e16m2(pack2n);
            vfloat16m2_t _max = vfmv_v_f_f16m2(max, vl);
            int j = 0;
            for (; j + pack2n <= cnt; j += pack2n) {
                vfloat16m2_t _input_data = vlse16_v_f16m2(ptr, inner_size * sizeof(__fp16), vl);
                _max = vfmax_vv_f16m2(_input_data, _max, vl);
                ptr += vl * inner_size;
            }
            vfloat16m1_t _min_f = vfmv_v_f_f16m1(max, 8);
            vfloat16m1_t _max0 = vfredmax_vs_f16m2_f16m1(vundefined_f16m1(), _max, _min_f, vl);
            max = vfmv_f_s_f16m1_f16(_max0);
            for (; j < cnt; j++) {
                max = fmax(max, *(input_data + j * inner_size + k));
            }

            ptr = input_data + k;
            __fp16 *ptr2 = exp_buffer + k;
            vfloat16m2_t _sum = vfmv_v_f_f16m2(0.0f, vl);
            j = 0;
            for (; j + pack2n <= cnt; j += pack2n) {
                vfloat16m2_t _input_data = vlse16_v_f16m2(ptr, inner_size * sizeof(__fp16), vl);
                _input_data = vfsub_vf_f16m2(_input_data, max, vl);
                vfloat16m2_t _output_data = exp_ps_vfloat16m2(_input_data, vl);
                vsse16_v_f16m2(ptr2, inner_size * sizeof(__fp16), _output_data, vl);
                _sum = vfadd_vv_f16m2(_sum, _output_data, vl);
                ptr += vl * inner_size;
                ptr2 += vl * inner_size;
            }
            vfloat16m1_t _0_f = vfmv_v_f_f16m1(0.0f, 8);
            vfloat16m1_t _sum2 = vfredosum_vs_f16m2_f16m1(_0_f, _sum, _0_f, vl);
            acc_exp = vfmv_f_s_f16m1_f16(_sum2);
            for (; j < cnt; j++) {
                __fp16 tmp = fast_exp16(*(input_data + j * inner_size + k) - max);
                exp_buffer[j * inner_size + k] = tmp;
                acc_exp += tmp;
            }

            ptr = exp_buffer + k;
            ptr2 = output_data + k;
            int n = cnt;
            while (n > 0) {
                size_t vl = vsetvl_e16m2(n);
                vfloat16m2_t _exp = vlse16_v_f16m2(ptr, inner_size * sizeof(__fp16), vl);
                vfloat16m2_t _output_data = vfdiv_vf_f16m2(_exp, acc_exp, vl);
                vsse16_v_f16m2(ptr2, inner_size * sizeof(__fp16), _output_data, vl);

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
