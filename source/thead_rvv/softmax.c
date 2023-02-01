/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

/* SHL version 2.1.x */

#include "rvv_mathfun.h"
#include "shl_thead_rvv.h"

static inline float fast_exp32(float y)
{
    union {
        float d;
        int16_t x[2];
    } data = {y};

    data.x[0] = 0;
    data.x[1] = (int16_t)(184 * y + 16250);

    return data.d;
}

static inline __fp16 fast_exp16(__fp16 y)
{
    union {
        __fp16 d;
        uint16_t x;
    } data = {y};

    data.x = (uint16_t)(y * 1477 + (15321));

    return data.d;
}

int shl_rvv_softmax_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_softmax_params *params)
{
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

    float *exp_buffer = (float *)shl_mem_alloc(inner_size * cnt * sizeof(float));
    for (int i = 0; i < outer_size; i++) {
        for (int k = 0; k < inner_size; k++) {
            float acc_exp = 0.0f;
            float max = -FLT_MAX;
            // Find max element value which we'll use to ensure numerical stability
            // taking advantage of the following equality:
            // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))

            float *ptr = input_data + k;
            vfloat32m2_t _output_data = vfmv_v_f_f32m2(max, 8);
            int j = 0;
            for (; j + 7 < cnt; j += 8) {
                vfloat32m2_t _input_data = vlse32_v_f32m2(ptr, 4 * inner_size, 8);
                _output_data = vfmax_vv_f32m2(_input_data, _output_data, 8);
                ptr += 8 * inner_size;
            }
            vfloat32m1_t _min_f = vfmv_v_f_f32m1(-FLT_MAX, 4);
            vfloat32m1_t _max = vfredmax_vs_f32m2_f32m1(_min_f, _output_data, _min_f, 8);
            max = vfmv_f_s_f32m1_f32(_max);

            for (; j < cnt; j++) {
                max = fmax(max, *(input_data + j * inner_size + k));
            }

            ptr = input_data + k;
            vfloat32m2_t _sum = vfmv_v_f_f32m2(0.0f, 8);

            j = 0;
            for (; j < cnt; j++) {
                float tmp = fast_exp32(*(input_data + j * inner_size + k) - max);
                exp_buffer[j * inner_size + k] = tmp;
                acc_exp += tmp;
            }

            ptr = exp_buffer + k;
            float *ptr2 = output_data + k;
            int n = cnt;
            while (n > 0) {
                size_t vl = vsetvl_e32m2(n);
                vfloat32m2_t _exp = vlse32_v_f32m2(ptr, 4 * inner_size, vl);
                vfloat32m2_t _output_data = vfdiv_vf_f32m2(_exp, acc_exp, vl);
                vsse32_v_f32m2(ptr2, 4 * inner_size, _output_data, vl);

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

int shl_rvv_softmax_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_softmax_params *params)
{
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

    for (int i = 0; i < outer_size; i++) {
        for (int k = 0; k < inner_size; k++) {
            __fp16 acc_exp = 0.0f;
            __fp16 max = -FLT_MAX;
            // Find max element value which we'll use to ensure numerical stability
            // taking advantage of the following equality:
            // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))

            int n = cnt;
            __fp16 *ptr = input_data + k;
            vfloat16m2_t _output_data = vfmv_v_f_f16m2(max, 16);
            int j = 0;
            for (; j + 15 < cnt; j += 16) {
                vfloat16m2_t _input_data = vlse16_v_f16m2(ptr, 2 * inner_size, 16);
                _output_data = vfmax_vv_f16m2(_input_data, _output_data, 16);
                ptr += 16 * inner_size;
            }
            vfloat16m1_t _min_f = vfmv_v_f_f16m1(-FLT_MAX, 8);
            vfloat16m1_t _max = vfredmax_vs_f16m2_f16m1(_min_f, _output_data, _min_f, 16);
            max = vfmv_f_s_f16m1_f16(_max);

            for (; j < cnt; j++) {
                max = fmax(max, *(input_data + j * inner_size + k));
            }

            n = cnt;
            ptr = input_data + k;
            vfloat16m2_t _sum = vfmv_v_f_f16m2(0.0f, 16);

            j = 0;
            for (; j + 15 < cnt; j += 16) {
                vfloat16m2_t _input_data = vlse16_v_f16m2(ptr, 2 * inner_size, 16);
                _input_data = vfsub_vf_f16m2(_input_data, max, 16);
                vfloat16m2_t _output_data = exp_ps_vfloat16m2(_input_data, 16);
                _sum = vfadd_vv_f16m2(_sum, _output_data, 16);
                ptr += 16 * inner_size;
            }
            vfloat16m1_t _0_f = vfmv_v_f_f16m1(0.0f, 8);
            vfloat16m1_t _sum2 = vfredosum_vs_f16m2_f16m1(_0_f, _sum, _0_f, 16);
            acc_exp = vfmv_f_s_f16m1_f16(_sum2);

            for (; j < cnt; j++) {
                acc_exp += fast_exp16(*(input_data + j * inner_size + k) - max);
            }
            n = cnt;
            ptr = input_data + k;
            __fp16 *ptr2 = output_data + k;
            while (n > 0) {
                size_t vl = vsetvl_e16m2(n);

                vfloat16m2_t _input_data = vlse16_v_f16m2(ptr, 2 * inner_size, vl);
                _input_data = vfsub_vf_f16m2(_input_data, max, vl);
                vfloat16m2_t _output_data = exp_ps_vfloat16m2(_input_data, vl);
                _output_data = vfdiv_vf_f16m2(_output_data, acc_exp, vl);
                vsse16_v_f16m2(ptr2, 2 * inner_size, _output_data, vl);

                ptr += vl * inner_size;
                ptr2 += vl * inner_size;
                n -= vl;
            }
        }
        input_data += inner_size * cnt;
        output_data += inner_size * cnt;
    }
    return CSINN_TRUE;
}

int shl_rvv_softmax_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_softmax_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_rvv_softmax_fp32);
}
