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

/* CSI-NN2 version 1.12.x */

#include <math.h>

#include "csi_c906.h"
#include "csi_utils.h"

int csi_c906_layer_norm_fp16(struct csi_tensor *input, struct csi_tensor *output,
                             struct csi_tensor *gamma, struct csi_tensor *beta,
                             struct layer_norm_params *params)
{
    int flatten_size = 0;
    flatten_size *= input->dim[0] * input->dim[1] * input->dim[2];

    __fp16 *sum = (__fp16 *)csi_mem_alloc(input->dim[1] * sizeof(__fp16));
    __fp16 *sum2 = (__fp16 *)csi_mem_alloc(input->dim[1] * sizeof(__fp16));
    __fp16 *input_data = input->data;
    __fp16 *output_data = output->data;
    __fp16 *gamma_data = gamma->data;
    __fp16 *beta_data = beta->data;

    __fp16 *p_input_data = input_data;
    __fp16 *p_output_data = output_data;

    size_t batch = input->dim[1];
    size_t output_depth = input->dim[2];
    for (int i = 0; i < batch; i++) {
        vfloat16m2_t _sum = vfmv_v_f_f16m2(0.0f, 16);
        for (int j = 0; j + 15 < output_depth; j += 16) {
            vfloat16m2_t _input_data = vle16_v_f16m2(p_input_data, 16);

            _sum = vfadd_vv_f16m2(_sum, _input_data, 16);
            p_input_data += 16;
        }

        vfloat16m1_t _0_f = vfmv_v_f_f16m1(0.0f, 8);
        vfloat16m1_t _sum2 = vfredosum_vs_f16m2_f16m1(_0_f, _sum, _0_f, 16);
        __fp16 tmp = vfmv_f_s_f16m1_f16(_sum2);
        tmp = tmp / output_depth;
        sum[i] = tmp;
    }

    p_input_data = input_data;
    p_output_data = output_data;
    for (int i = 0; i < batch; i++) {
        vfloat32m4_t _sum_f32 = vfmv_v_f_f32m4(0.0f, 16);
        vfloat16m2_t _sum = vfmv_v_f_f16m2(sum[i], 16);

        for (int j = 0; j + 15 < output_depth; j += 16) {
            vfloat16m2_t _input_data = vle16_v_f16m2(p_input_data, 16);
            _input_data = vfsub_vv_f16m2(_input_data, _sum, 16);
            vse16_v_f16m2(p_output_data, _input_data, 16);
            vfloat32m4_t _input_data_f32 = vfwmul_vv_f32m4(_input_data, _input_data, 16);
            _sum_f32 = vfadd_vv_f32m4(_input_data_f32, _sum_f32, 16);
            p_input_data += 16;
            p_output_data += 16;
        }
        vfloat32m1_t _0_f = vfmv_v_f_f32m1(0.0f, 4);
        vfloat32m1_t _sum2 = vfmv_v_f_f32m1(0.0f, 4);
        _sum2 = vfredosum_vs_f32m4_f32m1(_0_f, _sum_f32, _0_f, 16);
        float tmp = vfmv_f_s_f32m1_f32(_sum2);
        tmp /= output_depth;
        tmp = sqrtf(tmp);
        sum2[i] = tmp;
    }

    p_output_data = output_data;
    for (int i = 0; i < batch; i++) {
        __fp16 mul = 1.0f / sum2[i];
        vfloat16m2_t _sum = vfmv_v_f_f16m2(mul, 16);
        for (int j = 0; j + 15 < output_depth; j += 16) {
            vfloat16m2_t _output_data = vle16_v_f16m2(p_output_data, 16);
            _output_data = vfmul_vv_f16m2(_output_data, _sum, 16);
            vfloat16m2_t _gamma_data = vle16_v_f16m2(gamma_data + j, 16);
            _output_data = vfmul_vv_f16m2(_output_data, _gamma_data, 16);
            vfloat16m2_t _beta_data = vle16_v_f16m2(beta_data + j, 16);
            _output_data = vfadd_vv_f16m2(_output_data, _beta_data, 16);
            vse16_v_f16m2(p_output_data, _output_data, 16);
            p_output_data += 16;
        }
    }

    csi_mem_free(sum);
    csi_mem_free(sum2);

    return CSINN_TRUE;
}
