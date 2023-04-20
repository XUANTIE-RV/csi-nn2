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

/* SHL version 2.1.x */

#include "shl_thead_rvv.h"
int shl_rvv_prelu_fp16(struct csinn_tensor *input, struct csinn_tensor *alpha,
                       struct csinn_tensor *output, struct csinn_prelu_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *alpha_data = (__fp16 *)alpha->data;
    __fp16 *output_data = (__fp16 *)output->data;

    for (int n = 0; n < input->dim[0]; ++n) {
        for (int c = 0; c < input->dim[1]; ++c) {
            float a = alpha_data[c];
            int inner_size = input->dim[2] * input->dim[3];
            while (inner_size > 0) {
                int vl = vsetvl_e16m2(inner_size);
                vfloat16m2_t _input = vle16_v_f16m2(input_data, vl);
                vbool8_t _mask = vmflt_vf_f16m2_b8(_input, 0.0f, vl);
                vfloat16m2_t _res = vfmul_vf_f16m2_m(_mask, _input, _input, a, vl);
                vse16_v_f16m2(output_data, _res, vl);
                input_data += vl;
                output_data += vl;
                inner_size -= vl;
            }
        }
    }
    // requantize
    shl_rvv_siso_op_requantize_fp16(input, output);
    return CSINN_TRUE;
}
