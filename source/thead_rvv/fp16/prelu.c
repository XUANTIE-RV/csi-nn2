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

#include "shl_thead_rvv.h"

int shl_rvv_prelu_fp16(struct csinn_tensor *input, struct csinn_tensor *alpha,
                       struct csinn_tensor *output, struct csinn_prelu_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *alpha_data = (__fp16 *)alpha->data;
    __fp16 *output_data = (__fp16 *)output->data;

    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        const int packn = csrr_vlenb() / sizeof(__fp16);
        int inner_size = input->dim[2] * input->dim[3];
        for (int n = 0; n < input->dim[0]; ++n) {
            for (int c1 = 0; c1 < input->dim[1]; ++c1) {
                const __fp16 *in_ptr =
                    (__fp16 *)input_data + (n * input->dim[1] + c1) * inner_size * packn;
                __fp16 *out_ptr =
                    (__fp16 *)output_data + (n * input->dim[1] + c1) * inner_size * packn;
                const __fp16 *a_ptr = (__fp16 *)alpha_data + c1 * packn;
                vfloat16m1_t _a = vle16_v_f16m1(a_ptr, packn);
                for (int hw = 0; hw < inner_size; hw++) {
                    vfloat16m1_t _in = vle16_v_f16m1(input_data, packn);
                    vbool16_t _mask = vmflt_vf_f16m1_b16(_in, 0.0f, packn);
                    vfloat16m1_t _res = vfmul_vv_f16m1_m(_mask, _in, _in, _a, packn);
                    vse16_v_f16m1(out_ptr, _res, packn);
                    in_ptr += packn;
                    out_ptr += packn;
                }
            }
        }
        if (output->layout == CSINN_LAYOUT_NCHW) {
            output->dim[1] /= packn;
            output->dim[4] = packn;
            output->dim_count = 5;
            output->layout = CSINN_LAYOUT_NC1HWC0;
        }
    } else if (input->layout == CSINN_LAYOUT_NCHW) {
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
        if (output->layout == CSINN_LAYOUT_NC1HWC0) {
            const int packn = csrr_vlenb() / sizeof(__fp16);
            output->dim[1] *= packn;
            output->dim[4] = 0;
            output->dim_count = 4;
            output->layout = CSINN_LAYOUT_NCHW;
        }
    } else {
        shl_debug_error("prelu unsupported layout: %d\n", input->layout);
    }

    // requantize
    shl_rvv_siso_op_requantize_fp16(input, output);
    return CSINN_TRUE;
}
