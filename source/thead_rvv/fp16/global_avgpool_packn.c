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

int shl_rvv_global_avgpool2d_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_pool_params *params)
{
    if (input->layout == CSINN_LAYOUT_NCHW) {
        shl_rvv_tensor_ndarray_to_nc1xc0_replace_fp16(input);
    }
    if (output->layout == CSINN_LAYOUT_NCHW) {
        output->dim[1] /= input->dim[4];
        output->dim[4] = input->dim[4];
        output->dim_count = 5;
        output->layout = CSINN_LAYOUT_NC1HWC0;
    }
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;

    int batch = input->dim[0];
    int in_c = input->dim[1] * input->dim[4];
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int in_hw = in_h * in_w;

    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int vl = vsetvl_e16m1(packn);

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c + packn - 1 < in_c; c += packn) {
            /* avoid overflow */
            if (in_hw >= 1000) {
                vfloat32m2_t _acc = vfmv_v_f_f32m2(0.0f, vl);
                for (int i = 0; i < in_hw; i++) {
                    vfloat32m2_t _inputw = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(input_data, vl), vl);
                    _acc = vfadd_vv_f32m2(_acc, _inputw, vl);
                    input_data += packn;
                }
                vfloat32m2_t _avg = vfmul_vf_f32m2(_acc, 1.0f / in_hw, vl);
                vse16_v_f16m1(output_data, vfncvt_f_f_w_f16m1(_avg, vl), vl);
                output_data += packn;
            } else {
                vfloat16m1_t _acc = vle16_v_f16m1(input_data, vl);
                input_data += packn;
                for (int i = 1; i < in_hw; i++) {
                    _acc = vfadd_vv_f16m1(_acc, vle16_v_f16m1(input_data, vl), vl);
                    input_data += packn;
                }
                vfloat16m1_t _avg = vfmul_vf_f16m1(_acc, 1.0f / in_hw, vl);
                vse16_v_f16m1(output_data, _avg, vl);
                output_data += packn;
            }
        }
    }
    // requantize
    shl_rvv_siso_op_requantize_fp16(input, output);
    return CSINN_TRUE;
}
