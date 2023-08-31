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

/*************************************************************
 * note: support flexible vlen
 *************************************************************/
int shl_rvv_maxpool2x2s2_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
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
    int input_size = in_c * in_h * in_w;

    int out_h = output->dim[2];
    int out_w = output->dim[3];
    int output_size = in_c * out_h * out_w;

    int padded_in_h = in_h + params->pad_top + params->pad_down;
    int padded_in_w = in_w + params->pad_left + params->pad_right;
    int padded_in_hw = padded_in_w * padded_in_h;

    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int vl = vsetvl_e16m1(packn);

    __fp16 *input_ncxhwx = (__fp16 *)shl_mem_alloc(in_c * padded_in_hw * sizeof(__fp16));
    int tailstep = (padded_in_w - 2 * out_w + padded_in_w) * packn;

    for (int b = 0; b < batch; b++) {
        /* TODO: remove padding */
        shl_rvv_pad_input_packn_fp16(input_data, input_ncxhwx, in_c, in_h, in_w, padded_in_h,
                                     padded_in_w, params->pad_top, params->pad_left);

        for (int c = 0; c + packn - 1 < in_c; c += packn) {
            __fp16 *out0 = output_data + c * out_h * out_w;
            const __fp16 *line0 = input_ncxhwx + c * padded_in_h * padded_in_w;
            const __fp16 *line1 = line0 + padded_in_w * packn;

            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    vfloat16m1_t _max = vle16_v_f16m1(line0, vl);
                    _max = vfmax_vv_f16m1(_max, vle16_v_f16m1(line0 + packn, vl), vl);
                    _max = vfmax_vv_f16m1(_max, vle16_v_f16m1(line1, vl), vl);
                    _max = vfmax_vv_f16m1(_max, vle16_v_f16m1(line1 + packn, vl), vl);
                    vse16_v_f16m1(out0, _max, vl);

                    line0 += packn * 2;
                    line1 += packn * 2;
                    out0 += packn;
                }
                line0 += tailstep;
                line1 += tailstep;
            }
        }
        input_data += input_size;
        output_data += output_size;
    }

    shl_mem_free(input_ncxhwx);
    // requantize
    shl_rvv_siso_op_requantize_fp16(input, output);
    return CSINN_TRUE;
}
