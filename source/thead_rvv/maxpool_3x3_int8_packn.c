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

#include "shl_thead_rvv.h"

/*************************************************************
 * note: support flexible vlen
 *************************************************************/
int shl_rvv_maxpool3x3s2_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    int batch = input->dim[0];
    int in_c = input->dim[1];
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int input_size = in_c * in_h * in_w;

    int out_h = output->dim[2];
    int out_w = output->dim[3];
    int output_size = in_c * out_h * out_w;

    int padded_in_h = in_h + params->pad_top + params->pad_down;
    int padded_in_w = in_w + params->pad_left + params->pad_right;
    int padded_in_hw = padded_in_w * padded_in_h;

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);

    int8_t *input_ncxhwx = (int8_t *)shl_mem_alloc(in_c * padded_in_hw * sizeof(int8_t));
    int tailstep = (padded_in_w - 2 * out_w + padded_in_w) * packn;

    for (int b = 0; b < batch; b++) {
        shl_rvv_pad_input_packn_int8(input_data, input_ncxhwx, in_c, in_h, in_w, padded_in_h,
                                     padded_in_w, params->pad_top, params->pad_left,
                                     input->qinfo->zero_point);

        for (int c = 0; c + packn - 1 < in_c; c += packn) {
            int8_t *out0 = output_data + c * out_h * out_w;
            const int8_t *line0 = input_ncxhwx + c * padded_in_h * padded_in_w;
            const int8_t *line1 = line0 + padded_in_w * packn;
            const int8_t *line2 = line1 + padded_in_w * packn;

            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    vint8m1_t _max = vle8_v_i8m1(line0, vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line0 + packn * 1, vl), vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line0 + packn * 2, vl), vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line1, vl), vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line1 + packn * 1, vl), vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line1 + packn * 2, vl), vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line2, vl), vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line2 + packn * 1, vl), vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line2 + packn * 2, vl), vl);
                    vse8_v_i8m1(out0, _max, vl);

                    line0 += packn * 2;
                    line1 += packn * 2;
                    line2 += packn * 2;
                    out0 += packn;
                }
                line0 += tailstep;
                line1 += tailstep;
                line2 += tailstep;
            }
        }
        input_data += input_size;
        output_data += output_size;
    }
    shl_mem_free(input_ncxhwx);
    return CSINN_TRUE;
}

int shl_rvv_maxpool3x3s1_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    int batch = input->dim[0];
    int in_c = input->dim[1];
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int input_size = in_c * in_h * in_w;

    int out_h = output->dim[2];
    int out_w = output->dim[3];
    int output_size = in_c * out_h * out_w;

    int padded_in_h = in_h + params->pad_top + params->pad_down;
    int padded_in_w = in_w + params->pad_left + params->pad_right;
    int padded_in_hw = padded_in_w * padded_in_h;

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);

    int8_t *input_ncxhwx = (int8_t *)shl_mem_alloc(in_c * padded_in_hw * sizeof(int8_t));

    for (int b = 0; b < batch; b++) {
        shl_rvv_pad_input_packn_int8(input_data, input_ncxhwx, in_c, in_h, in_w, padded_in_h,
                                     padded_in_w, params->pad_top, params->pad_left,
                                     input->qinfo->zero_point);

        for (int c = 0; c + packn - 1 < in_c; c += packn) {
            int8_t *out0 = output_data + c * out_h * out_w;
            const int8_t *line0 = input_ncxhwx + c * padded_in_h * padded_in_w;
            const int8_t *line1 = line0 + padded_in_w * packn;
            const int8_t *line2 = line1 + padded_in_w * packn;

            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    vint8m1_t _max = vle8_v_i8m1(line0, vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line0 + packn * 1, vl), vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line0 + packn * 2, vl), vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line1, vl), vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line1 + packn * 1, vl), vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line1 + packn * 2, vl), vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line2, vl), vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line2 + packn * 1, vl), vl);
                    _max = vmax_vv_i8m1(_max, vle8_v_i8m1(line2 + packn * 2, vl), vl);
                    vse8_v_i8m1(out0, _max, vl);

                    line0 += packn * 1;
                    line1 += packn * 1;
                    line2 += packn * 1;
                    out0 += packn;
                }
                line0 += packn * 2;
                line1 += packn * 2;
                line2 += packn * 2;
            }
        }
        input_data += input_size;
        output_data += output_size;
    }
    shl_mem_free(input_ncxhwx);
    return CSINN_TRUE;
}
