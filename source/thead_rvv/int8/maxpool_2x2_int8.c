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

/****************************************************************************
 * note: VLEN = 128/256 ...
 * constrains: Input and outputs must all have same scale/zero_point
 ****************************************************************************/
int shl_rvv_maxpool2x2s2_int8(struct csinn_tensor *input, struct csinn_tensor *output,
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
    int out_hw = out_h * out_w;
    int output_size = in_c * out_h * out_w;

    int extend_h = 0;
    int extend_w = 0;

    if (in_h % 2 == 1 && params->pad_down == 1) {
        extend_h = 1;
        out_h--;
    }
    if (in_w % 2 == 1 && params->pad_right == 1) {
        extend_w = 1;
        out_w--;
    }

    int remain_w = in_w - 2 * out_w;
    int vl;
    int8_t input_zp = (int8_t)input->qinfo->zero_point;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const int8_t *line0 = input_data + c * in_h * in_w;
            const int8_t *line1 = line0 + in_w;
            int8_t *outptr = output_data + c * out_hw;

            for (int h = 0; h < out_h; h++) {
                int w = out_w;
                while (w > 0) {
                    vl = vsetvl_e8m1(w);
                    vint8m1_t _line0_0_14, _line0_1_15;
                    vint8m1_t _line1_0_14, _line1_1_15;

                    vlseg2e8_v_i8m1(&_line0_0_14, &_line0_1_15, line0, vl);
                    vlseg2e8_v_i8m1(&_line1_0_14, &_line1_1_15, line1, vl);

                    vint8m1_t _max0 = vmax_vv_i8m1(_line0_0_14, _line0_1_15, vl);
                    vint8m1_t _max1 = vmax_vv_i8m1(_line1_0_14, _line1_1_15, vl);
                    vint8m1_t _max = vmax_vv_i8m1(_max0, _max1, vl);

                    vse8_v_i8m1(outptr, _max, vl);
                    line0 += 2 * vl;
                    line1 += 2 * vl;
                    outptr += vl;
                    w -= vl;
                }
                if (extend_w) {
                    outptr[0] = line0[0] > line1[0] ? line0[0] : line1[0];
                    outptr[0] = outptr[0] > input_zp ? outptr[0] : input_zp;
                    outptr++;
                }
                line0 += remain_w + in_w;
                line1 += remain_w + in_w;
            }
            if (extend_h) {
                int w = out_w;
                while (w > 0) {
                    vl = vsetvl_e8m1(w);
                    vint8m1_t _line0_0_14, _line0_1_15;

                    vlseg2e8_v_i8m1(&_line0_0_14, &_line0_1_15, line0, vl);

                    vint8m1_t _max0 = vmax_vv_i8m1(_line0_0_14, _line0_1_15, vl);
                    vint8m1_t _max = vmax_vx_i8m1(_max0, input_zp, vl);

                    vse8_v_i8m1(outptr, _max, vl);
                    line0 += 2 * vl;
                    outptr += vl;
                    w -= vl;
                }

                if (extend_w) {
                    outptr[0] = line0[0] > input_zp ? line0[0] : input_zp;
                    outptr++;
                }
            }
        }
        input_data += input_size;
        output_data += output_size;
    }
    return CSINN_TRUE;
}

int shl_rvv_maxpool2x2s2_p1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
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
    int out_hw = out_h * out_w;
    int output_size = in_c * out_h * out_w;

    int extend_h = 0;
    int extend_w = 0;

    if (in_h % 2 == 0 && params->pad_down == 1) {
        extend_h = 1;
        out_h--;
    }
    if (in_w % 2 == 0 && params->pad_right == 1) {
        extend_w = 1;
        out_w--;
    }

    int remain_w = in_w - 2 * out_w + 1;
    int vl;
    int8_t input_zp = (int8_t)input->qinfo->zero_point;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const int8_t *line00 = input_data + c * in_h * in_w;
            int8_t *outptr = output_data + c * out_hw;

            // h top ---- w left
            outptr[0] = line00[0] > input_zp ? line00[0] : input_zp;
            outptr++;
            line00++;
            // h top ---- w mid
            int w = out_w - 1;
            while (w > 0) {
                vl = vsetvl_e8m1(w);
                vint8m1_t _line0_0_6, _line0_1_7;
                vlseg2e8_v_i8m1(&_line0_0_6, &_line0_1_7, line00, vl);
                vint8m1_t _max0 = vmax_vv_i8m1(_line0_0_6, _line0_1_7, vl);
                vint8m1_t _max = vmax_vx_i8m1(_max0, input_zp, vl);
                vse8_v_i8m1(outptr, _max, vl);
                line00 += 2 * vl;
                outptr += vl;
                w -= vl;
            }
            // h top ---- w right
            if (extend_w) {
                outptr[0] = line00[0] > input_zp ? line00[0] : input_zp;
                outptr++;
            }
            line00 += remain_w;

            // h mid
            const int8_t *line0 = line00;
            const int8_t *line1 = line0 + in_w;
            for (int h = 0; h < out_h - 1; h++) {
                // h mid ---- w left
                outptr[0] = line0[0] > line1[0] ? line0[0] : line1[0];
                outptr[0] = outptr[0] > input_zp ? outptr[0] : input_zp;
                outptr++;
                line0++;
                line1++;
                // h mid ---- w mid
                w = out_w - 1;
                while (w > 0) {
                    vl = vsetvl_e8m1(w);
                    vint8m1_t _line0_0_6, _line0_1_7;
                    vint8m1_t _line1_0_6, _line1_1_7;

                    vlseg2e8_v_i8m1(&_line0_0_6, &_line0_1_7, line0, vl);
                    vlseg2e8_v_i8m1(&_line1_0_6, &_line1_1_7, line1, vl);

                    vint8m1_t _max0 = vmax_vv_i8m1(_line0_0_6, _line0_1_7, vl);
                    vint8m1_t _max1 = vmax_vv_i8m1(_line1_0_6, _line1_1_7, vl);
                    vint8m1_t _max = vmax_vv_i8m1(_max0, _max1, vl);

                    vse8_v_i8m1(outptr, _max, vl);
                    line0 += 2 * vl;
                    line1 += 2 * vl;
                    outptr += vl;
                    w -= vl;
                }

                // h mid ---- w right
                if (extend_w) {
                    outptr[0] = line0[0] > line1[0] ? line0[0] : line1[0];
                    outptr[0] = outptr[0] > input_zp ? outptr[0] : input_zp;
                    outptr++;
                }
                line0 += remain_w + in_w;
                line1 += remain_w + in_w;
            }
            // h bottom
            if (extend_h) {
                // h bottom ---- w left
                outptr[0] = line0[0] > input_zp ? line0[0] : input_zp;
                outptr++;
                line0++;
                // h bottom ---- w mid
                w = out_w - 1;
                while (w > 0) {
                    vl = vsetvl_e8m1(w);
                    vint8m1_t _line0_0_6, _line0_1_7;

                    vlseg2e8_v_i8m1(&_line0_0_6, &_line0_1_7, line0, vl);

                    vint8m1_t _max0 = vmax_vv_i8m1(_line0_0_6, _line0_1_7, vl);
                    vint8m1_t _max = vmax_vx_i8m1(_max0, input_zp, vl);

                    vse8_v_i8m1(outptr, _max, vl);
                    line0 += 2 * vl;
                    outptr += vl;
                    w -= vl;
                }
                // h bottom ---- w right
                if (extend_w) {
                    outptr[0] = line0[0] > input_zp ? line0[0] : input_zp;
                }
            }
        }
        input_data += input_size;
        output_data += output_size;
    }
    return CSINN_TRUE;
}
