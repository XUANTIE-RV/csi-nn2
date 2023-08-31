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

/****************************************************************************
 * note: VLEN = 128/256 ...
 * constrains: Input and outputs must all have same scale/zero_point
 ****************************************************************************/
int shl_rvv_maxpool3x3s2_int8(struct csinn_tensor *input, struct csinn_tensor *output,
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
    int remain_w = in_w - 2 * out_w;
    int vl;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const int8_t *line0 = input_data + c * in_h * in_w;
            const int8_t *line1 = line0 + in_w;
            const int8_t *line2 = line1 + in_w;
            int8_t *outptr = output_data + c * out_hw;

            for (int h = 0; h < out_h; h++) {
                int w = out_w;
                while (w > 0) {
                    vl = vsetvl_e8m1(w);
                    vint8m1_t _line0_0_6, _line0_1_7;
                    vint8m1_t _line1_0_6, _line1_1_7;
                    vint8m1_t _line2_0_6, _line2_1_7;

                    vlseg2e8_v_i8m1(&_line0_0_6, &_line0_1_7, line0, vl);
                    line0 += 2;
                    vint8m1_t _line0_2_8 = vlse8_v_i8m1(line0, 2 * sizeof(int8_t), vl);
                    line0 += (vl - 1) * 2;
                    vint8m1_t _max0 =
                        vmax_vv_i8m1(_line0_2_8, vmax_vv_i8m1(_line0_0_6, _line0_1_7, vl), vl);

                    vlseg2e8_v_i8m1(&_line1_0_6, &_line1_1_7, line1, vl);
                    line1 += 2;
                    vint8m1_t _line1_2_8 = vlse8_v_i8m1(line1, 2 * sizeof(int8_t), vl);
                    line1 += (vl - 1) * 2;
                    vint8m1_t _max1 =
                        vmax_vv_i8m1(_line1_2_8, vmax_vv_i8m1(_line1_0_6, _line1_1_7, vl), vl);

                    vlseg2e8_v_i8m1(&_line2_0_6, &_line2_1_7, line2, vl);
                    line2 += 2;
                    vint8m1_t _line2_2_8 = vlse8_v_i8m1(line2, 2 * sizeof(int8_t), vl);
                    line2 += (vl - 1) * 2;
                    vint8m1_t _max2 =
                        vmax_vv_i8m1(_line2_2_8, vmax_vv_i8m1(_line2_0_6, _line2_1_7, vl), vl);

                    vint8m1_t _max = vmax_vv_i8m1(_max2, vmax_vv_i8m1(_max0, _max1, vl), vl);
                    vse8_v_i8m1(outptr, _max, vl);

                    outptr += vl;
                    w -= vl;
                }
                if (extend_w) {
                    int8_t max0 = line0[0] > line0[1] ? line0[0] : line0[1];
                    int8_t max1 = line1[0] > line1[1] ? line1[0] : line1[1];
                    int8_t max2 = line2[0] > line2[1] ? line2[0] : line2[1];
                    outptr[0] = max1 > max2 ? max1 : max2;
                    outptr[0] = outptr[0] > max0 ? outptr[0] : max0;
                    outptr++;
                }
                line0 += remain_w + in_w;
                line1 += remain_w + in_w;
                line2 += remain_w + in_w;
            }
            if (extend_h) {
                int w = out_w;
                while (w > 0) {
                    vl = vsetvl_e8m1(w);
                    vint8m1_t _line0_0_6, _line0_1_7;
                    vint8m1_t _line1_0_6, _line1_1_7;

                    vlseg2e8_v_i8m1(&_line0_0_6, &_line0_1_7, line0, vl);
                    line0 += 2;
                    vint8m1_t _line0_2_8 = vlse8_v_i8m1(line0, 2 * sizeof(int8_t), vl);
                    line0 += (vl - 1) * 2;
                    vint8m1_t _max0 =
                        vmax_vv_i8m1(_line0_2_8, vmax_vv_i8m1(_line0_0_6, _line0_1_7, vl), vl);

                    vlseg2e8_v_i8m1(&_line1_0_6, &_line1_1_7, line1, vl);
                    line1 += 2;
                    vint8m1_t _line1_2_8 = vlse8_v_i8m1(line1, 2 * sizeof(int8_t), vl);
                    line1 += (vl - 1) * 2;
                    vint8m1_t _max1 =
                        vmax_vv_i8m1(_line1_2_8, vmax_vv_i8m1(_line1_0_6, _line1_1_7, vl), vl);

                    vint8m1_t _max = vmax_vv_i8m1(_max0, _max1, vl);
                    vse8_v_i8m1(outptr, _max, vl);

                    outptr += vl;
                    w -= vl;
                }

                if (extend_w) {
                    int8_t max0 = line0[0] > line0[1] ? line0[0] : line0[1];
                    int8_t max1 = line1[0] > line1[1] ? line1[0] : line1[1];
                    outptr[0] = max0 > max1 ? max0 : max1;
                    outptr++;
                }
            }
        }
        input_data += input_size;
        output_data += output_size;
    }
    return CSINN_TRUE;
}

int shl_rvv_maxpool3x3s2_p1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
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

    int remain_w = in_w - 2 * out_w + 1;
    int vl;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const int8_t *line0 = input_data + c * in_h * in_w;
            const int8_t *line1 = line0 + in_w;
            int8_t *outptr = output_data + c * out_hw;

            // h top ---- w left
            int8_t max0 = line0[0] > line0[1] ? line0[0] : line0[1];
            int8_t max1 = line1[0] > line1[1] ? line1[0] : line1[1];
            outptr[0] = max0 > max1 ? max0 : max1;
            outptr++;
            line0++;
            line1++;
            // h top ---- w mid
            int w = out_w - 1;
            while (w > 0) {
                vl = vsetvl_e8m1(w);
                vint8m1_t _line0_0_6, _line0_1_7;
                vint8m1_t _line1_0_6, _line1_1_7;

                vlseg2e8_v_i8m1(&_line0_0_6, &_line0_1_7, line0, vl);
                line0 += 2;
                vint8m1_t _line0_2_8 = vlse8_v_i8m1(line0, 2 * sizeof(int8_t), vl);
                line0 += (vl - 1) * 2;
                vint8m1_t _max0 =
                    vmax_vv_i8m1(_line0_2_8, vmax_vv_i8m1(_line0_0_6, _line0_1_7, vl), vl);

                vlseg2e8_v_i8m1(&_line1_0_6, &_line1_1_7, line1, vl);
                line1 += 2;
                vint8m1_t _line1_2_8 = vlse8_v_i8m1(line1, 2 * sizeof(int8_t), vl);
                line1 += (vl - 1) * 2;
                vint8m1_t _max1 =
                    vmax_vv_i8m1(_line1_2_8, vmax_vv_i8m1(_line1_0_6, _line1_1_7, vl), vl);

                vint8m1_t _max = vmax_vv_i8m1(_max0, _max1, vl);
                vse8_v_i8m1(outptr, _max, vl);

                outptr += vl;
                w -= vl;
            }
            // h top ---- w right
            if (extend_w) {
                max0 = line0[0] > line0[1] ? line0[0] : line0[1];
                max1 = line1[0] > line1[1] ? line1[0] : line1[1];
                outptr[0] = max0 > max1 ? max0 : max1;
                outptr++;
            }
            line0 += remain_w;
            line1 += remain_w;

            // h mid
            const int8_t *line2 = line1 + in_w;
            int8_t max2 = 0;
            for (int h = 0; h < out_h - 1; h++) {
                // h mid ---- w left
                max0 = line0[0] > line0[1] ? line0[0] : line0[1];
                max1 = line1[0] > line1[1] ? line1[0] : line1[1];
                max2 = line2[0] > line2[1] ? line2[0] : line2[1];
                max1 = max1 > max2 ? max1 : max2;
                outptr[0] = max0 > max1 ? max0 : max1;
                outptr++;
                line0++;
                line1++;
                line2++;
                // h mid ---- w mid
                int w = out_w - 1;
                while (w > 0) {
                    vl = vsetvl_e8m1(w);
                    vint8m1_t _line0_0_6, _line0_1_7;
                    vint8m1_t _line1_0_6, _line1_1_7;
                    vint8m1_t _line2_0_6, _line2_1_7;

                    vlseg2e8_v_i8m1(&_line0_0_6, &_line0_1_7, line0, vl);
                    line0 += 2;
                    vint8m1_t _line0_2_8 = vlse8_v_i8m1(line0, 2 * sizeof(int8_t), vl);
                    line0 += (vl - 1) * 2;
                    vint8m1_t _max0 =
                        vmax_vv_i8m1(_line0_2_8, vmax_vv_i8m1(_line0_0_6, _line0_1_7, vl), vl);

                    vlseg2e8_v_i8m1(&_line1_0_6, &_line1_1_7, line1, vl);
                    line1 += 2;
                    vint8m1_t _line1_2_8 = vlse8_v_i8m1(line1, 2 * sizeof(int8_t), vl);
                    line1 += (vl - 1) * 2;
                    vint8m1_t _max1 =
                        vmax_vv_i8m1(_line1_2_8, vmax_vv_i8m1(_line1_0_6, _line1_1_7, vl), vl);

                    vlseg2e8_v_i8m1(&_line2_0_6, &_line2_1_7, line2, vl);
                    line2 += 2;
                    vint8m1_t _line2_2_8 = vlse8_v_i8m1(line2, 2 * sizeof(int8_t), vl);
                    line2 += (vl - 1) * 2;
                    vint8m1_t _max2 =
                        vmax_vv_i8m1(_line2_2_8, vmax_vv_i8m1(_line2_0_6, _line2_1_7, vl), vl);

                    vint8m1_t _max = vmax_vv_i8m1(_max2, vmax_vv_i8m1(_max0, _max1, vl), vl);
                    vse8_v_i8m1(outptr, _max, vl);

                    outptr += vl;
                    w -= vl;
                }
                // h mid ---- w right
                if (extend_w) {
                    max0 = line0[0] > line0[1] ? line0[0] : line0[1];
                    max1 = line1[0] > line1[1] ? line1[0] : line1[1];
                    max2 = line2[0] > line2[1] ? line2[0] : line2[1];
                    max1 = max1 > max2 ? max1 : max2;
                    outptr[0] = max0 > max1 ? max0 : max1;
                    outptr++;
                }
                line0 += in_w + remain_w;
                line1 += in_w + remain_w;
                line2 += in_w + remain_w;
            }

            // h bottom
            if (extend_h) {
                // h bottom ---- w left
                max0 = line0[0] > line0[1] ? line0[0] : line0[1];
                max1 = line1[0] > line1[1] ? line1[0] : line1[1];
                outptr[0] = max0 > max1 ? max0 : max1;
                outptr++;
                line0++;
                line1++;

                // h bottom ---- w mid
                int w = out_w - 1;
                while (w > 0) {
                    vl = vsetvl_e8m1(w);
                    vint8m1_t _line0_0_6, _line0_1_7;
                    vint8m1_t _line1_0_6, _line1_1_7;

                    vlseg2e8_v_i8m1(&_line0_0_6, &_line0_1_7, line0, vl);
                    line0 += 2;
                    vint8m1_t _line0_2_8 = vlse8_v_i8m1(line0, 2 * sizeof(int8_t), vl);
                    line0 += (vl - 1) * 2;
                    vint8m1_t _max0 =
                        vmax_vv_i8m1(_line0_2_8, vmax_vv_i8m1(_line0_0_6, _line0_1_7, vl), vl);

                    vlseg2e8_v_i8m1(&_line1_0_6, &_line1_1_7, line1, vl);
                    line1 += 2;
                    vint8m1_t _line1_2_8 = vlse8_v_i8m1(line1, 2 * sizeof(int8_t), vl);
                    line1 += (vl - 1) * 2;
                    vint8m1_t _max1 =
                        vmax_vv_i8m1(_line1_2_8, vmax_vv_i8m1(_line1_0_6, _line1_1_7, vl), vl);

                    vint8m1_t _max = vmax_vv_i8m1(_max0, _max1, vl);
                    vse8_v_i8m1(outptr, _max, vl);

                    outptr += vl;
                    w -= vl;
                }
                // h bottom ---- w right
                if (extend_w) {
                    max0 = line0[0] > line0[1] ? line0[0] : line0[1];
                    max1 = line1[0] > line1[1] ? line1[0] : line1[1];
                    outptr[0] = max0 > max1 ? max0 : max1;
                    outptr++;
                }
            }
        }
        input_data += input_size;
        output_data += output_size;
    }
    return CSINN_TRUE;
}

int shl_rvv_maxpool3x3s1_p1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
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

    int vl;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const int8_t *line1 = input_data + c * in_h * in_w;
            const int8_t *line2 = line1 + in_w;
            int8_t *outptr = output_data + c * out_h * out_w;
            // h top ---- w left
            int8_t max0 = line1[0] > line1[1] ? line1[0] : line1[1];
            int8_t max1 = line2[0] > line2[1] ? line2[0] : line2[1];
            outptr[0] = max0 > max1 ? max0 : max1;
            outptr++;
            // h top ---- w mid
            int w = out_w - 2;
            while (w > 0) {
                vl = vsetvl_e8m1(w);
                vint8m1_t _line1_0_3 = vle8_v_i8m1(line1, vl);
                line1++;
                vint8m1_t _line1_1_4 = vle8_v_i8m1(line1, vl);
                line1++;
                vint8m1_t _line1_2_5 = vle8_v_i8m1(line1, vl);
                line1 += vl - 2;
                vint8m1_t _max1 =
                    vmax_vv_i8m1(_line1_2_5, vmax_vv_i8m1(_line1_0_3, _line1_1_4, vl), vl);

                vint8m1_t _line2_0_3 = vle8_v_i8m1(line2, vl);
                line2++;
                vint8m1_t _line2_1_4 = vle8_v_i8m1(line2, vl);
                line2++;
                vint8m1_t _line2_2_5 = vle8_v_i8m1(line2, vl);
                line2 += vl - 2;
                vint8m1_t _max2 =
                    vmax_vv_i8m1(_line2_2_5, vmax_vv_i8m1(_line2_0_3, _line2_1_4, vl), vl);

                vint8m1_t _max = vmax_vv_i8m1(_max1, _max2, vl);
                vse8_v_i8m1(outptr, _max, vl);

                outptr += vl;
                w -= vl;
            }
            // h top ---- w right
            max0 = line1[0] > line1[1] ? line1[0] : line1[1];
            max1 = line2[0] > line2[1] ? line2[0] : line2[1];
            outptr[0] = max0 > max1 ? max0 : max1;
            outptr++;
            line1 += 2;  // bump next line: line1 --> line2
            line2 += 2;

            // h mid
            const int8_t *line0 = input_data + c * in_h * in_w;
            int8_t max2 = 0;
            for (int h = 0; h < out_h - 2; h++) {
                // h mid ---- w left
                max0 = line0[0] > line0[1] ? line0[0] : line0[1];
                max1 = line1[0] > line1[1] ? line1[0] : line1[1];
                max2 = line2[0] > line2[1] ? line2[0] : line2[1];
                max1 = max1 > max2 ? max1 : max2;
                outptr[0] = max0 > max1 ? max0 : max1;
                outptr++;
                // h mid ---- w mid
                w = out_w - 2;
                while (w > 0) {
                    vl = vsetvl_e8m1(w);
                    vint8m1_t _line0_0_3 = vle8_v_i8m1(line0, vl);
                    line0++;
                    vint8m1_t _line0_1_4 = vle8_v_i8m1(line0, vl);
                    line0++;
                    vint8m1_t _line0_2_5 = vle8_v_i8m1(line0, vl);
                    line0 += vl - 2;
                    vint8m1_t _max0 =
                        vmax_vv_i8m1(_line0_2_5, vmax_vv_i8m1(_line0_0_3, _line0_1_4, vl), vl);

                    vint8m1_t _line1_0_3 = vle8_v_i8m1(line1, vl);
                    line1++;
                    vint8m1_t _line1_1_4 = vle8_v_i8m1(line1, vl);
                    line1++;
                    vint8m1_t _line1_2_5 = vle8_v_i8m1(line1, vl);
                    line1 += vl - 2;
                    vint8m1_t _max1 =
                        vmax_vv_i8m1(_line1_2_5, vmax_vv_i8m1(_line1_0_3, _line1_1_4, vl), vl);

                    vint8m1_t _line2_0_3 = vle8_v_i8m1(line2, vl);
                    line2++;
                    vint8m1_t _line2_1_4 = vle8_v_i8m1(line2, vl);
                    line2++;
                    vint8m1_t _line2_2_5 = vle8_v_i8m1(line2, vl);
                    line2 += vl - 2;
                    vint8m1_t _max2 =
                        vmax_vv_i8m1(_line2_2_5, vmax_vv_i8m1(_line2_0_3, _line2_1_4, vl), vl);

                    vint8m1_t _max = vmax_vv_i8m1(_max2, vmax_vv_i8m1(_max0, _max1, vl), vl);
                    vse8_v_i8m1(outptr, _max, vl);

                    outptr += vl;
                    w -= vl;
                }
                // h mid ---- w right
                max0 = line0[0] > line0[1] ? line0[0] : line0[1];
                max1 = line1[0] > line1[1] ? line1[0] : line1[1];
                max2 = line2[0] > line2[1] ? line2[0] : line2[1];
                max1 = max1 > max2 ? max1 : max2;
                outptr[0] = max0 > max1 ? max0 : max1;
                outptr++;
                line0 += 2;
                line1 += 2;
                line2 += 2;
            }

            // h bottom ---- w left
            max0 = line0[0] > line0[1] ? line0[0] : line0[1];
            max1 = line1[0] > line1[1] ? line1[0] : line1[1];
            outptr[0] = max0 > max1 ? max0 : max1;
            outptr++;
            // h bottom ---- w mid
            w = out_w - 2;
            while (w > 0) {
                vl = vsetvl_e8m1(w);
                vint8m1_t _line0_0_3 = vle8_v_i8m1(line0, vl);
                line0++;
                vint8m1_t _line0_1_4 = vle8_v_i8m1(line0, vl);
                line0++;
                vint8m1_t _line0_2_5 = vle8_v_i8m1(line0, vl);
                line0 += vl - 2;
                vint8m1_t _max0 =
                    vmax_vv_i8m1(_line0_2_5, vmax_vv_i8m1(_line0_0_3, _line0_1_4, vl), vl);

                vint8m1_t _line1_0_3 = vle8_v_i8m1(line1, vl);
                line1++;
                vint8m1_t _line1_1_4 = vle8_v_i8m1(line1, vl);
                line1++;
                vint8m1_t _line1_2_5 = vle8_v_i8m1(line1, vl);
                line1 += vl - 2;
                vint8m1_t _max1 =
                    vmax_vv_i8m1(_line1_2_5, vmax_vv_i8m1(_line1_0_3, _line1_1_4, vl), vl);

                vint8m1_t _max = vmax_vv_i8m1(_max0, _max1, vl);
                vse8_v_i8m1(outptr, _max, vl);

                outptr += vl;
                w -= vl;
            }
            // h bottom ---- w right
            max0 = line0[0] > line0[1] ? line0[0] : line0[1];
            max1 = line1[0] > line1[1] ? line1[0] : line1[1];
            outptr[0] = max0 > max1 ? max0 : max1;
        }
        input_data += input_size;
        output_data += output_size;
    }
    return CSINN_TRUE;
}
