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

/*************************************************************
    note: VLEN = 128/256
*************************************************************/
/*
    pad_left = pad_top = 0
    pad_right = 0 or 1
    pad_down = 0 or 1
*/
int shl_rvv_maxpool2x2s2_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

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

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const float *line0 = input_data + c * in_h * in_w;
            const float *line1 = line0 + in_w;
            float *outptr = output_data + c * out_hw;

            for (int h = 0; h < out_h; h++) {
                int w = out_w;
                while (w > 0) {
                    vl = vsetvl_e32m1(w);
                    vfloat32m1_t _line0_0_6, _line0_1_7;
                    vfloat32m1_t _line1_0_6, _line1_1_7;

                    vlseg2e32_v_f32m1(&_line0_0_6, &_line0_1_7, line0, vl);
                    vlseg2e32_v_f32m1(&_line1_0_6, &_line1_1_7, line1, vl);

                    vfloat32m1_t _max0 = vfmax_vv_f32m1(_line0_0_6, _line0_1_7, vl);
                    vfloat32m1_t _max1 = vfmax_vv_f32m1(_line1_0_6, _line1_1_7, vl);
                    vfloat32m1_t _max = vfmax_vv_f32m1(_max0, _max1, vl);

                    vse32_v_f32m1(outptr, _max, vl);
                    line0 += 2 * vl;
                    line1 += 2 * vl;
                    outptr += vl;
                    w -= vl;
                }
                if (extend_w) {
                    outptr[0] = fmax(fmax(line0[0], line1[0]), 0.0f);
                    outptr++;
                }
                line0 += remain_w + in_w;
                line1 += remain_w + in_w;
            }
            if (extend_h) {
                int w = out_w;
                while (w > 0) {
                    vl = vsetvl_e32m1(w);
                    vfloat32m1_t _line0_0_6, _line0_1_7;

                    vlseg2e32_v_f32m1(&_line0_0_6, &_line0_1_7, line0, vl);

                    vfloat32m1_t _max0 = vfmax_vv_f32m1(_line0_0_6, _line0_1_7, vl);
                    vfloat32m1_t _max = vfmax_vf_f32m1(_max0, 0.0f, vl);

                    vse32_v_f32m1(outptr, _max, vl);
                    line0 += 2 * vl;
                    outptr += vl;
                    w -= vl;
                }

                if (extend_w) {
                    outptr[0] = fmax(line0[0], 0.0f);
                    outptr++;
                }
            }
        }
        input_data += input_size;
        output_data += output_size;
    }
    return CSINN_TRUE;
}

/*
    pad_left = pad_top = 1
    pad_right = 0 or 1
    pad_down = 0 or 1
*/
int shl_rvv_maxpool2x2s2_p1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

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

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const float *line00 = input_data + c * in_h * in_w;
            float *outptr = output_data + c * out_hw;

            // h top ---- w left
            outptr[0] = fmax(line00[0], 0.0f);
            outptr++;
            line00++;
            // h top ---- w mid
            int w = out_w - 1;
            while (w > 0) {
                vl = vsetvl_e32m1(w);
                vfloat32m1_t _line0_0_6, _line0_1_7;
                vlseg2e32_v_f32m1(&_line0_0_6, &_line0_1_7, line00, vl);
                vfloat32m1_t _max0 = vfmax_vv_f32m1(_line0_0_6, _line0_1_7, vl);
                vfloat32m1_t _max = vfmax_vf_f32m1(_max0, 0.0f, vl);
                vse32_v_f32m1(outptr, _max, vl);
                line00 += 2 * vl;
                outptr += vl;
                w -= vl;
            }
            // h top ---- w right
            if (extend_w) {
                outptr[0] = fmax(line00[0], 0.0f);
                outptr++;
            }
            line00 += remain_w;

            // h mid
            const float *line0 = line00;
            const float *line1 = line0 + in_w;
            for (int h = 0; h < out_h - 1; h++) {
                // h mid ---- w left
                outptr[0] = fmax(fmax(line0[0], line1[0]), 0.0f);
                outptr++;
                line0++;
                line1++;
                // h mid ---- w mid
                w = out_w - 1;
                while (w > 0) {
                    vl = vsetvl_e32m1(w);
                    vfloat32m1_t _line0_0_6, _line0_1_7;
                    vfloat32m1_t _line1_0_6, _line1_1_7;

                    vlseg2e32_v_f32m1(&_line0_0_6, &_line0_1_7, line0, vl);
                    vlseg2e32_v_f32m1(&_line1_0_6, &_line1_1_7, line1, vl);

                    vfloat32m1_t _max0 = vfmax_vv_f32m1(_line0_0_6, _line0_1_7, vl);
                    vfloat32m1_t _max1 = vfmax_vv_f32m1(_line1_0_6, _line1_1_7, vl);
                    vfloat32m1_t _max = vfmax_vv_f32m1(_max0, _max1, vl);

                    vse32_v_f32m1(outptr, _max, vl);
                    line0 += 2 * vl;
                    line1 += 2 * vl;
                    outptr += vl;
                    w -= vl;
                }

                // h mid ---- w right
                if (extend_w) {
                    outptr[0] = fmax(fmax(line0[0], line1[0]), 0.0f);
                    outptr++;
                }
                line0 += remain_w + in_w;
                line1 += remain_w + in_w;
            }
            // h bottom
            if (extend_h) {
                // h bottom ---- w left
                outptr[0] = fmax(line0[0], 0.0f);
                outptr++;
                line0++;
                // h bottom ---- w mid
                w = out_w - 1;
                while (w > 0) {
                    vl = vsetvl_e32m1(w);
                    vfloat32m1_t _line0_0_6, _line0_1_7;

                    vlseg2e32_v_f32m1(&_line0_0_6, &_line0_1_7, line0, vl);

                    vfloat32m1_t _max0 = vfmax_vv_f32m1(_line0_0_6, _line0_1_7, vl);
                    vfloat32m1_t _max = vfmax_vf_f32m1(_max0, 0.0f, vl);

                    vse32_v_f32m1(outptr, _max, vl);
                    line0 += 2 * vl;
                    outptr += vl;
                    w -= vl;
                }
                // h bottom ---- w right
                if (extend_w) {
                    outptr[0] = fmax(line0[0], 0.0f);
                }
            }
        }
        input_data += input_size;
        output_data += output_size;
    }
    return CSINN_TRUE;
}
