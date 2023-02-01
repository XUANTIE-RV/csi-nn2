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
    note: VLEN = 128/256
*************************************************************/
/*
    pad_left = pad_top = 0
    pad_right = 0 or 1
    pad_down = 0 or 1
*/
int shl_rvv_avgpool3x3s2_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
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

    int remain_w = in_w - 2 * out_w;
    float ratio = 0.11111111f;
    int vl;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const float *line0 = input_data + c * in_h * in_w;
            const float *line1 = line0 + in_w;
            const float *line2 = line1 + in_w;
            float *outptr = output_data + c * out_hw;

            for (int h = 0; h < out_h; h++) {
                ratio = 0.11111111f;
                int w = out_w;
                while (w > 0) {
                    vl = vsetvl_e32m1(w);
                    vfloat32m1_t _line0_0_6, _line0_1_7;
                    vfloat32m1_t _line1_0_6, _line1_1_7;
                    vfloat32m1_t _line2_0_6, _line2_1_7;

                    vlseg2e32_v_f32m1(&_line0_0_6, &_line0_1_7, line0, vl);
                    line0 += 2;
                    vfloat32m1_t _line0_2_8 = vlse32_v_f32m1(line0, 2 * sizeof(float), vl);
                    line0 += (vl - 1) * 2;
                    vfloat32m1_t _sum0 =
                        vfadd_vv_f32m1(_line0_2_8, vfadd_vv_f32m1(_line0_0_6, _line0_1_7, vl), vl);

                    vlseg2e32_v_f32m1(&_line1_0_6, &_line1_1_7, line1, vl);
                    line1 += 2;
                    vfloat32m1_t _line1_2_8 = vlse32_v_f32m1(line1, 2 * sizeof(float), vl);
                    line1 += (vl - 1) * 2;
                    vfloat32m1_t _sum1 =
                        vfadd_vv_f32m1(_line1_2_8, vfadd_vv_f32m1(_line1_0_6, _line1_1_7, vl), vl);

                    vlseg2e32_v_f32m1(&_line2_0_6, &_line2_1_7, line2, vl);
                    line2 += 2;
                    vfloat32m1_t _line2_2_8 = vlse32_v_f32m1(line2, 2 * sizeof(float), vl);
                    line2 += (vl - 1) * 2;
                    vfloat32m1_t _sum2 =
                        vfadd_vv_f32m1(_line2_2_8, vfadd_vv_f32m1(_line2_0_6, _line2_1_7, vl), vl);

                    vfloat32m1_t _sum = vfadd_vv_f32m1(_sum2, vfadd_vv_f32m1(_sum0, _sum1, vl), vl);
                    vfloat32m1_t _avg = vfmul_vf_f32m1(_sum, ratio, vl);
                    vse32_v_f32m1(outptr, _avg, vl);

                    outptr += vl;
                    w -= vl;
                }

                if (extend_w) {
                    ratio = (params->count_include_pad) ? 0.11111111f : 0.16666667f;
                    outptr[0] =
                        (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * ratio;
                    outptr++;
                }
                line0 += remain_w + in_w;
                line1 += remain_w + in_w;
                line2 += remain_w + in_w;
            }
            if (extend_h) {
                ratio = (params->count_include_pad) ? 0.11111111f : 0.16666667f;
                int w = out_w;
                while (w > 0) {
                    vl = vsetvl_e32m1(w);
                    vfloat32m1_t _line0_0_6, _line0_1_7;
                    vfloat32m1_t _line1_0_6, _line1_1_7;

                    vlseg2e32_v_f32m1(&_line0_0_6, &_line0_1_7, line0, vl);
                    line0 += 2;
                    vfloat32m1_t _line0_2_8 = vlse32_v_f32m1(line0, 2 * sizeof(float), vl);
                    line0 += (vl - 1) * 2;
                    vfloat32m1_t _sum0 =
                        vfadd_vv_f32m1(_line0_2_8, vfadd_vv_f32m1(_line0_0_6, _line0_1_7, vl), vl);

                    vlseg2e32_v_f32m1(&_line1_0_6, &_line1_1_7, line1, vl);
                    line1 += 2;
                    vfloat32m1_t _line1_2_8 = vlse32_v_f32m1(line1, 2 * sizeof(float), vl);
                    line1 += (vl - 1) * 2;
                    vfloat32m1_t _sum1 =
                        vfadd_vv_f32m1(_line1_2_8, vfadd_vv_f32m1(_line1_0_6, _line1_1_7, vl), vl);

                    vfloat32m1_t _sum = vfadd_vv_f32m1(_sum0, _sum1, vl);
                    vfloat32m1_t _avg = vfmul_vf_f32m1(_sum, ratio, vl);
                    vse32_v_f32m1(outptr, _avg, vl);

                    outptr += vl;
                    w -= vl;
                }
                if (extend_w) {
                    ratio = (params->count_include_pad) ? 0.11111111f : 0.25f;
                    outptr[0] = (line0[0] + line0[1] + line1[0] + line1[1]) * ratio;
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
int shl_rvv_avgpool3x3s2_p1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
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

    int remain_w = in_w - 2 * out_w + 1;
    float ratio = 0.11111111f;
    int vl;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const float *line0 = input_data + c * in_h * in_w;
            const float *line1 = line0 + in_w;
            float *outptr = output_data + c * out_hw;

            // h top ---- w left
            ratio = (params->count_include_pad) ? 0.11111111f : 0.25f;
            outptr[0] = (line0[0] + line0[1] + line1[0] + line1[1]) * ratio;
            outptr++;
            line0++;
            line1++;
            // h top ---- w mid
            ratio = (params->count_include_pad) ? 0.11111111f : 0.16666667f;
            int w = out_w - 1;
            while (w > 0) {
                vl = vsetvl_e32m1(w);
                vfloat32m1_t _line0_0_6, _line0_1_7;
                vfloat32m1_t _line1_0_6, _line1_1_7;

                vlseg2e32_v_f32m1(&_line0_0_6, &_line0_1_7, line0, vl);
                line0 += 2;
                vfloat32m1_t _line0_2_8 = vlse32_v_f32m1(line0, 2 * sizeof(float), vl);
                line0 += (vl - 1) * 2;
                vfloat32m1_t _sum0 =
                    vfadd_vv_f32m1(_line0_2_8, vfadd_vv_f32m1(_line0_0_6, _line0_1_7, vl), vl);

                vlseg2e32_v_f32m1(&_line1_0_6, &_line1_1_7, line1, vl);
                line1 += 2;
                vfloat32m1_t _line1_2_8 = vlse32_v_f32m1(line1, 2 * sizeof(float), vl);
                line1 += (vl - 1) * 2;
                vfloat32m1_t _sum1 =
                    vfadd_vv_f32m1(_line1_2_8, vfadd_vv_f32m1(_line1_0_6, _line1_1_7, vl), vl);

                vfloat32m1_t _sum = vfadd_vv_f32m1(_sum0, _sum1, vl);
                vfloat32m1_t _avg = vfmul_vf_f32m1(_sum, ratio, vl);
                vse32_v_f32m1(outptr, _avg, vl);

                outptr += vl;
                w -= vl;
            }

            // h top ---- w right
            ratio = (params->count_include_pad) ? 0.11111111f : 0.25f;
            if (extend_w) {
                outptr[0] = (line0[0] + line0[1] + line1[0] + line1[1]) * ratio;
                outptr++;
            }
            line0 += remain_w;
            line1 += remain_w;

            // h mid
            const float *line2 = line1 + in_w;
            for (int h = 0; h < out_h - 1; h++) {
                // h mid ---- w left
                ratio = (params->count_include_pad) ? 0.11111111f : 0.16666667f;
                outptr[0] =
                    (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * ratio;
                outptr++;
                line0++;
                line1++;
                line2++;
                // h mid ---- w mid
                ratio = 0.11111111f;
                int w = out_w - 1;
                while (w > 0) {
                    vl = vsetvl_e32m1(w);
                    vfloat32m1_t _line0_0_6, _line0_1_7;
                    vfloat32m1_t _line1_0_6, _line1_1_7;
                    vfloat32m1_t _line2_0_6, _line2_1_7;

                    vlseg2e32_v_f32m1(&_line0_0_6, &_line0_1_7, line0, vl);
                    line0 += 2;
                    vfloat32m1_t _line0_2_8 = vlse32_v_f32m1(line0, 2 * sizeof(float), vl);
                    line0 += (vl - 1) * 2;
                    vfloat32m1_t _sum0 =
                        vfadd_vv_f32m1(_line0_2_8, vfadd_vv_f32m1(_line0_0_6, _line0_1_7, vl), vl);

                    vlseg2e32_v_f32m1(&_line1_0_6, &_line1_1_7, line1, vl);
                    line1 += 2;
                    vfloat32m1_t _line1_2_8 = vlse32_v_f32m1(line1, 2 * sizeof(float), vl);
                    line1 += (vl - 1) * 2;
                    vfloat32m1_t _sum1 =
                        vfadd_vv_f32m1(_line1_2_8, vfadd_vv_f32m1(_line1_0_6, _line1_1_7, vl), vl);

                    vlseg2e32_v_f32m1(&_line2_0_6, &_line2_1_7, line2, vl);
                    line2 += 2;
                    vfloat32m1_t _line2_2_8 = vlse32_v_f32m1(line2, 2 * sizeof(float), vl);
                    line2 += (vl - 1) * 2;
                    vfloat32m1_t _sum2 =
                        vfadd_vv_f32m1(_line2_2_8, vfadd_vv_f32m1(_line2_0_6, _line2_1_7, vl), vl);

                    vfloat32m1_t _sum = vfadd_vv_f32m1(_sum2, vfadd_vv_f32m1(_sum0, _sum1, vl), vl);
                    vfloat32m1_t _avg = vfmul_vf_f32m1(_sum, ratio, vl);
                    vse32_v_f32m1(outptr, _avg, vl);

                    outptr += vl;
                    w -= vl;
                }

                // h mid ---- w right
                ratio = (params->count_include_pad) ? 0.11111111f : 0.16666667f;
                if (extend_w) {
                    outptr[0] =
                        (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * ratio;
                    outptr++;
                }
                line0 += in_w + remain_w;
                line1 += in_w + remain_w;
                line2 += in_w + remain_w;
            }

            // h bottom
            if (extend_h) {
                // h bottom ---- w left
                ratio = (params->count_include_pad) ? 0.11111111f : 0.25f;
                outptr[0] = (line0[0] + line0[1] + line1[0] + line1[1]) * ratio;
                outptr++;
                line0++;
                line1++;

                // h bottom ---- w mid
                ratio = (params->count_include_pad) ? 0.11111111f : 0.16666667f;
                int w = out_w - 1;
                while (w > 0) {
                    vl = vsetvl_e32m1(w);
                    vfloat32m1_t _line0_0_6, _line0_1_7;
                    vfloat32m1_t _line1_0_6, _line1_1_7;

                    vlseg2e32_v_f32m1(&_line0_0_6, &_line0_1_7, line0, vl);
                    line0 += 2;
                    vfloat32m1_t _line0_2_8 = vlse32_v_f32m1(line0, 2 * sizeof(float), vl);
                    line0 += (vl - 1) * 2;
                    vfloat32m1_t _sum0 =
                        vfadd_vv_f32m1(_line0_2_8, vfadd_vv_f32m1(_line0_0_6, _line0_1_7, vl), vl);

                    vlseg2e32_v_f32m1(&_line1_0_6, &_line1_1_7, line1, vl);
                    line1 += 2;
                    vfloat32m1_t _line1_2_8 = vlse32_v_f32m1(line1, 2 * sizeof(float), vl);
                    line1 += (vl - 1) * 2;
                    vfloat32m1_t _sum1 =
                        vfadd_vv_f32m1(_line1_2_8, vfadd_vv_f32m1(_line1_0_6, _line1_1_7, vl), vl);

                    vfloat32m1_t _sum = vfadd_vv_f32m1(_sum0, _sum1, vl);
                    vfloat32m1_t _avg = vfmul_vf_f32m1(_sum, ratio, vl);
                    vse32_v_f32m1(outptr, _avg, vl);

                    outptr += vl;
                    w -= vl;
                }
                // h bottom ---- w right
                ratio = (params->count_include_pad) ? 0.11111111f : 0.25f;
                if (extend_w) {
                    outptr[0] = (line0[0] + line0[1] + line1[0] + line1[1]) * ratio;
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
    pad_left = pad_right = pad_top = pad_down = 1
    in_w = out_w   in_h = out_h
*/
int shl_rvv_avgpool3x3s1_p1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
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
    int output_size = in_c * out_h * out_w;

    float ratio = 0.11111111f;
    int vl;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const float *line1 = input_data + c * in_h * in_w;
            const float *line2 = line1 + in_w;
            float *outptr = output_data + c * out_h * out_w;
            // h top ---- w left
            ratio = (params->count_include_pad) ? 0.11111111f : 0.25f;
            outptr[0] = (line1[0] + line1[1] + line2[0] + line2[1]) * ratio;
            outptr++;
            // h top ---- w mid
            ratio = (params->count_include_pad) ? 0.11111111f : 0.16666667f;
            int w = out_w - 2;
            while (w > 0) {
                vl = vsetvl_e32m1(w);
                vfloat32m1_t _line1_0_3 = vle32_v_f32m1(line1, vl);
                line1++;
                vfloat32m1_t _line1_1_4 = vle32_v_f32m1(line1, vl);
                line1++;
                vfloat32m1_t _line1_2_5 = vle32_v_f32m1(line1, vl);
                line1 += vl - 2;
                vfloat32m1_t _sum1 =
                    vfadd_vv_f32m1(_line1_2_5, vfadd_vv_f32m1(_line1_0_3, _line1_1_4, vl), vl);

                vfloat32m1_t _line2_0_3 = vle32_v_f32m1(line2, vl);
                line2++;
                vfloat32m1_t _line2_1_4 = vle32_v_f32m1(line2, vl);
                line2++;
                vfloat32m1_t _line2_2_5 = vle32_v_f32m1(line2, vl);
                line2 += vl - 2;
                vfloat32m1_t _sum2 =
                    vfadd_vv_f32m1(_line2_2_5, vfadd_vv_f32m1(_line2_0_3, _line2_1_4, vl), vl);

                vfloat32m1_t _sum = vfadd_vv_f32m1(_sum1, _sum2, vl);
                vfloat32m1_t _avg = vfmul_vf_f32m1(_sum, ratio, vl);
                vse32_v_f32m1(outptr, _avg, vl);

                outptr += vl;
                w -= vl;
            }
            // h top ---- w right
            ratio = (params->count_include_pad) ? 0.11111111f : 0.25f;
            outptr[0] = (line1[0] + line1[1] + line2[0] + line2[1]) * ratio;
            outptr++;
            line1 += 2;  // bump next line: line1 --> line2
            line2 += 2;

            // h mid
            const float *line0 = input_data + c * in_h * in_w;
            for (int h = 0; h < out_h - 2; h++) {
                // h mid ---- w left
                ratio = (params->count_include_pad) ? 0.11111111f : 0.16666667f;
                outptr[0] =
                    (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * ratio;
                outptr++;
                // h mid ---- w mid
                ratio = 0.11111111f;
                w = out_w - 2;
                while (w > 0) {
                    vl = vsetvl_e32m1(w);
                    vfloat32m1_t _line0_0_3 = vle32_v_f32m1(line0, vl);
                    line0++;
                    vfloat32m1_t _line0_1_4 = vle32_v_f32m1(line0, vl);
                    line0++;
                    vfloat32m1_t _line0_2_5 = vle32_v_f32m1(line0, vl);
                    line0 += vl - 2;
                    vfloat32m1_t _sum0 =
                        vfadd_vv_f32m1(_line0_2_5, vfadd_vv_f32m1(_line0_0_3, _line0_1_4, vl), vl);

                    vfloat32m1_t _line1_0_3 = vle32_v_f32m1(line1, vl);
                    line1++;
                    vfloat32m1_t _line1_1_4 = vle32_v_f32m1(line1, vl);
                    line1++;
                    vfloat32m1_t _line1_2_5 = vle32_v_f32m1(line1, vl);
                    line1 += vl - 2;
                    vfloat32m1_t _sum1 =
                        vfadd_vv_f32m1(_line1_2_5, vfadd_vv_f32m1(_line1_0_3, _line1_1_4, vl), vl);

                    vfloat32m1_t _line2_0_3 = vle32_v_f32m1(line2, vl);
                    line2++;
                    vfloat32m1_t _line2_1_4 = vle32_v_f32m1(line2, vl);
                    line2++;
                    vfloat32m1_t _line2_2_5 = vle32_v_f32m1(line2, vl);
                    line2 += vl - 2;
                    vfloat32m1_t _sum2 =
                        vfadd_vv_f32m1(_line2_2_5, vfadd_vv_f32m1(_line2_0_3, _line2_1_4, vl), vl);

                    vfloat32m1_t _sum = vfadd_vv_f32m1(_sum2, vfadd_vv_f32m1(_sum0, _sum1, vl), vl);
                    vfloat32m1_t _avg = vfmul_vf_f32m1(_sum, ratio, vl);
                    vse32_v_f32m1(outptr, _avg, vl);

                    outptr += vl;
                    w -= vl;
                }
                // h mid ---- w right
                ratio = (params->count_include_pad) ? 0.11111111f : 0.16666667f;
                outptr[0] =
                    (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * ratio;
                outptr++;
                line0 += 2;
                line1 += 2;
                line2 += 2;
            }

            // h bottom ---- w left
            ratio = (params->count_include_pad) ? 0.11111111f : 0.25f;
            outptr[0] = (line0[0] + line0[1] + line1[0] + line1[1]) * ratio;
            outptr++;
            // h bottom ---- w mid
            ratio = (params->count_include_pad) ? 0.11111111f : 0.16666667f;
            w = out_w - 2;
            while (w > 0) {
                vl = vsetvl_e32m1(w);
                vfloat32m1_t _line0_0_3 = vle32_v_f32m1(line0, vl);
                line0++;
                vfloat32m1_t _line0_1_4 = vle32_v_f32m1(line0, vl);
                line0++;
                vfloat32m1_t _line0_2_5 = vle32_v_f32m1(line0, vl);
                line0 += vl - 2;
                vfloat32m1_t _sum0 =
                    vfadd_vv_f32m1(_line0_2_5, vfadd_vv_f32m1(_line0_0_3, _line0_1_4, vl), vl);

                vfloat32m1_t _line1_0_3 = vle32_v_f32m1(line1, vl);
                line1++;
                vfloat32m1_t _line1_1_4 = vle32_v_f32m1(line1, vl);
                line1++;
                vfloat32m1_t _line1_2_5 = vle32_v_f32m1(line1, vl);
                line1 += vl - 2;
                vfloat32m1_t _sum1 =
                    vfadd_vv_f32m1(_line1_2_5, vfadd_vv_f32m1(_line1_0_3, _line1_1_4, vl), vl);

                vfloat32m1_t _sum = vfadd_vv_f32m1(_sum0, _sum1, vl);
                vfloat32m1_t _avg = vfmul_vf_f32m1(_sum, ratio, vl);
                vse32_v_f32m1(outptr, _avg, vl);

                outptr += vl;
                w -= vl;
            }
            // h bottom ---- w right
            ratio = (params->count_include_pad) ? 0.11111111f : 0.25f;
            outptr[0] = (line0[0] + line0[1] + line1[0] + line1[1]) * ratio;
        }
        input_data += input_size;
        output_data += output_size;
    }
    return CSINN_TRUE;
}
