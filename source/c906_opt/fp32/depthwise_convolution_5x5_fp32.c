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

#include "shl_c906.h"

#ifndef DWCONV5X5S1
#define DWCONV5X5S1 shl_c906_dwconv5x5s1
#endif

#ifndef DWCONV5X5S2
#define DWCONV5X5S2 shl_c906_dwconv5x5s2
#endif

/*
    TODO: support channel mult ??
          rvv optimization
*/
int DWCONV5X5S1(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                struct csinn_conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];  // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    float *input_padd_buf =
        (float *)shl_mem_alloc(in_c * (in_h + params->pad_top + params->pad_down) *
                               (in_w + params->pad_left + params->pad_right) * sizeof(float));

    shl_c906_pad_input(
        input_data, input_padd_buf, in_c, in_h, in_w, in_h + params->pad_top + params->pad_down,
        in_w + params->pad_left + params->pad_right, params->pad_top, params->pad_left);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

#pragma omp parallel for num_threads(1)
    for (int c = 0; c < in_c; c++) {
        float *out = output_data + c * out_h * out_w;
        float *outptr0 = out;
        float *outptr1 = outptr0 + out_w;

        const float bias0 = bias_data ? bias_data[c] : 0.0f;

        const float *img0 = input_padd_buf + c * in_h * in_w;
        const float *r0 = img0;
        const float *r1 = r0 + in_w;
        const float *r2 = r1 + in_w;
        const float *r3 = r2 + in_w;
        const float *r4 = r3 + in_w;
        const float *r5 = r4 + in_w;

        const float *kernel0 = kernel_data + c * 25;
        const float *k0 = kernel0;
        const float *k1 = k0 + 5;
        const float *k2 = k1 + 5;
        const float *k3 = k2 + 5;
        const float *k4 = k3 + 5;

        int h = 0;
        for (; h + 1 < out_h; h += 2) {
            for (int w = 0; w < out_w; w++) {
                float sum0 = bias0;
                float sum1 = bias0;

                sum0 +=
                    r0[0] * k0[0] + r0[1] * k0[1] + r0[2] * k0[2] + r0[3] * k0[3] + r0[4] * k0[4];

                sum0 +=
                    r1[0] * k1[0] + r1[1] * k1[1] + r1[2] * k1[2] + r1[3] * k1[3] + r1[4] * k1[4];
                sum1 +=
                    r1[0] * k0[0] + r1[1] * k0[1] + r1[2] * k0[2] + r1[3] * k0[3] + r1[4] * k0[4];

                sum0 +=
                    r2[0] * k2[0] + r2[1] * k2[1] + r2[2] * k2[2] + r2[3] * k2[3] + r2[4] * k2[4];
                sum1 +=
                    r2[0] * k1[0] + r2[1] * k1[1] + r2[2] * k1[2] + r2[3] * k1[3] + r2[4] * k1[4];

                sum0 +=
                    r3[0] * k3[0] + r3[1] * k3[1] + r3[2] * k3[2] + r3[3] * k3[3] + r3[4] * k3[4];
                sum1 +=
                    r3[0] * k2[0] + r3[1] * k2[1] + r3[2] * k2[2] + r3[3] * k2[3] + r3[4] * k2[4];

                sum0 +=
                    r4[0] * k4[0] + r4[1] * k4[1] + r4[2] * k4[2] + r4[3] * k4[3] + r4[4] * k4[4];
                sum1 +=
                    r4[0] * k3[0] + r4[1] * k3[1] + r4[2] * k3[2] + r4[3] * k3[3] + r4[4] * k3[4];

                sum1 +=
                    r5[0] * k4[0] + r5[1] * k4[1] + r5[2] * k4[2] + r5[3] * k4[3] + r5[4] * k4[4];

#ifdef FUSE_CONV_RELU
                sum0 = sum0 > 0 ? sum0 : 0;
                sum1 = sum1 > 0 ? sum1 : 0;
#endif  // FUSE_CONV_RELU

                *outptr0 = sum0;
                *outptr1 = sum1;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                outptr0++;
                outptr1++;
            }
            r0 += 4 + in_w;  // jump to next line
            r1 += 4 + in_w;
            r2 += 4 + in_w;
            r3 += 4 + in_w;
            r4 += 4 + in_w;
            r5 += 4 + in_w;

            outptr0 += out_w;
            outptr1 += out_w;
        }

        for (; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                float sum0 = bias0;
                sum0 +=
                    r0[0] * k0[0] + r0[1] * k0[1] + r0[2] * k0[2] + r0[3] * k0[3] + r0[4] * k0[4];
                sum0 +=
                    r1[0] * k1[0] + r1[1] * k1[1] + r1[2] * k1[2] + r1[3] * k1[3] + r1[4] * k1[4];
                sum0 +=
                    r2[0] * k2[0] + r2[1] * k2[1] + r2[2] * k2[2] + r2[3] * k2[3] + r2[4] * k2[4];
                sum0 +=
                    r3[0] * k3[0] + r3[1] * k3[1] + r3[2] * k3[2] + r3[3] * k3[3] + r3[4] * k3[4];
                sum0 +=
                    r4[0] * k4[0] + r4[1] * k4[1] + r4[2] * k4[2] + r4[3] * k4[3] + r4[4] * k4[4];

#ifdef FUSE_CONV_RELU
                sum0 = sum0 > 0 ? sum0 : 0;
#endif  // FUSE_CONV_RELU

                *outptr0 = sum0;
                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                outptr0++;
            }

            r0 += 4;
            r1 += 4;
            r2 += 4;
            r3 += 4;
            r4 += 4;
        }
    }

    shl_mem_free(input_padd_buf);
    return CSINN_TRUE;
}

/*
    TODO: support channel mult ??
          rvv optimization
*/

int DWCONV5X5S2(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                struct csinn_conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];  // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    float *input_padd_buf =
        (float *)shl_mem_alloc(in_c * (in_h + params->pad_top + params->pad_down) *
                               (in_w + params->pad_left + params->pad_right) * sizeof(float));

    shl_c906_pad_input(
        input_data, input_padd_buf, in_c, in_h, in_w, in_h + params->pad_top + params->pad_down,
        in_w + params->pad_left + params->pad_right, params->pad_top, params->pad_left);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

    const int tailstep = in_w - 2 * out_w + in_w;

#pragma omp parallel for num_threads(1)
    for (int c = 0; c < in_c; c++) {
        float *out = output_data + c * out_h * out_w;
        float *outptr0 = out;
        float *outptr1 = outptr0 + out_w;

        const float bias0 = bias_data ? bias_data[c] : 0.0f;

        const float *img0 = input_padd_buf + c * in_h * in_w;
        const float *r0 = img0;
        const float *r1 = r0 + in_w;
        const float *r2 = r1 + in_w;
        const float *r3 = r2 + in_w;
        const float *r4 = r3 + in_w;

        const float *kernel0 = kernel_data + c * 25;
        const float *k0 = kernel0;
        const float *k1 = k0 + 5;
        const float *k2 = k1 + 5;
        const float *k3 = k2 + 5;
        const float *k4 = k3 + 5;

        int h = 0;
        for (; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                float sum0 = bias0;

                sum0 +=
                    r0[0] * k0[0] + r0[1] * k0[1] + r0[2] * k0[2] + r0[3] * k0[3] + r0[4] * k0[4];
                sum0 +=
                    r1[0] * k1[0] + r1[1] * k1[1] + r1[2] * k1[2] + r1[3] * k1[3] + r1[4] * k1[4];
                sum0 +=
                    r2[0] * k2[0] + r2[1] * k2[1] + r2[2] * k2[2] + r2[3] * k2[3] + r2[4] * k2[4];
                sum0 +=
                    r3[0] * k3[0] + r3[1] * k3[1] + r3[2] * k3[2] + r3[3] * k3[3] + r3[4] * k3[4];
                sum0 +=
                    r4[0] * k4[0] + r4[1] * k4[1] + r4[2] * k4[2] + r4[3] * k4[3] + r4[4] * k4[4];

#ifdef FUSE_CONV_RELU
                sum0 = sum0 > 0 ? sum0 : 0;
#endif  // FUSE_CONV_RELU

                *outptr0 = sum0;

                r0 += 2;
                r1 += 2;
                r2 += 2;
                r3 += 2;
                r4 += 2;
                outptr0++;
            }
            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
            r3 += tailstep;
            r4 += tailstep;
        }
    }

    shl_mem_free(input_padd_buf);
    return CSINN_TRUE;
}
