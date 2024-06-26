/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

#include "c906/c906.h"

/*
    pad_left = pad_top = 0
    pad_right = 0 or 1
    pad_down = 0 or 1
*/
static int maxpool2x2s2(struct csinn_tensor *input, struct csinn_tensor *output,
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

    int out_w4 = out_w >> 2;
    int remain_w = in_w - 2 * out_w;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const float *line0 = input_data + c * in_h * in_w;
            const float *line1 = line0 + in_w;
            float *outptr = output_data + c * out_hw;

            for (int h = 0; h < out_h; h++) {
#if __riscv_vector == 128
                if (out_w4 > 0) {
                    // execution delay cycles for vlseg2e: >= 2 + 2
                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "mv             t0, %3\n\t"
                        "1:\n\t"
                        "vlseg2e.v      v0, (%0)\n\t"  // v0[0..3] = line0[0,2.4.6]   v1[0..3] =
                                                       // line0[1,3,5,7]
                        "vfmax.vv       v4, v0, v1\n\t"
                        "addi           %0, %0, 32\n\t"  // line0 += 8

                        "vlseg2e.v      v2, (%1)\n\t"
                        "vfmax.vv       v5, v2, v3\n\t"
                        "addi           %1, %1, 32\n\t"  // line1 += 8

                        "vfmax.vv       v6, v4, v5\n\t"
                        "vsw.v          v6, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"  // outptr += 4

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        : "=r"(line0),   // %0
                          "=r"(line1),   // %1
                          "=r"(outptr),  // %2
                          "=r"(out_w4)   // %3
                        : "0"(line0), "1"(line1), "2"(outptr), "3"(out_w4)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "t0");
                }
#else
                for (int w4 = 0; w4 < out_w4; w4++) {
                    float max00 = fmax(line0[0], line0[1]);
                    float max01 = fmax(line0[2], line0[3]);
                    float max02 = fmax(line0[4], line0[5]);
                    float max03 = fmax(line0[6], line0[7]);

                    float max10 = fmax(line1[0], line1[1]);
                    float max11 = fmax(line1[2], line1[3]);
                    float max12 = fmax(line1[4], line1[5]);
                    float max13 = fmax(line1[6], line1[7]);

                    outptr[0] = fmax(max00, max10);
                    outptr[1] = fmax(max01, max11);
                    outptr[2] = fmax(max02, max12);
                    outptr[3] = fmax(max03, max13);

                    line0 += 8;
                    line1 += 8;
                    outptr += 4;
                }
#endif  // __riscv_vector
                for (int i = out_w4 * 4; i < out_w; i++) {
                    float max0 = fmax(line0[0], line0[1]);
                    float max1 = fmax(line1[0], line1[1]);
                    outptr[0] = fmax(max0, max1);

                    line0 += 2;
                    line1 += 2;
                    outptr++;
                }
                if (extend_w) {
                    outptr[0] = fmax(line0[0], line1[0]);
                    outptr++;
                }
                line0 += remain_w + in_w;
                line1 += remain_w + in_w;
            }
            if (extend_h) {
#if __riscv_vector == 128
                if (out_w4 > 0) {
                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "mv             t0, %2\n\t"
                        "1:\n\t"
                        "vlseg2e.v      v0, (%0)\n\t"
                        "vfmax.vv       v2, v0, v1\n\t"
                        "addi           %0, %0, 32\n\t"

                        "vsw.v          v2, (%1)\n\t"
                        "addi           %1, %1, 16\n\t"

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        : "=r"(line0),   // %0
                          "=r"(outptr),  // %1
                          "=r"(out_w4)   // %2
                        : "0"(line0), "1"(outptr), "2"(out_w4)
                        : "cc", "memory", "v0", "v1", "v2", "t0");
                }
#else
                for (int w4 = 0; w4 < out_w4; w4++) {
                    outptr[0] = fmax(line0[0], line0[1]);
                    outptr[1] = fmax(line0[2], line0[3]);
                    outptr[2] = fmax(line0[4], line0[5]);
                    outptr[3] = fmax(line0[6], line0[7]);

                    line0 += 8;
                    outptr += 4;
                }
#endif  //__riscv_vector
                for (int i = out_w4 * 4; i < out_w; i++) {
                    outptr[0] = fmax(line0[0], line0[1]);
                    line0 += 2;
                    outptr++;
                }
                if (extend_w) {
                    outptr[0] = line0[0];
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
static int maxpool2x2s2_p1(struct csinn_tensor *input, struct csinn_tensor *output,
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

    int out_w4 = (out_w - 1) >> 2;
    int remain_w = in_w - 2 * out_w + 1;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const float *line00 = input_data + c * in_h * in_w;
            float *outptr = output_data + c * out_hw;

            // h top ---- w left
            outptr[0] = line00[0];
            outptr++;
            line00++;
            // h top ---- w mid
#if __riscv_vector == 128
            if (out_w4 > 0) {
                asm volatile(
                    "vsetvli        zero, zero, e32, m1\n\t"
                    "mv             t0, %2\n\t"
                    "1:\n\t"
                    "vlseg2e.v      v0, (%0)\n\t"
                    "vfmax.vv       v2, v0, v1\n\t"
                    "addi           %0, %0, 32\n\t"

                    "vsw.v          v2, (%1)\n\t"
                    "addi           %1, %1, 16\n\t"

                    "addi           t0, t0, -1\n\t"
                    "bnez           t0, 1b\n\t"

                    : "=r"(line00),  // %0
                      "=r"(outptr),  // %1
                      "=r"(out_w4)   // %2
                    : "0"(line00), "1"(outptr), "2"(out_w4)
                    : "cc", "memory", "v0", "v1", "v2", "t0");
            }
#else
            for (int w4 = 0; w4 < out_w4; w4++) {
                outptr[0] = fmax(line00[0], line00[1]);
                outptr[1] = fmax(line00[2], line00[3]);
                outptr[2] = fmax(line00[4], line00[5]);
                outptr[3] = fmax(line00[6], line00[7]);

                line00 += 8;
                outptr += 4;
            }
#endif  // __riscv_vector
            for (int j = out_w4 * 4 + 1; j < out_w; j++) {
                outptr[0] = fmax(line00[0], line00[1]);
                outptr++;
                line00 += 2;
            }
            // h top ---- w right
            if (extend_w) {
                outptr[0] = line00[0];
                outptr++;
            }
            line00 += remain_w;

            // h mid
            const float *line0 = line00;
            const float *line1 = line0 + in_w;
            for (int h = 0; h < out_h - 1; h++) {
                // h mid ---- w left
                outptr[0] = fmax(line0[0], line1[0]);
                outptr++;
                line0++;
                line1++;
                // h mid ---- w mid
#if __riscv_vector == 128
                if (out_w4 > 0) {
                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "mv             t0, %3\n\t"
                        "1:\n\t"
                        "vlseg2e.v      v0, (%0)\n\t"
                        "vfmax.vv       v4, v0, v1\n\t"
                        "addi           %0, %0, 32\n\t"

                        "vlseg2e.v      v2, (%1)\n\t"
                        "vfmax.vv       v5, v2, v3\n\t"
                        "addi           %1, %1, 32\n\t"

                        "vfmax.vv       v6, v4, v5\n\t"
                        "vsw.v          v6, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        : "=r"(line0),   // %0
                          "=r"(line1),   // %1
                          "=r"(outptr),  // %2
                          "=r"(out_w4)   // %3
                        : "0"(line0), "1"(line1), "2"(outptr), "3"(out_w4)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "t0");
                }
#else
                for (int w5 = 0; w5 < out_w4; w5++) {
                    float max00 = fmax(line0[0], line0[1]);
                    float max01 = fmax(line0[2], line0[3]);
                    float max02 = fmax(line0[4], line0[5]);
                    float max03 = fmax(line0[6], line0[7]);

                    float max10 = fmax(line1[0], line1[1]);
                    float max11 = fmax(line1[2], line1[3]);
                    float max12 = fmax(line1[4], line1[5]);
                    float max13 = fmax(line1[6], line1[7]);

                    outptr[0] = fmax(max00, max10);
                    outptr[1] = fmax(max01, max11);
                    outptr[2] = fmax(max02, max12);
                    outptr[3] = fmax(max03, max13);

                    line0 += 8;
                    line1 += 8;
                    outptr += 4;
                }
#endif  // __riscv_vector
                for (int i = out_w4 * 4 + 1; i < out_w; i++) {
                    float max0 = fmax(line0[0], line0[1]);
                    float max1 = fmax(line1[0], line1[1]);
                    outptr[0] = fmax(max0, max1);

                    line0 += 2;
                    line1 += 2;
                    outptr++;
                }
                // h mid ---- w right
                if (extend_w) {
                    outptr[0] = fmax(line0[0], line1[0]);
                    outptr++;
                }
                line0 += remain_w + in_w;
                line1 += remain_w + in_w;
            }
            // h bottom
            if (extend_h) {
                // h bottom ---- w left
                outptr[0] = line0[0];
                outptr++;
                line0++;
                // h bottom ---- w mid
#if __riscv_vector == 128
                if (out_w4 > 0) {
                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "mv             t0, %2\n\t"
                        "1:\n\t"
                        "vlseg2e.v      v0, (%0)\n\t"
                        "vfmax.vv       v2, v0, v1\n\t"
                        "addi           %0, %0, 32\n\t"

                        "vsw.v          v2, (%1)\n\t"
                        "addi           %1, %1, 16\n\t"

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        : "=r"(line0),   // %0
                          "=r"(outptr),  // %1
                          "=r"(out_w4)   // %2
                        : "0"(line0), "1"(outptr), "2"(out_w4)
                        : "cc", "memory", "v0", "v1", "v2", "t0");
                }
#else
                for (int w4 = 0; w4 < out_w4; w4++) {
                    outptr[0] = fmax(line0[0], line0[1]);
                    outptr[1] = fmax(line0[2], line0[3]);
                    outptr[2] = fmax(line0[4], line0[5]);
                    outptr[3] = fmax(line0[6], line0[7]);

                    line0 += 8;
                    outptr += 4;
                }
#endif  // __riscv_vector
                for (int i = out_w4 * 4 + 1; i < out_w; i++) {
                    outptr[0] = fmax(line0[0], line0[1]);
                    outptr++;
                    line0 += 2;
                }
                // h bottom ---- w right
                if (extend_w) {
                    outptr[0] = line0[0];
                }
            }
        }
        input_data += input_size;
        output_data += output_size;
    }
    return CSINN_TRUE;
}

/*
    pad_left = pad_top = 0
    pad_right = 0 or 1
    pad_down = 0 or 1
*/
static int maxpool3x3s2(struct csinn_tensor *input, struct csinn_tensor *output,
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

    int out_w4 = out_w >> 2;
    int remain_w = in_w - 2 * out_w;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const float *line0 = input_data + c * in_h * in_w;
            const float *line1 = line0 + in_w;
            const float *line2 = line1 + in_w;
            float *outptr = output_data + c * out_hw;

            for (int h = 0; h < out_h; h++) {
#if __riscv_vector == 128
                if (out_w4 > 0) {
                    int load_stride = 8;
                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "mv             t0, %4\n\t"  // t0 = out_w4
                        "1:\n\t"
                        "vlseg2e.v      v0, (%0)\n\t"  // v0[0..3] = line0[0,2.4.6]   v1[0..3] =
                                                       // line0[1,3,5,7]
                        "vfmax.vv       v6, v0, v1\n\t"
                        "addi           %0, %0, 8\n\t"     // line0 += 2
                        "vlsw.v         v9, (%0), %5\n\t"  // v9 = line0[2,4,6,8]
                        "vfmax.vv       v12, v6, v9\n\t"
                        "addi           %0, %0, 24\n\t"  // line0 += 6

                        "vlseg2e.v      v2, (%1)\n\t"
                        "vfmax.vv       v7, v2, v3\n\t"
                        "addi           %1, %1, 8\n\t"  // line1 += 2
                        "vlsw.v         v10, (%1), %5\n\t"
                        "vfmax.vv       v13, v7, v10\n\t"
                        "addi           %1, %1, 24\n\t"  // line1 += 6

                        "vlseg2e.v      v4, (%2)\n\t"
                        "vfmax.vv       v8, v4, v5\n\t"
                        "addi           %2, %2, 8\n\t"  // line2 += 2
                        "vlsw.v         v11, (%2), %5\n\t"
                        "vfmax.vv       v14, v8, v11\n\t"
                        "addi           %2, %2, 24\n\t"  // line2 += 6

                        "vfmax.vv       v15, v12, v13\n\t"
                        "vfmax.vv       v15, v14, v15\n\t"

                        "vsw.v          v15, (%3)\n\t"
                        "addi           %3, %3, 16\n\t"  // outptr += 4

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        : "=r"(line0),       // %0
                          "=r"(line1),       // %1
                          "=r"(line2),       // %2
                          "=r"(outptr),      // %3
                          "=r"(out_w4),      // %4
                          "=r"(load_stride)  // %5
                        : "0"(line0), "1"(line1), "2"(line2), "3"(outptr), "4"(out_w4),
                          "5"(load_stride)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                          "v9", "v10", "v11", "v12", "v13", "v14", "v15", "t0");
                }
#else
                for (int w4 = 0; w4 < out_w4; w4++) {
                    float max00 = fmax(line0[0], fmax(line0[1], line0[2]));
                    float max01 = fmax(line0[2], fmax(line0[3], line0[4]));
                    float max02 = fmax(line0[4], fmax(line0[5], line0[6]));
                    float max03 = fmax(line0[6], fmax(line0[7], line0[8]));

                    float max10 = fmax(line1[0], fmax(line1[1], line1[2]));
                    float max11 = fmax(line1[2], fmax(line1[3], line1[4]));
                    float max12 = fmax(line1[4], fmax(line1[5], line1[6]));
                    float max13 = fmax(line1[6], fmax(line1[7], line1[8]));

                    float max20 = fmax(line2[0], fmax(line2[1], line2[2]));
                    float max21 = fmax(line2[2], fmax(line2[3], line2[4]));
                    float max22 = fmax(line2[4], fmax(line2[5], line2[6]));
                    float max23 = fmax(line2[6], fmax(line2[7], line2[8]));

                    outptr[0] = fmax(max00, fmax(max10, max20));
                    outptr[1] = fmax(max01, fmax(max11, max21));
                    outptr[2] = fmax(max02, fmax(max12, max22));
                    outptr[3] = fmax(max03, fmax(max13, max23));

                    line0 += 8;
                    line1 += 8;
                    line2 += 8;
                    outptr += 4;
                }
#endif  // __riscv_vector
                for (int i = out_w4 * 4; i < out_w; i++) {
                    float max0 = fmax(line0[0], fmax(line0[1], line0[2]));
                    float max1 = fmax(line1[0], fmax(line1[1], line1[2]));
                    float max2 = fmax(line2[0], fmax(line2[1], line2[2]));
                    outptr[0] = fmax(max0, fmax(max1, max2));

                    line0 += 2;
                    line1 += 2;
                    line2 += 2;
                    outptr++;
                }
                if (extend_w) {
                    float max0 = fmax(line0[0], line0[1]);
                    float max1 = fmax(line1[0], line1[1]);
                    float max2 = fmax(line2[0], line2[1]);
                    outptr[0] = fmax(max0, fmax(max1, max2));
                    outptr++;
                }
                line0 += remain_w + in_w;
                line1 += remain_w + in_w;
                line2 += remain_w + in_w;
            }
            if (extend_h) {
#if __riscv_vector == 128
                if (out_w4 > 0) {
                    int load_stride = 8;
                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "mv             t0, %3\n\t"
                        "1:\n\t"
                        "vlseg2e.v      v0, (%0)\n\t"
                        "vfmax.vv       v4, v0, v1\n\t"
                        "addi           %0, %0, 8\n\t"
                        "vlsw.v         v6, (%0), %4\n\t"
                        "vfmax.vv       v8, v4, v6\n\t"
                        "addi           %0, %0, 24\n\t"

                        "vlseg2e.v      v2, (%1)\n\t"
                        "vfmax.vv       v5, v2, v3\n\t"
                        "addi           %1, %1, 8\n\t"
                        "vlsw.v         v7, (%1), %4\n\t"
                        "vfmax.vv       v9, v5, v7\n\t"
                        "addi           %1, %1, 24\n\t"

                        "vfmax.vv       v10, v8, v9\n\t"
                        "vsw.v          v10, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        : "=r"(line0),       // %0
                          "=r"(line1),       // %1
                          "=r"(outptr),      // %2
                          "=r"(out_w4),      // %3
                          "=r"(load_stride)  // %4
                        : "0"(line0), "1"(line1), "2"(outptr), "3"(out_w4), "4"(load_stride)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                          "v9", "v10", "t0");
                }
#else
                for (int w4 = 0; w4 < out_w4; w4++) {
                    float max00 = fmax(line0[0], fmax(line0[1], line0[2]));
                    float max01 = fmax(line0[2], fmax(line0[3], line0[4]));
                    float max02 = fmax(line0[4], fmax(line0[5], line0[6]));
                    float max03 = fmax(line0[6], fmax(line0[7], line0[8]));

                    float max10 = fmax(line1[0], fmax(line1[1], line1[2]));
                    float max11 = fmax(line1[2], fmax(line1[3], line1[4]));
                    float max12 = fmax(line1[4], fmax(line1[5], line1[6]));
                    float max13 = fmax(line1[6], fmax(line1[7], line1[8]));

                    outptr[0] = fmax(max00, max10);
                    outptr[1] = fmax(max01, max11);
                    outptr[2] = fmax(max02, max12);
                    outptr[3] = fmax(max03, max13);

                    line0 += 8;
                    line1 += 8;
                    outptr += 4;
                }
#endif  // __riscv_vector
                for (int i = out_w4 * 4; i < out_w; i++) {
                    float max0 = fmax(line0[0], fmax(line0[1], line0[2]));
                    float max1 = fmax(line1[0], fmax(line1[1], line1[2]));
                    outptr[0] = fmax(max0, max1);

                    line0 += 2;
                    line1 += 2;
                    outptr++;
                }
                if (extend_w) {
                    float max0 = fmax(line0[0], line0[1]);
                    float max1 = fmax(line1[0], line1[1]);
                    outptr[0] = fmax(max0, max1);
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
static int maxpool3x3s2_p1(struct csinn_tensor *input, struct csinn_tensor *output,
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

    int out_w4 = (out_w - 1) >> 2;
    int remain_w = in_w - 2 * out_w + 1;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const float *line0 = input_data + c * in_h * in_w;
            const float *line1 = line0 + in_w;
            float *outptr = output_data + c * out_hw;

            // h top ---- w left
            outptr[0] = fmax(fmax(line0[0], line0[1]), fmax(line1[0], line1[1]));
            outptr++;
            line0++;
            line1++;
            // h top ---- w mid
#if __riscv_vector == 128
            if (out_w4 > 0) {
                int load_stride = 8;
                asm volatile(
                    "vsetvli        zero, zero, e32, m1\n\t"
                    "mv             t0, %3\n\t"  // t0 = out_w4
                    "1:\n\t"
                    "vlseg2e.v      v0, (%0)\n\t"  // v0[0..3] = line0[0,2.4.6]   v1[0..3] =
                                                   // line0[1,3,5,7]
                    "vfmax.vv       v4, v0, v1\n\t"
                    "addi           %0, %0, 8\n\t"     // line0 += 2
                    "vlsw.v         v6, (%0), %4\n\t"  // v9 = line0[2,4,6,8]
                    "vfmax.vv       v8, v4, v6\n\t"
                    "addi           %0, %0, 24\n\t"  // line0 += 6

                    "vlseg2e.v      v2, (%1)\n\t"
                    "vfmax.vv       v5, v2, v3\n\t"
                    "addi           %1, %1, 8\n\t"  // line1 += 2
                    "vlsw.v         v7, (%1), %4\n\t"
                    "vfmax.vv       v9, v5, v7\n\t"
                    "addi           %1, %1, 24\n\t"  // line1 += 6

                    "vfmax.vv       v10, v8, v9\n\t"
                    "vsw.v          v10, (%2)\n\t"
                    "addi           %2, %2, 16\n\t"  // outptr += 4

                    "addi           t0, t0, -1\n\t"
                    "bnez           t0, 1b\n\t"

                    : "=r"(line0),       // %0
                      "=r"(line1),       // %1
                      "=r"(outptr),      // %2
                      "=r"(out_w4),      // %3
                      "=r"(load_stride)  // %4
                    : "0"(line0), "1"(line1), "2"(outptr), "3"(out_w4), "4"(load_stride)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                      "v10", "t0");
            }
#else
            for (int w4 = 0; w4 < out_w4; w4++) {
                float max00 = fmax(line0[0], fmax(line0[1], line0[2]));
                float max01 = fmax(line0[2], fmax(line0[3], line0[4]));
                float max02 = fmax(line0[4], fmax(line0[5], line0[6]));
                float max03 = fmax(line0[6], fmax(line0[7], line0[8]));

                float max10 = fmax(line1[0], fmax(line1[1], line1[2]));
                float max11 = fmax(line1[2], fmax(line1[3], line1[4]));
                float max12 = fmax(line1[4], fmax(line1[5], line1[6]));
                float max13 = fmax(line1[6], fmax(line1[7], line1[8]));

                outptr[0] = fmax(max00, max10);
                outptr[1] = fmax(max01, max11);
                outptr[2] = fmax(max02, max12);
                outptr[3] = fmax(max03, max13);

                line0 += 8;
                line1 += 8;
                outptr += 4;
            }
#endif  // __riscv_vector
            for (int i = out_w4 * 4 + 1; i < out_w; i++) {
                float max0 = fmax(line0[0], fmax(line0[1], line0[2]));
                float max1 = fmax(line1[0], fmax(line1[1], line1[2]));
                outptr[0] = fmax(max0, max1);

                line0 += 2;
                line1 += 2;
                outptr++;
            }
            // h top ---- w right
            if (extend_w) {
                outptr[0] = fmax(fmax(line0[0], line0[1]), fmax(line1[0], line1[1]));
                outptr++;
            }
            line0 += remain_w;
            line1 += remain_w;

            // h mid
            const float *line2 = line1 + in_w;
            for (int h = 0; h < out_h - 1; h++) {
                // h mid ---- w left
                float max0 = fmax(line0[0], line0[1]);
                float max1 = fmax(line1[0], line1[1]);
                float max2 = fmax(line2[0], line2[1]);
                outptr[0] = fmax(max0, fmax(max1, max2));
                outptr++;
                line0++;
                line1++;
                line2++;
                // h mid ---- w mid
#if __riscv_vector == 128
                if (out_w4 > 0) {
                    int load_stride = 8;
                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "mv             t0, %4\n\t"
                        "1:\n\t"
                        "vlseg2e.v      v0, (%0)\n\t"
                        "vfmax.vv       v6, v0, v1\n\t"
                        "addi           %0, %0, 8\n\t"
                        "vlsw.v         v9, (%0), %5\n\t"
                        "vfmax.vv       v12, v6, v9\n\t"
                        "addi           %0, %0, 24\n\t"

                        "vlseg2e.v      v2, (%1)\n\t"
                        "vfmax.vv       v7, v2, v3\n\t"
                        "addi           %1, %1, 8\n\t"
                        "vlsw.v         v10, (%1), %5\n\t"
                        "vfmax.vv       v13, v7, v10\n\t"
                        "addi           %1, %1, 24\n\t"

                        "vlseg2e.v      v4, (%2)\n\t"
                        "vfmax.vv       v8, v4, v5\n\t"
                        "addi           %2, %2, 8\n\t"
                        "vlsw.v         v11, (%2), %5\n\t"
                        "vfmax.vv       v14, v8, v11\n\t"
                        "addi           %2, %2, 24\n\t"

                        "vfmax.vv       v15, v12, v13\n\t"
                        "vfmax.vv       v15, v14, v15\n\t"

                        "vsw.v          v15, (%3)\n\t"
                        "addi           %3, %3, 16\n\t"

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        : "=r"(line0),       // %0
                          "=r"(line1),       // %1
                          "=r"(line2),       // %2
                          "=r"(outptr),      // %3
                          "=r"(out_w4),      // %4
                          "=r"(load_stride)  // %5
                        : "0"(line0), "1"(line1), "2"(line2), "3"(outptr), "4"(out_w4),
                          "5"(load_stride)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                          "v9", "v10", "v11", "v12", "v13", "v14", "v15", "t0");
                }
#else
                for (int w4 = 0; w4 < out_w4; w4++) {
                    float max00 = fmax(line0[0], fmax(line0[1], line0[2]));
                    float max01 = fmax(line0[2], fmax(line0[3], line0[4]));
                    float max02 = fmax(line0[4], fmax(line0[5], line0[6]));
                    float max03 = fmax(line0[6], fmax(line0[7], line0[8]));

                    float max10 = fmax(line1[0], fmax(line1[1], line1[2]));
                    float max11 = fmax(line1[2], fmax(line1[3], line1[4]));
                    float max12 = fmax(line1[4], fmax(line1[5], line1[6]));
                    float max13 = fmax(line1[6], fmax(line1[7], line1[8]));

                    float max20 = fmax(line2[0], fmax(line2[1], line2[2]));
                    float max21 = fmax(line2[2], fmax(line2[3], line2[4]));
                    float max22 = fmax(line2[4], fmax(line2[5], line2[6]));
                    float max23 = fmax(line2[6], fmax(line2[7], line2[8]));

                    outptr[0] = fmax(max00, fmax(max10, max20));
                    outptr[1] = fmax(max01, fmax(max11, max21));
                    outptr[2] = fmax(max02, fmax(max12, max22));
                    outptr[3] = fmax(max03, fmax(max13, max23));

                    line0 += 8;
                    line1 += 8;
                    line2 += 8;
                    outptr += 4;
                }
#endif  // __riscv_vector
                for (int i = out_w4 * 4 + 1; i < out_w; i++) {
                    float max0 = fmax(line0[0], fmax(line0[1], line0[2]));
                    float max1 = fmax(line1[0], fmax(line1[1], line1[2]));
                    float max2 = fmax(line2[0], fmax(line2[1], line2[2]));
                    outptr[0] = fmax(max0, fmax(max1, max2));

                    line0 += 2;
                    line1 += 2;
                    line2 += 2;
                    outptr++;
                }
                // h mid ---- w right
                if (extend_w) {
                    float max0 = fmax(line0[0], line0[1]);
                    float max1 = fmax(line1[0], line1[1]);
                    float max2 = fmax(line2[0], line2[1]);
                    outptr[0] = fmax(max0, fmax(max1, max2));
                    outptr++;
                }
                line0 += in_w + remain_w;
                line1 += in_w + remain_w;
                line2 += in_w + remain_w;
            }

            // h bottom
            if (extend_h) {
                // h bottom ---- w left
                outptr[0] = fmax(fmax(line0[0], line0[1]), fmax(line1[0], line1[1]));
                outptr++;
                line0++;
                line1++;

                // h bottom ---- w mid
#if __riscv_vector == 128
                if (out_w4 > 0) {
                    int load_stride = 8;
                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "mv             t0, %3\n\t"
                        "1:\n\t"
                        "vlseg2e.v      v0, (%0)\n\t"
                        "vfmax.vv       v4, v0, v1\n\t"
                        "addi           %0, %0, 8\n\t"
                        "vlsw.v         v6, (%0), %4\n\t"
                        "vfmax.vv       v8, v4, v6\n\t"
                        "addi           %0, %0, 24\n\t"

                        "vlseg2e.v      v2, (%1)\n\t"
                        "vfmax.vv       v5, v2, v3\n\t"
                        "addi           %1, %1, 8\n\t"
                        "vlsw.v         v7, (%1), %4\n\t"
                        "vfmax.vv       v9, v5, v7\n\t"
                        "addi           %1, %1, 24\n\t"

                        "vfmax.vv       v10, v8, v9\n\t"
                        "vsw.v          v10, (%2)\n\t"
                        "addi           %2, %2, 16\n\t"

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        : "=r"(line0),       // %0
                          "=r"(line1),       // %1
                          "=r"(outptr),      // %2
                          "=r"(out_w4),      // %3
                          "=r"(load_stride)  // %4
                        : "0"(line0), "1"(line1), "2"(outptr), "3"(out_w4), "4"(load_stride)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                          "v9", "v10", "t0");
                }
#else
                for (int w4 = 0; w4 < out_w4; w4++) {
                    float max00 = fmax(line0[0], fmax(line0[1], line0[2]));
                    float max01 = fmax(line0[2], fmax(line0[3], line0[4]));
                    float max02 = fmax(line0[4], fmax(line0[5], line0[6]));
                    float max03 = fmax(line0[6], fmax(line0[7], line0[8]));

                    float max10 = fmax(line1[0], fmax(line1[1], line1[2]));
                    float max11 = fmax(line1[2], fmax(line1[3], line1[4]));
                    float max12 = fmax(line1[4], fmax(line1[5], line1[6]));
                    float max13 = fmax(line1[6], fmax(line1[7], line1[8]));

                    outptr[0] = fmax(max00, max10);
                    outptr[1] = fmax(max01, max11);
                    outptr[2] = fmax(max02, max12);
                    outptr[3] = fmax(max03, max13);

                    line0 += 8;
                    line1 += 8;
                    outptr += 4;
                }
#endif  // __riscv_vector
                for (int i = out_w4 * 4 + 1; i < out_w; i++) {
                    float max0 = fmax(line0[0], fmax(line0[1], line0[2]));
                    float max1 = fmax(line1[0], fmax(line1[1], line1[2]));
                    outptr[0] = fmax(max0, max1);

                    line0 += 2;
                    line1 += 2;
                    outptr++;
                }
                // h bottom ---- w right
                if (extend_w) {
                    outptr[0] = fmax(fmax(line0[0], line0[1]), fmax(line1[0], line1[1]));
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
static int maxpool3x3s1_p1(struct csinn_tensor *input, struct csinn_tensor *output,
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

    int out_w4 = (out_w - 2) >> 2;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) {
            const float *line1 = input_data + c * in_h * in_w;
            const float *line2 = line1 + in_w;
            float *outptr = output_data + c * out_h * out_w;
            // h top ---- w left
            outptr[0] = fmax(fmax(line1[0], line1[1]), fmax(line2[0], line2[1]));
            outptr++;
            // h top ---- w mid
#if __riscv_vector == 128
            if (out_w4 > 0) {
                asm volatile(
                    "vsetvli        zero, zero, e32, m1\n\t"
                    "mv             t0, %3\n\t"
                    "1:\n\t"
                    "vlw.v          v0, (%0)\n\t"
                    "addi           %0, %0, 4\n\t"
                    "vlw.v          v1, (%0)\n\t"
                    "addi           %0, %0, 4\n\t"
                    "vlw.v          v2, (%0)\n\t"
                    "addi           %0, %0, 8\n\t"
                    "vfmax.vv       v3, v0, v1\n\t"
                    "vfmax.vv       v4, v2, v3\n\t"

                    "vlw.v          v5, (%1)\n\t"
                    "addi           %1, %1, 4\n\t"
                    "vlw.v          v6, (%1)\n\t"
                    "addi           %1, %1, 4\n\t"
                    "vlw.v          v7, (%1)\n\t"
                    "addi           %1, %1, 8\n\t"
                    "vfmax.vv       v8, v5, v6\n\t"
                    "vfmax.vv       v9, v7, v8\n\t"

                    "vfmax.vv       v10, v4, v9\n\t"
                    "vsw.v          v10, (%2)\n\t"
                    "addi           %2, %2, 16\n\t"

                    "addi           t0, t0, -1\n\t"
                    "bnez           t0, 1b\n\t"

                    : "=r"(line1),   // %0
                      "=r"(line2),   // %1
                      "=r"(outptr),  // %2
                      "=r"(out_w4)   // %3
                    : "0"(line1), "1"(line2), "2"(outptr), "3"(out_w4)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                      "v10", "t0");
            }
#else
            for (int w4 = 0; w4 < out_w4; w4++) {
                float max10 = fmax(line1[0], fmax(line1[1], line1[2]));
                float max11 = fmax(line1[1], fmax(line1[2], line1[3]));
                float max12 = fmax(line1[2], fmax(line1[3], line1[4]));
                float max13 = fmax(line1[3], fmax(line1[4], line1[5]));

                float max20 = fmax(line2[0], fmax(line2[1], line2[2]));
                float max21 = fmax(line2[1], fmax(line2[2], line2[3]));
                float max22 = fmax(line2[2], fmax(line2[3], line2[4]));
                float max23 = fmax(line2[3], fmax(line2[4], line2[5]));

                outptr[0] = fmax(max10, max20);
                outptr[1] = fmax(max11, max21);
                outptr[2] = fmax(max12, max22);
                outptr[3] = fmax(max13, max23);

                line1 += 4;
                line2 += 4;
                outptr += 4;
            }
#endif  // __riscv_vctor
            for (int i = out_w4 * 4; i < out_w - 2; i++) {
                float max1 = fmax(line1[0], fmax(line1[1], line1[2]));
                float max2 = fmax(line2[0], fmax(line2[1], line2[2]));
                outptr[0] = fmax(max1, max2);
                outptr++;
                line1++;
                line2++;
            }
            // h top ---- w right
            outptr[0] = fmax(fmax(line1[0], line1[1]), fmax(line2[0], line2[1]));
            outptr++;
            line1 += 2;  // bump next line: line1 --> line2
            line2 += 2;

            // h mid
            const float *line0 = input_data + c * in_h * in_w;
            for (int h = 0; h < out_h - 2; h++) {
                // h mid ---- w left
                float max0 = fmax(line0[0], line0[1]);
                float max1 = fmax(line1[0], line1[1]);
                float max2 = fmax(line2[0], line2[1]);
                outptr[0] = fmax(max0, fmax(max1, max2));
                outptr++;
                // h mid ---- w mid
#if __riscv_vector == 128
                if (out_w4 > 0) {
                    asm volatile(
                        "vsetvli        zero, zero, e32, m1\n\t"
                        "mv             t0, %4\n\t"
                        "1:\n\t"
                        "vlw.v          v0, (%0)\n\t"
                        "addi           %0, %0, 4\n\t"
                        "vlw.v          v1, (%0)\n\t"
                        "addi           %0, %0, 4\n\t"
                        "vlw.v          v2, (%0)\n\t"
                        "addi           %0, %0, 8\n\t"
                        "vfmax.vv       v3, v0, v1\n\t"
                        "vfmax.vv       v4, v2, v3\n\t"

                        "vlw.v          v5, (%1)\n\t"
                        "addi           %1, %1, 4\n\t"
                        "vlw.v          v6, (%1)\n\t"
                        "addi           %1, %1, 4\n\t"
                        "vlw.v          v7, (%1)\n\t"
                        "addi           %1, %1, 8\n\t"
                        "vfmax.vv       v8, v5, v6\n\t"
                        "vfmax.vv       v9, v7, v8\n\t"

                        "vlw.v          v10, (%2)\n\t"
                        "addi           %2, %2, 4\n\t"
                        "vlw.v          v11, (%2)\n\t"
                        "addi           %2, %2, 4\n\t"
                        "vlw.v          v12, (%2)\n\t"
                        "addi           %2, %2, 8\n\t"
                        "vfmax.vv       v13, v10, v11\n\t"
                        "vfmax.vv       v14, v12, v13\n\t"

                        "vfmax.vv       v15, v4, v9\n\t"
                        "vfmax.vv       v15, v14, v15\n\t"
                        "vsw.v          v15, (%3)\n\t"
                        "addi           %3, %3, 16\n\t"

                        "addi           t0, t0, -1\n\t"
                        "bnez           t0, 1b\n\t"

                        : "=r"(line0),   // %0
                          "=r"(line1),   // %1
                          "=r"(line2),   // %2
                          "=r"(outptr),  // %3
                          "=r"(out_w4)   // %4
                        : "0"(line0), "1"(line1), "2"(line2), "3"(outptr), "4"(out_w4)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                          "v9", "v10", "v11", "v12", "v13", "v14", "v15", "t0");
                }
#else
                for (int w4 = 0; w4 < out_w4; w4++) {
                    float max00 = fmax(line0[0], fmax(line0[1], line0[2]));
                    float max01 = fmax(line0[1], fmax(line0[2], line0[3]));
                    float max02 = fmax(line0[2], fmax(line0[3], line0[4]));
                    float max03 = fmax(line0[3], fmax(line0[4], line0[5]));

                    float max10 = fmax(line1[0], fmax(line1[1], line1[2]));
                    float max11 = fmax(line1[1], fmax(line1[2], line1[3]));
                    float max12 = fmax(line1[2], fmax(line1[3], line1[4]));
                    float max13 = fmax(line1[3], fmax(line1[4], line1[5]));

                    float max20 = fmax(line2[0], fmax(line2[1], line2[2]));
                    float max21 = fmax(line2[1], fmax(line2[2], line2[3]));
                    float max22 = fmax(line2[2], fmax(line2[3], line2[4]));
                    float max23 = fmax(line2[3], fmax(line2[4], line2[5]));

                    outptr[0] = fmax(max00, fmax(max10, max20));
                    outptr[1] = fmax(max01, fmax(max11, max21));
                    outptr[2] = fmax(max02, fmax(max12, max22));
                    outptr[3] = fmax(max03, fmax(max13, max23));

                    line0 += 4;
                    line1 += 4;
                    line2 += 4;
                    outptr += 4;
                }
#endif  // __riscv_vctor
                for (int i = out_w4 * 4; i < out_w - 2; i++) {
                    float max0 = fmax(line0[0], fmax(line0[1], line0[2]));
                    float max1 = fmax(line1[0], fmax(line1[1], line1[2]));
                    float max2 = fmax(line2[0], fmax(line2[1], line2[2]));
                    outptr[0] = fmax(max0, fmax(max1, max2));

                    outptr++;
                    line0++;
                    line1++;
                    line2++;
                }
                // h mid ---- w right
                float max0_0 = fmax(line0[0], line0[1]);
                float max1_0 = fmax(line1[0], line1[1]);
                float max2_0 = fmax(line2[0], line2[1]);
                outptr[0] = fmax(max0_0, fmax(max1_0, max2_0));

                outptr++;
                line0 += 2;
                line1 += 2;
                line2 += 2;
            }

            // h bottom ---- w left
            outptr[0] = fmax(fmax(line0[0], line0[1]), fmax(line1[0], line1[1]));
            outptr++;
            // h bottom ---- w mid
#if __riscv_vector == 128
            if (out_w4 > 0) {
                asm volatile(
                    "vsetvli        zero, zero, e32, m1\n\t"
                    "mv             t0, %3\n\t"
                    "1:\n\t"
                    "vlw.v          v0, (%0)\n\t"
                    "addi           %0, %0, 4\n\t"
                    "vlw.v          v1, (%0)\n\t"
                    "addi           %0, %0, 4\n\t"
                    "vlw.v          v2, (%0)\n\t"
                    "addi           %0, %0, 8\n\t"
                    "vfmax.vv       v3, v0, v1\n\t"
                    "vfmax.vv       v4, v2, v3\n\t"

                    "vlw.v          v5, (%1)\n\t"
                    "addi           %1, %1, 4\n\t"
                    "vlw.v          v6, (%1)\n\t"
                    "addi           %1, %1, 4\n\t"
                    "vlw.v          v7, (%1)\n\t"
                    "addi           %1, %1, 8\n\t"
                    "vfmax.vv       v8, v5, v6\n\t"
                    "vfmax.vv       v9, v7, v8\n\t"

                    "vfmax.vv       v10, v4, v9\n\t"
                    "vsw.v          v10, (%2)\n\t"
                    "addi           %2, %2, 16\n\t"

                    "addi           t0, t0, -1\n\t"
                    "bnez           t0, 1b\n\t"

                    : "=r"(line0),   // %0
                      "=r"(line1),   // %1
                      "=r"(outptr),  // %2
                      "=r"(out_w4)   // %3
                    : "0"(line0), "1"(line1), "2"(outptr), "3"(out_w4)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                      "v10", "t0");
            }
#else
            for (int w4 = 0; w4 < out_w4; w4++) {
                float max00 = fmax(line0[0], fmax(line0[1], line0[2]));
                float max01 = fmax(line0[1], fmax(line0[2], line0[3]));
                float max02 = fmax(line0[2], fmax(line0[3], line0[4]));
                float max03 = fmax(line0[3], fmax(line0[4], line0[5]));

                float max10 = fmax(line1[0], fmax(line1[1], line1[2]));
                float max11 = fmax(line1[1], fmax(line1[2], line1[3]));
                float max12 = fmax(line1[2], fmax(line1[3], line1[4]));
                float max13 = fmax(line1[3], fmax(line1[4], line1[5]));

                outptr[0] = fmax(max00, max10);
                outptr[1] = fmax(max01, max11);
                outptr[2] = fmax(max02, max12);
                outptr[3] = fmax(max03, max13);

                line0 += 4;
                line1 += 4;
                outptr += 4;
            }
#endif  // __riscv_vctor
            for (int i = out_w4 * 4; i < out_w - 2; i++) {
                float max0 = fmax(line0[0], fmax(line0[1], line0[2]));
                float max1 = fmax(line1[0], fmax(line1[1], line1[2]));
                outptr[0] = fmax(max0, max1);
                outptr++;
                line0++;
                line1++;
            }
            // h bottom ---- w right
            outptr[0] = fmax(fmax(line0[0], line0[1]), fmax(line1[0], line1[1]));
        }
        input_data += input_size;
        output_data += output_size;
    }
    return CSINN_TRUE;
}

int shl_c906_maxpool2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params)
{
    int32_t input_h = input->dim[2];
    int32_t input_w = input->dim[3];
    int32_t kernel_h = params->filter_height;
    int32_t kernel_w = params->filter_width;
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t pad_left = params->pad_left;
    int32_t pad_right = params->pad_right;
    int32_t pad_top = params->pad_top;
    int32_t pad_down = params->pad_down;
    struct csinn_callback *cb = params->base.cb;
    cb->exec = NULL;

    if (params->base.sess->base_run_mode == CSINN_RM_CPU_GRAPH) {
        struct shl_c906_option *option = shl_c906_get_graph_option(params->base.sess);
        if (option && option->base.use_packn_layout) {
            shl_debug_error("%s: unsupport packn\n", __func__);
            return CSINN_UNSUPPORT_LAYOUT;
        }
    }

    // global maxpool2d
    if (input_h == kernel_h && input_w == kernel_w) {
        cb->exec = shl_c906_global_maxpool2d_f32;
        return CSINN_TRUE;
    }

    if (stride_h == 2 && stride_w == 2) {
        if (kernel_h == 2 && kernel_w == 2) {  // 2x2s2
            if (pad_left == 0 && pad_top == 0) {
                // adjust pad according to ceil_mode (ceil mode on caffe pytorch..)
                if (input_h % 2 == 1 && params->ceil_mode == 1) {
                    if (params->pad_down == 0) params->pad_down++;
                }
                if (input_w % 2 == 1 && params->ceil_mode == 1) {
                    if (params->pad_right == 0) params->pad_right++;
                }
                // end consider ceil_mode 2x2s2p0
                cb->exec = maxpool2x2s2;
            } else if (pad_left == 1 && pad_top == 1) {
                cb->exec = maxpool2x2s2_p1;
            }
        } else if (kernel_h == 3 && kernel_w == 3) {  // 3x3s2
            if (pad_left == 0 && pad_top == 0) {
                // adjust pad according to ceil_mode (ceil mode on caffe pytorch..)
                if (input_h % 2 == 0 && params->ceil_mode == 1) {
                    if (params->pad_down == 0)
                        params->pad_down++;  // origin pad_down mast be equal to zero ?
                }
                if (input_w % 2 == 0 && params->ceil_mode == 1) {
                    if (params->pad_right == 0) params->pad_right++;
                }
                // end consider ceil_mode 3x3s2p0
                cb->exec = maxpool3x3s2;
            } else if (pad_left == 1 && pad_top == 1) {
                if (params->ceil_mode == 0) {
                    cb->exec = maxpool3x3s2_p1;
                } else {
                    if ((input_w % 2 == 0 && pad_right == 1) ||
                        (input_h % 2 == 0 && pad_down == 1)) {
                        cb->exec = shl_ref_maxpool2d_f32;
                    } else {
                        cb->exec = maxpool3x3s2_p1;
                    }
                }
            }
        }
    } else if (stride_h == 1 && stride_w == 1) {
        if (kernel_h == 3 && kernel_w == 3) {
            if (pad_left == 1 && pad_top == 1 && pad_right == 1 && pad_down == 1) {
                cb->exec = maxpool3x3s1_p1;
            }
        }
    }

    if (cb->exec == NULL) {
        shl_debug_warning(
            "maxpool is not optimized to achieve under this condition on C906, call reference func "
            "replaced.\n");
        cb->exec = shl_ref_maxpool2d_f32;
    }
    return CSINN_TRUE;
}
