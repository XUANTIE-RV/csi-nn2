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

/************************************************************************************************************
    c906 vlen = 128, 128/16 = 8 --> pack8, if vlen = 256  256/16 = 16 --> pack16
    input, kernel, bias, output layout:
        input:  [c/8, in_h, in_w, 8]
        kernel: [c/8, k_h*k_w, 8]
        bias:   [c/8, 8]
        output: [c/8, out_h, out_w, 8]

    constraint: in_channel = out_channel and is a multiple of 8
                No reference implementation
**************************************************************************************************************/

/*
    (1) Algorithm works as follows:
        out_h2:     out_h2_w4_loop  -->  out_h2_w2  -->  out_h2_w1
        out_h_tail: out_h1_w4_loop  -->  out_h1_w2  -->  out_h1_w1

    (2) register definition:
        t0:         i_out_h
        t1:         i_out_w
        v0:         bias_data
        v1-v9:      [ k00, k01, k02, k10, k11, k12, k20, k21, k22 ]
        v10-v19:    r00-r05 / r10-r15 / r20-r25 / r30-r35
        v24-v27:    outptr0[0-3]    line0
        v28-v31:    outptr1[0-3]    line1

    Due to pack8, both kxx and rxx actually occupy a v register

    TODO: how to pack for input / kernel / bias / output
          padding

*/

int shl_c906_dwconv3x3s1_pack8_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                    struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];  // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    for (int c = 0; c < in_c / 8; c++) {
        __fp16 *out = output_data + c * out_h * out_w * 8;
        __fp16 *outptr0 = out;
        __fp16 *outptr1 = outptr0 + out_w * 8;

        const __fp16 *img0 = input_data + c * in_h * in_w * 8;
        const __fp16 *r0 = img0;
        const __fp16 *r1 = r0 + in_w * 8;
        const __fp16 *r2 = r1 + in_w * 8;
        const __fp16 *r3 = r2 + in_w * 8;

        const __fp16 *kernel0 = kernel_data + c * 9 * 8;

        const __fp16 *bias0 = NULL;
        if (bias_data && bias->dim_count != 0) {
            bias0 = bias_data + c * 8;
        }

        asm volatile(
            "vsetvli        zero, zero, e16, m1\n\t"

            "vmv.v.x        v0, zero\n\t"  // clear v0
            "beqz           %5, 0f\n\t"    // if bias_data = NULL  clear v0
            "vle.v          v0, (%5)\n\t"

            "0:\n\t"  // init global v register hold 9 kernel_data

            "vle.v          v1, (%0)\n\t"    // k00
            "addi           %0, %0, 16\n\t"  // kernel += 8
            "vle.v          v2, (%0)\n\t"    // k01
            "addi           %0, %0, 16\n\t"
            "vle.v          v3, (%0)\n\t"  // k02
            "addi           %0, %0, 16\n\t"
            "vle.v          v4, (%0)\n\t"  // k10
            "addi           %0, %0, 16\n\t"
            "vle.v          v5, (%0)\n\t"  // k11
            "addi           %0, %0, 16\n\t"
            "vle.v          v6, (%0)\n\t"  // k12
            "addi           %0, %0, 16\n\t"
            "vle.v          v7, (%0)\n\t"  // k20
            "addi           %0, %0, 16\n\t"
            "vle.v          v8, (%0)\n\t"  // k21
            "addi           %0, %0, 16\n\t"
            "vle.v          v9, (%0)\n\t"  // k22

            "srai           t0, %8, 1\n\t"  // t0 = out_h >> 1
            "beqz           t0, 6f\n\t"

            "1:\n\t"  // out_h2_loop

            "srai           t1, %9, 2\n\t"  // t1 = out_w >> 2
            "beqz           t1, 3f\n\t"

            // pre-load rxx
            "vle.v          v10, (%1)\n\t"  // r00
            "addi           %1, %1, 16\n\t"

            "vle.v          v11, (%1)\n\t"  // r01
            "addi           %1, %1, 16\n\t"

            "vle.v          v12, (%1)\n\t"  // r02
            "addi           %1, %1, 16\n\t"

            // load 24 times, mac 72 times
            "2:\n\t"  // out_w4_loop

            "vmv.v.x        v24, zero\n\t"

            "vle.v          v13, (%2)\n\t"  // r10
            "addi           %2, %2, 16\n\t"

            "vmv.v.x        v25, zero\n\t"

            "vfmacc.vv      v24, v1, v10\n\t"  // k00 * r00    out[0][0]

            "vmv.v.x        v26, zero\n\t"

            "vle.v          v14, (%2)\n\t"  // r11
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v25, v1, v11\n\t"  // k00 * r01    out[1][0]
            "vmv.v.x        v27, zero\n\t"
            "vfmacc.vv      v26, v1, v12\n\t"  // k00 * r02    out[2][0]
            "vfmacc.vv      v24, v4, v13\n\t"  // k10 * r10    out[0][3]

            "vmv.v.x        v28, zero\n\t"

            "vle.v          v15, (%1)\n\t"  // r03
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v25, v2, v12\n\t"  // k01 * r02    out[1][1]
            "vmv.v.x        v29, zero\n\t"
            "vfmacc.vv      v24, v5, v14\n\t"  // k11 * r11    out[0][4]
            "vfmacc.vv      v28, v1, v13\n\t"  // k00 * r10    out[4][0]

            "vle.v          v16, (%2)\n\t"  // r12
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v26, v2, v15\n\t"  // k01 * r03    out[2][1]
            "vmv.v.x        v30, zero\n\t"
            "vfmacc.vv      v25, v3, v15\n\t"  // k02 * r03    out[1][2]
            "vfmacc.vv      v29, v1, v14\n\t"  // k01 * r11    out[5][0]

            "vle.v          v17, (%1)\n\t"  // r04
            "addi           %1, %1, 16\n\t"

            "vmv.v.x        v31, zero\n\t"
            "vfmacc.vv      v24, v2, v11\n\t"  // k01 * r01    out[0][1]
            "vfmacc.vv      v27, v1, v15\n\t"  // k00 * r03    out[3][0]
            "vfmacc.vv      v28, v2, v14\n\t"  // k01 * r11    out[4][1]

            "vle.v          v18, (%2)\n\t"  // r13
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v29, v2, v16\n\t"  // k01 * r12    out[5][1]
            "vfmacc.vv      v30, v1, v16\n\t"  // k00 * r12    out[6][0]
            "vfmacc.vv      v24, v3, v12\n\t"  // k02 * r02    out[0][2]

            "vle.v          v19, (%1)\n\t"    // r05
            "addi           %1, %1, -16\n\t"  // r0 -= 8  ********* bump r0 to next 4 element addr
                                              // ************

            "vfmacc.vv      v26, v3, v17\n\t"   // k02 * r04    out[2][2]
            "vfmacc.vv      v27, v2, v17\n\t"   // k01 * r04    out[3][1]
            "vfmacc.vv      v28, v3, v16\n\t "  // k02 * r12    out[4][2]

            "vle.v          v10, (%2)\n\t"  // r14
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v25, v4, v14\n\t"  // k10 * r11    out[1][3]
            "vfmacc.vv      v29, v3, v18\n\t"  // k02 * r13    out[5][2]
            "vfmacc.vv      v30, v2, v18\n\t"  // k01 * r13    out[6][1]
            "vfmacc.vv      v31, v1, v18\n\t"  // k00 * r13    out[7][0]

            "vle.v          v11, (%3)\n\t"  // r20
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v27, v4, v18\n\t"  // k10 * r13    out[3][3]
            "vfmacc.vv      v24, v6, v16\n\t"  // k12 * r12    out[0][5]
            "vfmacc.vv      v26, v4, v16\n\t"  // k10 * r12    out[2][3]
            "vfmacc.vv      v25, v5, v16\n\t"  // k11 * r12    out[1][4]

            "vle.v          v12, (%2)\n\t"    // r15
            "addi           %2, %2, -16\n\t"  // r1 -= 8  ********* bump r1 to next 4 element addr
                                              // ************

            "vfmacc.vv      v30, v3, v10\n\t"  // k02 * r14    out[6][2]
            "vfmacc.vv      v31, v2, v10\n\t"  // k01 * r14    out[7][1]
            "vfmacc.vv      v27, v3, v19\n\t"  // k02 * r05    out[3][2]

            "vle.v          v13, (%3)\n\t"  // r21
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v25, v6, v18\n\t"  // k12 * r13    out[1][5]
            "vfmacc.vv      v26, v5, v18\n\t"  // k11 * r13    out[2][4]
            "vfmacc.vv      v28, v4, v11\n\t"  // k10 * r20    out[4][3]

            "vle.v          v14, (%4)\n\t"  // r30
            "addi           %4, %4, 16\n\t"

            "vfmacc.vv      v27, v5, v10\n\t"  // k11 * r14    out[3][4]
            "vfmacc.vv      v31, v3, v12\n\t"  // k02 * r15    out[7][2]
            "vfmacc.vv      v24, v7, v11\n\t"  // k20 * r20    out[0][6]

            "vle.v          v15, (%3)\n\t"  // r22
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v25, v7, v13\n\t"  // k20 * r21    out[1][6]
            "vfmacc.vv      v26, v6, v10\n\t"  // k12 * r14    out[2][5]
            "vfmacc.vv      v29, v4, v13\n\t"  // k10 * r21    out[5][3]

            "vle.v          v16, (%4)\n\t"  // r31
            "addi           %4, %4, 16\n\t"

            "vfmacc.vv      v27, v6, v12\n\t"  // k12 * r15    out[3][5]
            "vfmacc.vv      v28, v5, v13\n\t"  // k11 * r21    out[4][4]
            "vfmacc.vv      v30, v4, v15\n\t"  // k10 * r22    out[6][3]

            "vle.v          v17, (%3)\n\t"  // r23
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v24, v8, v13\n\t"  // k21 * r21    out[0][7]
            "vfmacc.vv      v25, v8, v15\n\t"  // k21 * r22    out[1][7]
            "vfmacc.vv      v29, v5, v15\n\t"  // k11 * r22    out[5][5]

            "vle.v          v18, (%4)\n\t"  // r32
            "addi           %4, %4, 16\n\t"

            "vfmacc.vv      v26, v7, v15\n\t"  // k20 * r22    out[2][6]
            "vfmacc.vv      v28, v6, v15\n\t"  // k12 * r22    out[4][5]
            "vfmacc.vv      v24, v9, v15\n\t"  // k22 * r22    out[0][8]

            "vle.v          v19, (%3)\n\t"  // r24
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v30, v5, v17\n\t"  // k11 * r23    out[6][4]
            "vfmacc.vv      v29, v6, v17\n\t"  // k12 * r23    out[5][5]

            "vfadd.vv       v24, v24, v0\n\t"  // out0 += bias

            "vfmacc.vv      v27, v7, v17\n\t"  // k20 * r23    out[3][6]
            "vfmacc.vv      v31, v4, v17\n\t"  // k10 * r23    out[7][3]

            "vle.v          v13, (%4)\n\t"  // r33
            "addi           %4, %4, 16\n\t"

            "vse.v          v24, (%6)\n\t"  // store out0
            "addi           %6, %6, 16\n\t"

            "vfmacc.vv      v26, v8, v17\n\t"  // k21 * r23    out[2][7]
            "vfmacc.vv      v28, v7, v14\n\t"  // k20 * r30    out[4][6]
            "vfmacc.vv      v29, v7, v16\n\t"  // k20 * r31    out[5][6]
            "vfmacc.vv      v30, v6, v19\n\t"  // k12 * r24    out[6][5]

            "vle.v          v14, (%3)\n\t"    // r25
            "addi           %3, %3, -16\n\t"  // r2 -= 8  ********* bump r2 to next 4 element addr
                                              // ************

            "vfmacc.vv      v25, v9, v17\n\t"  // k22 * r23    out[1][8]
            "vfmacc.vv      v27, v8, v19\n\t"  // k21 * r24    out[3][7]
            "vfmacc.vv      v28, v8, v16\n\t"  // k21 * r31    out[4][7]
            "vfmacc.vv      v31, v5, v19\n\t"  // k11 * r24    out[7][4]

            "vle.v          v10, (%1)\n\t"  // r00
            "addi           %1, %1, 16\n\t"

            "vfadd.vv       v25, v25, v0\n\t"  // out1 += bias

            "vfmacc.vv      v26, v9, v19\n\t"  // k22 * r24    out[2][8]
            "vfmacc.vv      v29, v8, v18\n\t"  // k21 * r32    out[5][7]
            "vfmacc.vv      v30, v7, v18\n\t"  // k20 * r32    out[6][6]

            "vle.v          v15, (%4)\n\t"  // r34
            "addi           %4, %4, 16\n\t"

            "vse.v          v25, (%6)\n\t"  // store out1
            "addi           %6, %6, 16\n\t"

            "vfadd.vv       v26, v26, v0\n\t"  // out2 += bias

            "vfmacc.vv      v27, v9, v14\n\t"  // k22 * r25    out[3][8]
            "vfmacc.vv      v28, v9, v18\n\t"  // k22 * r32    out[4][8]
            "vfmacc.vv      v31, v6, v14\n\t"  // k12 * r25    out[7][5]

            "vle.v          v11, (%1)\n\t"  // r01
            "addi           %1, %1, 16\n\t"

            "vse.v          v26, (%6)\n\t"  // store out2
            "addi           %6, %6, 16\n\t"

            "vfadd.vv       v27, v27, v0\n\t"  // out3 += bias

            "vfmacc.vv      v29, v9, v13\n\t"  // k22 * r33    out[5][8]
            "vfmacc.vv      v30, v8, v13\n\t"  // k21 * r33    out[6][7]
            "vfmacc.vv      v31, v7, v13\n\t"  // k20 * r33    out[7][6]

            "vse.v          v27, (%6)\n\t"  // store out3
            "addi           %6, %6, 16\n\t"

            "vfadd.vv       v28, v28, v0\n\t"  // out4 += bias

            "vle.v          v16, (%4)\n\t"    // r35
            "addi           %4, %4, -16\n\t"  // r3 -= 8  ********* bump r3 to next 4 element addr
                                              // ************

            "vfmacc.vv      v30, v9, v15\n\t"  // k22 * r34    out[6][8]
            "vfmacc.vv      v31, v8, v15\n\t"  // k21 * r34    out[7][7]

            "vse.v          v28, (%7)\n\t"  // store out4
            "addi           %7, %7, 16\n\t"

            "vfadd.vv       v29, v29, v0\n\t"  // out5 += bias

            "vle.v          v12, (%1)\n\t"  // r02
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v31, v9, v16\n\t"  // k22 * r35    out[7][8]

            "vse.v          v29, (%7)\n\t"  // store out5
            "addi           %7, %7, 16\n\t"

            "vfadd.vv       v30, v30, v0\n\t"  // out6 += bias
            "vfadd.vv       v31, v31, v0\n\t"  // out7 += bias

            "vse.v          v30, (%7)\n\t"  // store out6
            "addi           %7, %7, 16\n\t"

            "vse.v          v31, (%7)\n\t"  // store out7
            "addi           %7, %7, 16\n\t"

            "addi           t1, t1, -1\n\t"
            "bnez           t1, 2b\n\t"

            "addi           %1, %1, -48\n\t"  // r0 -= 24  ********* bump r0 to origin addr
                                              // ************

            "3:\n\t"  // out_w2

            "andi           t1, %9, 3\n\t"  // t1 = out_w & 3
            "srai           t2, t1, 1\n\t"  // t2 = (out_w & 3) >> 1
            "beqz           t2, 4f\n\t"

            // load 16 times, mac 36 times
            "vmv.v.x        v24, zero\n\t"

            "vle.v          v10, (%1)\n\t"  // r00
            "addi           %1, %1, 16\n\t"

            "vmv.v.x        v25, zero\n\t"

            "vle.v          v11, (%1)\n\t"  // r01
            "addi           %1, %1, 16\n\t"

            "vmv.v.x        v28, zero\n\t"

            "vfmacc.vv      v24, v1, v10\n\t"  // k00 * r00    out[0][0]

            "vle.v          v12, (%4)\n\t"  // r30
            "addi           %4, %4, 16\n\t"

            "vmv.v.x        v29, zero\n\t"

            "vfmacc.vv      v25, v1, v11\n\t"  // k00 * r01    out[1][0]

            "vle.v          v13, (%4)\n\t"  // r31
            "addi           %4, %4, 16\n\t"

            "vfmacc.vv      v28, v7, v12\n\t"  // k20 * r30    out[2][6]

            "vle.v          v14, (%1)\n\t"  // r02
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v24, v2, v11\n\t"  // k01 * r01    out[0][1]
            "vfmacc.vv      v29, v7, v13\n\t"  // k20 * r31    out[3][6]

            "vle.v          v15, (%4)\n\t"  // r32
            "addi           %4, %4, 16\n\t"

            "vfmacc.vv      v28, v8, v13\n\t"  // k21 * r31    out[2][7]
            "vfmacc.vv      v25, v2, v14\n\t"  // k01 * r02    out[1][1]

            "vle.v          v16, (%1)\n\t"    // r03
            "addi           %1, %1, -16\n\t"  // r0 -= 8  ********* bump r0 to next 2 element addr
                                              // ************

            "vfmacc.vv      v24, v3, v14\n\t"  // k02 * r02    out[0][2]
            "vfmacc.vv      v29, v8, v15\n\t"  // k21 * r32    out[3][7]

            "vle.v          v17, (%4)\n\t"    // r33
            "addi           %4, %4, -16\n\t"  // r3 -= 8  ********* bump r3 to next 2 element addr
                                              // ************

            "vfmacc.vv      v28, v9, v15\n\t"  // k22 * r32    out[2][8]
            "vfmacc.vv      v25, v3, v16\n\t"  // k02 * r03    out[1][2]

            "vle.v          v10, (%2)\n\t"  // r10
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v29, v9, v17\n\t"  // k22 * r33    out[3][8]

            "vle.v          v11, (%2)\n\t"  // r11
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v24, v4, v10\n\t"  // k10 * r10    out[0][3]
            "vfmacc.vv      v28, v1, v10\n\t"  // k00 * r10    out[2][0]

            "vle.v          v12, (%2)\n\t"  // r12
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v25, v4, v11\n\t"  // k10 * r11    out[1][3]
            "vfmacc.vv      v29, v1, v11\n\t"  // k00 * r11    out[3][0]
            "vfmacc.vv      v24, v5, v11\n\t"  // k11 * r11    out[0][4]
            "vfmacc.vv      v28, v2, v11\n\t"  // k01 * r11    out[2][1]

            "vle.v          v13, (%2)\n\t"    // r13
            "addi           %2, %2, -16\n\t"  // r1 -= 8  ********* bump r1 to next 2 element addr
                                              // ************

            "vfmacc.vv      v25, v5, v12\n\t"  // k11 * r12    out[1][4]
            "vfmacc.vv      v29, v2, v12\n\t"  // k01 * r12    out[3][1]
            "vfmacc.vv      v24, v6, v12\n\t"  // k12 * r12    out[0][4]
            "vfmacc.vv      v28, v3, v12\n\t"  // k02 * r12    out[2][2]

            "vle.v          v14, (%3)\n\t"  // r20
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v25, v6, v13\n\t"  // k12 * r13    out[1][5]
            "vfmacc.vv      v29, v3, v13\n\t"  // k02 * r13    out[3][2]

            "vle.v          v15, (%3)\n\t"  // r21
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v24, v7, v14\n\t"  // k20 * r20    out[0][6]
            "vfmacc.vv      v28, v4, v14\n\t"  // k10 * r20    out[2][3]

            "vle.v          v16, (%3)\n\t"  // r22
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v25, v7, v15\n\t"  // k20 * r21    out[1][6]
            "vfmacc.vv      v29, v4, v15\n\t"  // k10 * r21    out[3][3]
            "vfmacc.vv      v24, v8, v15\n\t"  // k21 * r21    out[0][7]
            "vfmacc.vv      v28, v5, v15\n\t"  // k11 * r21    out[2][4]

            "vle.v          v17, (%3)\n\t"    // r23
            "addi           %3, %3, -16\n\t"  // r2 -= 8  ********* bump r2 to next 2 element addr
                                              // ************

            "vfmacc.vv      v25, v8, v16\n\t"  // k21 * r22    out[1][7]
            "vfmacc.vv      v29, v5, v16\n\t"  // k11 * r22    out[3][4]
            "vfmacc.vv      v24, v9, v16\n\t"  // k22 * r22    out[0][8]
            "vfmacc.vv      v28, v6, v16\n\t"  // k12 * r22    out[2][5]

            "vfmacc.vv      v25, v9, v17\n\t"  // k22 * r23    out[1][8]
            "vfmacc.vv      v29, v6, v17\n\t"  // k12 * r23    out[3][5]

            "vfadd.vv       v24, v24, v0\n\t"
            "vfadd.vv       v25, v25, v0\n\t"
            "vfadd.vv       v28, v28, v0\n\t"
            "vfadd.vv       v29, v29, v0\n\t"  // add bias

            "vse.v          v24, (%6)\n\t"  // store outptr[0][0]
            "addi           %6, %6,16\n\t"

            "vse.v          v25, (%6)\n\t"  // store outptr[0][1]
            "addi           %6, %6, 16\n\t"

            "vse.v          v28, (%7)\n\t"  // store outptr[1][0]
            "addi           %7, %7,16\n\t"

            "vse.v          v29, (%7)\n\t"  // store outptr[1][1]
            "addi           %7, %7, 16\n\t"

            "4:\n\t"  // out_w1

            "andi           t2, t1, 1\n\t"  // t2 = (out_w & 3) & 1
            "beqz           t2, 5f\n\t"

            // load 12 times, mac 18 times
            "vmv.v.x        v24, zero\n\t"

            "vle.v          v10, (%1)\n\t"  // r00
            "addi           %1, %1, 16\n\t"

            "vmv.v.x        v28, zero\n\t"

            "vle.v          v11, (%1)\n\t"  // r01
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v24, v1, v10\n\t"  // k00 * r00    out[0][0]

            "vle.v          v12, (%1)\n\t"    // r02
            "addi           %1, %1, -16\n\t"  // r0 -= 4  ********* bump r0 to next 1 element addr
                                              // ************

            "vfmacc.vv      v24, v2, v11\n\t"  // k01 * r01    out[0][1]

            "vle.v          v13, (%2)\n\t"  // r10
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v24, v3, v12\n\t"  // k02 * r02    out[0][2]

            "vle.v          v14, (%2)\n\t"  // r11
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v28, v1, v13\n\t"  // k00 * r10    out[1][0]
            "vfmacc.vv      v24, v4, v13\n\t"  // k10 * r10    out[0][3]

            "vle.v          v15, (%2)\n\t"    // r12
            "addi           %2, %2, -16\n\t"  // r1 -= 4  ********* bump r1 to next 1 element addr
                                              // ************

            "vfmacc.vv      v28, v2, v14\n\t"  // k01 * r11    out[1][1]
            "vfmacc.vv      v24, v5, v14\n\t"  // k11 * r11    out[0][4]

            "vle.v          v16, (%3)\n\t"  // r20
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v28, v3, v15\n\t"  // k02 * r12    out[1][2]
            "vfmacc.vv      v24, v6, v15\n\t"  // k12 * r12    out[0][5]

            "vle.v          v17, (%3)\n\t"  // r21
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v28, v4, v16\n\t"  // k10 * r20    out[1][3]
            "vfmacc.vv      v24, v7, v16\n\t"  // k20 * r20    out[0][6]

            "vle.v          v18, (%3)\n\t"    // r22
            "addi           %3, %3, -16\n\t"  // r2 -= 4  ********* bump r2 to next 1 element addr
                                              // ************

            "vfmacc.vv      v28, v5, v17\n\t"  // k11 * r21    out[1][4]
            "vfmacc.vv      v24, v8, v17\n\t"  // k21 * r21    out[0][7]

            "vle.v          v10, (%4)\n\t"  // r30
            "addi           %4, %4, 16\n\t"

            "vfmacc.vv      v28, v6, v18\n\t"  // k12 * r22    out[1][5]
            "vfmacc.vv      v24, v9, v18\n\t"  // k22 * r22    out[0][8]

            "vle.v          v11, (%4)\n\t"  // r31
            "addi           %4, %4, 16\n\t"

            "vfmacc.vv      v28, v7, v10\n\t"  // k20 * r30    out[1][6]
            "vfadd.vv       v24, v24, v0\n\t"  // add bias

            "vle.v          v12, (%4)\n\t"    // r32
            "addi           %4, %4, -16\n\t"  // r3 -= 4  ********* bump r3 to next 1 element addr
                                              // ************

            "vfmacc.vv      v28, v8, v11\n\t"  // k21 * r31    out[1][7]

            "vse.v          v24, (%6)\n\t"  // store outptr[0][0]
            "addi           %6, %6, 16\n\t"

            "vfmacc.vv      v28, v9, v12\n\t"  // k22 * r32    out[1][8]
            "vfadd.vv       v28, v28, v0\n\t"  // add bias

            "vse.v          v28, (%7)\n\t"  // store outptr[1][0]
            "addi           %7, %7, 16\n\t"

            "5:\n\t"                         // out_h2_loop cnt
            "addi           t2, %10, 2\n\t"  // in_w + 2
            "slli           t2, t2, 4\n\t"   // (in_w + 2) * 2 * 8
            "slli           t3, %9, 4\n\t"   // out_w * 2 * 8

            "add            %1, %1, t2\n\t"
            "add            %2, %2, t2\n\t"
            "add            %3, %3, t2\n\t"
            "add            %4, %4, t2\n\t"  // r0/r1/r2/r3 += (in_w + 2) * 8(packn)

            "add            %6, %6, t3\n\t"
            "add            %7, %7, t3\n\t"  // outprt0/outptr1 += out_w * 8(packn)

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 1b\n\t"

            "6:\n\t"  // out_h1

            "andi           t0, %8, 1\n\t"  // t0 = out_h & 1
            "beqz           t0, 10f\n\t"

            "srai           t1, %9, 2\n\t"  // t1 = out_w >> 2
            "beqz           t1, 8f\n\t"

            // pre-load rxx
            "vle.v          v10, (%1)\n\t"  // r00
            "addi           %1, %1, 16\n\t"

            "vle.v          v11, (%1)\n\t"  // r01
            "addi           %1, %1, 16\n\t"

            // load 18 times, mac 36 次
            "7:\n\t"  // out_w4_loop

            "vmv.v.x        v24, zero\n\t"

            "vle.v          v12, (%1)\n\t"  // r02
            "addi           %1, %1, 16\n\t"
            "vmv.v.x        v25, zero\n\t"

            "vfmacc.vv      v24, v1, v10\n\t"  // k00 * r00    out[0][0]

            "vle.v          v13, (%1)\n\t"  // r03
            "addi           %1, %1, 16\n\t"

            "vmv.v.x        v26, zero\n\t"

            "vfmacc.vv      v25, v1, v11\n\t"  // k00 * r01    out[1][0]

            "vle.v          v14, (%1)\n\t"  // r04
            "addi           %1, %1, 16\n\t"
            "vmv.v.x        v27, zero\n\t"

            "vfmacc.vv      v24, v2, v11\n\t"  // k01 * r01    out[0][1]
            "vfmacc.vv      v26, v1, v12\n\t"  // k00 * r02    out[2][0]

            "vle.v          v15, (%1)\n\t"    // r05
            "addi           %1, %1, -16\n\t"  // r0 -= 8  ********* bump r0 to next 4 elements addr
                                              // ************

            "vfmacc.vv      v25, v2, v12\n\t"  // k01 * r02    out[1][1]
            "vfmacc.vv      v27, v1, v13\n\t"  // k00 * r03    out[3][0]

            "vle.v          v16, (%2)\n\t"  // r10
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v26, v2, v13\n\t"  // k01 * r03    out[2][1]
            "vfmacc.vv      v24, v3, v12\n\t"  // k02 * r02    out[0][2]
            "vfmacc.vv      v25, v3, v13\n\t"  // k02 * r03    out[1][2]

            "vle.v          v17, (%2)\n\t"  // r11
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v27, v2, v14\n\t"  // k01 * r04    out[3][1]
            "vfmacc.vv      v26, v3, v14\n\t"  // k02 * r04    out[2][2]

            "vle.v          v18, (%2)\n\t"  // r12
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v24, v4, v16\n\t"  // k10 * r10    out[0][3]
            "vfmacc.vv      v27, v3, v15\n\t"  // k02 * r05    out[3][2]

            "vle.v          v19, (%2)\n\t"  // r13
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v25, v4, v17\n\t"  // k10 * r11    out[1][3]
            "vfmacc.vv      v24, v5, v17\n\t"  // k11 * r11    out[0][4]

            "vle.v          v12, (%2)\n\t"  // r14
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v26, v4, v18\n\t"  // k10 * r12    out[2][3]
            "vfmacc.vv      v25, v5, v18\n\t"  // k12 * r13    out[1][4]

            "vle.v          v13, (%2)\n\t"    // r15
            "addi           %2, %2, -16\n\t"  // r1 -= 8  ********* bump r1 to next 4 elements addr
                                              // ************

            "vfmacc.vv      v27, v4, v19\n\t"  // k10 * r13    out[3][3]
            "vfmacc.vv      v24, v6, v18\n\t"  // k12 * r12    out[0][5]

            "vle.v          v14, (%3)\n\t"  // r20
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v25, v6, v19\n\t"  // k12 * r13    out[1][5]
            "vfmacc.vv      v26, v5, v19\n\t"  // k11 * r13    out[2][4]
            "vfmacc.vv      v27, v5, v12\n\t"  // k11 * r14    out[3][4]

            "vle.v          v15, (%3)\n\t"  // r21
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v24, v7, v14\n\t"  // k20 * r20    out[0][6]
            "vfmacc.vv      v26, v6, v12\n\t"  // k12 * r14    out[2][5]

            "vle.v          v16, (%3)\n\t"  // r22
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v27, v6, v13\n\t"  // k12 * r15    out[3][5]
            "vfmacc.vv      v25, v7, v15\n\t"  // k20 * r21    out[1][6]

            "vle.v          v17, (%3)\n\t"  // r23
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v24, v8, v15\n\t"  // k21 * r21    out[0][7]
            "vfmacc.vv      v26, v7, v16\n\t"  // k20 * r22    out[2][6]

            "vle.v          v18, (%3)\n\t"  // r24
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v25, v8, v16\n\t"  // k21 * r22    out[1][7]
            "vfmacc.vv      v27, v7, v17\n\t"  // k20 * r23    out[3][6]

            "vle.v          v19, (%3)\n\t"    // r25
            "addi           %3, %3, -16\n\t"  // r2 -= 8  ********* bump r2 to next 4 elements addr
                                              // ************

            "vfmacc.vv      v24, v9, v16\n\t"  // k22 * r22    out[0][8]
            "vfmacc.vv      v26, v8, v17\n\t"  // k21 * r23    out[2][7]

            "vle.v          v10, (%1)\n\t"  // r00
            "addi           %1, %1, 16\n\t"

            "vfadd.vv       v24, v24, v0\n\t"

            "vfmacc.vv      v25, v9, v17\n\t"  // k22 * r23    out[1][8]
            "vfmacc.vv      v27, v8, v18\n\t"  // k21 * r24    out[3][7]

            "vse.v          v24, (%6)\n\t"
            "addi           %6, %6, 16\n\t"  // store out0

            "vfadd.vv       v25, v25, v0\n\t"

            "vle.v          v11, (%1)\n\t"  // r01
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v26, v9, v18\n\t"  // k22 * r24    out[2][8]

            "vse.v          v25, (%6)\n\t"
            "addi           %6, %6, 16\n\t"  // store out1

            "vfmacc.vv      v27, v9, v19\n\t"  // k22 * r25    out[3][8]

            "vfadd.vv       v26, v26, v0\n\t"
            "vfadd.vv       v27, v27, v0\n\t"  // add bias

            "vse.v          v26, (%6)\n\t"
            "addi           %6, %6, 16\n\t"  // store out2

            "vse.v          v27, (%6)\n\t"
            "addi           %6, %6, 16\n\t"  // store out3

            "addi           t1, t1, -1\n\t"
            "bnez           t1, 7b\n\t"

            "addi           %1, %1, -32\n\t"  // r0 -= 16  ********* bump r0 to origin addr
                                              // ************

            "8:\n\t"  // out_w2

            "andi           t1, %9, 3\n\t"  // t1 = out_w & 3
            "srai           t2, t1, 1\n\t"  // t2 = (out_w & 3) >> 1
            "beqz           t2, 9f\n\t"

            // load 12 times, mac 18 times
            "vmv.v.x        v24, zero\n\t"

            "vle.v          v10, (%1)\n\t"  // r00
            "addi           %1, %1, 16\n\t"

            "vmv.v.x        v25, zero\n\t"

            "vle.v          v11, (%1)\n\t"  // r01
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v24, v1, v10\n\t"  // k00 * r00    out[0][0]

            "vle.v          v12, (%1)\n\t"  // r02
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v25, v1, v11\n\t"  // k00 * r01    out[1][0]
            "vfmacc.vv      v24, v2, v11\n\t"  // k01 * r01    out[0][1]

            "vle.v          v13, (%1)\n\t"    // r03
            "addi           %1, %1, -16\n\t"  // r0 -= 8  ********* bump r0 to next 2 elements addr
                                              // ************

            "vfmacc.vv      v25, v2, v12\n\t"  // k01 * r02    out[1][1]
            "vfmacc.vv      v24, v3, v12\n\t"  // k02 * r02    out[0][2]

            "vle.v          v14, (%2)\n\t"  // r10
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v25, v3, v13\n\t"  // k02 * r03    out[1][2]

            "vle.v          v15, (%2)\n\t"  // r11
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v24, v4, v14\n\t"  // k10 * r10    out[0][3]

            "vle.v          v16, (%2)\n\t"  // r12
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v25, v4, v15\n\t"  // k10 * r11    out[1][3]
            "vfmacc.vv      v24, v5, v15\n\t"  // k11 * r11    out[0][4]

            "vle.v          v17, (%2)\n\t"    // r13
            "addi           %2, %2, -16\n\t"  // r1 -= 8  ********* bump r1 to next 2 elements addr
                                              // ************

            "vfmacc.vv      v25, v5, v16\n\t"  // k11 * r12    out[1][4]
            "vfmacc.vv      v24, v6, v16\n\t"  // k12 * r12    out[0][5]

            "vle.v          v10, (%3)\n\t"  // r20
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v25, v6, v17\n\t"  // k12 * r13    out[1][5]

            "vle.v          v11, (%3)\n\t"  // r21
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v24, v7, v10\n\t"  // k20 * r20    out[0][6]

            "vle.v          v12, (%3)\n\t"  // r22
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v25, v7, v11\n\t"  // k20 * r21    out[1][6]
            "vfmacc.vv      v24, v8, v11\n\t"  // k21 * r21    out[0][7]

            "vle.v          v13, (%3)\n\t"    // r23
            "addi           %3, %3, -16\n\t"  // r2 -= 8  ********* bump r2 to next 2 elements addr
                                              // ************

            "vfmacc.vv      v25, v8, v12\n\t"  // k21 * r22    out[1][7]
            "vfmacc.vv      v24, v9, v12\n\t"  // k22 * r22    out[0][8]

            "vfmacc.vv      v25, v9, v13\n\t"  // k22 * r23    out[1][8]

            "vfadd.vv       v24, v24, v0\n\t"
            "vfadd.vv       v25, v25, v0\n\t"

            "vse.v          v24, (%6)\n\t"
            "addi           %6, %6, 16\n\t"

            "vse.v          v25, (%6)\n\t"
            "addi           %6, %6, 16\n\t"

            "9:\n\t"  // out_w1

            "andi           t2, t1, 1\n\t"  // t2 = (out_w & 3) & 1
            "beqz           t2, 10f\n\t"

            // load 9 times, mac 9 times
            "vle.v          v10, (%1)\n\t"  // r00
            "addi           %1, %1, 16\n\t"

            "vmv.v.x        v24, zero\n\t"

            "vle.v          v11, (%1)\n\t"  // r01
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v24, v1, v10\n\t"  // k00 * r00    out[0][0]

            "vle.v          v12, (%1)\n\t"    // r02
            "addi           %1, %1, -16\n\t"  // r0 -= 4  ********* bump r0 to next 1 elements addr
                                              // ************

            "vfmacc.vv      v24, v2, v11\n\t"  // k01 * r01    out[0][1]

            "vle.v          v13, (%2)\n\t"  // r10
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v24, v3, v12\n\t"  // k02 * r02    out[0][2]

            "vle.v          v14, (%2)\n\t"  // r11
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v24, v4, v13\n\t"  // k10 * r10    out[0][3]

            "vle.v          v15, (%2)\n\t"    // r12
            "addi           %2, %2, -16\n\t"  // r1 -= 4  ********* bump r1 to next 1 elements addr
                                              // ************

            "vfmacc.vv      v24, v5, v14\n\t"  // k11 * r11    out[0][4]

            "vle.v          v16, (%3)\n\t"  // r20
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v24, v6, v15\n\t"  // k12 * r12    out[0][5]

            "vle.v          v17, (%3)\n\t"  // r21
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v24, v7, v16\n\t"  // k20 * r20    out[0][6]

            "vle.v          v18, (%3)\n\t"    // r22
            "addi           %3, %3, -16\n\t"  // r2 -= 4  ********* bump r2 to next 1 elements addr
                                              // ************

            "vfmacc.vv      v24, v8, v17\n\t"  // k21 * r21    out[0][7]
            "vfmacc.vv      v24, v9, v18\n\t"  // k22 * r22    out[0][8]

            "vfadd.vv       v24, v24, v0\n\t"

            "vse.v          v24, (%6)\n\t"
            "addi           %6, %6, 16\n\t"

            "10:\n\t"
            // updata addr
            "addi           %1, %1, 32\n\t"  // r0 += 2 * 2(bytes) * 8(packn)
            "addi           %2, %2, 32\n\t"  // r1 += 2 * 2(bytes) * 8(packn)
            "addi           %3, %3, 32\n\t"  // r2 += 2 * 2(bytes) * 8(packn)

            : "=r"(kernel0),  // %0
              "=r"(r0),       // %1
              "=r"(r1),       // %2
              "=r"(r2),       // %3
              "=r"(r3),       // %4
              "=r"(bias0),    // %5
              "=r"(outptr0),  // %6
              "=r"(outptr1),  // %7
              "=r"(out_h),    // %8
              "=r"(out_w),    // %9
              "=r"(in_w)      // %10
            : "0"(kernel0), "1"(r0), "2"(r1), "3"(r2), "4"(r3), "5"(bias0), "6"(outptr0),
              "7"(outptr1), "8"(out_h), "9"(out_w), "10"(in_w)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31", "t0", "t1", "t2", "t3");
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}

/*
    (1) Algorithm works as follows:
        out_h1_loop:     out_h1_w4_loop  -->  out_h1_w2  -->  out_h1_w1

    (2) register definition:
        t0:         i_out_h
        t1:         i_out_w
        v0:         bias_data
        v1-v9:      [ k00, k01, k02, k10, k11, k12, k20, k21, k22 ]
        v10-v20:    r00-r08 / r10-r18 / r20-r28
        v28-v31:    output_data

    Due to pack8, both kxx and rxx actually occupy a v register

    TODO: how to pack for input / kernel / bias / output
          padding
*/
int shl_c906_dwconv3x3s2_pack8_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                    struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];  // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int tailstep = (in_w - 2 * out_w + in_w) * 8;

    for (int c = 0; c < in_c / 8; c++) {
        __fp16 *out = output_data + c * out_h * out_w * 8;
        __fp16 *outptr0 = out;

        const __fp16 *img0 = input_data + c * in_h * in_w * 8;
        const __fp16 *r0 = img0;
        const __fp16 *r1 = r0 + in_w * 8;
        const __fp16 *r2 = r1 + in_w * 8;

        const __fp16 *kernel0 = kernel_data + c * 9 * 8;

        const __fp16 *bias0 = NULL;
        if (bias_data && bias->dim_count != 0) {
            bias0 = bias_data + c * 8;
        }

        asm volatile(
            "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 4

            "vmv.v.x        v0, zero\n\t"  // clear v0
            "beqz           %4, 0f\n\t"    // if bias_data = NULL  clear v0
            "vle.v          v0, (%4)\n\t"

            "0:\n\t"

            "vle.v          v1, (%0)\n\t"    // k00
            "addi           %0, %0, 16\n\t"  // kernel += 8
            "vle.v          v2, (%0)\n\t"    // k01
            "addi           %0, %0, 16\n\t"
            "vle.v          v3, (%0)\n\t"  // k02
            "addi           %0, %0, 16\n\t"
            "vle.v          v4, (%0)\n\t"  // k10
            "addi           %0, %0, 16\n\t"
            "vle.v          v5, (%0)\n\t"  // k11
            "addi           %0, %0, 16\n\t"
            "vle.v          v6, (%0)\n\t"  // k12
            "addi           %0, %0, 16\n\t"
            "vle.v          v7, (%0)\n\t"  // k20
            "addi           %0, %0, 16\n\t"
            "vle.v          v8, (%0)\n\t"  // k21
            "addi           %0, %0, 16\n\t"
            "vle.v          v9, (%0)\n\t"  // k22

            "mv             t0, %6\n\t"  // i_out_h = out_h

            "1:\n\t"  // out_h1_loop

            "srai           t1, %7, 2\n\t"  // t1 = out_w >> 2
            "beqz           t1, 3f\n\t"

            // pre-load rxx
            "vle.v          v10, (%1)\n\t"   // r00
            "addi           %1, %1, 16\n\t"  // r0 += 8

            "vle.v          v11, (%1)\n\t"  // r01
            "addi           %1, %1, 16\n\t"

            "vle.v          v12, (%1)\n\t"  // r02
            "addi           %1, %1, 16\n\t"

            // load 27 times, mac 36 times
            "2:\n\t"  // out_w4_loop

            "vmv.v.x        v28, zero\n\t"
            "vmv.v.x        v29, zero\n\t"
            "vmv.v.x        v30, zero\n\t"
            "vmv.v.x        v31, zero\n\t"

            "vle.v          v13, (%1)\n\t"  // r03
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v28, v1, v10\n\t"  // k00 * r00    out0

            "vle.v          v14, (%1)\n\t"  // r04
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v29, v1, v12\n\t"  // k00 * r02    out1

            "vle.v          v15, (%1)\n\t"  // r05
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v28, v2, v11\n\t"  // k01 * r01    out0

            "vle.v          v16, (%1)\n\t"  // r06
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v29, v2, v13\n\t"  // k01 * r03    out1

            "vle.v          v17, (%1)\n\t"  // r07
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v28, v3, v12\n\t"  // k02 * r02    out0

            "vle.v          v18, (%1)\n\t"  // r08
            // "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v29, v3, v14\n\t"  // k02 * r04    out1

            "vle.v          v10, (%2)\n\t"  // r10
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v30, v1, v14\n\t"  // k00 * r04    out2

            "vle.v          v11, (%2)\n\t"  // r11
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v31, v1, v16\n\t"  // k00 * r06    out3

            "vle.v          v12, (%2)\n\t"  // r12
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v30, v2, v15\n\t"  // k01 * r05    out2

            "vle.v          v13, (%2)\n\t"  // r13
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v31, v2, v17\n\t"  // k01 * r07    out3

            "vle.v          v14, (%2)\n\t"  // r14
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v30, v3, v16\n\t"  // k02 * r06    out2

            "vle.v          v15, (%2)\n\t"  // r15
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v31, v3, v18\n\t"  // k02 * r08    out3

            "vle.v          v16, (%2)\n\t"  // r16
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v28, v4, v10\n\t"  // k10 * r10    out0

            "vle.v          v17, (%2)\n\t"  // r17
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v29, v4, v12\n\t"  // k10 * r12    out1

            "vle.v          v18, (%2)\n\t"  // r18
            // "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v28, v5, v11\n\t"  // k11 * r11    out0

            "vle.v          v10, (%3)\n\t"  // r20
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v29, v5, v13\n\t"  // k11 * r13    out1

            "vle.v          v11, (%3)\n\t"  // r21
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v28, v6, v12\n\t"  // k12 * r12    out0

            "vle.v          v12, (%3)\n\t"  // r22
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v29, v6, v14\n\t"  // k12 * r14    out1

            "vle.v          v13, (%3)\n\t"  // r23
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v30, v4, v14\n\t"  // k10 * r14    out2

            "vle.v          v14, (%3)\n\t"  // r24
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v31, v4, v16\n\t"  // k10 * r16    out3

            "vle.v          v19, (%3)\n\t"  // r25
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v30, v5, v15\n\t"  // k11 * r15    out2

            "vle.v          v20, (%3)\n\t"  // r26
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v31, v5, v17\n\t"  // k11 * r17    out3

            "vle.v          v15, (%3)\n\t"  // r27
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v30, v6, v16\n\t"  // k12 * r16    out2

            "vle.v          v16, (%3)\n\t"  // r28
            // "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v28, v7, v10\n\t"  // k20 * r20    out0
            "vfmacc.vv      v31, v6, v18\n\t"  // k12 * r18    out3

            "vle.v          v10, (%1)\n\t"  // r00  ******** load r00-r02 for next loop *******
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v28, v8, v11\n\t"  // k21 * r21    out0

            "vle.v          v11, (%1)\n\t"  // r01
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v28, v9, v12\n\t"  // k22 * r22    out0
            "vfmacc.vv      v29, v7, v12\n\t"  // k20 * r22    out1

            "vle.v          v12, (%1)\n\t"  // r02
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v29, v8, v13\n\t"  // k21 * r23    out1
            "vfmacc.vv      v29, v9, v14\n\t"  // k22 * r24    out1
            "vfmacc.vv      v30, v7, v14\n\t"  // k20 * r24    out2
            "vfmacc.vv      v31, v7, v20\n\t"  // k20 * r26    out3
            "vfmacc.vv      v30, v8, v19\n\t"  // k21 * r25    out2
            "vfmacc.vv      v31, v8, v15\n\t"  // k21 * r27    out3
            "vfmacc.vv      v30, v9, v20\n\t"  // k22 * r26    out2
            "vfmacc.vv      v31, v9, v16\n\t"  // k22 * r28    out3

            "vfadd.vv       v28, v28, v0\n\t"
            "vfadd.vv       v29, v29, v0\n\t"
            "vfadd.vv       v30, v30, v0\n\t"
            "vfadd.vv       v31, v31, v0\n\t"  // add bias

            "vse.v          v28, (%5)\n\t"
            "addi           %5, %5, 16\n\t"

            "vse.v          v29, (%5)\n\t"
            "addi           %5, %5, 16\n\t"

            "vse.v          v30, (%5)\n\t"
            "addi           %5, %5, 16\n\t"

            "vse.v          v31, (%5)\n\t"
            "addi           %5, %5, 16\n\t"

            "addi           t1, t1, -1\n\t"  // loop cnt
            "bnez           t1, 2b\n\t"

            "addi           %1, %1, -48\n\t"  // r0 -= 24  ********* bump r0 to origin addr
                                              // ************

            "3:\n\t"  // out_w2

            "andi           t1, %7, 3\n\t"  // t1 = out_w & 3
            "srai           t2, t1, 1\n\t"  // t2 = (out_w & 3) >> 1
            "beqz           t2, 4f\n\t"

            // load 15 times, mac 18 times
            "vle.v          v10, (%1)\n\t"  // r00
            "addi           %1, %1, 16\n\t"

            "vmv.v.x        v28, zero\n\t"

            "vle.v          v11, (%1)\n\t"  // r01
            "addi           %1, %1, 16\n\t"

            "vmv.v.x        v29, zero\n\t"

            "vle.v          v12, (%1)\n\t"  // r02
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v28, v1, v10\n\t"  // k00 * r00    out0

            "vle.v          v13, (%2)\n\t"  // r10
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v29, v1, v12\n\t"  // k00 * r02    out1

            "vle.v          v14, (%1)\n\t"  // r03
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v28, v2, v11\n\t"  // k01 * r01    out0

            "vle.v          v15, (%2)\n\t"  // r11
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v29, v2, v14\n\t"  // k01 * r03    out1

            "vle.v          v16, (%1)\n\t"  // r04

            "vfmacc.vv      v28, v3, v12\n\t"  // k02 * r02    out0

            "vle.v          v17, (%2)\n\t"  // r12
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v29, v3, v16\n\t"  // k02 * r04    out1

            "vle.v          v18, (%3)\n\t"  // r20
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v28, v4, v13\n\t"  // k10 * r10    out0

            "vle.v          v19, (%2)\n\t"  // r13
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v29, v4, v17\n\t"  // k10 * r12    out1
            "vfmacc.vv      v28, v6, v17\n\t"  // k12 * r12    out0

            "vle.v          v20, (%3)\n\t"  // r21
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v29, v5, v19\n\t"  // k11 * r13    out1
            "vfmacc.vv      v28, v5, v15\n\t"  // k11 * r11    out0

            "vle.v          v10, (%2)\n\t"  // r14
            // "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v28, v7, v18\n\t"  // k20 * r20    out0

            "vle.v          v11, (%3)\n\t"  // r22
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v29, v6, v10\n\t"  // k12 * r14    out1

            "vle.v          v12, (%3)\n\t"  // r23
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v28, v8, v20\n\t"  // k21 * r21    out0
            "vfmacc.vv      v29, v7, v11\n\t"  // k20 * r22    out1

            "vle.v          v13, (%3)\n\t"  // r24
            // "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v29, v8, v12\n\t"  // k21 * r23    out1
            "vfmacc.vv      v28, v9, v11\n\t"  // k22 * r22    out0
            "vfmacc.vv      v29, v9, v13\n\t"  // k22 * r24    out1

            "vfadd.vv       v28, v28, v0\n\t"
            "vfadd.vv       v29, v29, v0\n\t"  // add bias

            "vse.v          v28, (%5)\n\t"
            "addi           %5, %5, 16\n\t"

            "vse.v          v29, (%5)\n\t"
            "addi           %5, %5, 16\n\t"

            "4:\n\t"  // out_w1

            "andi           t2, t1, 1\n\t"  // t2 = (out_w & 3) & 1
            "beqz           t2, 5f\n\t"

            // load 9 times, mac 9 times
            "vle.v          v10, (%1)\n\t"  // r00
            "addi           %1, %1, 16\n\t"

            "vmv.v.x        v28, zero\n\t"

            "vle.v          v11, (%2)\n\t"  // r10
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v28, v1, v10\n\t"  // k00 * r00

            "vle.v          v12, (%3)\n\t"
            "addi           %3, %3, 16\n\t"  // r20

            "vfmacc.vv      v28, v4, v11\n\t"  // k10 * r10

            "vle.v          v13, (%1)\n\t"  // r01
            "addi           %1, %1, 16\n\t"

            "vfmacc.vv      v28, v7, v12\n\t"  // k20 * r20

            "vle.v          v14, (%2)\n\t"  // r11
            "addi           %2, %2, 16\n\t"

            "vfmacc.vv      v28, v2, v13\n\t"  // k01 * r01

            "vle.v          v15, (%3)\n\t"  // r21
            "addi           %3, %3, 16\n\t"

            "vfmacc.vv      v28, v5, v14\n\t"  // k11 * r11

            "vle.v          v16, (%1)\n\t"  // r02

            "vfmacc.vv      v28, v8, v15\n\t"  // k21 * r21

            "vle.v          v17, (%2)\n\t"  // r12

            "vfmacc.vv      v28, v3, v16\n\t"  // k02 * r02

            "vle.v          v18, (%3)\n\t"  // r22

            "vfmacc.vv      v28, v6, v17\n\t"  // k12 * r12
            "vfmacc.vv      v28, v9, v18\n\t"  // k22 * r22

            "vfadd.vv       v28, v28, v0\n\t"  // add bias

            "vse.v          v28, (%5)\n\t"
            "addi           %5, %5, 16\n\t"

            "5:\n\t"

            "slli           t2, %8, 1\n\t"  // t2 = tailstep * 2
            "add            %1, %1, t2\n\t"
            "add            %2, %2, t2\n\t"
            "add            %3, %3, t2\n\t"  // r0/r1/r2 += tailstep

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 1b\n\t"

            : "=r"(kernel0),  // %0
              "=r"(r0),       // %1
              "=r"(r1),       // %2
              "=r"(r2),       // %3
              "=r"(bias0),    // %4
              "=r"(outptr0),  // %5
              "=r"(out_h),    // %6
              "=r"(out_w),    // %7
              "=r"(tailstep)  // %8
            : "0"(kernel0), "1"(r0), "2"(r1), "3"(r2), "4"(bias0), "5"(outptr0), "6"(out_h),
              "7"(out_w), "8"(tailstep)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v28", "v29",
              "v30", "v31", "t0", "t1", "t2");
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}
