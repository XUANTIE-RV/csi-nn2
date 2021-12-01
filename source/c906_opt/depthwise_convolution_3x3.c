/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

/* CSI-NN2 version 1.10.x */

#include "csi_c906.h"

#ifndef DWCONV3X3S1
#define DWCONV3X3S1 csi_c906_dwconv3x3s1
#endif

#ifndef DWCONV3X3S2
#define DWCONV3X3S2 csi_c906_dwconv3x3s2
#endif



/*
    (1) Algorithm works as follows:
        out_h2:     out_h2_w8_loop --> out_h2_w4 --> out_h2_wtail
        out_h_tail: out_h1_w8_loop --> out_h1_w4 --> out_h1_wtail

        out_h2_w8:                                    out_h2_w4:                                      ||    out_h1_w8:               out_h1_w4:
            outptr0[0-7]:        outptr1[0-7]:            outptr0[0-3]:         outptr1[0-3]          ||        outptr0[0-7]:            outptr0[0-3]:
                k00 * r0[0-7]        k00 * r1[0-7]            k00 * r0[0-3]          k00 * r1[0-3]    ||            k00 * r0[0-7]            k00 * r0[0-3]
                k01 * r0[1-8]        k01 * r1[1-8]            k01 * r0[1-4]          k01 * r1[1-4]    ||            k01 * r0[1-8]            k01 * r0[1-4]
                k02 * r0[2-9]        k02 * r1[2-9]            k02 * r0[2-5]          k02 * r1[2-5]    ||            k02 * r0[2-9]            k02 * r0[2-5]
                k10 * r1[0-7]        k10 * r2[0-7]            k10 * r1[0-3]          k10 * r2[0-3]    ||            k10 * r1[0-7]            k10 * r1[0-3]
                k11 * r1[1-8]        k11 * r2[1-8]            k11 * r1[1-4]          k11 * r2[1-4]    ||            k11 * r1[1-8]            k11 * r1[1-4]
                k12 * r1[2-9]        k12 * r2[2-9]            k12 * r1[2-5]          k12 * r2[2-5]    ||            k12 * r1[2-9]            k12 * r1[2-5]
                k20 * r2[0-7]        k20 * r3[0-7]            k20 * r2[0-3]          k20 * r3[0-3]    ||            k20 * r2[0-7]            k20 * r2[0-3]
                k21 * r2[1-8]        k21 * r3[1-8]            k21 * r2[1-4]          k21 * r3[1-4]    ||            k21 * r2[1-8]            k21 * r2[1-4]
                k22 * r2[2-9]        k22 * r3[2-9]            k22 * r2[2-5]          k22 * r3[2-5]    ||            k22 * r2[2-9]            k22 * r2[2-5]

            h2_w8_loop execution process:

                load r0[0-7]  -->  load r0[1-8]  -->  load r0[2-9]  -->     // Load r0[0-7] r0[1-8] r0[-9] before the loop to facilitate pipeline work

            --> load bias0[0-7]  -->  load r3[0-7]  -->  load bias1[0-7]  -->  load r3[1-8]  -->  k00*r0[0-7] / k20*r3[0-7]  -->
            -
            -   load r3[2-9]  -->  k01*r0[1-8] / k21*r3[1-8]  -->  load r1[0-7]  -->  k02*r0[2-9] / k22*r3[2-9]  -->  load r1[1-8]  -->  k10*r1[0-7] / k00*r1[0-7]  -->
            -
            -   load r1[2-9]  -->  k11*r1[1-8] / k01*r1[1-8]  -->  load r2[0-7]  -->  k12*r1[2-9] / k02*r1[2-9]  -->  load r2[1-8]  -->  k20*r2[0-7] / k10*r2[0-7]  -->
            -
            -   load r2[2-9]  -->  k21*r2[1-8] / k11*r2[1-8]  -->  load r0[0-7]  -->  k22*r2[2-9] / k12*r2[2-9]  -->  load r0[1-8]  -->  load r0[2-9]  ----------------
            -                                                                                                                                                         -
            -----------------------------------------------------------------------------------------------------------------------------------------------------------


            h1_w8_loop execution process:

                load r0[0-7]  -->  load r0[1-8]  -->  load r0[2-9]  -->

            --> load bias0[0-7]  -->  k00*r0[0-7]  -->  load r1[0-7]  -->  k01*r0[1-8]  -->  load r1[1-8]  --> k02*r0[2-9]  -->  load r1[2-9]  -->  k10*r1[0-7]  -->
            -
            -   load r2[0-7]  -->  k11*r1[1-8]  -->  load r2[1-8]  -->  k12*r1[2-9]  -->  load r2[2-9]  -->  k20*r2[0-7]  -->  load r0[0-7]  -->   k21*r2[1-8]  -->
            -
            -   load r0[1-8]  -->  k22*r2[2-9]  -->  load r0[2-9]  -------------------------------------------------------------------------------------------------
            -                                                                                                                                                      -
            --------------------------------------------------------------------------------------------------------------------------------------------------------

    (2) register definition:
        t0:         i_out_h
        t1-t2:      i_out_w
        v0-v1:      bias0[0-7], output_data(acc)
        v2-v3:      bias1[0-7], output_data(acc)
        v4-v9:      r0  v4,v5:r0[0-7]  v6,v7:r0[1-8]   v8,v9:r0[2-9]
        v10-v15:    r3
        v16-v21:    r1
        v22-v27:    r2
        ft0-ft8:    [ k00,k01,k02,k10,k11,k12,k20,k21,k22 ]
        ft11:       constant float 0.0f, used by fusing relu

    (3) // TODO: support channel mult ??
                 opt padding

*/

int DWCONV3X3S1(struct csi_tensor *input,
                struct csi_tensor *output,
                struct csi_tensor *kernel,
                struct csi_tensor *bias,
                struct conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];       // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    float *input_padd_buf = (float *)csi_mem_alloc(in_c * (in_h + params->pad_top + params->pad_down) * (in_w + params->pad_left + params->pad_right) * sizeof(float));

    csi_c906_pad_input(input_data, input_padd_buf, in_c, in_h, in_w, in_h + params->pad_top + params->pad_down, in_w + params->pad_left + params->pad_right, params->pad_top, params->pad_left);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

#pragma omp parallel for num_threads(8)
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

        const float *kernel0 = kernel_data + c * 9;

#if __riscv_vector == 128

        asm volatile(
            "vsetvli        zero, zero, e32, m2\n\t"

#ifdef  FUSE_CONV_RELU
            "fmv.w.x        ft11, zero\n\t"
#endif  // FUSE_CONV_RELU

            "flw            ft0, 0(%0)\n\t"     // k00
            "flw            ft1, 4(%0)\n\t"     // k01
            "flw            ft2, 8(%0)\n\t"     // k02
            "flw            ft3, 12(%0)\n\t"    // k10
            "flw            ft4, 16(%0)\n\t"    // k11
            "flw            ft5, 20(%0)\n\t"    // k12
            "flw            ft6, 24(%0)\n\t"    // k20
            "flw            ft7, 28(%0)\n\t"    // k21
            "flw            ft8, 32(%0)\n\t"    // k22

            "srai           t0, %7, 1\n\t"      // t0 = out_h >> 1
            "beqz           t0, 7f\n\t"

        "1:\n\t"        // out_h_loop2

            "srai           t1, %8, 3\n\t"      // t1 = out_w >> 3
            "beqz           t1, 3f\n\t"

            "vsetvli        zero, zero, e32, m2\n\t"    // set vl = 8
            "vlw.v          v4, (%1)\n\t"       // r0[0-7]
            "addi           %1, %1, 4\n\t"      // r0++
            "vlw.v          v6, (%1)\n\t"       // r0[1-8]
            "addi           %1, %1, 4\n\t"      // r0++
            "vlw.v          v8, (%1)\n\t"       // r0[2-9]

            "2:\n\t"     // out_w_loop8

                "vfmv.v.f       v0, %20\n\t"        // bias0[0-7]
                "addi           %1, %1, 24\n\t"     // r0 += 6

                "vlw.v          v10, (%4)\n\t"      // r3[0-7]
                "addi           %4, %4, 4\n\t"      // r3++
                "vfmv.v.f       v2, %20\n\t"        // bias1[0-7]

                "vlw.v          v12, (%4)\n\t"      // r3[1-8]
                "addi           %4, %4, 4\n\t"      // r3++

                "vfmacc.vf      v0, ft0, v4\n\t"    // k00 * r0[0-7]
                "vfmacc.vf      v2, ft6, v10\n\t"   // k20 * r3[0-7]

                "vlw.v          v14, (%4)\n\t"      // r3[2-9]
                "addi           %4, %4, 24\n\t"     // r3 += 6

                "vfmacc.vf      v0, ft1, v6\n\t"    // k01 * r0[1-8]
                "vfmacc.vf      v2, ft7, v12\n\t"   // k21 * r3[1-8]

                "vlw.v          v16, (%2)\n\t"      // r1[0-7]
                "addi           %2, %2, 4\n\t"      // r1++

                "vfmacc.vf      v0, ft2, v8\n\t"    // k02 * r0[2-9]
                "vfmacc.vf      v2, ft8, v14\n\t"   // k22 * r3[2-9]

                "vlw.v          v18, (%2)\n\t"      // r1[1-8]
                "addi           %2, %2, 4\n\t"      // r1++

                "vfmacc.vf      v0, ft3, v16\n\t"   // k10 * r1[0-7]
                "vfmacc.vf      v2, ft0, v16\n\t"   // k00 * r1[0-7]

                "vlw.v          v20, (%2)\n\t"      // r1[2-9]
                "addi           %2, %2, 24\n\t"     // r1 += 6

                "vfmacc.vf      v0, ft4, v18\n\t"   // k11 * r1[1-8]
                "vfmacc.vf      v2, ft1, v18\n\t"   // k01 * r1[1-8]

                "vlw.v          v22, (%3)\n\t"      // r2[0-7]
                "addi           %3, %3, 4\n\t"      // r2++

                "vfmacc.vf      v0, ft5, v20\n\t"   // k12 * r1[2-9]
                "vfmacc.vf      v2, ft2, v20\n\t"   // k02 * r1[2-9]

                "vlw.v          v24, (%3)\n\t"      // r2[1-8]
                "addi           %3, %3, 4\n\t"      // r2++

                "vfmacc.vf      v0, ft6, v22\n\t"   // k20 * r2[0-7]
                "vfmacc.vf      v2, ft3, v22\n\t"   // k10 * r2[0-7]

                "vlw.v          v26, (%3)\n\t"      // r2[2-9]
                "addi           %3, %3, 24\n\t"     // r2 += 6

                "vfmacc.vf      v0, ft7, v24\n\t"   // k21 * r2[1-8]
                "vfmacc.vf      v2, ft4, v24\n\t"   // k11 * r2[1-8]

                "vlw.v          v4, (%1)\n\t"       // r0[0-7]  load r0 for next loop
                "addi           %1, %1, 4\n\t"      // r0++

                "vfmacc.vf      v0, ft8, v26\n\t"   // k22 * r2[2-9]

                "vlw.v          v6, (%1)\n\t"       // r0[1-8]
                "addi           %1, %1, 4\n\t"      // r0++

#ifdef  FUSE_CONV_RELU
                "vfmax.vf       v0, v0, ft11\n\t"   // **** relu ****
#endif  // FUSE_CONV_RELU

                "vsw.v          v0, (%5)\n\t"       // store line0 8 elements on outptr0
                "addi           %5, %5, 32\n\t"     // outptr0 += 8

                "vfmacc.vf      v2, ft5, v26\n\t"   // k12 * r2[2-9]

                "vlw.v          v8, (%1)\n\t"       // r0[2-9]

#ifdef  FUSE_CONV_RELU
                "vfmax.vf       v2, v2, ft11\n\t"   // **** relu ****
#endif  // FUSE_CONV_RELU

                "vsw.v          v2, (%6)\n\t"       // store line1 8 elements on outptr1
                "addi           %6, %6, 32\n\t"     // outptr1 += 8

                "addi           t1, t1, -1\n\t"
                "bnez           t1, 2b\n\t"

                "addi           %1, %1, -8\n\t"     // r0 -= 2  ********* bump r0 to origin addr ************

            "3:\n\t"     // out_w4 // h2循环中只有执行一次的机会
                "andi           t1, %8, 7\n\t"      // t1 = out_w & 7
                "srai           t2, t1, 2\n\t"      // t2 = (out_w & 7) >> 2
                "beqz           t2, 4f\n\t"

                "vsetvli        zero, zero, e32, m1\n\t"    // set vl = 4

                "vlw.v          v4, (%1)\n\t"       // r0[0-3]
                "addi           %1, %1, 4\n\t"      // r0++

                "vfmv.v.f       v0, %20\n\t"        // bias0[0-3]

                "vlw.v          v10, (%4)\n\t"      // r3[0-3]
                "addi           %4, %4, 4\n\t"      // r3++

                "vfmv.v.f       v2, %20\n\t"        // bias1[0-3]

                "vlw.v          v5, (%1)\n\t"       // r0[1-4]
                "addi           %1, %1, 4\n\t"      // r0++

                "vlw.v          v11, (%4)\n\t"      // r3[1-4]
                "addi           %4, %4, 4\n\t"      // r3++

                "vfmacc.vf      v0, ft0, v4\n\t"    // k00 * r0[0-3]
                "vfmacc.vf      v2, ft6, v10\n\t"   // k20 * r3[0-3]

                "vlw.v          v6, (%1)\n\t"       // r0[2-5]
                "addi           %1, %1, 8\n\t"      // r0 += 2

                "vlw.v          v12, (%4)\n\t"      // r3[2-5]
                "addi           %4, %4, 8\n\t"      // r3 += 2

                "vfmacc.vf      v0, ft1, v5\n\t"    // k01 * r0[1-4]
                "vfmacc.vf      v2, ft7, v11\n\t"   // k21 * r3[1-4]

                "vlw.v          v16, (%2)\n\t"      // r1[0-3]
                "addi           %2, %2, 4\n\t"      // r1++

                "vfmacc.vf      v0, ft2, v6\n\t"    // k02 * r0[2-5]
                "vfmacc.vf      v2, ft8, v12\n\t"   // k22 * r3[2-5]

                "vlw.v          v17, (%2)\n\t"      // r1[1-4]
                "addi           %2, %2, 4\n\t"      // r1++

                "vfmacc.vf      v0, ft3, v16\n\t"   // k10 * r1[0-3]
                "vfmacc.vf      v2, ft0, v16\n\t"   // k00 * r1[0-3]

                "vlw.v          v18, (%2)\n\t"      // r1[2-5]
                "addi           %2, %2, 8\n\t"      // r1 += 2

                "vfmacc.vf      v0, ft4, v17\n\t"   // k11 * r1[1-4]
                "vfmacc.vf      v2, ft1, v17\n\t"   // k01 * r1[1-4]

                "vlw.v          v22, (%3)\n\t"      // r2[0-3]
                "addi           %3, %3, 4\n\t"      // r2++

                "vfmacc.vf      v0, ft5, v18\n\t"   // k12 * r1[2-5]
                "vfmacc.vf      v2, ft2, v18\n\t"   // k02 * r1[2-5]]

                "vlw.v          v23, (%3)\n\t"      // r2[1-4]
                "addi           %3, %3, 4\n\t"      // r2++

                "vfmacc.vf      v0, ft6, v22\n\t"   // k20 * r2[0-3]
                "vfmacc.vf      v2, ft3, v22\n\t"   // k10 * r2[0-3]

                "vlw.v          v24, (%3)\n\t"      // r2[2-5]
                "addi           %3, %3, 8\n\t"      // r2 += 2

                "vfmacc.vf      v0, ft7, v23\n\t"   // k21 * r2[1-4]
                "vfmacc.vf      v2, ft4, v23\n\t"   // k11 * r2[1-4]

                "vfmacc.vf      v0, ft8, v24\n\t"   // k22 * r2[2-5]
                "vfmacc.vf      v2, ft5, v24\n\t"   // k12 * r2[2-5]

#ifdef  FUSE_CONV_RELU
                "vfmax.vf       v0, v0, ft11\n\t"   // **** relu ****
                "vfmax.vf       v2, v2, ft11\n\t"   // **** relu ****
#endif  // FUSE_CONV_RELU

                "vsw.v          v0, (%5)\n\t"       // store line0 4 elements on outptr0
                "addi           %5, %5, 16\n\t"     // outptr0 += 4
                "vsw.v          v2, (%6)\n\t"       // store line1 4 elements on outptr1
                "addi           %6, %6, 16\n\t"     // outptr1 += 4

            "4:\n\t"     // out_w_tail
                "andi           t2, t1, 3\n\t"      // t2 = (out_w & 7) & 3
                "beqz           t2, 6f\n\t"

                "vfmv.v.f       v0, %20\n\t"        // bias0[0-3] / bias1[0-3]
                "li             t5, 3\n\t"
                "vsetvli        zero, t5, e32, m1\n\t"  // set vl = 3

                "vlw.v          v5, (%0)\n\t"       // k0
                "addi           %0, %0, 12\n\t"
                "vlw.v          v6, (%0)\n\t"       // k1
                "addi           %0, %0, 12\n\t"
                "vlw.v          v7, (%0)\n\t"       // k2

            "5:\n\t"    // out_w_tail

                "vlw.v          v4, (%1)\n\t"       // r0
                "addi           %1, %1, 4\n\t"      // r0++

                "vlw.v          v16, (%2)\n\t"      // r1
                "addi           %2, %2, 4\n\t"      // r1++

                "vlw.v          v22, (%3)\n\t"      // r2
                "addi           %3, %3, 4\n\t"      // r2++

                "vlw.v          v10, (%4)\n\t"      // r3
                "addi           %4, %4, 4\n\t"      // r3++

                "vfmul.vv       v8, v4, v5\n\t"     // r0 * k0
                "vfmacc.vv      v8, v16, v6\n\t"    // += r1 * k1
                "vfmacc.vv      v8, v22, v7\n\t"    // += r2 * k2

                "vfredsum.vs    v11, v8, v0\n\t"    // v11[0] = v0[0] + sum(v8[0..2])
                "vfmv.f.s       ft9, v11\n\t"       // ft9 = v11[0]


                "vfmul.vv       v9, v16, v5\n\t"    // r1 * k0
                "vfmacc.vv      v9, v22, v6\n\t"    // += r2 * k1
                "vfmacc.vv      v9, v10, v7\n\t"    // += r3 * k2

                "vfredsum.vs    v12, v9, v0\n\t"    // v12[0] = v0[0] + sum(v9[0..2])
                "vfmv.f.s       ft10, v12\n\t"      // ft10 = v12[0]

#ifdef  FUSE_CONV_RELU
                "fmax.s         ft9, ft9, ft11\n\t"     // **** relu ****
                "fmax.s         ft10, ft10, ft11\n\t"   // **** relu ****
#endif  // FUSE_CONV_RELU

                "fsw            ft9, 0(%5)\n\t"
                "addi           %5, %5, 4\n\t"
                "fsw            ft10, 0(%6)\n\t"
                "addi           %6, %6, 4\n\t"

                "addi           t2, t2, -1\n\t"
                "bnez           t2, 5b\n\t"

                "addi           %0, %0, -24\n\t"    // kernel -= 6  ********* bump kernel_data to origin addr ************

        "6:\n\t"        // out_h_loop2 cnt

            "slli           t3, %9, 2\n\t"      // in_w * 4
            "addi           t3, t3, 8\n\t"      // in_w * 4 + 8

            "slli           t4, %8, 2\n\t"      // out_w * 4

            "add            %1, %1, t3\n\t"     // r0 += 2 + in_w
            "add            %2, %2, t3\n\t"     // r1 += 2 + in_w
            "add            %3, %3, t3\n\t"     // r2 += 2 + in_w
            "add            %4, %4, t3\n\t"     // r3 += 2 + in_w

            "add            %5, %5, t4\n\t"     // outptr0 += out_w
            "add            %6, %6, t4\n\t"     // outptr1 += out_w

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 1b\n\t"

        "7:\n\t"         // out_h_tail // 只有执行一次的机会
            "andi           t0, %7, 1\n\t"      // t0 = out_h & 1
            "beqz           t0, 12f\n\t"

            "srai           t1, %8, 3\n\t"      // t1 = out_w >> 3
            "beqz           t1, 9f\n\t"

            "vsetvli        zero, zero, e32, m2\n\t"    // set vl = 8
            "vlw.v          v4, (%1)\n\t"       // r0[0-7]
            "addi           %1, %1, 4\n\t"      // r0++
            "vlw.v          v6, (%1)\n\t"       // r0[1-8]
            "addi           %1, %1, 4\n\t"      // r0++
            "vlw.v          v8, (%1)\n\t"       // r0[2-9]

            "8:\n\t"     // out_w_loop8 (可以考虑用m1，指令更多，但是还可以再错开，便于流水?)

                "vfmv.v.f       v0, %20\n\t"        // bias0[0-7]
                "addi           %1, %1, 24\n\t"     // r0 += 6

                "vfmacc.vf      v0, ft0, v4\n\t"    // k00 * r0[0-7]

                "vlw.v          v16, (%2)\n\t"      // r1[0-7]
                "addi           %2, %2, 4\n\t"      // r1++

                "vfmacc.vf      v0, ft1, v6\n\t"    // k01 * r0[1-8]

                "vlw.v          v18, (%2)\n\t"      // r1[1-8]
                "addi           %2, %2, 4\n\t"      // r1++

                "vfmacc.vf      v0, ft2, v8\n\t"    // k02 * r0[2-9]

                "vlw.v          v20, (%2)\n\t"      // r1[2-9]
                "addi           %2, %2, 24\n\t"     // r1 += 6

                "vfmacc.vf      v0, ft3, v16\n\t"   // k10 * r1[0-7]

                "vlw.v          v22, (%3)\n\t"      // r2[0-7]
                "addi           %3, %3, 4\n\t"      // r2++

                "vfmacc.vf      v0, ft4, v18\n\t"   // k11 * r1[1-8]

                "vlw.v          v24, (%3)\n\t"      // r2[1-8]
                "addi           %3, %3, 4\n\t"      // r2++

                "vfmacc.vf      v0, ft5, v20\n\t"   // k12 * r1[2-9]

                "vlw.v          v26, (%3)\n\t"      // r2[2-9]
                "addi           %3, %3, 24\n\t"     // r2 += 6

                "vfmacc.vf      v0, ft6, v22\n\t"   // k20 * r2[0-7]

                "vlw.v          v4, (%1)\n\t"       // r0[0-7]
                "addi           %1, %1, 4\n\t"      // r0++

                "vfmacc.vf      v0, ft7, v24\n\t"   // k21 * r2[1-8]

                "vlw.v          v6, (%1)\n\t"       // r0[1-8]
                "addi           %1, %1, 4\n\t"      // r0++

                "vfmacc.vf      v0, ft8, v26\n\t"   // k22 * r2[2-9]

                "vlw.v          v8, (%1)\n\t"       // r0[2-9]

#ifdef  FUSE_CONV_RELU
                "vfmax.vf       v0, v0, ft11\n\t"       // **** relu ****
#endif  // FUSE_CONV_RELU

                "vsw.v          v0, (%5)\n\t"       // store line0 8 elements on outptr0
                "addi           %5, %5, 32\n\t"     // outptr0 += 8

                "addi           t1, t1, -1\n\t"
                "bnez           t1, 8b\n\t"

                "addi           %1, %1, -8\n\t"     // r0 -= 8  ********* bump r0 to origin addr ************

            "9:\n\t"     // out_w4
                "andi           t1, %8, 7\n\t"      // t1 = out_w & 7
                "srai           t2, t1, 2\n\t"      // t2 = (out_w & 7) >> 2
                "beqz           t2, 10f\n\t"

                "vsetvli        zero, zero, e32, m1\n\t"    // set vl = 4

                "vlw.v          v4, (%1)\n\t"       // r0[0-3]
                "addi           %1, %1, 4\n\t"      // r0++

                "vfmv.v.f       v0, %20\n\t"        // bias0[0-3]

                "vlw.v          v5, (%1)\n\t"       // r0[1-4]
                "addi           %1, %1, 4\n\t"      // r0++

                "vfmacc.vf      v0, ft0, v4\n\t"    // k00 * r0[0-3]

                "vlw.v          v6, (%1)\n\t"       // r0[2-5]
                "addi           %1, %1, 8\n\t"      // r0 += 2

                "vfmacc.vf      v0, ft1, v5\n\t"    // k01 * r0[1-4]

                "vlw.v          v16, (%2)\n\t"      // r1[0-3]
                "addi           %2, %2, 4\n\t"     // r1++

                "vfmacc.vf      v0, ft2, v6\n\t"    // k02 * r0[2-5]

                "vlw.v          v17, (%2)\n\t"      // r1[1-4]
                "addi           %2, %2, 4\n\t"      // r1++

                "vfmacc.vf      v0, ft3, v16\n\t"   // k10 * r1[0-3]

                "vlw.v          v18, (%2)\n\t"      // r1[2-5]
                "addi           %2, %2, 8\n\t"      // r1 += 2

                "vfmacc.vf      v0, ft4, v17\n\t"   // k11 * r1[1-4]

                "vlw.v          v22, (%3)\n\t"      // r2[0-3]
                "addi           %3, %3, 4\n\t"      // r2++

                "vfmacc.vf      v0, ft5, v18\n\t"   // k12 * r1[2-5]

                "vlw.v          v23, (%3)\n\t"      // r2[1-4]
                "addi           %3, %3, 4\n\t"      // r2++

                "vfmacc.vf      v0, ft6, v22\n\t"   // k20 * r2[0-3]

                "vlw.v          v24, (%3)\n\t"      // r2[2-5]
                "addi           %3, %3, 8\n\t"      // r2 += 2

                "vfmacc.vf      v0, ft7, v23\n\t"   // k21 * r2[1-4]

                "vfmacc.vf      v0, ft8, v24\n\t"   // k22 * r2[2-5]

#ifdef  FUSE_CONV_RELU
                "vfmax.vf       v0, v0, ft11\n\t"       // **** relu ****
#endif  // FUSE_CONV_RELU

                "vsw.v          v0, (%5)\n\t"       // store line0 4 elements on outptr0
                "addi           %5, %5, 16\n\t"     // outptr0 += 4

            "10:\n\t"       // out_w_tail
                "andi           t2, t1, 3\n\t"
                "beqz           t2, 12f\n\t"

                "vfmv.v.f       v0, %20\n\t"        // bias0[0-3]
                "li             t5, 3\n\t"
                "vsetvli        zero, t5, e32, m1\n\t"  // set vl = 3

                "vlw.v          v5, (%0)\n\t"       // k0
                "addi           %0, %0, 12\n\t"
                "vlw.v          v6, (%0)\n\t"       // k1
                "addi           %0, %0, 12\n\t"
                "vlw.v          v7, (%0)\n\t"       // k2

            "11:\n\t"       // out_w_tail

                "vlw.v          v4, (%1)\n\t"       // r0
                "addi           %1, %1, 4\n\t"      // r0++

                "vlw.v          v16, (%2)\n\t"      // r1
                "addi           %2, %2, 4\n\t"      // r1++

                "vlw.v          v22, (%3)\n\t"      // r2
                "addi           %3, %3, 4\n\t"      // r2++

                "vfmul.vv       v8, v4, v5\n\t"     // r0 * k0
                "vfmacc.vv      v8, v16, v6\n\t"    // += r1 * k1
                "vfmacc.vv      v8, v22, v7\n\t"    // += r2 * k2

                "vfredsum.vs    v11, v8, v0\n\t"    // v11[0] = v0[0] + sum(v8[0..2])
                "vfmv.f.s       ft9, v11\n\t"       // ft9 = v11[0]

#ifdef  FUSE_CONV_RELU
                "fmax.s         ft9, ft9, ft11\n\t"     // **** relu ****
#endif  // FUSE_CONV_RELU

                "fsw            ft9, 0(%5)\n\t"
                "addi           %5, %5, 4\n\t"

                "addi           t2, t2, -1\n\t"
                "bnez           t2, 11b\n\t"

        "12:\n\t"
            // updata addr
            "addi           %1, %1, 8\n\t"      // r0 += 2
            "addi           %2, %2, 8\n\t"      // r1 += 2
            "addi           %3, %3, 8\n\t"      // r2 += 2

            :"=r"(kernel0),     // %0
            "=r"(r0),           // %1
            "=r"(r1),           // %2
            "=r"(r2),           // %3
            "=r"(r3),           // %4
            "=r"(outptr0),      // %5
            "=r"(outptr1),      // %6
            "=r"(out_h),        // %7
            "=r"(out_w),        // %8
            "=r"(in_w)          // %9
            :"0"(kernel0),
            "1"(r0),
            "2"(r1),
            "3"(r2),
            "4"(r3),
            "5"(outptr0),
            "6"(outptr1),
            "7"(out_h),
            "8"(out_w),
            "9"(in_w),
            "f"(bias0)          // %20
            :"cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
             "v22", "v23", "v24", "v25", "v26", "v27", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11", "t0", "t1", "t2", "t3", "t4", "t5"
        );
    }
#else
        const float *k0 = kernel0;
        const float *k1 = k0 + 3;
        const float *k2 = k1 + 3;

        int h = 0;
        for (; h + 1 < out_h; h += 2)
        {
            for (int w = 0; w < out_w; w++) {
                float sum0 = bias0;
                float sum1 = bias0;

                sum0 += r0[0] * k0[0] + r0[1] * k0[1] + r0[2] * k0[2];

                sum0 += r1[0] * k1[0] + r1[1] * k1[1] + r1[2] * k1[2];
                sum1 += r1[0] * k0[0] + r1[1] * k0[1] + r1[2] * k0[2];

                sum0 += r2[0] * k2[0] + r2[1] * k2[1] + r2[2] * k2[2];
                sum1 += r2[0] * k1[0] + r2[1] * k1[1] + r2[2] * k1[2];

                sum1 += r3[0] * k2[0] + r3[1] * k2[1] + r3[2] * k2[2];

#ifdef  FUSE_CONV_RELU
                sum0 = sum0 > 0 ? sum0 : 0;
                sum1 = sum1 > 0 ? sum1 : 0;
#endif  // FUSE_CONV_RELU

                *outptr0 = sum0;
                *outptr1 = sum1;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0++;
                outptr1++;
            }
            r0 += 2 + in_w;     // jump to next line
            r1 += 2 + in_w;
            r2 += 2 + in_w;
            r3 += 2 + in_w;

            outptr0 += out_w;
            outptr1 += out_w;
        }

        for (; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                float sum0 = bias0;
                sum0 += r0[0] * k0[0] + r0[1] * k0[1] + r0[2] * k0[2];
                sum0 += r1[0] * k1[0] + r1[1] * k1[1] + r1[2] * k1[2];
                sum0 += r2[0] * k2[0] + r2[1] * k2[1] + r2[2] * k2[2];

#ifdef  FUSE_CONV_RELU
                sum0 = sum0 > 0 ? sum0 : 0;
#endif  // FUSE_CONV_RELU

                *outptr0 = sum0;
                r0++;
                r1++;
                r2++;
                outptr0++;
            }

            r0 += 2;
            r1 += 2;
            r2 += 2;
        }
    }
#endif  // __riscv_vector

    csi_mem_free(input_padd_buf);
    return CSINN_TRUE;
}


/*
    (1) Algorithm works as follows:
        out_h1_loop: out_w4_loop  -->  out_w_tail

        k00*r00    k00*r02    k00*r04    k00*r06
        k01*r01    k01*r03    k01*r05    k01*r07
        k02*r02    k02*r04    k02*r06    k02*r08
        ----------------------------------------
        k10*r10    k10*r12    k10*r14    k10*r16
        k11*r11    k11*r13    k11*r15    k11*r17
        k12*r12    k12*r14    k12*r16    k12*r18
        ----------------------------------------
        k20*r20    k20*r22    k20*r24    k20*r26
        k21*r21    k21*r23    k21*r25    k21*r27
        k22*r22    k22*r24    k22*r26    k22*r28

    计算 k * r 时可以用 .vv 也可以用 .vf

    (2) register definition:
        t0:         i_out_h loop cnt
        t1-t2:      i_out_w loop cnt
        t3:         load stride 2 for r0-r2
        t4:         constant 3 for setting vl = 3
        ft0:        hold 1 output data
        ft1-ft9:    [ k00, k01, k02, k10, k11, k12, k20, k21, k22 ]
        ft11:       constant float 0.0f, used by fusing relu
        v0:         bias, acc
        v4-v5:      r0[0,2.4.6]   r0[1,3,5,7]
        v1:         r0[2,4,6,8]
        v6-v7:      r1[0,2.4.6]   r1[1,3,5,7]
        v2:         r1[2,4,6,8]
        v8-v9:      r2[0,2.4.6]   r2[1,3,5,7]
        v3:         r2[2,4,6,8]
        v10-v12:    k0, k1, k2
        v20-v21:    [ acc(kx1*rx), acc(kx2*rx) ]

    (3) //TODO: support channel mult ??
                Staggered instructions
*/

int DWCONV3X3S2(struct csi_tensor *input,
                struct csi_tensor *output,
                struct csi_tensor *kernel,
                struct csi_tensor *bias,
                struct conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];       // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    float *input_padd_buf = (float *)csi_mem_alloc(in_c * (in_h + params->pad_top + params->pad_down) * (in_w + params->pad_left + params->pad_right) * sizeof(float));

    csi_c906_pad_input(input_data, input_padd_buf, in_c, in_h, in_w, in_h + params->pad_top + params->pad_down, in_w + params->pad_left + params->pad_right, params->pad_top, params->pad_left);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

    int tailstep = in_w - 2 * out_w + in_w;

#pragma omp parallel for num_threads(8)
    for (int c = 0; c < in_c; c++) {

        float *out = output_data + c * out_h * out_w;
        float *outptr0 = out;

        const float bias0 = bias_data ? bias_data[c] : 0.0f;

        const float *img0 = input_padd_buf + c * in_h * in_w;
        const float *r0 = img0;
        const float *r1 = r0 + in_w;
        const float *r2 = r1 + in_w;

        const float *kernel0 = kernel_data + c * 9;

#if __riscv_vector == 128

        asm volatile(
            "vsetvli        zero, zero, e32, m1\n\t"
            "li             t3, 8\n\t"          //  load stride for r_x

#ifdef  FUSE_CONV_RELU
            "fmv.w.x        ft11, zero\n\t"
#endif  // FUSE_CONV_RELU

            "flw            ft1, (%0)\n\t"
            "flw            ft2, 4(%0)\n\t"
            "flw            ft3, 8(%0)\n\t"
            "flw            ft4, 12(%0)\n\t"
            "flw            ft5, 16(%0)\n\t"
            "flw            ft6, 20(%0)\n\t"
            "flw            ft7, 24(%0)\n\t"
            "flw            ft8, 28(%0)\n\t"
            "flw            ft9, 32(%0)\n\t"      // load k00 - k22

            "vlw.v          v10, (%0)\n\t"       // k0
            "addi           %0, %0, 12\n\t"
            "vlw.v          v11, (%0)\n\t"       // k1
            "addi           %0, %0, 12\n\t"
            "vlw.v          v12, (%0)\n\t"       // k2

            "vfmv.v.f       v0, %16\n\t"        // bias0

            "mv             t0, %5\n\t"         // i_out_h = out_h

        "1:\n\t"        // out_h

            "srai           t1, %6, 2\n\t"      // t1 = out_w >> 2
            "beqz           t1, 3f\n\t"
            "vsetvli        zero, zero, e32, m1\n\t"

            // pre-load rxx
            "vlseg2e.v      v4, (%1)\n\t"       // v4[0..3] = r0[0,2.4.6]   v5[0..3] = r0[1,3,5,7]
            "addi           %1, %1, 8\n\t"      // r0 += 2
            "vlsw.v         v1, (%1), t3\n\t"   // r0[2,4,6,8]
            "addi           %1, %1, 24\n\t"

            "2:\n\t"        // out_w_loop4

                "vlseg2e.v      v6, (%2)\n\t"       // v6[0..3] = r1[0,2.4.6]   v7[0..3] = r1[1,3,5,7]
                "addi           %2, %2, 8\n\t"
                "vfmul.vf       v20, v4, ft1\n\t"   // = k00 * r0[0,2,4,6]
                "vfmul.vf       v21, v5, ft2\n\t"   // = k01 * r0[1,3,5,7]
                "vlsw.v         v2, (%2), t3\n\t"
                "addi           %2, %2, 24\n\t"
                "vfmacc.vf      v0, ft3, v1\n\t"    // += k02 * r0[2,4,6,8]


                "vlseg2e.v      v8, (%3)\n\t"       // v8[0..3] = r2[0,2.4.6]   v9[0..3] = r2[1,3,5,7]
                "addi           %3, %3, 8\n\t"
                "vfmacc.vf      v20, ft4, v6\n\t"   // += k10 * r1[0,2,4,6]
                "vfmacc.vf      v21, ft5, v7\n\t"   // += k11 * r1[1,3,5,7]
                "vlsw.v         v3, (%3), t3\n\t"
                "addi           %3, %3, 24\n\t"
                "vfmacc.vf      v0, ft6, v2\n\t"    // += k12 * r1[2,4,6,8]


                "vlseg2e.v      v4, (%1)\n\t"       // v4[0..3] = r0[0,2.4.6]   v5[0..3] = r0[1,3,5,7]
                "addi           %1, %1, 8\n\t"      // r0 += 2
                "vfmacc.vf      v20, ft7, v8\n\t"   // += k20 * r2[0,2,4,6]
                "vfmacc.vf      v21, ft8, v9\n\t"   // += k21 * r2[1,3,5,7]
                "vlsw.v         v1, (%1), t3\n\t"   // r0[2,4,6,8]
                "addi           %1, %1, 24\n\t"
                "vfmacc.vf      v0, ft9, v3\n\t"    // += k22 * r2[2,4,6,8]


                "vfadd.vv       v2, v20, v21\n\t"
                "vfadd.vv       v0, v0, v2\n\t"

#ifdef  FUSE_CONV_RELU
                "vfmax.vf       v0, v0, ft11\n\t"   // **** relu ****
#endif  // FUSE_CONV_RELU

                "vsw.v          v0, (%4)\n\t"
                "addi           %4, %4, 16\n\t"     // outptr += 16

                "vfmv.v.f       v0, %16\n\t"        // bias0

                "addi           t1, t1, -1\n\t"
                "bnez           t1, 2b\n\t"

                "addi           %1, %1, -32\n\t"    // r0 -= 8  ********* bump r0 to origin addr ************

            "3:\n\t"        // out_w_tail
                "andi           t2, %6, 3\n\t"      // t2 = out_w & 3
                "beqz           t2, 5f\n\t"


            "4:\n\t"        // out_w_tail
                "vlw.v          v4, (%1)\n\t"       // r0
                "addi           %1, %1, 8\n\t"
                "vlw.v          v6, (%2)\n\t"       // r1
                "addi           %2, %2, 8\n\t"
                "vlw.v          v8, (%3)\n\t"       // r2
                "addi           %3, %3, 8\n\t"

                "vfmul.vv       v20, v4, v10\n\t"   // r0 * k0
                "vfmacc.vv      v20, v6, v11\n\t"   // += r1 * k1
                "vfmacc.vv      v20, v8, v12\n\t"   // += r2 * k2

                "li             t4, 3\n\t"
                "vsetvli        zero, t4, e32, m1\n\t"  // set vl = 3
                "vfredsum.vs    v21, v20, v0\n\t"       // v21[0] = v0[0](bias) + sum(v20[0..2])

                "vfmv.f.s       ft0, v21\n\t"           // ft0 = v21[0]

#ifdef  FUSE_CONV_RELU
                "fmax.s         ft0, ft0, ft11\n\t"     // **** relu ****
#endif  // FUSE_CONV_RELU

                "fsw            ft0, 0(%4)\n\t"
                "addi           %4, %4, 4\n\t"          // bump output_data pointer

                "addi           t2, t2, -1\n\t"
                "bnez           t2, 4b\n\t"

        "5:\n\t"
                "slli           t2, %7, 2\n\t"      // t2 = tailstep * 4
                "add            %1, %1, t2\n\t"
                "add            %2, %2, t2\n\t"
                "add            %3, %3, t2\n\t"     // r0/r1/r2 += tailstep

                "addi           t0, t0, -1\n\t"
                "bnez           t0, 1b\n\t"

            :"=r"(kernel0),     // %0
            "=r"(r0),           // %1
            "=r"(r1),           // %2
            "=r"(r2),           // %3
            "=r"(outptr0),      // %4
            "=r"(out_h),        // %5
            "=r"(out_w),        // %6
            "=r"(tailstep)      // %7
            :"0"(kernel0),
            "1"(r0),
            "2"(r1),
            "3"(r2),
            "4"(outptr0),
            "5"(out_h),
            "6"(out_w),
            "7"(tailstep),
            "f"(bias0)          // %16
            :"cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v20", "v21",
             "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft11", "t0", "t1", "t2", "t3", "t4"
        );
    }
#else
        const float *k0 = kernel0;
        const float *k1 = k0 + 3;
        const float *k2 = k1 + 3;
        int h = 0;
        for (; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                float sum0 = bias0;
                sum0 += r0[0] * k0[0] + r0[1] * k0[1] + r0[2] * k0[2];
                sum0 += r1[0] * k1[0] + r1[1] * k1[1] + r1[2] * k1[2];
                sum0 += r2[0] * k2[0] + r2[1] * k2[1] + r2[2] * k2[2];

#ifdef  FUSE_CONV_RELU
                sum0 = sum0 > 0 ? sum0 : 0;
#endif  // FUSE_CONV_RELU

                *outptr0 = sum0;
                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr0++;
            }
            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
#endif  // __riscv_vector

    csi_mem_free(input_padd_buf);
    return CSINN_TRUE;
}
