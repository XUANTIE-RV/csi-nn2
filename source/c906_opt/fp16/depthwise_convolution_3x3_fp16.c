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

#include "c906/c906.h"

/*
    (1) Algorithm works as follows:
        out_h2:     out_h2_w8_loop --> out_h2_w4 --> out_h2_wtail
        out_h_tail: out_h1_w8_loop --> out_h1_w4 --> out_h1_wtail

        out_h2_w8:                                    out_h2_w4: ||    out_h1_w8: out_h1_w4:
            outptr0[0-7]:        outptr1[0-7]:            outptr0[0-3]:         outptr1[0-3] ||
   outptr0[0-7]:            outptr0[0-3]: k00 * r0[0-7]        k00 * r1[0-7]            k00 *
   r0[0-3]          k00 * r1[0-3]    ||            k00 * r0[0-7]            k00 * r0[0-3] k01 *
   r0[1-8]        k01 * r1[1-8]            k01 * r0[1-4]          k01 * r1[1-4]    ||            k01
   * r0[1-8]            k01 * r0[1-4] k02 * r0[2-9]        k02 * r1[2-9]            k02 * r0[2-5]
   k02 * r1[2-5]    ||            k02 * r0[2-9]            k02 * r0[2-5] k10 * r1[0-7]        k10 *
   r2[0-7]            k10 * r1[0-3]          k10 * r2[0-3]    ||            k10 * r1[0-7] k10 *
   r1[0-3] k11 * r1[1-8]        k11 * r2[1-8]            k11 * r1[1-4]          k11 * r2[1-4]    ||
   k11 * r1[1-8]            k11 * r1[1-4] k12 * r1[2-9]        k12 * r2[2-9]            k12 *
   r1[2-5]          k12 * r2[2-5]    ||            k12 * r1[2-9]            k12 * r1[2-5] k20 *
   r2[0-7]        k20 * r3[0-7]            k20 * r2[0-3]          k20 * r3[0-3]    ||            k20
   * r2[0-7]            k20 * r2[0-3] k21 * r2[1-8]        k21 * r3[1-8]            k21 * r2[1-4]
   k21 * r3[1-4]    ||            k21 * r2[1-8]            k21 * r2[1-4] k22 * r2[2-9]        k22 *
   r3[2-9]            k22 * r2[2-5]          k22 * r3[2-5]    ||            k22 * r2[2-9] k22 *
   r2[2-5]


    (2) register definition:
        t0:         i_out_h
        t1-t2:      i_out_w
        v0:         bias0[0-7], output_data(acc)
        v2:         bias1[0-7], output_data(acc)
        v4,v6,v8:   r0  v4:r0[0-7]  v6:r0[1-8]   v8:r0[2-9]
        v10,v12,v14:r3
        v16,v18,v20:r1
        v22,v24,v26:r2
        ft0-ft8:    [ k00,k01,k02,k10,k11,k12,k20,k21,k22 ]

    (3) // TODO: support channel mult ??
                 opt padding

*/

int shl_c906_dwconv3x3s1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = NULL;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];  // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    __fp16 *kernel_fp16 = NULL;
    if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
        int size = csinn_tensor_size(kernel);
        int8_t *kernel_int8 = (int8_t *)kernel->data;
        kernel_fp16 = (__fp16 *)shl_mem_alloc(size * sizeof(__fp16));
        if (kernel->quant_channel > 1) {
            const int maxk = kernel->dim[2] * kernel->dim[3];
            for (int c = 0; c < in_c; c++) {
                int32_t zp = kernel->qinfo[c].zero_point;
                float scale = kernel->qinfo[c].scale;
                shl_rvv_dequantize_i8_to_f16(kernel_int8 + c * maxk, kernel_fp16 + c * maxk, maxk,
                                             zp, scale);
            }
        } else {
            int32_t zp = kernel->qinfo->zero_point;
            float scale = kernel->qinfo->scale;
            shl_rvv_dequantize_i8_to_f16(kernel_int8, kernel_fp16, size, zp, scale);
        }
        kernel_data = kernel_fp16;
    } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
        kernel_data = (__fp16 *)kernel->data;
    } else {
        shl_debug_error("kernel unsupport dtype: %d\n", kernel->dtype);
        return CSINN_FALSE;
    }

    __fp16 *input_padd_buf =
        (__fp16 *)shl_mem_alloc(in_c * (in_h + params->pad_top + params->pad_down) *
                                (in_w + params->pad_left + params->pad_right) * sizeof(__fp16));

    shl_c906_pad_input_fp16(
        input_data, input_padd_buf, in_c, in_h, in_w, in_h + params->pad_top + params->pad_down,
        in_w + params->pad_left + params->pad_right, params->pad_top, params->pad_left);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

#pragma omp parallel for num_threads(1)
    for (int c = 0; c < in_c; c++) {
        __fp16 *out = output_data + c * out_h * out_w;
        __fp16 *outptr0 = out;
        __fp16 *outptr1 = outptr0 + out_w;

        const __fp16 bias0 = bias_data ? bias_data[c] : 0.0f;

        const __fp16 *img0 = input_padd_buf + c * in_h * in_w;
        const __fp16 *r0 = img0;
        const __fp16 *r1 = r0 + in_w;
        const __fp16 *r2 = r1 + in_w;
        const __fp16 *r3 = r2 + in_w;

        const __fp16 *kernel0 = kernel_data + c * 9;

        asm volatile(
            "vsetvli        zero, zero, e16, m1\n\t"

            "flh            ft0, 0(%0)\n\t"   // k00
            "flh            ft1, 2(%0)\n\t"   // k01
            "flh            ft2, 4(%0)\n\t"   // k02
            "flh            ft3, 6(%0)\n\t"   // k10
            "flh            ft4, 8(%0)\n\t"   // k11
            "flh            ft5, 10(%0)\n\t"  // k12
            "flh            ft6, 12(%0)\n\t"  // k20
            "flh            ft7, 14(%0)\n\t"  // k21
            "flh            ft8, 16(%0)\n\t"  // k22

            "srai           t0, %7, 1\n\t"  // t0 = out_h >> 1
            "beqz           t0, 7f\n\t"

            "1:\n\t"  // out_h_loop2

            "srai           t1, %8, 3\n\t"  // t1 = out_w >> 3
            "beqz           t1, 3f\n\t"

            "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 8

            // pre-load rxx
            "vle.v          v4, (%1)\n\t"   // r0[0-7]
            "addi           %1, %1, 2\n\t"  // r0++
            "vle.v          v6, (%1)\n\t"   // r0[1-8]
            "addi           %1, %1, 2\n\t"  // r0++
            "vle.v          v8, (%1)\n\t"   // r0[2-9]

            "2:\n\t"  // out_w_loop8

            "vfmv.v.f       v0, %20\n\t"     // bias0[0-7]
            "addi           %1, %1, 12\n\t"  // r0 += 6

            "vle.v          v10, (%4)\n\t"  // r3[0-7]
            "addi           %4, %4, 2\n\t"  // r3++
            "vfmv.v.f       v2, %20\n\t"    // bias1[0-7]

            "vle.v          v12, (%4)\n\t"  // r3[1-8]
            "addi           %4, %4, 2\n\t"  // r3++

            "vfmacc.vf      v0, ft0, v4\n\t"   // k00 * r0[0-7]
            "vfmacc.vf      v2, ft6, v10\n\t"  // k20 * r3[0-7]

            "vle.v          v14, (%4)\n\t"   // r3[2-9]
            "addi           %4, %4, 12\n\t"  // r3 += 6

            "vfmacc.vf      v0, ft1, v6\n\t"   // k01 * r0[1-8]
            "vfmacc.vf      v2, ft7, v12\n\t"  // k21 * r3[1-8]

            "vle.v          v16, (%2)\n\t"  // r1[0-7]
            "addi           %2, %2, 2\n\t"  // r1++

            "vfmacc.vf      v0, ft2, v8\n\t"   // k02 * r0[2-9]
            "vfmacc.vf      v2, ft8, v14\n\t"  // k22 * r3[2-9]

            "vle.v          v18, (%2)\n\t"  // r1[1-8]
            "addi           %2, %2, 2\n\t"  // r1++

            "vfmacc.vf      v0, ft3, v16\n\t"  // k10 * r1[0-7]
            "vfmacc.vf      v2, ft0, v16\n\t"  // k00 * r1[0-7]

            "vle.v          v20, (%2)\n\t"   // r1[2-9]
            "addi           %2, %2, 12\n\t"  // r1 += 6

            "vfmacc.vf      v0, ft4, v18\n\t"  // k11 * r1[1-8]
            "vfmacc.vf      v2, ft1, v18\n\t"  // k01 * r1[1-8]

            "vle.v          v22, (%3)\n\t"  // r2[0-7]
            "addi           %3, %3, 2\n\t"  // r2++

            "vfmacc.vf      v0, ft5, v20\n\t"  // k12 * r1[2-9]
            "vfmacc.vf      v2, ft2, v20\n\t"  // k02 * r1[2-9]

            "vle.v          v24, (%3)\n\t"  // r2[1-8]
            "addi           %3, %3, 2\n\t"  // r2++

            "vfmacc.vf      v0, ft6, v22\n\t"  // k20 * r2[0-7]
            "vfmacc.vf      v2, ft3, v22\n\t"  // k10 * r2[0-7]

            "vle.v          v26, (%3)\n\t"   // r2[2-9]
            "addi           %3, %3, 12\n\t"  // r2 += 6

            "vfmacc.vf      v0, ft7, v24\n\t"  // k21 * r2[1-8]
            "vfmacc.vf      v2, ft4, v24\n\t"  // k11 * r2[1-8]

            "vle.v          v4, (%1)\n\t"   // r0[0-7]  load r0 for next loop
            "addi           %1, %1, 2\n\t"  // r0++

            "vfmacc.vf      v0, ft8, v26\n\t"  // k22 * r2[2-9]

            "vle.v          v6, (%1)\n\t"   // r0[1-8]
            "addi           %1, %1, 2\n\t"  // r0++

            "vfmacc.vf      v2, ft5, v26\n\t"  // k12 * r2[2-9]

            "vle.v          v8, (%1)\n\t"  // r0[2-9]

            "vse.v          v0, (%5)\n\t"    // store line0 8 elements on outptr0
            "addi           %5, %5, 16\n\t"  // outptr0 += 8
            "vse.v          v2, (%6)\n\t"    // store line1 8 elements on outptr1
            "addi           %6, %6, 16\n\t"  // outptr1 += 8

            "addi           t1, t1, -1\n\t"
            "bnez           t1, 2b\n\t"

            "addi           %1, %1, -4\n\t"  // r0 -= 2  ********* bump r0 to origin addr
                                             // ************

            "3:\n\t"                        // out_w4 : can only be executed once in h2 loop
            "andi           t1, %8, 7\n\t"  // t1 = out_w & 7
            "srai           t2, t1, 2\n\t"  // t2 = (out_w & 7) >> 2
            "beqz           t2, 4f\n\t"

            "li             t5, 4\n\t"
            "vsetvli        zero, t5, e16, m1\n\t"  // set vl = 8 actually low 4 used

            // "vsetvli        zero, zero, e16, m1\n\t"    // set vl = 8 actually low 4 used

            "vle.v          v4, (%1)\n\t"   // r0[0-3]  [4-7] unused
            "addi           %1, %1, 2\n\t"  // r0++

            "vfmv.v.f       v0, %20\n\t"  // bias0[0-3]

            "vle.v          v10, (%4)\n\t"  // r3[0-3]
            "addi           %4, %4, 2\n\t"  // r3++

            "vfmv.v.f       v2, %20\n\t"  // bias1[0-3]

            "vle.v          v5, (%1)\n\t"   // r0[1-4]
            "addi           %1, %1, 2\n\t"  // r0++

            "vle.v          v11, (%4)\n\t"  // r3[1-4]
            "addi           %4, %4, 2\n\t"  // r3++

            "vfmacc.vf      v0, ft0, v4\n\t"   // k00 * r0[0-3]
            "vfmacc.vf      v2, ft6, v10\n\t"  // k20 * r3[0-3]

            "vle.v          v6, (%1)\n\t"   // r0[2-5]
            "addi           %1, %1, 4\n\t"  // r0 += 2

            "vle.v          v12, (%4)\n\t"  // r3[2-5]
            "addi           %4, %4, 4\n\t"  // r3 += 2

            "vfmacc.vf      v0, ft1, v5\n\t"   // k01 * r0[1-4]
            "vfmacc.vf      v2, ft7, v11\n\t"  // k21 * r3[1-4]

            "vle.v          v16, (%2)\n\t"  // r1[0-3]
            "addi           %2, %2, 2\n\t"  // r1++

            "vfmacc.vf      v0, ft2, v6\n\t"   // k02 * r0[2-5]
            "vfmacc.vf      v2, ft8, v12\n\t"  // k22 * r3[2-5]

            "vle.v          v17, (%2)\n\t"  // r1[1-4]
            "addi           %2, %2, 2\n\t"  // r1++

            "vfmacc.vf      v0, ft3, v16\n\t"  // k10 * r1[0-3]
            "vfmacc.vf      v2, ft0, v16\n\t"  // k00 * r1[0-3]

            "vle.v          v18, (%2)\n\t"  // r1[2-5]
            "addi           %2, %2, 4\n\t"  // r1 += 2

            "vfmacc.vf      v0, ft4, v17\n\t"  // k11 * r1[1-4]
            "vfmacc.vf      v2, ft1, v17\n\t"  // k01 * r1[1-4]

            "vle.v          v22, (%3)\n\t"  // r2[0-3]
            "addi           %3, %3, 2\n\t"  // r2++

            "vfmacc.vf      v0, ft5, v18\n\t"  // k12 * r1[2-5]
            "vfmacc.vf      v2, ft2, v18\n\t"  // k02 * r1[2-5]]

            "vle.v          v23, (%3)\n\t"  // r2[1-4]
            "addi           %3, %3, 2\n\t"  // r2++

            "vfmacc.vf      v0, ft6, v22\n\t"  // k20 * r2[0-3]
            "vfmacc.vf      v2, ft3, v22\n\t"  // k10 * r2[0-3]

            "vle.v          v24, (%3)\n\t"  // r2[2-5]
            "addi           %3, %3, 4\n\t"  // r2 += 2

            "vfmacc.vf      v0, ft7, v23\n\t"  // k21 * r2[1-4]
            "vfmacc.vf      v2, ft4, v23\n\t"  // k11 * r2[1-4]

            "vfmacc.vf      v0, ft8, v24\n\t"  // k22 * r2[2-5]
            "vfmacc.vf      v2, ft5, v24\n\t"  // k12 * r2[2-5]

            "vse.v          v0, (%5)\n\t"   // store line0 4 elements on outptr0
            "addi           %5, %5, 8\n\t"  // outptr0 += 4
            "vse.v          v2, (%6)\n\t"   // store line1 4 elements on outptr1
            "addi           %6, %6, 8\n\t"  // outptr1 += 4

            "4:\n\t"                        // out_w_tail
            "andi           t2, t1, 3\n\t"  // t2 = (out_w & 7) & 3
            "beqz           t2, 6f\n\t"

            "vfmv.v.f       v0, %20\n\t"  // bias0[0-3] / bias1[0-3]
            "li             t5, 3\n\t"
            "vsetvli        zero, t5, e16, m1\n\t"  // set vl = 3

            "vle.v          v5, (%0)\n\t"  // k0
            "addi           %0, %0, 6\n\t"
            "vle.v          v6, (%0)\n\t"  // k1
            "addi           %0, %0, 6\n\t"
            "vle.v          v7, (%0)\n\t"  // k2

            "5:\n\t"  // out_w_tail

            "vle.v          v4, (%1)\n\t"   // r0
            "addi           %1, %1, 2\n\t"  // r0++

            "vle.v          v16, (%2)\n\t"  // r1
            "addi           %2, %2, 2\n\t"  // r1++

            "vle.v          v22, (%3)\n\t"  // r2
            "addi           %3, %3, 2\n\t"  // r2++

            "vle.v          v10, (%4)\n\t"  // r3
            "addi           %4, %4, 2\n\t"  // r3++

            "vfmul.vv       v8, v4, v5\n\t"   // r0 * k0
            "vfmacc.vv      v8, v16, v6\n\t"  // += r1 * k1
            "vfmacc.vv      v8, v22, v7\n\t"  // += r2 * k2

            "vfredsum.vs    v11, v8, v0\n\t"  // v11[0] = v0[0] + sum(v8[0..2])
            "vfmv.f.s       ft9, v11\n\t"     // ft9 = v11[0]

            "vfmul.vv       v9, v16, v5\n\t"  // r1 * k0
            "vfmacc.vv      v9, v22, v6\n\t"  // += r2 * k1
            "vfmacc.vv      v9, v10, v7\n\t"  // += r3 * k2

            "vfredsum.vs    v12, v9, v0\n\t"  // v12[0] = v0[0] + sum(v9[0..2])
            "vfmv.f.s       ft10, v12\n\t"    // ft10 = v12[0]

            "fsh            ft9, 0(%5)\n\t"
            "addi           %5, %5, 2\n\t"
            "fsh            ft10, 0(%6)\n\t"
            "addi           %6, %6, 2\n\t"

            "addi           t2, t2, -1\n\t"
            "bnez           t2, 5b\n\t"

            "addi           %0, %0, -12\n\t"  // kernel -= 6  ********* bump kernel_data to origin
                                              // addr ************

            "6:\n\t"  // out_h_loop2 cnt

            "slli           t3, %9, 1\n\t"  // in_w * 2
            "addi           t3, t3, 4\n\t"  // in_w * 2 + 4

            "slli           t4, %8, 1\n\t"  // out_w * 2

            "add            %1, %1, t3\n\t"  // r0 += 2 + in_w
            "add            %2, %2, t3\n\t"  // r1 += 2 + in_w
            "add            %3, %3, t3\n\t"  // r2 += 2 + in_w
            "add            %4, %4, t3\n\t"  // r3 += 2 + in_w

            "add            %5, %5, t4\n\t"  // outptr0 += out_w
            "add            %6, %6, t4\n\t"  // outptr1 += out_w

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 1b\n\t"

            "7:\n\t"                        // out_h_tail // 只有执行一次的机会
            "andi           t0, %7, 1\n\t"  // t0 = out_h & 1
            "beqz           t0, 12f\n\t"

            "srai           t1, %8, 3\n\t"  // t1 = out_w >> 3
            "beqz           t1, 9f\n\t"

            "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 8

            // pre-load rxx
            "vle.v          v4, (%1)\n\t"   // r0[0-7]
            "addi           %1, %1, 2\n\t"  // r0++
            "vle.v          v6, (%1)\n\t"   // r0[1-8]
            "addi           %1, %1, 2\n\t"  // r0++
            "vle.v          v8, (%1)\n\t"   // r0[2-9]

            "8:\n\t"  // out_w_loop8

            "vfmv.v.f       v0, %20\n\t"     // bias0[0-7]
            "addi           %1, %1, 12\n\t"  // r0 += 6

            "vfmacc.vf      v0, ft0, v4\n\t"  // k00 * r0[0-7]

            "vle.v          v16, (%2)\n\t"  // r1[0-7]
            "addi           %2, %2, 2\n\t"  // r1++

            "vfmacc.vf      v0, ft1, v6\n\t"  // k01 * r0[1-8]

            "vle.v          v18, (%2)\n\t"  // r1[1-8]
            "addi           %2, %2, 2\n\t"  // r1++

            "vfmacc.vf      v0, ft2, v8\n\t"  // k02 * r0[2-9]

            "vle.v          v20, (%2)\n\t"   // r1[2-9]
            "addi           %2, %2, 12\n\t"  // r1 += 6

            "vfmacc.vf      v0, ft3, v16\n\t"  // k10 * r1[0-7]

            "vle.v          v22, (%3)\n\t"  // r2[0-7]
            "addi           %3, %3, 2\n\t"  // r2++

            "vfmacc.vf      v0, ft4, v18\n\t"  // k11 * r1[1-8]

            "vle.v          v24, (%3)\n\t"  // r2[1-8]
            "addi           %3, %3, 2\n\t"  // r2++

            "vfmacc.vf      v0, ft5, v20\n\t"  // k12 * r1[2-9]

            "vle.v          v26, (%3)\n\t"   // r2[2-9]
            "addi           %3, %3, 12\n\t"  // r2 += 6

            "vfmacc.vf      v0, ft6, v22\n\t"  // k20 * r2[0-7]

            "vle.v          v4, (%1)\n\t"   // r0[0-7]
            "addi           %1, %1, 2\n\t"  // r0++

            "vfmacc.vf      v0, ft7, v24\n\t"  // k21 * r2[1-8]

            "vle.v          v6, (%1)\n\t"   // r0[1-8]
            "addi           %1, %1, 2\n\t"  // r0++

            "vfmacc.vf      v0, ft8, v26\n\t"  // k22 * r2[2-9]

            "vle.v          v8, (%1)\n\t"  // r0[2-9]

            "vse.v          v0, (%5)\n\t"    // store line0 8 elements on outptr0
            "addi           %5, %5, 16\n\t"  // outptr0 += 8

            "addi           t1, t1, -1\n\t"
            "bnez           t1, 8b\n\t"

            "addi           %1, %1, -4\n\t"  // r0 -= 2  ********* bump r0 to origin addr
                                             // ************

            "9:\n\t"                        // out_w4
            "andi           t1, %8, 7\n\t"  // t1 = out_w & 7
            "srai           t2, t1, 2\n\t"  // t2 = (out_w & 7) >> 2
            "beqz           t2, 10f\n\t"

            "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 4

            "vle.v          v4, (%1)\n\t"   // r0[0-3]
            "addi           %1, %1, 2\n\t"  // r0++

            "vfmv.v.f       v0, %20\n\t"  // bias0[0-3]

            "vle.v          v5, (%1)\n\t"   // r0[1-4]
            "addi           %1, %1, 2\n\t"  // r0++

            "vfmacc.vf      v0, ft0, v4\n\t"  // k00 * r0[0-3]

            "vle.v          v6, (%1)\n\t"   // r0[2-5]
            "addi           %1, %1, 4\n\t"  // r0 += 2

            "vfmacc.vf      v0, ft1, v5\n\t"  // k01 * r0[1-4]

            "vle.v          v16, (%2)\n\t"  // r1[0-3]
            "addi           %2, %2, 2\n\t"  // r1++

            "vfmacc.vf      v0, ft2, v6\n\t"  // k02 * r0[2-5]

            "vle.v          v17, (%2)\n\t"  // r1[1-4]
            "addi           %2, %2, 2\n\t"  // r1++

            "vfmacc.vf      v0, ft3, v16\n\t"  // k10 * r1[0-3]

            "vle.v          v18, (%2)\n\t"  // r1[2-5]
            "addi           %2, %2, 4\n\t"  // r1 += 2

            "vfmacc.vf      v0, ft4, v17\n\t"  // k11 * r1[1-4]

            "vle.v          v22, (%3)\n\t"  // r2[0-3]
            "addi           %3, %3, 2\n\t"  // r2++

            "vfmacc.vf      v0, ft5, v18\n\t"  // k12 * r1[2-5]

            "vle.v          v23, (%3)\n\t"  // r2[1-4]
            "addi           %3, %3, 2\n\t"  // r2++

            "vfmacc.vf      v0, ft6, v22\n\t"  // k20 * r2[0-3]

            "vle.v          v24, (%3)\n\t"  // r2[2-5]
            "addi           %3, %3, 4\n\t"  // r2 += 2

            "vfmacc.vf      v0, ft7, v23\n\t"  // k21 * r2[1-4]

            "vfmacc.vf      v0, ft8, v24\n\t"  // k22 * r2[2-5]

            "vse.v          v0, (%5)\n\t"    // store line0 4 elements on outptr0
            "addi           %5, %5, 16\n\t"  // outptr0 += 4

            "10:\n\t"  // out_w_tail
            "andi           t2, t1, 3\n\t"
            "beqz           t2, 12f\n\t"

            "vfmv.v.f       v0, %20\n\t"  // bias0[0-3]
            "li             t5, 3\n\t"
            "vsetvli        zero, t5, e16, m1\n\t"  // set vl = 3

            "vle.v          v5, (%0)\n\t"  // k0
            "addi           %0, %0, 6\n\t"
            "vle.v          v6, (%0)\n\t"  // k1
            "addi           %0, %0, 6\n\t"
            "vle.v          v7, (%0)\n\t"  // k2

            "11:\n\t"  // out_w_tail

            "vle.v          v4, (%1)\n\t"   // r0
            "addi           %1, %1, 2\n\t"  // r0++

            "vle.v          v16, (%2)\n\t"  // r1
            "addi           %2, %2, 2\n\t"  // r1++

            "vle.v          v22, (%3)\n\t"  // r2
            "addi           %3, %3, 2\n\t"  // r2++

            "vfmul.vv       v8, v4, v5\n\t"   // r0 * k0
            "vfmacc.vv      v8, v16, v6\n\t"  // += r1 * k1
            "vfmacc.vv      v8, v22, v7\n\t"  // += r2 * k2

            "vfredsum.vs    v11, v8, v0\n\t"  // v11[0] = v0[0] + sum(v8[0..2])
            "vfmv.f.s       ft9, v11\n\t"     // ft9 = v11[0]

            "fsh            ft9, 0(%5)\n\t"
            "addi           %5, %5, 2\n\t"

            "addi           t2, t2, -1\n\t"
            "bnez           t2, 11b\n\t"

            "12:\n\t"
            // updata addr
            "addi           %1, %1, 4\n\t"  // r0 += 2
            "addi           %2, %2, 4\n\t"  // r1 += 2
            "addi           %3, %3, 4\n\t"  // r2 += 2

            : "=r"(kernel0),  // %0
              "=r"(r0),       // %1
              "=r"(r1),       // %2
              "=r"(r2),       // %3
              "=r"(r3),       // %4
              "=r"(outptr0),  // %5
              "=r"(outptr1),  // %6
              "=r"(out_h),    // %7
              "=r"(out_w),    // %8
              "=r"(in_w)      // %9
            : "0"(kernel0), "1"(r0), "2"(r1), "3"(r2), "4"(r3), "5"(outptr0), "6"(outptr1),
              "7"(out_h), "8"(out_w), "9"(in_w),
              "f"(bias0)  // %20
            : "cc", "memory", "v0", "v2", "v4", "v6", "v8", "v10", "v12", "v14", "v16", "v18",
              "v20", "v22", "v23", "v24", "v26", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6",
              "ft7", "ft8", "ft9", "ft10", "t0", "t1", "t2", "t3", "t4", "t5");
    }

    shl_mem_free(input_padd_buf);
    if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
        shl_mem_free(kernel_fp16);
        return CSINN_TRUE;
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}

/*
    (1) Algorithm works as follows:
        out_h1_loop: out_w8_loop  -->  out_w4  -->  out_w_tail

    (2) register definition:
        t0:         i_out_h loop cnt
        t1-t2:      i_out_w loop cnt
        t3:         load stride 2 for r0-r2
        t4:         constant 3/4 for setting vl = 3/4
        ft0:        hold 1 output data
        v3:         bias
        ft1-ft2:    [ k00, k01, k02, k10, k11, k12, k20, k21, k22 ]
        v10-v18:    [ k0, k1, k2 ]
        v19-v21:    [ acc(kx0*rx), acc(kx1*rx), acc(kx2*rx) ]

    (3) //TODO: support channel mult ??
                Staggered instructions

*/

int shl_c906_dwconv3x3s2_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = NULL;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];  // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    __fp16 *kernel_fp16 = NULL;
    if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
        int size = csinn_tensor_size(kernel);
        int8_t *kernel_int8 = (int8_t *)kernel->data;
        kernel_fp16 = (__fp16 *)shl_mem_alloc(size * sizeof(__fp16));
        if (kernel->quant_channel > 1) {
            const int maxk = kernel->dim[2] * kernel->dim[3];
            for (int c = 0; c < in_c; c++) {
                int32_t zp = kernel->qinfo[c].zero_point;
                float scale = kernel->qinfo[c].scale;
                shl_rvv_dequantize_i8_to_f16(kernel_int8 + c * maxk, kernel_fp16 + c * maxk, maxk,
                                             zp, scale);
            }
        } else {
            int32_t zp = kernel->qinfo->zero_point;
            float scale = kernel->qinfo->scale;
            shl_rvv_dequantize_i8_to_f16(kernel_int8, kernel_fp16, size, zp, scale);
        }
        kernel_data = kernel_fp16;
    } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
        kernel_data = (__fp16 *)kernel->data;
    } else {
        shl_debug_error("kernel unsupport dtype: %d\n", kernel->dtype);
        return CSINN_FALSE;
    }

    __fp16 *input_padd_buf =
        (__fp16 *)shl_mem_alloc(in_c * (in_h + params->pad_top + params->pad_down) *
                                (in_w + params->pad_left + params->pad_right) * sizeof(__fp16));

    shl_c906_pad_input_fp16(
        input_data, input_padd_buf, in_c, in_h, in_w, in_h + params->pad_top + params->pad_down,
        in_w + params->pad_left + params->pad_right, params->pad_top, params->pad_left);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

    int tailstep = in_w - 2 * out_w + in_w;

#pragma omp parallel for num_threads(1)
    for (int c = 0; c < in_c; c++) {
        __fp16 *out = output_data + c * out_h * out_w;
        __fp16 *outptr0 = out;

        const __fp16 bias0 = bias_data ? bias_data[c] : 0.0f;

        const __fp16 *img0 = input_padd_buf + c * in_h * in_w;
        const __fp16 *r0 = img0;
        const __fp16 *r1 = r0 + in_w;
        const __fp16 *r2 = r1 + in_w;

        const __fp16 *kernel0 = kernel_data + c * 9;

        asm volatile(
            "vsetvli        zero, zero, e16, m1\n\t"
            "li             t3, 4\n\t"  //  load stride for r_x

            "flh            ft1, (%0)\n\t"
            "flh            ft2, 2(%0)\n\t"
            "flh            ft3, 4(%0)\n\t"
            "flh            ft4, 6(%0)\n\t"
            "flh            ft5, 8(%0)\n\t"
            "flh            ft6, 10(%0)\n\t"
            "flh            ft7, 12(%0)\n\t"
            "flh            ft8, 14(%0)\n\t"
            "flh            ft9, 16(%0)\n\t"  // load k00 - k22

            "vle.v          v10, (%0)\n\t"  // k0
            "addi           %0, %0, 6\n\t"
            "vle.v          v11, (%0)\n\t"  // k1
            "addi           %0, %0, 6\n\t"
            "vle.v          v12, (%0)\n\t"  // k2

            "vfmv.v.f       v0, %16\n\t"  // bias0

            "mv             t0, %5\n\t"  // i_out_h = out_h

            "1:\n\t"  // out_h

            "srai           t1, %6, 3\n\t"  // t1 = out_w >> 3
            "beqz           t1, 3f\n\t"
            "vsetvli        zero, zero, e16, m1\n\t"

            // pre-load rxx
            "vlseg2e.v      v4, (%1)\n\t"      // v4[0..7] = r0[0,2,4,6,8,10,12,14]   v5[0..7] =
                                               // r0[1,3,5,7,9,11,13,15]
            "addi           %1, %1, 4\n\t"     // r0 += 2
            "vlse.v         v1, (%1), t3\n\t"  // r0[2,4,6,8,10,12,14,16]
            "addi           %1, %1, 28\n\t"

            "2:\n\t"  // out_w_loop8

            "vfmv.v.f       v0, %16\n\t"  // bias0

            "vlseg2e.v      v6, (%2)\n\t"  // v6[0..7] = r1[0,2,4,6,8,10,12,14]   v7[0..7] =
                                           // r1[1,3,5,7,9,11,13,15]
            "addi           %2, %2, 4\n\t"
            "vfmul.vf       v20, v4, ft1\n\t"  // = k00 * r0[0,2,4,6,8,10,12,14]
            "vfmul.vf       v21, v5, ft2\n\t"  // = k01 * r0[1,3,5,7,9,11,13,15]
            "vlse.v         v2, (%2), t3\n\t"  // r1[2,4,6,8,10,12,14,16]
            "addi           %2, %2, 28\n\t"
            "vfmacc.vf      v0, ft3, v1\n\t"  // += k02 * r0[2,4,6,8,10,12,14,16]

            "vlseg2e.v      v8, (%3)\n\t"  // v8[0..7] = r2[0,2,4,6,8,10,12,14]   v9[0..7] =
                                           // r2[1,3,5,7,9,11,13,15]
            "addi           %3, %3, 4\n\t"
            "vfmacc.vf      v20, ft4, v6\n\t"  // += k10 * r1[0,2,4,6,8,10,12,14]
            "vfmacc.vf      v21, ft5, v7\n\t"  // += k11 * r1[1,3,5,7,9,11,13,15]
            "vlse.v         v3, (%3), t3\n\t"
            "addi           %3, %3, 28\n\t"
            "vfmacc.vf      v0, ft6, v2\n\t"  // += k12 * r1[2,4,6,8,10,12,14,16]

            "vlseg2e.v      v4, (%1)\n\t"      // v4[0..3] = r0[0,2,4,6,8,10,12,14]   v5[0..3] =
                                               // r0[1,3,5,7,9,11,13,15]
            "addi           %1, %1, 4\n\t"     // r0 += 2
            "vfmacc.vf      v20, ft7, v8\n\t"  // += k20 * r2[0,2,4,6,8,10,12,14]
            "vfmacc.vf      v21, ft8, v9\n\t"  // += k21 * r2[1,3,5,7,9,11,13,15]
            "vlse.v         v1, (%1), t3\n\t"  // r0[2,4,6,8,10,12,14,16]
            "addi           %1, %1, 28\n\t"
            "vfmacc.vf      v0, ft9, v3\n\t"  // += k22 * r2[2,4,6,8,10,12,14,16]

            "vfadd.vv       v2, v20, v21\n\t"
            "vfadd.vv       v0, v0, v2\n\t"

            "vse.v          v0, (%4)\n\t"
            "addi           %4, %4, 16\n\t"  // outptr += 8

            "addi           t1, t1, -1\n\t"
            "bnez           t1, 2b\n\t"

            "addi           %1, %1, -32\n\t"  // r0 -= 16  ********* bump r0 to origin addr
                                              // ************

            "3:\n\t"                        // out_w4
            "andi           t1, %6, 7\n\t"  // t1 = out_w & 7
            "srai           t2, t1, 2\n\t"  // t2 = (out_w & 7) >> 2
            "beqz           t2, 4f\n\t"

            "li             t4, 4\n\t"
            "vsetvli        zero, t4, e16, m1\n\t"  // set vl = 4

            "vfmv.v.f       v0, %16\n\t"  // bias0

            "vlseg2e.v      v4, (%1)\n\t"   // v4[0..3] = r0[0,2,4,6]   v5[0..3] = r0[1,3,5,7]
            "addi           %1, %1, 4\n\t"  // r0 += 2

            "vlse.v         v1, (%1), t3\n\t"  // r0[2,4,6,8]
            "addi           %1, %1, 12\n\t"

            "vfmul.vf       v20, v4, ft1\n\t"  // = k00 * r0[0,2,4,6]
            "vfmul.vf       v21, v5, ft2\n\t"  // = k01 * r0[1,3,5,7]

            "vlseg2e.v      v6, (%2)\n\t"   // v6[0..3] = r1[0,2,4,6]   v7[0..3] = r1[1,3,5,7]
            "addi           %2, %2, 4\n\t"  // r1 += 2

            "vfmacc.vf      v0, ft3, v1\n\t"  // += k02 * r0[2,4,6,8]

            "vlse.v         v2, (%2), t3\n\t"  // r1[2,4,6,8]
            "addi           %2, %2, 12\n\t"

            "vfmacc.vf      v20, ft4, v6\n\t"  // += k10 * r1[0,2,4,6]
            "vfmacc.vf      v21, ft5, v7\n\t"  // += k11 * r1[1,3,5,7]

            "vlseg2e.v      v8, (%3)\n\t"  // v8[0..3] = r2[0,2,4,6]   v9[0..3] = r2[1,3,5,7]
            "addi           %3, %3, 4\n\t"

            "vfmacc.vf      v0, ft6, v2\n\t"  // += k12 * r1[2,4,6,8]

            "vlse.v         v3, (%3), t3\n\t"  // r2[2,4,6,8]
            "addi           %3, %3, 12\n\t"

            "vfmacc.vf      v20, ft7, v8\n\t"  // += k20 * r2[0,2,4,6]
            "vfmacc.vf      v21, ft8, v9\n\t"  // += k21 * r2[1,3,5,7]
            "vfmacc.vf      v0, ft9, v3\n\t"   // += k22 * r2[2,4,6,8]

            "vfadd.vv       v2, v20, v21\n\t"
            "vfadd.vv       v0, v0, v2\n\t"

            "vse.v          v0, (%4)\n\t"
            "addi           %4, %4, 8\n\t"  // outptr += 4

            "4:\n\t"                        // out_w_tail
            "andi           t2, t1, 3\n\t"  // t2 = out_w & 3
            "beqz           t2, 6f\n\t"

            "li             t4, 3\n\t"
            "vsetvli        zero, t4, e16, m1\n\t"  // set vl = 3

            "vfmv.v.f       v0, %16\n\t"  // bias0

            "5:\n\t"                       // out_w_tail
            "vle.v          v4, (%1)\n\t"  // r0
            "addi           %1, %1, 4\n\t"
            "vle.v          v6, (%2)\n\t"  // r1
            "addi           %2, %2, 4\n\t"
            "vle.v          v8, (%3)\n\t"  // r2
            "addi           %3, %3, 4\n\t"

            "vfmul.vv       v20, v4, v10\n\t"  // r0 * k0
            "vfmacc.vv      v20, v6, v11\n\t"  // += r1 * k1
            "vfmacc.vv      v20, v8, v12\n\t"  // += r2 * k2

            "vfredsum.vs    v21, v20, v0\n\t"  // v21[0] = v0[0](bias) + sum(v20[0..2])

            "vfmv.f.s       ft0, v21\n\t"  // ft0 = v21[0]

            "fsh            ft0, 0(%4)\n\t"
            "addi           %4, %4, 2\n\t"  // bump output_data pointer

            "addi           t2, t2, -1\n\t"
            "bnez           t2, 5b\n\t"

            "6:\n\t"
            "slli           t2, %7, 1\n\t"  // t2 = tailstep * 2
            "add            %1, %1, t2\n\t"
            "add            %2, %2, t2\n\t"
            "add            %3, %3, t2\n\t"  // r0/r1/r2 += tailstep

            "addi           t0, t0, -1\n\t"
            "bnez           t0, 1b\n\t"

            : "=r"(kernel0),  // %0
              "=r"(r0),       // %1
              "=r"(r1),       // %2
              "=r"(r2),       // %3
              "=r"(outptr0),  // %4
              "=r"(out_h),    // %5
              "=r"(out_w),    // %6
              "=r"(tailstep)  // %7
            : "0"(kernel0), "1"(r0), "2"(r1), "3"(r2), "4"(outptr0), "5"(out_h), "6"(out_w),
              "7"(tailstep),
              "f"(bias0)  // %16
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v20", "v21", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
              "ft8", "ft9", "ft11", "t0", "t1", "t2", "t3", "t4");
    }

    shl_mem_free(input_padd_buf);
    if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
        shl_mem_free(kernel_fp16);
        return CSINN_TRUE;
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}
