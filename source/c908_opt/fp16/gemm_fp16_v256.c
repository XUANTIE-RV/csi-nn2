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

#include "c908/c908.h"

/*************************************************************
 * note: VLEN = 256
 * VS kernel 12 x 16
 * input matrix and kernel matrix have been reordered
 *************************************************************/
#ifdef SHL_UNUSED_REGISTER_BLK
static inline void kernel_m8n48_fp16_v256(__fp16 *dst, __fp16 *sa, __fp16 *sb, int m, int k, int n,
                                          int ldc, __fp16 *bias)
{
    asm volatile(
        "li             a0, 48\n\t"
        "divw           t1, %[n], a0\n\t"  // t1 = n24
        "remw           t2, %[n], a0\n\t"  // t2 = n % 24 (n_tail)
        "srai           t3, %[k], 1\n\t"   // t3 = k2
        "andi           t4, %[k], 1\n\t"   // t4 = k1

        "srai           t0, %[m], 3\n\t"  // t0 = m8
        "beqz           t0, 19f\n\t"

        // m8
        "1:\n\t"
        "li             s1, 16\n\t"
        "vsetvli        zero, s1, e16, m1\n\t"  // set vl = 8
        // load 8 bias_data for 8 out_channels
        "flh            fs0, 0(%[bias_ptr])\n\t"
        "flh            fs1, 2(%[bias_ptr])\n\t"
        "flh            fs2, 4(%[bias_ptr])\n\t"
        "flh            fs3, 6(%[bias_ptr])\n\t"
        "flh            fs4, 8(%[bias_ptr])\n\t"
        "flh            fs5, 10(%[bias_ptr])\n\t"
        "flh            fs6, 12(%[bias_ptr])\n\t"
        "flh            fs7, 14(%[bias_ptr])\n\t"

        "mv             s1, t1\n\t"  // s1 = n24

        // init output addr
        "slli           t5, %[ldc], 1\n\t"  // t5_tmp = ldc * 2
        "mv             a0, %[output_ptr]\n\t"
        "add            a1, a0, t5\n\t"
        "add            a2, a1, t5\n\t"
        "add            a3, a2, t5\n\t"
        "add            a4, a3, t5\n\t"
        "add            a5, a4, t5\n\t"
        "add            a6, a5, t5\n\t"
        "add            a7, a6, t5\n\t"  // ******* 移到m8外面

        "mv             s3, %[input_ptr]\n\t"  // s3 hold input data start addr

        "beqz           t1, 6f\n\t"  // if n24==0, jump to m8n16
        // m8n24
        "2:\n\t"
        // init out_tmp = bias
        "vfmv.v.f       v8, fs0\n\t"
        "vfmv.v.f       v9, fs0\n\t"
        "vfmv.v.f       v10, fs0\n\t"
        "vfmv.v.f       v11, fs1\n\t"
        "vfmv.v.f       v12, fs1\n\t"
        "vfmv.v.f       v13, fs1\n\t"
        "vfmv.v.f       v14, fs2\n\t"
        "vfmv.v.f       v15, fs2\n\t"
        "vfmv.v.f       v16, fs2\n\t"
        "vfmv.v.f       v17, fs3\n\t"
        "vfmv.v.f       v18, fs3\n\t"
        "vfmv.v.f       v19, fs3\n\t"
        "vfmv.v.f       v20, fs4\n\t"
        "vfmv.v.f       v21, fs4\n\t"
        "vfmv.v.f       v22, fs4\n\t"
        "vfmv.v.f       v23, fs5\n\t"
        "vfmv.v.f       v24, fs5\n\t"
        "vfmv.v.f       v25, fs5\n\t"
        "vfmv.v.f       v26, fs6\n\t"
        "vfmv.v.f       v27, fs6\n\t"
        "vfmv.v.f       v28, fs6\n\t"
        "vfmv.v.f       v29, fs7\n\t"
        "vfmv.v.f       v30, fs7\n\t"
        "vfmv.v.f       v31, fs7\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (s3)\n\t"
        "addi           s3, s3, 32\n\t"
        "vle16.v        v2, (s3)\n\t"
        "addi           s3, s3, 32\n\t"
        "vle16.v        v3, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"
        "flh            ft2, 4(s2)\n\t"
        "flh            ft3, 6(s2)\n\t"
        "flh            ft4, 8(s2)\n\t"
        "flh            ft5, 10(s2)\n\t"
        "flh            ft6, 12(s2)\n\t"
        "flh            ft7, 14(s2)\n\t"

        "beqz           t3, 4f\n\t"  // if k2 == 0, jump to m8n24k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m8n24k2
        "3:\n\t"

        "vle16.v        v4, (s3)\n\t"
        "addi           s3, s3, 32\n\t"
        "vle16.v        v5, (s3)\n\t"
        "addi           s3, s3, 32\n\t"
        "vle16.v        v6, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft0, v2\n\t"
        "vfmacc.vf      v10, ft0, v3\n\t"
        "flh            fa0, 16(s2)\n\t"
        "vfmacc.vf      v11, ft1, v1\n\t"
        "vfmacc.vf      v12, ft1, v2\n\t"
        "vfmacc.vf      v13, ft1, v3\n\t"
        "flh            fa1, 18(s2)\n\t"
        "vfmacc.vf      v14, ft2, v1\n\t"
        "vfmacc.vf      v15, ft2, v2\n\t"
        "vfmacc.vf      v16, ft2, v3\n\t"
        "flh            fa2, 20(s2)\n\t"
        "vfmacc.vf      v17, ft3, v1\n\t"
        "vfmacc.vf      v18, ft3, v2\n\t"
        "vfmacc.vf      v19, ft3, v3\n\t"
        "flh            fa3, 22(s2)\n\t"
        "vfmacc.vf      v20, ft4, v1\n\t"
        "vfmacc.vf      v21, ft4, v2\n\t"
        "vfmacc.vf      v22, ft4, v3\n\t"
        "flh            fa4, 24(s2)\n\t"
        "vfmacc.vf      v23, ft5, v1\n\t"
        "vfmacc.vf      v24, ft5, v2\n\t"
        "vfmacc.vf      v25, ft5, v3\n\t"
        "flh            fa5, 26(s2)\n\t"
        "vfmacc.vf      v26, ft6, v1\n\t"
        "vfmacc.vf      v27, ft6, v2\n\t"
        "vfmacc.vf      v28, ft6, v3\n\t"
        "flh            fa6, 28(s2)\n\t"
        "vfmacc.vf      v29, ft7, v1\n\t"
        "vfmacc.vf      v30, ft7, v2\n\t"
        "vfmacc.vf      v31, ft7, v3\n\t"
        "flh            fa7, 30(s2)\n\t"  // 0
        "addi           s2, s2, 32\n\t"   // += 16 elements, bump kernel to next k2 addr

        "vle16.v        v1, (s3)\n\t"
        "addi           s3, s3, 32\n\t"
        "vle16.v        v2, (s3)\n\t"
        "addi           s3, s3, 32\n\t"
        "vle16.v        v3, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        "vfmacc.vf      v8, fa0, v4\n\t"
        "vfmacc.vf      v9, fa0, v5\n\t"
        "vfmacc.vf      v10, fa0, v6\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v11, fa1, v4\n\t"
        "vfmacc.vf      v12, fa1, v5\n\t"
        "vfmacc.vf      v13, fa1, v6\n\t"
        "flh            ft1, 2(s2)\n\t"
        "vfmacc.vf      v14, fa2, v4\n\t"
        "vfmacc.vf      v15, fa2, v5\n\t"
        "vfmacc.vf      v16, fa2, v6\n\t"
        "flh            ft2, 4(s2)\n\t"
        "vfmacc.vf      v17, fa3, v4\n\t"
        "vfmacc.vf      v18, fa3, v5\n\t"
        "vfmacc.vf      v19, fa3, v6\n\t"
        "flh            ft3, 6(s2)\n\t"
        "vfmacc.vf      v20, fa4, v4\n\t"
        "vfmacc.vf      v21, fa4, v5\n\t"
        "vfmacc.vf      v22, fa4, v6\n\t"
        "flh            ft4, 8(s2)\n\t"
        "vfmacc.vf      v23, fa5, v4\n\t"
        "vfmacc.vf      v24, fa5, v5\n\t"
        "vfmacc.vf      v25, fa5, v6\n\t"
        "flh            ft5, 10(s2)\n\t"
        "vfmacc.vf      v26, fa6, v4\n\t"
        "vfmacc.vf      v27, fa6, v5\n\t"
        "vfmacc.vf      v28, fa6, v6\n\t"
        "flh            ft6, 12(s2)\n\t"
        "vfmacc.vf      v29, fa7, v4\n\t"
        "vfmacc.vf      v30, fa7, v5\n\t"
        "vfmacc.vf      v31, fa7, v6\n\t"
        "flh            ft7, 14(s2)\n\t"  // 1

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 3b\n\t"

        // m8n24k1
        "4:\n\t"
        "beqz           t4, 5f\n\t"  // if k1 == 0, jump to end kernel_m8n24

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft0, v2\n\t"
        "vfmacc.vf      v10, ft0, v3\n\t"
        "vfmacc.vf      v11, ft1, v1\n\t"
        "vfmacc.vf      v12, ft1, v2\n\t"
        "vfmacc.vf      v13, ft1, v3\n\t"
        "vfmacc.vf      v14, ft2, v1\n\t"
        "vfmacc.vf      v15, ft2, v2\n\t"
        "vfmacc.vf      v16, ft2, v3\n\t"
        "vfmacc.vf      v17, ft3, v1\n\t"
        "vfmacc.vf      v18, ft3, v2\n\t"
        "vfmacc.vf      v19, ft3, v3\n\t"
        "vfmacc.vf      v20, ft4, v1\n\t"
        "vfmacc.vf      v21, ft4, v2\n\t"
        "vfmacc.vf      v22, ft4, v3\n\t"
        "vfmacc.vf      v23, ft5, v1\n\t"
        "vfmacc.vf      v24, ft5, v2\n\t"
        "vfmacc.vf      v25, ft5, v3\n\t"
        "vfmacc.vf      v26, ft6, v1\n\t"
        "vfmacc.vf      v27, ft6, v2\n\t"
        "vfmacc.vf      v28, ft6, v3\n\t"
        "vfmacc.vf      v29, ft7, v1\n\t"
        "vfmacc.vf      v30, ft7, v2\n\t"
        "vfmacc.vf      v31, ft7, v3\n\t"

        "addi           s3, s3, 96\n\t"  // ********************

        // end kernel_m8n24
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           s3, s3, -96\n\t"  // pb -= 24

        "vse16.v        v8, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v11, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v14, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v17, (a3)\n\t"
        "addi           a3, a3, 32\n\t"
        "vse16.v        v20, (a4)\n\t"
        "addi           a4, a4, 32\n\t"
        "vse16.v        v23, (a5)\n\t"
        "addi           a5, a5, 32\n\t"
        "vse16.v        v26, (a6)\n\t"
        "addi           a6, a6, 32\n\t"
        "vse16.v        v29, (a7)\n\t"
        "addi           a7, a7, 32\n\t"

        "vse16.v        v9, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v12, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v15, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v18, (a3)\n\t"
        "addi           a3, a3, 32\n\t"
        "vse16.v        v21, (a4)\n\t"
        "addi           a4, a4, 32\n\t"
        "vse16.v        v24, (a5)\n\t"
        "addi           a5, a5, 32\n\t"
        "vse16.v        v27, (a6)\n\t"
        "addi           a6, a6, 32\n\t"
        "vse16.v        v30, (a7)\n\t"
        "addi           a7, a7, 32\n\t"

        "vse16.v        v10, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v13, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v16, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v19, (a3)\n\t"
        "addi           a3, a3, 32\n\t"
        "vse16.v        v22, (a4)\n\t"
        "addi           a4, a4, 32\n\t"
        "vse16.v        v25, (a5)\n\t"
        "addi           a5, a5, 32\n\t"
        "vse16.v        v28, (a6)\n\t"
        "addi           a6, a6, 32\n\t"
        "vse16.v        v31, (a7)\n\t"
        "addi           a7, a7, 32\n\t"

        "addi           s1, s1, -1\n\t"
        "bnez           s1, 2b\n\t"

        // m8n16
        "6:\n\t"
        "andi           s1, t2, 32\n\t"  // s1 = bool_n16
        "beqz           s1, 10f\n\t"     // if n16==0, jump to m8n8

        // init out_tmp = bias
        "vfmv.v.f       v16, fs0\n\t"
        "vfmv.v.f       v17, fs0\n\t"
        "vfmv.v.f       v18, fs1\n\t"
        "vfmv.v.f       v19, fs1\n\t"
        "vfmv.v.f       v20, fs2\n\t"
        "vfmv.v.f       v21, fs2\n\t"
        "vfmv.v.f       v22, fs3\n\t"
        "vfmv.v.f       v23, fs3\n\t"
        "vfmv.v.f       v24, fs4\n\t"
        "vfmv.v.f       v25, fs4\n\t"
        "vfmv.v.f       v26, fs5\n\t"
        "vfmv.v.f       v27, fs5\n\t"
        "vfmv.v.f       v28, fs6\n\t"
        "vfmv.v.f       v29, fs6\n\t"
        "vfmv.v.f       v30, fs7\n\t"
        "vfmv.v.f       v31, fs7\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (s3)\n\t"
        "addi           s3, s3, 32\n\t"
        "vle16.v        v2, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"
        "flh            ft2, 4(s2)\n\t"
        "flh            ft3, 6(s2)\n\t"
        "flh            ft4, 8(s2)\n\t"
        "flh            ft5, 10(s2)\n\t"
        "flh            ft6, 12(s2)\n\t"
        "flh            ft7, 14(s2)\n\t"

        "beqz           t3, 8f\n\t"  // if k2 == 0, jump to m8n16k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m8n16k2
        "7:\n\t"
        "vle16.v        v4, (s3)\n\t"
        "addi           s3, s3, 32\n\t"
        "vle16.v        v5, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        "vfmacc.vf      v16, ft0, v1\n\t"
        "vfmacc.vf      v17, ft0, v2\n\t"
        "flh            fa0, 16(s2)\n\t"
        "vfmacc.vf      v18, ft1, v1\n\t"
        "vfmacc.vf      v19, ft1, v2\n\t"
        "flh            fa1, 18(s2)\n\t"
        "vfmacc.vf      v20, ft2, v1\n\t"
        "vfmacc.vf      v21, ft2, v2\n\t"
        "flh            fa2, 20(s2)\n\t"
        "vfmacc.vf      v22, ft3, v1\n\t"
        "vfmacc.vf      v23, ft3, v2\n\t"
        "flh            fa3, 22(s2)\n\t"
        "vfmacc.vf      v24, ft4, v1\n\t"
        "vfmacc.vf      v25, ft4, v2\n\t"
        "flh            fa4, 24(s2)\n\t"
        "vfmacc.vf      v26, ft5, v1\n\t"
        "vfmacc.vf      v27, ft5, v2\n\t"
        "flh            fa5, 26(s2)\n\t"
        "vfmacc.vf      v28, ft6, v1\n\t"
        "vfmacc.vf      v29, ft6, v2\n\t"
        "flh            fa6, 28(s2)\n\t"
        "vfmacc.vf      v30, ft7, v1\n\t"
        "vfmacc.vf      v31, ft7, v2\n\t"
        "flh            fa7, 30(s2)\n\t"  // 0
        "addi           s2, s2, 32\n\t"   // += 16 elements, bump kernel to next k2 addr

        "vle16.v        v1, (s3)\n\t"
        "addi           s3, s3, 32\n\t"
        "vle16.v        v2, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        "vfmacc.vf      v16, fa0, v4\n\t"
        "vfmacc.vf      v17, fa0, v5\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v18, fa1, v4\n\t"
        "vfmacc.vf      v19, fa1, v5\n\t"
        "flh            ft1, 2(s2)\n\t"
        "vfmacc.vf      v20, fa2, v4\n\t"
        "vfmacc.vf      v21, fa2, v5\n\t"
        "flh            ft2, 4(s2)\n\t"
        "vfmacc.vf      v22, fa3, v4\n\t"
        "vfmacc.vf      v23, fa3, v5\n\t"
        "flh            ft3, 6(s2)\n\t"
        "vfmacc.vf      v24, fa4, v4\n\t"
        "vfmacc.vf      v25, fa4, v5\n\t"
        "flh            ft4, 8(s2)\n\t"
        "vfmacc.vf      v26, fa5, v4\n\t"
        "vfmacc.vf      v27, fa5, v5\n\t"
        "flh            ft5, 10(s2)\n\t"
        "vfmacc.vf      v28, fa6, v4\n\t"
        "vfmacc.vf      v29, fa6, v5\n\t"
        "flh            ft6, 12(s2)\n\t"
        "vfmacc.vf      v30, fa7, v4\n\t"
        "vfmacc.vf      v31, fa7, v5\n\t"
        "flh            ft7, 14(s2)\n\t"  // 1

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 7b\n\t"

        // m8n16k1
        "8:\n\t"
        "beqz           t4, 9f\n\t"  // if k1 == 0, jump to end kernel_m8n16

        "vfmacc.vf      v16, ft0, v1\n\t"
        "vfmacc.vf      v17, ft0, v2\n\t"
        "vfmacc.vf      v18, ft1, v1\n\t"
        "vfmacc.vf      v19, ft1, v2\n\t"
        "vfmacc.vf      v20, ft2, v1\n\t"
        "vfmacc.vf      v21, ft2, v2\n\t"
        "vfmacc.vf      v22, ft3, v1\n\t"
        "vfmacc.vf      v23, ft3, v2\n\t"
        "vfmacc.vf      v24, ft4, v1\n\t"
        "vfmacc.vf      v25, ft4, v2\n\t"
        "vfmacc.vf      v26, ft5, v1\n\t"
        "vfmacc.vf      v27, ft5, v2\n\t"
        "vfmacc.vf      v28, ft6, v1\n\t"
        "vfmacc.vf      v29, ft6, v2\n\t"
        "vfmacc.vf      v30, ft7, v1\n\t"
        "vfmacc.vf      v31, ft7, v2\n\t"

        "addi           s3, s3, 64\n\t"  // ********************

        // end kernel_m8n16
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           s3, s3, -64\n\t"  // pb -= 16

        "vse16.v        v16, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v18, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v20, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v22, (a3)\n\t"
        "addi           a3, a3, 32\n\t"
        "vse16.v        v24, (a4)\n\t"
        "addi           a4, a4, 32\n\t"
        "vse16.v        v26, (a5)\n\t"
        "addi           a5, a5, 32\n\t"
        "vse16.v        v28, (a6)\n\t"
        "addi           a6, a6, 32\n\t"
        "vse16.v        v30, (a7)\n\t"
        "addi           a7, a7, 32\n\t"

        "vse16.v        v17, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v19, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v21, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v23, (a3)\n\t"
        "addi           a3, a3, 32\n\t"
        "vse16.v        v25, (a4)\n\t"
        "addi           a4, a4, 32\n\t"
        "vse16.v        v27, (a5)\n\t"
        "addi           a5, a5, 32\n\t"
        "vse16.v        v29, (a6)\n\t"
        "addi           a6, a6, 32\n\t"
        "vse16.v        v31, (a7)\n\t"
        "addi           a7, a7, 32\n\t"

        // m8n8
        "10:\n\t"
        "andi           s1, t2, 16\n\t"  // s1 = bool_n8
        "beqz           s1, 14f\n\t"     // if n8==0, jump to m8n_tail

        // init out_tmp = bias
        "vfmv.v.f       v24, fs0\n\t"
        "vfmv.v.f       v25, fs1\n\t"
        "vfmv.v.f       v26, fs2\n\t"
        "vfmv.v.f       v27, fs3\n\t"
        "vfmv.v.f       v28, fs4\n\t"
        "vfmv.v.f       v29, fs5\n\t"
        "vfmv.v.f       v30, fs6\n\t"
        "vfmv.v.f       v31, fs7\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"
        "flh            ft2, 4(s2)\n\t"
        "flh            ft3, 6(s2)\n\t"
        "flh            ft4, 8(s2)\n\t"
        "flh            ft5, 10(s2)\n\t"
        "flh            ft6, 12(s2)\n\t"
        "flh            ft7, 14(s2)\n\t"

        "beqz           t3, 12f\n\t"  // if k2 == 0, jump to m8n8k1
        "mv             t5, t3\n\t"   // t5 = k2

        // m8n4k2
        "11:\n\t"
        "vle16.v        v4, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa0, 16(s2)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "flh            fa1, 18(s2)\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "flh            fa2, 20(s2)\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"
        "flh            fa3, 22(s2)\n\t"
        "vfmacc.vf      v28, ft4, v1\n\t"
        "flh            fa4, 24(s2)\n\t"
        "vfmacc.vf      v29, ft5, v1\n\t"
        "flh            fa5, 26(s2)\n\t"
        "vfmacc.vf      v30, ft6, v1\n\t"
        "flh            fa6, 28(s2)\n\t"
        "vfmacc.vf      v31, ft7, v1\n\t"
        "flh            fa7, 30(s2)\n\t"  // 0
        "addi           s2, s2, 32\n\t"   // += 16 elements, bump kernel to next k2 addr

        "vle16.v        v1, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        "vfmacc.vf      v24, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v25, fa1, v4\n\t"
        "flh            ft1, 2(s2)\n\t"
        "vfmacc.vf      v26, fa2, v4\n\t"
        "flh            ft2, 4(s2)\n\t"
        "vfmacc.vf      v27, fa3, v4\n\t"
        "flh            ft3, 6(s2)\n\t"
        "vfmacc.vf      v28, fa4, v4\n\t"
        "flh            ft4, 8(s2)\n\t"
        "vfmacc.vf      v29, fa5, v4\n\t"
        "flh            ft5, 10(s2)\n\t"
        "vfmacc.vf      v30, fa6, v4\n\t"
        "flh            ft6, 12(s2)\n\t"
        "vfmacc.vf      v31, fa7, v4\n\t"
        "flh            ft7, 14(s2)\n\t"  // 1

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 11b\n\t"

        // m8n8k1
        "12:\n\t"
        "beqz           t4, 13f\n\t"  // if k1 == 0, jump to end kernel_m8n8

        "vfmacc.vf      v24, ft0, v1\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"
        "vfmacc.vf      v28, ft4, v1\n\t"
        "vfmacc.vf      v29, ft5, v1\n\t"
        "vfmacc.vf      v30, ft6, v1\n\t"
        "vfmacc.vf      v31, ft7, v1\n\t"

        "addi           s3, s3, 32\n\t"  // ********************

        // end kernel_m8n8
        "13:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           s3, s3, -32\n\t"  // pb -= 8

        "vse16.v        v24, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v25, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v26, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v27, (a3)\n\t"
        "addi           a3, a3, 32\n\t"
        "vse16.v        v28, (a4)\n\t"
        "addi           a4, a4, 32\n\t"
        "vse16.v        v29, (a5)\n\t"
        "addi           a5, a5, 32\n\t"
        "vse16.v        v30, (a6)\n\t"
        "addi           a6, a6, 32\n\t"
        "vse16.v        v31, (a7)\n\t"
        "addi           a7, a7, 32\n\t"

        // m8n_tail
        "14:\n\t"
        "andi           s1, t2, 15\n\t"         // s1 = bool_n_tail
        "beqz           a1, 18f\n\t"            // if n4==0, jump to m8n_tail
        "vsetvli        zero, s1, e16, m1\n\t"  // set vl = n_tail
        "slli           t6, s1, 1\n\t"          // t6 = 2 * n_tail
        // init out_tmp = bias
        "vfmv.v.f       v24, fs0\n\t"
        "vfmv.v.f       v25, fs1\n\t"
        "vfmv.v.f       v26, fs2\n\t"
        "vfmv.v.f       v27, fs3\n\t"
        "vfmv.v.f       v28, fs4\n\t"
        "vfmv.v.f       v29, fs5\n\t"
        "vfmv.v.f       v30, fs6\n\t"
        "vfmv.v.f       v31, fs7\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (s3)\n\t"
        "add            s3, s3, t6\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"
        "flh            ft2, 4(s2)\n\t"
        "flh            ft3, 6(s2)\n\t"
        "flh            ft4, 8(s2)\n\t"
        "flh            ft5, 10(s2)\n\t"
        "flh            ft6, 12(s2)\n\t"
        "flh            ft7, 14(s2)\n\t"

        "beqz           t3, 16f\n\t"  // if k2 == 0, jump to m8n_tailk1
        "mv             t5, t3\n\t"   // t5 = k2

        // m8n_tailk2
        "15:\n\t"
        "vle16.v        v4, (s3)\n\t"
        "add            s3, s3, t6\n\t"

        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa0, 16(s2)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "flh            fa1, 18(s2)\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "flh            fa2, 20(s2)\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"
        "flh            fa3, 22(s2)\n\t"
        "vfmacc.vf      v28, ft4, v1\n\t"
        "flh            fa4, 24(s2)\n\t"
        "vfmacc.vf      v29, ft5, v1\n\t"
        "flh            fa5, 26(s2)\n\t"
        "vfmacc.vf      v30, ft6, v1\n\t"
        "flh            fa6, 28(s2)\n\t"
        "vfmacc.vf      v31, ft7, v1\n\t"
        "flh            fa7, 30(s2)\n\t"  // 0
        "addi           s2, s2, 32\n\t"   // += 16 elements, bump kernel to next k2 addr

        "vle16.v        v1, (s3)\n\t"
        "add            s3, s3, t6\n\t"

        "vfmacc.vf      v24, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v25, fa1, v4\n\t"
        "flh            ft1, 2(s2)\n\t"
        "vfmacc.vf      v26, fa2, v4\n\t"
        "flh            ft2, 4(s2)\n\t"
        "vfmacc.vf      v27, fa3, v4\n\t"
        "flh            ft3, 6(s2)\n\t"
        "vfmacc.vf      v28, fa4, v4\n\t"
        "flh            ft4, 8(s2)\n\t"
        "vfmacc.vf      v29, fa5, v4\n\t"
        "flh            ft5, 10(s2)\n\t"
        "vfmacc.vf      v30, fa6, v4\n\t"
        "flh            ft6, 12(s2)\n\t"
        "vfmacc.vf      v31, fa7, v4\n\t"
        "flh            ft7, 14(s2)\n\t"  // 1

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 15b\n\t"

        // m8n_tailk1
        "16:\n\t"
        "beqz           t4, 17f\n\t"  // if k1 == 0, jump to end kernel_m8n4

        "vfmacc.vf      v24, ft0, v1\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"
        "vfmacc.vf      v28, ft4, v1\n\t"
        "vfmacc.vf      v29, ft5, v1\n\t"
        "vfmacc.vf      v30, ft6, v1\n\t"
        "vfmacc.vf      v31, ft7, v1\n\t"

        "add            s3, s3, t6\n\t"  // ********************

        // end kernel_m8n_tail
        "17:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "sub            s3, s3, t6\n\t"  // pb -= n_tail

        "vse16.v        v24, (a0)\n\t"
        "add            a0, a0, t6\n\t"
        "vse16.v        v25, (a1)\n\t"
        "add            a1, a1, t6\n\t"
        "vse16.v        v26, (a2)\n\t"
        "add            a2, a2, t6\n\t"
        "vse16.v        v27, (a3)\n\t"
        "add            a3, a3, t6\n\t"
        "vse16.v        v28, (a4)\n\t"
        "add            a4, a4, t6\n\t"
        "vse16.v        v29, (a5)\n\t"
        "add           a5, a5, t6\n\t"
        "vse16.v        v30, (a6)\n\t"
        "add            a6, a6, t6\n\t"
        "vse16.v        v31, (a7)\n\t"
        "add            a7, a7, t6\n\t"

        // end kernel_m8
        "18:\n\t"
        "addi           %[bias_ptr], %[bias_ptr], 16\n\t"  // bias_data += 8
        "slli           t6, %[k], 4\n\t"
        "add            %[kernel_ptr], %[kernel_ptr], t6\n\t"  // kernel_data += 8 * k
        "slli           t6, %[ldc], 4\n\t"
        "add            %[output_ptr], %[output_ptr], t6\n\t"  // output_data += 8 * ldc

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        // ending
        "19:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)
        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v1", "v2", "v3", "v4", "v5", "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
        "v29", "v30", "v31",
        // We use these general-purpose registers.
        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6",
        "s1", "s2", "s3", "fs0", "fs1", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fa0", "fa1",
        "fa2", "fa3", "fa4", "fa5", "fa6", "fa7", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6",
        "ft7");
}

static inline void kernel_m4n48_fp16_v256(__fp16 *dst, __fp16 *sa, __fp16 *sb, int m, int k, int n,
                                          int ldc, __fp16 *bias)
{
    asm volatile(
        "li             a0, 48\n\t"
        "divw           t1, %[n], a0\n\t"  // t1 = n12
        "remw           t2, %[n], a0\n\t"  // t2 = n % 12 (n_tail)
        "srai           t3, %[k], 1\n\t"   // t3 = k2
        "andi           t4, %[k], 1\n\t"   // t4 = k1

        // m4
        "1:\n\t"
        "li             a0, 16\n\t"
        "vsetvli        zero, a0, e16, m1\n\t"  // set vl = 4
        // load 8 bias_data for 8 out_channels
        "flh            fs0, 0(%[bias_ptr])\n\t"
        "flh            fs1, 2(%[bias_ptr])\n\t"
        "flh            fs2, 4(%[bias_ptr])\n\t"
        "flh            fs3, 6(%[bias_ptr])\n\t"

        // init output addr
        "slli           t5, %[ldc], 1\n\t"  // t5_tmp = ldc * 2
        "mv             a0, %[output_ptr]\n\t"
        "add            a1, a0, t5\n\t"
        "add            a2, a1, t5\n\t"
        "add            a3, a2, t5\n\t"

        "beqz           t1, 6f\n\t"  // if n12==0, jump to m4n8
        // m4n12
        "2:\n\t"
        // init out_tmp = bias
        "vfmv.v.f       v8, fs0\n\t"
        "vfmv.v.f       v9, fs0\n\t"
        "vfmv.v.f       v10, fs0\n\t"
        "vfmv.v.f       v11, fs1\n\t"
        "vfmv.v.f       v12, fs1\n\t"
        "vfmv.v.f       v13, fs1\n\t"
        "vfmv.v.f       v14, fs2\n\t"
        "vfmv.v.f       v15, fs2\n\t"
        "vfmv.v.f       v16, fs2\n\t"
        "vfmv.v.f       v17, fs3\n\t"
        "vfmv.v.f       v18, fs3\n\t"
        "vfmv.v.f       v19, fs3\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v3, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"
        "flh            ft2, 4(s2)\n\t"
        "flh            ft3, 6(s2)\n\t"

        "beqz           t3, 4f\n\t"  // if k2 == 0, jump to m4n12k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m4n12k2
        "3:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v5, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v6, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft0, v2\n\t"
        "vfmacc.vf      v10, ft0, v3\n\t"
        "flh            fa0, 8(s2)\n\t"
        "vfmacc.vf      v11, ft1, v1\n\t"
        "vfmacc.vf      v12, ft1, v2\n\t"
        "vfmacc.vf      v13, ft1, v3\n\t"
        "flh            fa1, 10(s2)\n\t"
        "vfmacc.vf      v14, ft2, v1\n\t"
        "vfmacc.vf      v15, ft2, v2\n\t"
        "vfmacc.vf      v16, ft2, v3\n\t"
        "flh            fa2, 12(s2)\n\t"
        "vfmacc.vf      v17, ft3, v1\n\t"
        "vfmacc.vf      v18, ft3, v2\n\t"
        "vfmacc.vf      v19, ft3, v3\n\t"
        "flh            fa3, 14(s2)\n\t"
        "addi           s2, s2, 16\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v3, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, fa0, v4\n\t"
        "vfmacc.vf      v9, fa0, v5\n\t"
        "vfmacc.vf      v10, fa0, v6\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v11, fa1, v4\n\t"
        "vfmacc.vf      v12, fa1, v5\n\t"
        "vfmacc.vf      v13, fa1, v6\n\t"
        "flh            ft1, 2(s2)\n\t"
        "vfmacc.vf      v14, fa2, v4\n\t"
        "vfmacc.vf      v15, fa2, v5\n\t"
        "vfmacc.vf      v16, fa2, v6\n\t"
        "flh            ft2, 4(s2)\n\t"
        "vfmacc.vf      v17, fa3, v4\n\t"
        "vfmacc.vf      v18, fa3, v5\n\t"
        "vfmacc.vf      v19, fa3, v6\n\t"
        "flh            ft3, 6(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 3b\n\t"

        // m4n12k1
        "4:\n\t"
        "beqz           t4, 5f\n\t"  // if k1 == 0, jump to end kernel_m4n12

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft0, v2\n\t"
        "vfmacc.vf      v10, ft0, v3\n\t"
        "vfmacc.vf      v11, ft1, v1\n\t"
        "vfmacc.vf      v12, ft1, v2\n\t"
        "vfmacc.vf      v13, ft1, v3\n\t"
        "vfmacc.vf      v14, ft2, v1\n\t"
        "vfmacc.vf      v15, ft2, v2\n\t"
        "vfmacc.vf      v16, ft2, v3\n\t"
        "vfmacc.vf      v17, ft3, v1\n\t"
        "vfmacc.vf      v18, ft3, v2\n\t"
        "vfmacc.vf      v19, ft3, v3\n\t"

        "addi           %[input_ptr], %[input_ptr], 96\n\t"  // ********************

        // end kernel_m4n12
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -96\n\t"  // pb -= 24

        "vse16.v        v8, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v11, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v14, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v17, (a3)\n\t"
        "addi           a3, a3, 32\n\t"

        "vse16.v        v9, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v12, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v15, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v18, (a3)\n\t"
        "addi           a3, a3, 32\n\t"

        "vse16.v        v10, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v13, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v16, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v19, (a3)\n\t"
        "addi           a3, a3, 32\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m4n8
        "6:\n\t"
        "andi           t1, t2, 32\n\t"  // s1 = bool_n8
        "beqz           t1, 10f\n\t"     // if n8==0, jump to m4n4

        // init out_tmp = bias
        "vfmv.v.f       v8, fs0\n\t"
        "vfmv.v.f       v9, fs0\n\t"
        "vfmv.v.f       v10, fs1\n\t"
        "vfmv.v.f       v11, fs1\n\t"
        "vfmv.v.f       v12, fs2\n\t"
        "vfmv.v.f       v13, fs2\n\t"
        "vfmv.v.f       v14, fs3\n\t"
        "vfmv.v.f       v15, fs3\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"
        "flh            ft2, 4(s2)\n\t"
        "flh            ft3, 6(s2)\n\t"

        "beqz           t3, 8f\n\t"  // if k2 == 0, jump to m4n8k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m4n8k2
        "7:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v5, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft0, v2\n\t"
        "flh            fa0, 8(s2)\n\t"
        "vfmacc.vf      v10, ft1, v1\n\t"
        "vfmacc.vf      v11, ft1, v2\n\t"
        "flh            fa1, 10(s2)\n\t"
        "vfmacc.vf      v12, ft2, v1\n\t"
        "vfmacc.vf      v13, ft2, v2\n\t"
        "flh            fa2, 12(s2)\n\t"
        "vfmacc.vf      v14, ft3, v1\n\t"
        "vfmacc.vf      v15, ft3, v2\n\t"
        "flh            fa3, 14(s2)\n\t"
        "addi           s2, s2, 16\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, fa0, v4\n\t"
        "vfmacc.vf      v9, fa0, v5\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v10, fa1, v4\n\t"
        "vfmacc.vf      v11, fa1, v5\n\t"
        "flh            ft1, 2(s2)\n\t"
        "vfmacc.vf      v12, fa2, v4\n\t"
        "vfmacc.vf      v13, fa2, v5\n\t"
        "flh            ft2, 4(s2)\n\t"
        "vfmacc.vf      v14, fa3, v4\n\t"
        "vfmacc.vf      v15, fa3, v5\n\t"
        "flh            ft3, 6(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 7b\n\t"

        // m4n8k1
        "8:\n\t"
        "beqz           t4, 9f\n\t"  // if k1 == 0, jump to end kernel_m4n8

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft0, v2\n\t"
        "vfmacc.vf      v10, ft1, v1\n\t"
        "vfmacc.vf      v11, ft1, v2\n\t"
        "vfmacc.vf      v12, ft2, v1\n\t"
        "vfmacc.vf      v13, ft2, v2\n\t"
        "vfmacc.vf      v14, ft3, v1\n\t"
        "vfmacc.vf      v15, ft3, v2\n\t"

        "addi           %[input_ptr], %[input_ptr], 64\n\t"  // ********************

        // end kernel_m4n8
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -64\n\t"  // pb -= 8

        "vse16.v        v8, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v10, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v12, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v14, (a3)\n\t"
        "addi           a3, a3, 32\n\t"

        "vse16.v        v9, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v11, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v13, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v15, (a3)\n\t"
        "addi           a3, a3, 32\n\t"

        // m4n4
        "10:\n\t"
        "andi           t1, t2, 16\n\t"  // s1 = bool_n4
        "beqz           t1, 14f\n\t"     // if n4==0, jump to m4n_tail

        // init out_tmp = bias
        "vfmv.v.f       v8, fs0\n\t"
        "vfmv.v.f       v9, fs1\n\t"
        "vfmv.v.f       v10, fs2\n\t"
        "vfmv.v.f       v11, fs3\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"
        "flh            ft2, 4(s2)\n\t"
        "flh            ft3, 6(s2)\n\t"

        "beqz           t3, 12f\n\t"  // if k2 == 0, jump to m4n4k1
        "mv             t5, t3\n\t"   // t5 = k2

        // m4n4k2
        "11:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, ft0, v1\n\t"
        "flh            fa0, 8(s2)\n\t"
        "vfmacc.vf      v9, ft1, v1\n\t"
        "flh            fa1, 10(s2)\n\t"
        "vfmacc.vf      v10, ft2, v1\n\t"
        "flh            fa2, 12(s2)\n\t"
        "vfmacc.vf      v11, ft3, v1\n\t"
        "flh            fa3, 14(s2)\n\t"
        "addi           s2, s2, 16\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v9, fa1, v4\n\t"
        "flh            ft1, 2(s2)\n\t"
        "vfmacc.vf      v10, fa2, v4\n\t"
        "flh            ft2, 4(s2)\n\t"
        "vfmacc.vf      v11, fa3, v4\n\t"
        "flh            ft3, 6(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 11b\n\t"

        // m4n4k1
        "12:\n\t"
        "beqz           t4, 13f\n\t"  // if k1 == 0, jump to end kernel_m4n4

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft1, v1\n\t"
        "vfmacc.vf      v10, ft2, v1\n\t"
        "vfmacc.vf      v11, ft3, v1\n\t"

        "addi           %[input_ptr], %[input_ptr], 32\n\t"  // ********************

        // end kernel_m4n4
        "13:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -32\n\t"  // pb -= 4

        "vse16.v        v8, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v9, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v10, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v11, (a3)\n\t"
        "addi           a3, a3, 32\n\t"

        // m4n_tail
        "14:\n\t"
        "andi           t1, t2, 15\n\t"         // s1 = bool_n_tail
        "beqz           t1, 18f\n\t"            // if bool_n_tail==0, jump to ending
        "vsetvli        zero, t1, e16, m1\n\t"  // set vl = n_tail
        "slli           t6, t1, 1\n\t"          // t6 = 2 * n_tail
        // init out_tmp = bias
        "vfmv.v.f       v8, fs0\n\t"
        "vfmv.v.f       v9, fs1\n\t"
        "vfmv.v.f       v10, fs2\n\t"
        "vfmv.v.f       v11, fs3\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"
        "flh            ft2, 4(s2)\n\t"
        "flh            ft3, 6(s2)\n\t"

        "beqz           t3, 16f\n\t"  // if k2 == 0, jump to m4n_tailk1
        "mv             t5, t3\n\t"   // t5 = k2

        // m4n_tailk2
        "15:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vfmacc.vf      v8, ft0, v1\n\t"
        "flh            fa0, 8(s2)\n\t"
        "vfmacc.vf      v9, ft1, v1\n\t"
        "flh            fa1, 10(s2)\n\t"
        "vfmacc.vf      v10, ft2, v1\n\t"
        "flh            fa2, 12(s2)\n\t"
        "vfmacc.vf      v11, ft3, v1\n\t"
        "flh            fa3, 14(s2)\n\t"
        "addi           s2, s2, 16\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vfmacc.vf      v8, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v9, fa1, v4\n\t"
        "flh            ft1, 2(s2)\n\t"
        "vfmacc.vf      v10, fa2, v4\n\t"
        "flh            ft2, 4(s2)\n\t"
        "vfmacc.vf      v11, fa3, v4\n\t"
        "flh            ft3, 6(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 15b\n\t"

        // m4n_tailk1
        "16:\n\t"
        "beqz           t4, 17f\n\t"  // if k1 == 0, jump to end kernel_m4n4

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft1, v1\n\t"
        "vfmacc.vf      v10, ft2, v1\n\t"
        "vfmacc.vf      v11, ft3, v1\n\t"

        "add            %[input_ptr], %[input_ptr], t6\n\t"  // ********************

        // end kernel_m8n_tail
        "17:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "sub            %[input_ptr], %[input_ptr], t6\n\t"  // pb -= n_tail

        "vse16.v        v8, (a0)\n\t"
        "add            a0, a0, t6\n\t"
        "vse16.v        v9, (a1)\n\t"
        "add            a1, a1, t6\n\t"
        "vse16.v        v10, (a2)\n\t"
        "add            a2, a2, t6\n\t"
        "vse16.v        v11, (a3)\n\t"
        "add            a3, a3, t6\n\t"

        // ending
        "18:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)

        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v1", "v2", "v3", "v4", "v5", "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19",
        // We use these general-purpose registers.
        "a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s2", "fs0", "fs1", "fs2",
        "fs3", "fa0", "fa1", "fa2", "fa3", "ft0", "ft1", "ft2", "ft3");
}

static inline void kernel_m2n48_fp16_v256(__fp16 *dst, __fp16 *sa, __fp16 *sb, int m, int k, int n,
                                          int ldc, __fp16 *bias)
{
    asm volatile(
        "li             a0, 48\n\t"
        "divw           t1, %[n], a0\n\t"  // t1 = n12
        "remw           t2, %[n], a0\n\t"  // t2 = n % 12 (n_tail)
        "srai           t3, %[k], 1\n\t"   // t3 = k2
        "andi           t4, %[k], 1\n\t"   // t4 = k1

        // m4
        "1:\n\t"
        "li             a0, 16\n\t"
        "vsetvli        zero, a0, e16, m1\n\t"  // set vl = 4
        // load 8 bias_data for 8 out_channels
        "flh            fs0, 0(%[bias_ptr])\n\t"
        "flh            fs1, 2(%[bias_ptr])\n\t"

        // init output addr
        "slli           t5, %[ldc], 1\n\t"  // t5_tmp = ldc * 2
        "mv             a0, %[output_ptr]\n\t"
        "add            a1, a0, t5\n\t"

        "beqz           t1, 6f\n\t"  // if n12==0, jump to m4n8
        // m4n12
        "2:\n\t"
        // init out_tmp = bias
        "vfmv.v.f       v8, fs0\n\t"
        "vfmv.v.f       v9, fs0\n\t"
        "vfmv.v.f       v10, fs0\n\t"
        "vfmv.v.f       v11, fs1\n\t"
        "vfmv.v.f       v12, fs1\n\t"
        "vfmv.v.f       v13, fs1\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v3, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"

        "beqz           t3, 4f\n\t"  // if k2 == 0, jump to m4n12k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m4n12k2
        "3:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v5, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v6, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft0, v2\n\t"
        "vfmacc.vf      v10, ft0, v3\n\t"
        "flh            fa0, 4(s2)\n\t"
        "vfmacc.vf      v11, ft1, v1\n\t"
        "vfmacc.vf      v12, ft1, v2\n\t"
        "vfmacc.vf      v13, ft1, v3\n\t"
        "flh            fa1, 6(s2)\n\t"
        "addi           s2, s2, 8\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v3, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, fa0, v4\n\t"
        "vfmacc.vf      v9, fa0, v5\n\t"
        "vfmacc.vf      v10, fa0, v6\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v11, fa1, v4\n\t"
        "vfmacc.vf      v12, fa1, v5\n\t"
        "vfmacc.vf      v13, fa1, v6\n\t"
        "flh            ft1, 2(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 3b\n\t"

        // m4n12k1
        "4:\n\t"
        "beqz           t4, 5f\n\t"  // if k1 == 0, jump to end kernel_m4n12

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft0, v2\n\t"
        "vfmacc.vf      v10, ft0, v3\n\t"
        "vfmacc.vf      v11, ft1, v1\n\t"
        "vfmacc.vf      v12, ft1, v2\n\t"
        "vfmacc.vf      v13, ft1, v3\n\t"

        "addi           %[input_ptr], %[input_ptr], 96\n\t"  // ********************

        // end kernel_m4n12
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -96\n\t"  // pb -= 24

        "vse16.v        v8, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v11, (a1)\n\t"
        "addi           a1, a1, 32\n\t"

        "vse16.v        v9, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v12, (a1)\n\t"
        "addi           a1, a1, 32\n\t"

        "vse16.v        v10, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v13, (a1)\n\t"
        "addi           a1, a1, 32\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m4n8
        "6:\n\t"
        "andi           t1, t2, 32\n\t"  // s1 = bool_n8
        "beqz           t1, 10f\n\t"     // if n8==0, jump to m4n4

        // init out_tmp = bias
        "vfmv.v.f       v8, fs0\n\t"
        "vfmv.v.f       v9, fs0\n\t"
        "vfmv.v.f       v10, fs1\n\t"
        "vfmv.v.f       v11, fs1\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"

        "beqz           t3, 8f\n\t"  // if k2 == 0, jump to m4n8k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m4n8k2
        "7:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v5, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft0, v2\n\t"
        "flh            fa0, 4(s2)\n\t"
        "vfmacc.vf      v10, ft1, v1\n\t"
        "vfmacc.vf      v11, ft1, v2\n\t"
        "flh            fa1, 6(s2)\n\t"
        "addi           s2, s2, 8\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, fa0, v4\n\t"
        "vfmacc.vf      v9, fa0, v5\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v10, fa1, v4\n\t"
        "vfmacc.vf      v11, fa1, v5\n\t"
        "flh            ft1, 2(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 7b\n\t"

        // m4n8k1
        "8:\n\t"
        "beqz           t4, 9f\n\t"  // if k1 == 0, jump to end kernel_m4n8

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft0, v2\n\t"
        "vfmacc.vf      v10, ft1, v1\n\t"
        "vfmacc.vf      v11, ft1, v2\n\t"

        "addi           %[input_ptr], %[input_ptr], 64\n\t"  // ********************

        // end kernel_m4n8
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -64\n\t"  // pb -= 8

        "vse16.v        v8, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v10, (a1)\n\t"
        "addi           a1, a1, 32\n\t"

        "vse16.v        v9, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v11, (a1)\n\t"
        "addi           a1, a1, 32\n\t"

        // m4n4
        "10:\n\t"
        "andi           t1, t2, 16\n\t"  // s1 = bool_n4
        "beqz           t1, 14f\n\t"     // if n4==0, jump to m4n_tail

        // init out_tmp = bias
        "vfmv.v.f       v8, fs0\n\t"
        "vfmv.v.f       v9, fs1\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"

        "beqz           t3, 12f\n\t"  // if k2 == 0, jump to m4n4k1
        "mv             t5, t3\n\t"   // t5 = k2

        // m4n4k2
        "11:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, ft0, v1\n\t"
        "flh            fa0, 4(s2)\n\t"
        "vfmacc.vf      v9, ft1, v1\n\t"
        "flh            fa1, 6(s2)\n\t"
        "addi           s2, s2, 8\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v9, fa1, v4\n\t"
        "flh            ft1, 2(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 11b\n\t"

        // m4n4k1
        "12:\n\t"
        "beqz           t4, 13f\n\t"  // if k1 == 0, jump to end kernel_m4n4

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft1, v1\n\t"

        "addi           %[input_ptr], %[input_ptr], 32\n\t"  // ********************

        // end kernel_m4n4
        "13:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -32\n\t"  // pb -= 4

        "vse16.v        v8, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v9, (a1)\n\t"
        "addi           a1, a1, 32\n\t"

        // m4n_tail
        "14:\n\t"
        "andi           t1, t2, 15\n\t"         // s1 = bool_n_tail
        "beqz           t1, 18f\n\t"            // if bool_n_tail==0, jump to ending
        "vsetvli        zero, t1, e16, m1\n\t"  // set vl = n_tail
        "slli           t6, t1, 1\n\t"          // t6 = 2 * n_tail
        // init out_tmp = bias
        "vfmv.v.f       v8, fs0\n\t"
        "vfmv.v.f       v9, fs1\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"

        "beqz           t3, 16f\n\t"  // if k2 == 0, jump to m4n_tailk1
        "mv             t5, t3\n\t"   // t5 = k2

        // m4n_tailk2
        "15:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vfmacc.vf      v8, ft0, v1\n\t"
        "flh            fa0, 4(s2)\n\t"
        "vfmacc.vf      v9, ft1, v1\n\t"
        "flh            fa1, 6(s2)\n\t"
        "addi           s2, s2, 8\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vfmacc.vf      v8, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v9, fa1, v4\n\t"
        "flh            ft1, 2(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 15b\n\t"

        // m4n_tailk1
        "16:\n\t"
        "beqz           t4, 17f\n\t"  // if k1 == 0, jump to end kernel_m4n4

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft1, v1\n\t"

        "add            %[input_ptr], %[input_ptr], t6\n\t"  // ********************

        // end kernel_m8n_tail
        "17:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "sub            %[input_ptr], %[input_ptr], t6\n\t"  // pb -= n_tail

        "vse16.v        v8, (a0)\n\t"
        "add            a0, a0, t6\n\t"
        "vse16.v        v9, (a1)\n\t"
        "add            a1, a1, t6\n\t"

        // ending
        "18:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)

        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v1", "v2", "v3", "v4", "v5", "v6", "v8", "v9", "v10", "v11", "v12", "v13",
        // We use these general-purpose registers.
        "a0", "a1", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s2", "fs0", "fs1", "fa0", "fa1",
        "ft0", "ft1");
}

static inline void kernel_m1n48_fp16_v256(__fp16 *dst, __fp16 *sa, __fp16 *sb, int m, int k, int n,
                                          int ldc, __fp16 *bias)
{
    asm volatile(
        "li             a0, 48\n\t"
        "divw           t1, %[n], a0\n\t"  // t1 = n12
        "remw           t2, %[n], a0\n\t"  // t2 = n % 12 (n_tail)
        "srai           t3, %[k], 1\n\t"   // t3 = k2
        "andi           t4, %[k], 1\n\t"   // t4 = k1

        // m4
        "1:\n\t"
        "li             a0, 16\n\t"
        "vsetvli        zero, a0, e16, m1\n\t"  // set vl = 4
        // load 8 bias_data for 8 out_channels
        "flh            fs0, 0(%[bias_ptr])\n\t"

        // init output addr
        "mv             a0, %[output_ptr]\n\t"
        "beqz           t1, 6f\n\t"  // if n12==0, jump to m4n8
        // m4n12
        "2:\n\t"
        // init out_tmp = bias
        "vfmv.v.f       v8, fs0\n\t"
        "vfmv.v.f       v9, fs0\n\t"
        "vfmv.v.f       v10, fs0\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v3, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"

        "beqz           t3, 4f\n\t"  // if k2 == 0, jump to m4n12k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m4n12k2
        "3:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v5, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v6, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft0, v2\n\t"
        "vfmacc.vf      v10, ft0, v3\n\t"
        "flh            fa0, 2(s2)\n\t"
        "addi           s2, s2, 4\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v3, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, fa0, v4\n\t"
        "vfmacc.vf      v9, fa0, v5\n\t"
        "vfmacc.vf      v10, fa0, v6\n\t"
        "flh            ft0, 0(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 3b\n\t"

        // m4n12k1
        "4:\n\t"
        "beqz           t4, 5f\n\t"  // if k1 == 0, jump to end kernel_m4n12

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft0, v2\n\t"
        "vfmacc.vf      v10, ft0, v3\n\t"

        "addi           %[input_ptr], %[input_ptr], 96\n\t"  // ********************

        // end kernel_m4n12
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -96\n\t"  // pb -= 24

        "vse16.v        v8, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v9, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v10, (a0)\n\t"
        "addi           a0, a0, 32\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m4n8
        "6:\n\t"
        "andi           t1, t2, 32\n\t"  // s1 = bool_n8
        "beqz           t1, 10f\n\t"     // if n8==0, jump to m4n4

        // init out_tmp = bias
        "vfmv.v.f       v8, fs0\n\t"
        "vfmv.v.f       v9, fs0\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"

        "beqz           t3, 8f\n\t"  // if k2 == 0, jump to m4n8k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m4n8k2
        "7:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v5, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft0, v2\n\t"
        "flh            fa0, 2(s2)\n\t"
        "addi           s2, s2, 4\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, fa0, v4\n\t"
        "vfmacc.vf      v9, fa0, v5\n\t"
        "flh            ft0, 0(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 7b\n\t"

        // m4n8k1
        "8:\n\t"
        "beqz           t4, 9f\n\t"  // if k1 == 0, jump to end kernel_m4n8

        "vfmacc.vf      v8, ft0, v1\n\t"
        "vfmacc.vf      v9, ft0, v2\n\t"

        "addi           %[input_ptr], %[input_ptr], 64\n\t"  // ********************

        // end kernel_m4n8
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -64\n\t"  // pb -= 8

        "vse16.v        v8, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v9, (a0)\n\t"
        "addi           a0, a0, 32\n\t"

        // m4n4
        "10:\n\t"
        "andi           t1, t2, 16\n\t"  // s1 = bool_n4
        "beqz           t1, 14f\n\t"     // if n4==0, jump to m4n_tail

        // init out_tmp = bias
        "vfmv.v.f       v8, fs0\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"

        "beqz           t3, 12f\n\t"  // if k2 == 0, jump to m4n4k1
        "mv             t5, t3\n\t"   // t5 = k2

        // m4n4k2
        "11:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, ft0, v1\n\t"
        "flh            fa0, 2(s2)\n\t"
        "addi           s2, s2, 4\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v8, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 11b\n\t"

        // m4n4k1
        "12:\n\t"
        "beqz           t4, 13f\n\t"  // if k1 == 0, jump to end kernel_m4n4

        "vfmacc.vf      v8, ft0, v1\n\t"

        "addi           %[input_ptr], %[input_ptr], 32\n\t"  // ********************

        // end kernel_m4n4
        "13:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -32\n\t"  // pb -= 4

        "vse16.v        v8, (a0)\n\t"
        "addi           a0, a0, 32\n\t"

        // m4n_tail
        "14:\n\t"
        "andi           t1, t2, 15\n\t"         // s1 = bool_n_tail
        "beqz           t1, 18f\n\t"            // if bool_n_tail==0, jump to ending
        "vsetvli        zero, t1, e16, m1\n\t"  // set vl = n_tail
        "slli           t6, t1, 1\n\t"          // t6 = 2 * n_tail
        // init out_tmp = bias
        "vfmv.v.f       v8, fs0\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"

        "beqz           t3, 16f\n\t"  // if k2 == 0, jump to m4n_tailk1
        "mv             t5, t3\n\t"   // t5 = k2

        // m4n_tailk2
        "15:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vfmacc.vf      v8, ft0, v1\n\t"
        "flh            fa0, 2(s2)\n\t"
        "addi           s2, s2, 4\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vfmacc.vf      v8, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 15b\n\t"

        // m4n_tailk1
        "16:\n\t"
        "beqz           t4, 17f\n\t"  // if k1 == 0, jump to end kernel_m4n4

        "vfmacc.vf      v8, ft0, v1\n\t"

        "add            %[input_ptr], %[input_ptr], t6\n\t"  // ********************

        // end kernel_m8n_tail
        "17:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "sub            %[input_ptr], %[input_ptr], t6\n\t"  // pb -= n_tail

        "vse16.v        v8, (a0)\n\t"
        "add            a0, a0, t6\n\t"

        // ending
        "18:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)

        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v1", "v2", "v3", "v4", "v5", "v6", "v8", "v9", "v10",
        // We use these general-purpose registers.
        "a0", "a1", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s2", "fs0", "fa0", "ft0");
}

/**************************************************************
 * dst - output:[m, n]
 * sa - kernel: [m, k]
 * sb - input:  [k, n]
 **************************************************************/
void shl_c908_gemm_8x48_fp16_v256(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, __fp16 *bias,
                                  int m, int k, int n, int ldc)
{
    __fp16 *kernel_ptr = (__fp16 *)sa;
    __fp16 *input_ptr = (__fp16 *)sb;
    __fp16 *output_ptr = dst;

    bool flag_bias = 1;  // default: conv2d layer include bias
    if (bias == NULL) {
        flag_bias = 0;
        bias = (__fp16 *)shl_mem_alloc(m * sizeof(__fp16));
    }
    __fp16 *bias_ptr = bias;

    int tail = m % 8;
    if (m > 8) {
        kernel_m8n48_fp16_v256(output_ptr, kernel_ptr, input_ptr, m, k, n, ldc, bias_ptr);
        output_ptr += (m - tail) * n;
        kernel_ptr += (m - tail) * k;
        bias_ptr += (m - tail);
    }
    if (tail & 4) {
        kernel_m4n48_fp16_v256(output_ptr, kernel_ptr, input_ptr, m, k, n, ldc, bias_ptr);
        output_ptr += 4 * n;
        kernel_ptr += 4 * k;
        bias_ptr += 4;
    }
    if (tail & 2) {
        kernel_m2n48_fp16_v256(output_ptr, kernel_ptr, input_ptr, m, k, n, ldc, bias_ptr);
        output_ptr += 2 * n;
        kernel_ptr += 2 * k;
        bias_ptr += 2;
    }
    if (tail & 1) {
        kernel_m1n48_fp16_v256(output_ptr, kernel_ptr, input_ptr, m, k, n, ldc, bias_ptr);
        output_ptr += 1 * n;
        kernel_ptr += 1 * k;
        bias_ptr += 1;
    }
    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
#endif

static inline void kernel_m8n32_fp16_v256(__fp16 *dst, __fp16 *sa, __fp16 *sb, int m, int k, int n,
                                          int ldc, __fp16 *bias)
{
    asm volatile(
        "srai           t1, %[n], 5\n\t"   // t1 = n32
        "andi           t2, %[n], 31\n\t"  // t2 = n & 31u (n_tail)
        "srai           t3, %[k], 1\n\t"   // t3 = k2
        "andi           t4, %[k], 1\n\t"   // t4 = k1

        "srai           t0, %[m], 3\n\t"  // t0 = m8
        "beqz           t0, 15f\n\t"

        // m8
        "1:\n\t"
        "li             s1, 16\n\t"
        "vsetvli        zero, s1, e16, m1\n\t"  // set vl = 16
        // load 8 bias_data for 8 out_channels
        "flh            fs0, 0(%[bias_ptr])\n\t"
        "flh            fs1, 2(%[bias_ptr])\n\t"
        "flh            fs2, 4(%[bias_ptr])\n\t"
        "flh            fs3, 6(%[bias_ptr])\n\t"
        "flh            fs4, 8(%[bias_ptr])\n\t"
        "flh            fs5, 10(%[bias_ptr])\n\t"
        "flh            fs6, 12(%[bias_ptr])\n\t"
        "flh            fs7, 14(%[bias_ptr])\n\t"

        "mv             s1, t1\n\t"  // s1 = n32

        // init output addr
        "slli           t5, %[ldc], 1\n\t"  // t5_tmp = ldc * 2
        "mv             a0, %[output_ptr]\n\t"
        "add            a1, a0, t5\n\t"
        "add            a2, a1, t5\n\t"
        "add            a3, a2, t5\n\t"
        "add            a4, a3, t5\n\t"
        "add            a5, a4, t5\n\t"
        "add            a6, a5, t5\n\t"
        "add            a7, a6, t5\n\t"  // ******* 移到m8外面

        "mv             s3, %[input_ptr]\n\t"  // s3 hold input data start addr

        "beqz           t1, 6f\n\t"  // if n32==0, jump to m8n16
        // m8n32
        "2:\n\t"
        // init out_tmp = bias
        "vfmv.v.f       v16, fs0\n\t"
        "vfmv.v.f       v17, fs0\n\t"
        "vfmv.v.f       v18, fs1\n\t"
        "vfmv.v.f       v19, fs1\n\t"
        "vfmv.v.f       v20, fs2\n\t"
        "vfmv.v.f       v21, fs2\n\t"
        "vfmv.v.f       v22, fs3\n\t"
        "vfmv.v.f       v23, fs3\n\t"
        "vfmv.v.f       v24, fs4\n\t"
        "vfmv.v.f       v25, fs4\n\t"
        "vfmv.v.f       v26, fs5\n\t"
        "vfmv.v.f       v27, fs5\n\t"
        "vfmv.v.f       v28, fs6\n\t"
        "vfmv.v.f       v29, fs6\n\t"
        "vfmv.v.f       v30, fs7\n\t"
        "vfmv.v.f       v31, fs7\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (s3)\n\t"
        "addi           s3, s3, 32\n\t"
        "vle16.v        v2, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"
        "flh            ft2, 4(s2)\n\t"
        "flh            ft3, 6(s2)\n\t"
        "flh            ft4, 8(s2)\n\t"
        "flh            ft5, 10(s2)\n\t"
        "flh            ft6, 12(s2)\n\t"
        "flh            ft7, 14(s2)\n\t"

        "beqz           t3, 4f\n\t"  // if k2 == 0, jump to m8n32k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m8n32k2
        "3:\n\t"
        "vle16.v        v4, (s3)\n\t"
        "addi           s3, s3, 32\n\t"
        "vle16.v        v5, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        "vfmacc.vf      v16, ft0, v1\n\t"
        "vfmacc.vf      v17, ft0, v2\n\t"
        "flh            fa0, 16(s2)\n\t"
        "vfmacc.vf      v18, ft1, v1\n\t"
        "vfmacc.vf      v19, ft1, v2\n\t"
        "flh            fa1, 18(s2)\n\t"
        "vfmacc.vf      v20, ft2, v1\n\t"
        "vfmacc.vf      v21, ft2, v2\n\t"
        "flh            fa2, 20(s2)\n\t"
        "vfmacc.vf      v22, ft3, v1\n\t"
        "vfmacc.vf      v23, ft3, v2\n\t"
        "flh            fa3, 22(s2)\n\t"
        "vfmacc.vf      v24, ft4, v1\n\t"
        "vfmacc.vf      v25, ft4, v2\n\t"
        "flh            fa4, 24(s2)\n\t"
        "vfmacc.vf      v26, ft5, v1\n\t"
        "vfmacc.vf      v27, ft5, v2\n\t"
        "flh            fa5, 26(s2)\n\t"
        "vfmacc.vf      v28, ft6, v1\n\t"
        "vfmacc.vf      v29, ft6, v2\n\t"
        "flh            fa6, 28(s2)\n\t"
        "vfmacc.vf      v30, ft7, v1\n\t"
        "vfmacc.vf      v31, ft7, v2\n\t"
        "flh            fa7, 30(s2)\n\t"  // 0
        "addi           s2, s2, 32\n\t"   // += 16 elements, bump kernel to next k2 addr

        "vle16.v        v1, (s3)\n\t"
        "addi           s3, s3, 32\n\t"
        "vle16.v        v2, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        "vfmacc.vf      v16, fa0, v4\n\t"
        "vfmacc.vf      v17, fa0, v5\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v18, fa1, v4\n\t"
        "vfmacc.vf      v19, fa1, v5\n\t"
        "flh            ft1, 2(s2)\n\t"
        "vfmacc.vf      v20, fa2, v4\n\t"
        "vfmacc.vf      v21, fa2, v5\n\t"
        "flh            ft2, 4(s2)\n\t"
        "vfmacc.vf      v22, fa3, v4\n\t"
        "vfmacc.vf      v23, fa3, v5\n\t"
        "flh            ft3, 6(s2)\n\t"
        "vfmacc.vf      v24, fa4, v4\n\t"
        "vfmacc.vf      v25, fa4, v5\n\t"
        "flh            ft4, 8(s2)\n\t"
        "vfmacc.vf      v26, fa5, v4\n\t"
        "vfmacc.vf      v27, fa5, v5\n\t"
        "flh            ft5, 10(s2)\n\t"
        "vfmacc.vf      v28, fa6, v4\n\t"
        "vfmacc.vf      v29, fa6, v5\n\t"
        "flh            ft6, 12(s2)\n\t"
        "vfmacc.vf      v30, fa7, v4\n\t"
        "vfmacc.vf      v31, fa7, v5\n\t"
        "flh            ft7, 14(s2)\n\t"  // 1

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 3b\n\t"

        // m8n32k1
        "4:\n\t"
        "beqz           t4, 5f\n\t"  // if k1 == 0, jump to end kernel_m8n16

        "vfmacc.vf      v16, ft0, v1\n\t"
        "vfmacc.vf      v17, ft0, v2\n\t"
        "vfmacc.vf      v18, ft1, v1\n\t"
        "vfmacc.vf      v19, ft1, v2\n\t"
        "vfmacc.vf      v20, ft2, v1\n\t"
        "vfmacc.vf      v21, ft2, v2\n\t"
        "vfmacc.vf      v22, ft3, v1\n\t"
        "vfmacc.vf      v23, ft3, v2\n\t"
        "vfmacc.vf      v24, ft4, v1\n\t"
        "vfmacc.vf      v25, ft4, v2\n\t"
        "vfmacc.vf      v26, ft5, v1\n\t"
        "vfmacc.vf      v27, ft5, v2\n\t"
        "vfmacc.vf      v28, ft6, v1\n\t"
        "vfmacc.vf      v29, ft6, v2\n\t"
        "vfmacc.vf      v30, ft7, v1\n\t"
        "vfmacc.vf      v31, ft7, v2\n\t"

        "addi           s3, s3, 64\n\t"  // ********************

        // end kernel_m8n32
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           s3, s3, -64\n\t"  // pb -= 32

        "vse16.v        v16, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v18, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v20, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v22, (a3)\n\t"
        "addi           a3, a3, 32\n\t"
        "vse16.v        v24, (a4)\n\t"
        "addi           a4, a4, 32\n\t"
        "vse16.v        v26, (a5)\n\t"
        "addi           a5, a5, 32\n\t"
        "vse16.v        v28, (a6)\n\t"
        "addi           a6, a6, 32\n\t"
        "vse16.v        v30, (a7)\n\t"
        "addi           a7, a7, 32\n\t"

        "vse16.v        v17, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v19, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v21, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v23, (a3)\n\t"
        "addi           a3, a3, 32\n\t"
        "vse16.v        v25, (a4)\n\t"
        "addi           a4, a4, 32\n\t"
        "vse16.v        v27, (a5)\n\t"
        "addi           a5, a5, 32\n\t"
        "vse16.v        v29, (a6)\n\t"
        "addi           a6, a6, 32\n\t"
        "vse16.v        v31, (a7)\n\t"
        "addi           a7, a7, 32\n\t"

        "addi           s1, s1, -1\n\t"
        "bnez           s1, 2b\n\t"

        // m8n16
        "6:\n\t"
        "andi           s1, t2, 16\n\t"  // s1 = n16
        "beqz           s1, 10f\n\t"     // if n8==0, jump to m8n_tail

        // init out_tmp = bias
        "vfmv.v.f       v24, fs0\n\t"
        "vfmv.v.f       v25, fs1\n\t"
        "vfmv.v.f       v26, fs2\n\t"
        "vfmv.v.f       v27, fs3\n\t"
        "vfmv.v.f       v28, fs4\n\t"
        "vfmv.v.f       v29, fs5\n\t"
        "vfmv.v.f       v30, fs6\n\t"
        "vfmv.v.f       v31, fs7\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"
        "flh            ft2, 4(s2)\n\t"
        "flh            ft3, 6(s2)\n\t"
        "flh            ft4, 8(s2)\n\t"
        "flh            ft5, 10(s2)\n\t"
        "flh            ft6, 12(s2)\n\t"
        "flh            ft7, 14(s2)\n\t"

        "beqz           t3, 8f\n\t"  // if k2 == 0, jump to m8n8k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m8n16k2
        "7:\n\t"
        "vle16.v        v4, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa0, 16(s2)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "flh            fa1, 18(s2)\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "flh            fa2, 20(s2)\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"
        "flh            fa3, 22(s2)\n\t"
        "vfmacc.vf      v28, ft4, v1\n\t"
        "flh            fa4, 24(s2)\n\t"
        "vfmacc.vf      v29, ft5, v1\n\t"
        "flh            fa5, 26(s2)\n\t"
        "vfmacc.vf      v30, ft6, v1\n\t"
        "flh            fa6, 28(s2)\n\t"
        "vfmacc.vf      v31, ft7, v1\n\t"
        "flh            fa7, 30(s2)\n\t"  // 0
        "addi           s2, s2, 32\n\t"   // += 16 elements, bump kernel to next k2 addr

        "vle16.v        v1, (s3)\n\t"
        "addi           s3, s3, 32\n\t"

        "vfmacc.vf      v24, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v25, fa1, v4\n\t"
        "flh            ft1, 2(s2)\n\t"
        "vfmacc.vf      v26, fa2, v4\n\t"
        "flh            ft2, 4(s2)\n\t"
        "vfmacc.vf      v27, fa3, v4\n\t"
        "flh            ft3, 6(s2)\n\t"
        "vfmacc.vf      v28, fa4, v4\n\t"
        "flh            ft4, 8(s2)\n\t"
        "vfmacc.vf      v29, fa5, v4\n\t"
        "flh            ft5, 10(s2)\n\t"
        "vfmacc.vf      v30, fa6, v4\n\t"
        "flh            ft6, 12(s2)\n\t"
        "vfmacc.vf      v31, fa7, v4\n\t"
        "flh            ft7, 14(s2)\n\t"  // 1

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 7b\n\t"

        // m8n16k1
        "8:\n\t"
        "beqz           t4, 9f\n\t"  // if k1 == 0, jump to end kernel_m8n8

        "vfmacc.vf      v24, ft0, v1\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"
        "vfmacc.vf      v28, ft4, v1\n\t"
        "vfmacc.vf      v29, ft5, v1\n\t"
        "vfmacc.vf      v30, ft6, v1\n\t"
        "vfmacc.vf      v31, ft7, v1\n\t"

        "addi           s3, s3, 32\n\t"  // ********************

        // end kernel_m8n16
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           s3, s3, -32\n\t"  // pb -= 16

        "vse16.v        v24, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v25, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v26, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v27, (a3)\n\t"
        "addi           a3, a3, 32\n\t"
        "vse16.v        v28, (a4)\n\t"
        "addi           a4, a4, 32\n\t"
        "vse16.v        v29, (a5)\n\t"
        "addi           a5, a5, 32\n\t"
        "vse16.v        v30, (a6)\n\t"
        "addi           a6, a6, 32\n\t"
        "vse16.v        v31, (a7)\n\t"
        "addi           a7, a7, 32\n\t"

        // m8n_tail
        "10:\n\t"
        "andi           s1, t2, 15\n\t"         // s1 = bool_n_tail
        "beqz           s1, 14f\n\t"            // if n_tail==0, jump to end m8
        "vsetvli        zero, s1, e16, m1\n\t"  // set vl = n_tail
        "slli           t6, s1, 1\n\t"          // t6 = 2 * n_tail
        // init out_tmp = bias
        "vfmv.v.f       v24, fs0\n\t"
        "vfmv.v.f       v25, fs1\n\t"
        "vfmv.v.f       v26, fs2\n\t"
        "vfmv.v.f       v27, fs3\n\t"
        "vfmv.v.f       v28, fs4\n\t"
        "vfmv.v.f       v29, fs5\n\t"
        "vfmv.v.f       v30, fs6\n\t"
        "vfmv.v.f       v31, fs7\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (s3)\n\t"
        "add            s3, s3, t6\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"
        "flh            ft2, 4(s2)\n\t"
        "flh            ft3, 6(s2)\n\t"
        "flh            ft4, 8(s2)\n\t"
        "flh            ft5, 10(s2)\n\t"
        "flh            ft6, 12(s2)\n\t"
        "flh            ft7, 14(s2)\n\t"

        "beqz           t3, 12f\n\t"  // if k2 == 0, jump to m8n_tailk1
        "mv             t5, t3\n\t"   // t5 = k2

        // m8n_tailk2
        "11:\n\t"
        "vle16.v        v4, (s3)\n\t"
        "add            s3, s3, t6\n\t"

        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa0, 16(s2)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "flh            fa1, 18(s2)\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "flh            fa2, 20(s2)\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"
        "flh            fa3, 22(s2)\n\t"
        "vfmacc.vf      v28, ft4, v1\n\t"
        "flh            fa4, 24(s2)\n\t"
        "vfmacc.vf      v29, ft5, v1\n\t"
        "flh            fa5, 26(s2)\n\t"
        "vfmacc.vf      v30, ft6, v1\n\t"
        "flh            fa6, 28(s2)\n\t"
        "vfmacc.vf      v31, ft7, v1\n\t"
        "flh            fa7, 30(s2)\n\t"  // 0
        "addi           s2, s2, 32\n\t"   // += 16 elements, bump kernel to next k2 addr

        "vle16.v        v1, (s3)\n\t"
        "add            s3, s3, t6\n\t"

        "vfmacc.vf      v24, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v25, fa1, v4\n\t"
        "flh            ft1, 2(s2)\n\t"
        "vfmacc.vf      v26, fa2, v4\n\t"
        "flh            ft2, 4(s2)\n\t"
        "vfmacc.vf      v27, fa3, v4\n\t"
        "flh            ft3, 6(s2)\n\t"
        "vfmacc.vf      v28, fa4, v4\n\t"
        "flh            ft4, 8(s2)\n\t"
        "vfmacc.vf      v29, fa5, v4\n\t"
        "flh            ft5, 10(s2)\n\t"
        "vfmacc.vf      v30, fa6, v4\n\t"
        "flh            ft6, 12(s2)\n\t"
        "vfmacc.vf      v31, fa7, v4\n\t"
        "flh            ft7, 14(s2)\n\t"  // 1

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 11b\n\t"

        // m8n_tailk1
        "12:\n\t"
        "beqz           t4, 13f\n\t"  // if k1 == 0, jump to end kernel_m8n4

        "vfmacc.vf      v24, ft0, v1\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"
        "vfmacc.vf      v28, ft4, v1\n\t"
        "vfmacc.vf      v29, ft5, v1\n\t"
        "vfmacc.vf      v30, ft6, v1\n\t"
        "vfmacc.vf      v31, ft7, v1\n\t"

        "add            s3, s3, t6\n\t"  // ********************

        // end kernel_m8n_tail
        "13:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "sub            s3, s3, t6\n\t"  // pb -= n_tail

        "vse16.v        v24, (a0)\n\t"
        "add            a0, a0, t6\n\t"
        "vse16.v        v25, (a1)\n\t"
        "add            a1, a1, t6\n\t"
        "vse16.v        v26, (a2)\n\t"
        "add            a2, a2, t6\n\t"
        "vse16.v        v27, (a3)\n\t"
        "add            a3, a3, t6\n\t"
        "vse16.v        v28, (a4)\n\t"
        "add            a4, a4, t6\n\t"
        "vse16.v        v29, (a5)\n\t"
        "add            a5, a5, t6\n\t"
        "vse16.v        v30, (a6)\n\t"
        "add            a6, a6, t6\n\t"
        "vse16.v        v31, (a7)\n\t"
        "add            a7, a7, t6\n\t"

        // end kernel_m8
        "14:\n\t"
        "addi           %[bias_ptr], %[bias_ptr], 16\n\t"  // bias_data += 8
        "slli           t6, %[k], 4\n\t"
        "add            %[kernel_ptr], %[kernel_ptr], t6\n\t"  // kernel_data += 8 * k
        "slli           t6, %[ldc], 4\n\t"
        "add            %[output_ptr], %[output_ptr], t6\n\t"  // output_data += 8 * ldc

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        // ending
        "15:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)
        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v1", "v2", "v4", "v5", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
        "v25", "v26", "v27", "v28", "v29", "v30", "v31",
        // We use these general-purpose registers.
        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6",
        "s1", "s2", "s3", "fs0", "fs1", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fa0", "fa1",
        "fa2", "fa3", "fa4", "fa5", "fa6", "fa7", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6",
        "ft7");
}

static inline void kernel_m4n32_fp16_v256(__fp16 *dst, __fp16 *sa, __fp16 *sb, int m, int k, int n,
                                          int ldc, __fp16 *bias)
{
    asm volatile(
        "srai           t1, %[n], 5\n\t"   // t1 = n8
        "andi           t2, %[n], 31\n\t"  // t2 = n & 7u (n_tail)
        "srai           t3, %[k], 1\n\t"   // t3 = k2
        "andi           t4, %[k], 1\n\t"   // t4 = k1

        // m4
        "1:\n\t"
        "li             a0, 16\n\t"
        "vsetvli        zero, a0, e16, m1\n\t"  // set vl = 8
        // load 4 bias_data for 4 out_channels
        "flh            fs0, 0(%[bias_ptr])\n\t"
        "flh            fs1, 2(%[bias_ptr])\n\t"
        "flh            fs2, 4(%[bias_ptr])\n\t"
        "flh            fs3, 6(%[bias_ptr])\n\t"

        // init output addr
        "slli           t5, %[ldc], 1\n\t"  // t5_tmp = ldc * 2
        "mv             a0, %[output_ptr]\n\t"
        "add            a1, a0, t5\n\t"
        "add            a2, a1, t5\n\t"
        "add            a3, a2, t5\n\t"

        "beqz           t1, 6f\n\t"  // if n8==0, jump to m4n4
        // m4n8
        "2:\n\t"
        // init out_tmp = bias
        "vfmv.v.f       v16, fs0\n\t"
        "vfmv.v.f       v17, fs0\n\t"
        "vfmv.v.f       v18, fs1\n\t"
        "vfmv.v.f       v19, fs1\n\t"
        "vfmv.v.f       v20, fs2\n\t"
        "vfmv.v.f       v21, fs2\n\t"
        "vfmv.v.f       v22, fs3\n\t"
        "vfmv.v.f       v23, fs3\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"
        "flh            ft2, 4(s2)\n\t"
        "flh            ft3, 6(s2)\n\t"

        "beqz           t3, 4f\n\t"  // if k2 == 0, jump to m4n8k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m4n8k2
        "3:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v5, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v16, ft0, v1\n\t"
        "vfmacc.vf      v17, ft0, v2\n\t"
        "flh            fa0, 8(s2)\n\t"
        "vfmacc.vf      v18, ft1, v1\n\t"
        "vfmacc.vf      v19, ft1, v2\n\t"
        "flh            fa1, 10(s2)\n\t"
        "vfmacc.vf      v20, ft2, v1\n\t"
        "vfmacc.vf      v21, ft2, v2\n\t"
        "flh            fa2, 12(s2)\n\t"
        "vfmacc.vf      v22, ft3, v1\n\t"
        "vfmacc.vf      v23, ft3, v2\n\t"
        "flh            fa3, 14(s2)\n\t"
        "addi           s2, s2, 16\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v16, fa0, v4\n\t"
        "vfmacc.vf      v17, fa0, v5\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v18, fa1, v4\n\t"
        "vfmacc.vf      v19, fa1, v5\n\t"
        "flh            ft1, 2(s2)\n\t"
        "vfmacc.vf      v20, fa2, v4\n\t"
        "vfmacc.vf      v21, fa2, v5\n\t"
        "flh            ft2, 4(s2)\n\t"
        "vfmacc.vf      v22, fa3, v4\n\t"
        "vfmacc.vf      v23, fa3, v5\n\t"
        "flh            ft3, 6(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 3b\n\t"

        // m4n8k1
        "4:\n\t"
        "beqz           t4, 5f\n\t"  // if k1 == 0, jump to end kernel_m4n8

        "vfmacc.vf      v16, ft0, v1\n\t"
        "vfmacc.vf      v17, ft0, v2\n\t"
        "vfmacc.vf      v18, ft1, v1\n\t"
        "vfmacc.vf      v19, ft1, v2\n\t"
        "vfmacc.vf      v20, ft2, v1\n\t"
        "vfmacc.vf      v21, ft2, v2\n\t"
        "vfmacc.vf      v22, ft3, v1\n\t"
        "vfmacc.vf      v23, ft3, v2\n\t"

        "addi           %[input_ptr], %[input_ptr], 64\n\t"  // ********************

        // end kernel_m4n8
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -64\n\t"  // pb -= 8

        "vse16.v        v16, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v18, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v20, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v22, (a3)\n\t"
        "addi           a3, a3, 32\n\t"

        "vse16.v        v17, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v19, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v21, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v23, (a3)\n\t"
        "addi           a3, a3, 32\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m4n4
        "6:\n\t"
        "andi           t1, t2, 16\n\t"  // s1 = n4
        "beqz           t1, 10f\n\t"     // if n4==0, jump to m4n_tail

        // init out_tmp = bias
        "vfmv.v.f       v16, fs0\n\t"
        "vfmv.v.f       v17, fs1\n\t"
        "vfmv.v.f       v18, fs2\n\t"
        "vfmv.v.f       v19, fs3\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"
        "flh            ft2, 4(s2)\n\t"
        "flh            ft3, 6(s2)\n\t"

        "beqz           t3, 8f\n\t"  // if k2 == 0, jump to m4n4k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m4n4k2
        "7:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v16, ft0, v1\n\t"
        "flh            fa0, 8(s2)\n\t"
        "vfmacc.vf      v17, ft1, v1\n\t"
        "flh            fa1, 10(s2)\n\t"
        "vfmacc.vf      v18, ft2, v1\n\t"
        "flh            fa2, 12(s2)\n\t"
        "vfmacc.vf      v19, ft3, v1\n\t"
        "flh            fa3, 14(s2)\n\t"
        "addi           s2, s2, 16\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v16, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v17, fa1, v4\n\t"
        "flh            ft1, 2(s2)\n\t"
        "vfmacc.vf      v18, fa2, v4\n\t"
        "flh            ft2, 4(s2)\n\t"
        "vfmacc.vf      v19, fa3, v4\n\t"
        "flh            ft3, 6(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 7b\n\t"

        // m4n4k1
        "8:\n\t"
        "beqz           t4, 9f\n\t"  // if k1 == 0, jump to end kernel_m4n4

        "vfmacc.vf      v16, ft0, v1\n\t"
        "vfmacc.vf      v17, ft1, v1\n\t"
        "vfmacc.vf      v18, ft2, v1\n\t"
        "vfmacc.vf      v19, ft3, v1\n\t"

        "addi           %[input_ptr], %[input_ptr], 32\n\t"  // ********************

        // end kernel_m4n4
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -32\n\t"  // pb -= 4

        "vse16.v        v16, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v17, (a1)\n\t"
        "addi           a1, a1, 32\n\t"
        "vse16.v        v18, (a2)\n\t"
        "addi           a2, a2, 32\n\t"
        "vse16.v        v19, (a3)\n\t"
        "addi           a3, a3, 32\n\t"

        // m4n_tail
        "10:\n\t"
        "andi           t1, t2, 15\n\t"         // s1 = bool_n_tail
        "beqz           t1, 14f\n\t"            // if n4==0, jump to m4n_tail
        "vsetvli        zero, t1, e16, m1\n\t"  // set vl = n_tail
        "slli           t6, t1, 1\n\t"          // t6 = 4 * n_tail
        // init out_tmp = bias
        "vfmv.v.f       v16, fs0\n\t"
        "vfmv.v.f       v17, fs1\n\t"
        "vfmv.v.f       v18, fs2\n\t"
        "vfmv.v.f       v19, fs3\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"
        "flh            ft2, 4(s2)\n\t"
        "flh            ft3, 6(s2)\n\t"

        "beqz           t3, 12f\n\t"  // if k2 == 0, jump to m4n_tailk1
        "mv             t5, t3\n\t"   // t5 = k2

        // m4n_tailk2
        "11:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vfmacc.vf      v16, ft0, v1\n\t"
        "flh            fa0, 8(s2)\n\t"
        "vfmacc.vf      v17, ft1, v1\n\t"
        "flh            fa1, 10(s2)\n\t"
        "vfmacc.vf      v18, ft2, v1\n\t"
        "flh            fa2, 12(s2)\n\t"
        "vfmacc.vf      v19, ft3, v1\n\t"
        "flh            fa3, 14(s2)\n\t"
        "addi           s2, s2, 16\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vfmacc.vf      v16, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v17, fa1, v4\n\t"
        "flh            ft1, 2(s2)\n\t"
        "vfmacc.vf      v18, fa2, v4\n\t"
        "flh            ft2, 4(s2)\n\t"
        "vfmacc.vf      v19, fa3, v4\n\t"
        "flh            ft3, 6(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 11b\n\t"

        // m4n_tailk1
        "12:\n\t"
        "beqz           t4, 13f\n\t"  // if k1 == 0, jump to end kernel_m4n4

        "vfmacc.vf      v16, ft0, v1\n\t"
        "vfmacc.vf      v17, ft1, v1\n\t"
        "vfmacc.vf      v18, ft2, v1\n\t"
        "vfmacc.vf      v19, ft3, v1\n\t"

        "add            %[input_ptr], %[input_ptr], t6\n\t"  // ********************

        // end kernel_m4n_tail
        "13:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "sub            %[input_ptr], %[input_ptr], t6\n\t"  // pb -= n_tail

        "vse16.v        v16, (a0)\n\t"
        "add            a0, a0, t6\n\t"
        "vse16.v        v17, (a1)\n\t"
        "add            a1, a1, t6\n\t"
        "vse16.v        v18, (a2)\n\t"
        "add            a2, a2, t6\n\t"
        "vse16.v        v19, (a3)\n\t"
        "add            a3, a3, t6\n\t"

        // end kernel_m4
        "14:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)
        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v1", "v2", "v4", "v5", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        // We use these general-purpose registers.
        "a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s2", "fs0", "fs1", "fs2",
        "fs3", "fa0", "fa1", "fa2", "fa3", "ft0", "ft1", "ft2", "ft3");
}

static inline void kernel_m2n32_fp16_v256(__fp16 *dst, __fp16 *sa, __fp16 *sb, int m, int k, int n,
                                          int ldc, __fp16 *bias)
{
    asm volatile(
        "srai           t1, %[n], 5\n\t"   // t1 = n8
        "andi           t2, %[n], 31\n\t"  // t2 = n & 7u (n_tail)
        "srai           t3, %[k], 1\n\t"   // t3 = k2
        "andi           t4, %[k], 1\n\t"   // t4 = k1

        // m4
        "1:\n\t"
        "li             a0, 16\n\t"
        "vsetvli        zero, a0, e16, m1\n\t"  // set vl = 8
        // load 4 bias_data for 4 out_channels
        "flh            fs0, 0(%[bias_ptr])\n\t"
        "flh            fs1, 2(%[bias_ptr])\n\t"

        // init output addr
        "slli           t5, %[ldc], 1\n\t"  // t5_tmp = ldc * 2
        "mv             a0, %[output_ptr]\n\t"
        "add            a1, a0, t5\n\t"

        "beqz           t1, 6f\n\t"  // if n8==0, jump to m4n4
        // m4n8
        "2:\n\t"
        // init out_tmp = bias
        "vfmv.v.f       v16, fs0\n\t"
        "vfmv.v.f       v17, fs0\n\t"
        "vfmv.v.f       v18, fs1\n\t"
        "vfmv.v.f       v19, fs1\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"

        "beqz           t3, 4f\n\t"  // if k2 == 0, jump to m4n8k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m4n8k2
        "3:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v5, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v16, ft0, v1\n\t"
        "vfmacc.vf      v17, ft0, v2\n\t"
        "flh            fa0, 4(s2)\n\t"
        "vfmacc.vf      v18, ft1, v1\n\t"
        "vfmacc.vf      v19, ft1, v2\n\t"
        "flh            fa1, 6(s2)\n\t"
        "addi           s2, s2, 8\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v16, fa0, v4\n\t"
        "vfmacc.vf      v17, fa0, v5\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v18, fa1, v4\n\t"
        "vfmacc.vf      v19, fa1, v5\n\t"
        "flh            ft1, 2(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 3b\n\t"

        // m4n8k1
        "4:\n\t"
        "beqz           t4, 5f\n\t"  // if k1 == 0, jump to end kernel_m4n8

        "vfmacc.vf      v16, ft0, v1\n\t"
        "vfmacc.vf      v17, ft0, v2\n\t"
        "vfmacc.vf      v18, ft1, v1\n\t"
        "vfmacc.vf      v19, ft1, v2\n\t"

        "addi           %[input_ptr], %[input_ptr], 64\n\t"  // ********************

        // end kernel_m4n8
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -64\n\t"  // pb -= 8

        "vse16.v        v16, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v18, (a1)\n\t"
        "addi           a1, a1, 32\n\t"

        "vse16.v        v17, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v19, (a1)\n\t"
        "addi           a1, a1, 32\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m4n4
        "6:\n\t"
        "andi           t1, t2, 16\n\t"  // s1 = n4
        "beqz           t1, 10f\n\t"     // if n4==0, jump to m4n_tail

        // init out_tmp = bias
        "vfmv.v.f       v16, fs0\n\t"
        "vfmv.v.f       v17, fs1\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"

        "beqz           t3, 8f\n\t"  // if k2 == 0, jump to m4n4k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m4n4k2
        "7:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v16, ft0, v1\n\t"
        "flh            fa0, 4(s2)\n\t"
        "vfmacc.vf      v17, ft1, v1\n\t"
        "flh            fa1, 6(s2)\n\t"
        "addi           s2, s2, 8\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v16, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v17, fa1, v4\n\t"
        "flh            ft1, 2(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 7b\n\t"

        // m4n4k1
        "8:\n\t"
        "beqz           t4, 9f\n\t"  // if k1 == 0, jump to end kernel_m4n4

        "vfmacc.vf      v16, ft0, v1\n\t"
        "vfmacc.vf      v17, ft1, v1\n\t"

        "addi           %[input_ptr], %[input_ptr], 32\n\t"  // ********************

        // end kernel_m4n4
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -32\n\t"  // pb -= 4

        "vse16.v        v16, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v17, (a1)\n\t"
        "addi           a1, a1, 32\n\t"

        // m4n_tail
        "10:\n\t"
        "andi           t1, t2, 15\n\t"         // s1 = bool_n_tail
        "beqz           t1, 14f\n\t"            // if n4==0, jump to m4n_tail
        "vsetvli        zero, t1, e16, m1\n\t"  // set vl = n_tail
        "slli           t6, t1, 1\n\t"          // t6 = 4 * n_tail
        // init out_tmp = bias
        "vfmv.v.f       v16, fs0\n\t"
        "vfmv.v.f       v17, fs1\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"
        "flh            ft1, 2(s2)\n\t"

        "beqz           t3, 12f\n\t"  // if k2 == 0, jump to m4n_tailk1
        "mv             t5, t3\n\t"   // t5 = k2

        // m4n_tailk2
        "11:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vfmacc.vf      v16, ft0, v1\n\t"
        "flh            fa0, 4(s2)\n\t"
        "vfmacc.vf      v17, ft1, v1\n\t"
        "flh            fa1, 6(s2)\n\t"
        "addi           s2, s2, 8\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vfmacc.vf      v16, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"
        "vfmacc.vf      v17, fa1, v4\n\t"
        "flh            ft1, 2(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 11b\n\t"

        // m4n_tailk1
        "12:\n\t"
        "beqz           t4, 13f\n\t"  // if k1 == 0, jump to end kernel_m4n4

        "vfmacc.vf      v16, ft0, v1\n\t"
        "vfmacc.vf      v17, ft1, v1\n\t"

        "add            %[input_ptr], %[input_ptr], t6\n\t"  // ********************

        // end kernel_m4n_tail
        "13:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "sub            %[input_ptr], %[input_ptr], t6\n\t"  // pb -= n_tail

        "vse16.v        v16, (a0)\n\t"
        "add            a0, a0, t6\n\t"
        "vse16.v        v17, (a1)\n\t"
        "add            a1, a1, t6\n\t"

        // end kernel_m4
        "14:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)
        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v1", "v2", "v4", "v5", "v16", "v17", "v18", "v19",
        // We use these general-purpose registers.
        "a0", "a1", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s2", "fs0", "fs1", "fa0", "fa1",
        "ft0", "ft1");
}

static inline void kernel_m1n32_fp16_v256(__fp16 *dst, __fp16 *sa, __fp16 *sb, int m, int k, int n,
                                          int ldc, __fp16 *bias)
{
    asm volatile(
        "srai           t1, %[n], 5\n\t"   // t1 = n8
        "andi           t2, %[n], 31\n\t"  // t2 = n & 7u (n_tail)
        "srai           t3, %[k], 1\n\t"   // t3 = k2
        "andi           t4, %[k], 1\n\t"   // t4 = k1

        // m4
        "1:\n\t"
        "li             a0, 16\n\t"
        "vsetvli        zero, a0, e16, m1\n\t"  // set vl = 8
        // load 4 bias_data for 4 out_channels
        "flh            fs0, 0(%[bias_ptr])\n\t"

        // init output addr
        "mv             a0, %[output_ptr]\n\t"

        "beqz           t1, 6f\n\t"  // if n8==0, jump to m4n4
        // m4n8
        "2:\n\t"
        // init out_tmp = bias
        "vfmv.v.f       v16, fs0\n\t"
        "vfmv.v.f       v17, fs0\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"

        "beqz           t3, 4f\n\t"  // if k2 == 0, jump to m4n8k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m4n8k2
        "3:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v5, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v16, ft0, v1\n\t"
        "vfmacc.vf      v17, ft0, v2\n\t"
        "flh            fa0, 2(s2)\n\t"
        "addi           s2, s2, 4\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"
        "vle16.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v16, fa0, v4\n\t"
        "vfmacc.vf      v17, fa0, v5\n\t"
        "flh            ft0, 0(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 3b\n\t"

        // m4n8k1
        "4:\n\t"
        "beqz           t4, 5f\n\t"  // if k1 == 0, jump to end kernel_m4n8

        "vfmacc.vf      v16, ft0, v1\n\t"
        "vfmacc.vf      v17, ft0, v2\n\t"

        "addi           %[input_ptr], %[input_ptr], 64\n\t"  // ********************

        // end kernel_m4n8
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -64\n\t"  // pb -= 8

        "vse16.v        v16, (a0)\n\t"
        "addi           a0, a0, 32\n\t"
        "vse16.v        v17, (a0)\n\t"
        "addi           a0, a0, 32\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m4n4
        "6:\n\t"
        "andi           t1, t2, 16\n\t"  // s1 = n4
        "beqz           t1, 10f\n\t"     // if n4==0, jump to m4n_tail

        // init out_tmp = bias
        "vfmv.v.f       v16, fs0\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"

        "beqz           t3, 8f\n\t"  // if k2 == 0, jump to m4n4k1
        "mv             t5, t3\n\t"  // t5 = k2

        // m4n4k2
        "7:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v16, ft0, v1\n\t"
        "flh            fa0, 2(s2)\n\t"
        "addi           s2, s2, 4\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vfmacc.vf      v16, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 7b\n\t"

        // m4n4k1
        "8:\n\t"
        "beqz           t4, 9f\n\t"  // if k1 == 0, jump to end kernel_m4n4

        "vfmacc.vf      v16, ft0, v1\n\t"

        "addi           %[input_ptr], %[input_ptr], 32\n\t"  // ********************

        // end kernel_m4n4
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -32\n\t"  // pb -= 4

        "vse16.v        v16, (a0)\n\t"
        "addi           a0, a0, 32\n\t"

        // m4n_tail
        "10:\n\t"
        "andi           t1, t2, 15\n\t"         // s1 = bool_n_tail
        "beqz           t1, 14f\n\t"            // if n4==0, jump to m4n_tail
        "vsetvli        zero, t1, e16, m1\n\t"  // set vl = n_tail
        "slli           t6, t1, 1\n\t"          // t6 = 4 * n_tail
        // init out_tmp = bias
        "vfmv.v.f       v16, fs0\n\t"

        "mv             s2, %[kernel_ptr]\n\t"  // s2 hold kernel 4 lines start addr

        // pre-load pb (input_data)
        "vle16.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        // pre-load pa(kernel_data)
        "flh            ft0, 0(s2)\n\t"

        "beqz           t3, 12f\n\t"  // if k2 == 0, jump to m4n_tailk1
        "mv             t5, t3\n\t"   // t5 = k2

        // m4n_tailk2
        "11:\n\t"
        "vle16.v        v4, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vfmacc.vf      v16, ft0, v1\n\t"
        "flh            fa0, 2(s2)\n\t"
        "addi           s2, s2, 4\n\t"  // += 8 elements, bump kernel to next k2 addr

        "vle16.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vfmacc.vf      v16, fa0, v4\n\t"
        "flh            ft0, 0(s2)\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 11b\n\t"

        // m4n_tailk1
        "12:\n\t"
        "beqz           t4, 13f\n\t"  // if k1 == 0, jump to end kernel_m4n4

        "vfmacc.vf      v16, ft0, v1\n\t"

        "add            %[input_ptr], %[input_ptr], t6\n\t"  // ********************

        // end kernel_m4n_tail
        "13:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "sub            %[input_ptr], %[input_ptr], t6\n\t"  // pb -= n_tail

        "vse16.v        v16, (a0)\n\t"
        "add            a0, a0, t6\n\t"

        // end kernel_m4
        "14:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)
        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v1", "v2", "v4", "v5", "v16", "v17",
        // We use these general-purpose registers.
        "a0", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s2", "fs0", "fa0", "ft0");
}

/**************************************************************
 * dst - output:[m, n]
 * sa - kernel: [m, k]
 * sb - input:  [k, n]
 **************************************************************/
void shl_c908_gemm_8x32_fp16_v256(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, __fp16 *bias,
                                  int m, int k, int n, int ldc)
{
    __fp16 *kernel_ptr = (__fp16 *)sa;
    __fp16 *input_ptr = (__fp16 *)sb;
    __fp16 *output_ptr = dst;

    bool flag_bias = 1;  // default: conv2d layer include bias
    if (bias == NULL) {
        flag_bias = 0;
        bias = (__fp16 *)shl_mem_alloc(m * sizeof(__fp16));
    }
    __fp16 *bias_ptr = bias;

    int tail = m % 8;
    if (m > 8) {
        kernel_m8n32_fp16_v256(output_ptr, kernel_ptr, input_ptr, m, k, n, ldc, bias_ptr);
        output_ptr += (m - tail) * n;
        kernel_ptr += (m - tail) * k;
        bias_ptr += (m - tail);
    }
    if (tail & 4) {
        kernel_m4n32_fp16_v256(output_ptr, kernel_ptr, input_ptr, m, k, n, ldc, bias_ptr);
        output_ptr += 4 * n;
        kernel_ptr += 4 * k;
        bias_ptr += 4;
    }
    if (tail & 2) {
        kernel_m2n32_fp16_v256(output_ptr, kernel_ptr, input_ptr, m, k, n, ldc, bias_ptr);
        output_ptr += 2 * n;
        kernel_ptr += 2 * k;
        bias_ptr += 2;
    }
    if (tail & 1) {
        kernel_m1n32_fp16_v256(output_ptr, kernel_ptr, input_ptr, m, k, n, ldc, bias_ptr);
        output_ptr += 1 * n;
        kernel_ptr += 1 * k;
        bias_ptr += 1;
    }
    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
