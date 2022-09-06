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

/* CSI-NN2 version 2.0.x */

#include "shl_c908.h"

/*************************************************************
 * note: VLEN = 128
 * input matrix and kernel matrix have been reordered
 *************************************************************/

static inline void kernel_m8n8_int8(int8_t *dst, int8_t *sa, int8_t *sb, int m, int k, int n,
                                    int32_t *bias, int32_t out_zp, int32_t *mult, int32_t *shift)
{
    asm volatile(
        "srai           t0, %[m], 3\n\t"  // t0 = m8
        "beqz           t0, 15f\n\t"

        // m8
        "1:\n\t"
        "srai           t1, %[n], 3\n\t"        // t1 = n8
        "mv             t2, %[output_ptr]\n\t"  // init output addr
        "mv             t3, %[input_ptr]\n\t"   // t3 hold input data start addr

        "beqz           t1, 6f\n\t"  // if n8==0, jump to m8n4
        // m8n8
        "2:\n\t"
        "li             t6, 4\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        // init out_tmp = bias
        "lw             t4, 0(%[bias_ptr])\n\t"  // bias_ptr[0]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v17, t4\n\t"
        "lw             t4, 4(%[bias_ptr])\n\t"  // bias_ptr[1]
        "vmv.v.x        v18, t4\n\t"
        "vmv.v.x        v19, t4\n\t"
        "lw             t4, 8(%[bias_ptr])\n\t"  // bias_ptr[2]
        "vmv.v.x        v20, t4\n\t"
        "vmv.v.x        v21, t4\n\t"
        "lw             t4, 12(%[bias_ptr])\n\t"  // bias_ptr[3]
        "vmv.v.x        v22, t4\n\t"
        "vmv.v.x        v23, t4\n\t"
        "lw             t4, 16(%[bias_ptr])\n\t"  // bias_ptr[4]
        "vmv.v.x        v24, t4\n\t"
        "vmv.v.x        v25, t4\n\t"
        "lw             t4, 20(%[bias_ptr])\n\t"  // bias_ptr[5]
        "vmv.v.x        v26, t4\n\t"
        "vmv.v.x        v27, t4\n\t"
        "lw             t4, 24(%[bias_ptr])\n\t"  // bias_ptr[6]
        "vmv.v.x        v28, t4\n\t"
        "vmv.v.x        v29, t4\n\t"
        "lw             t4, 28(%[bias_ptr])\n\t"  // bias_ptr[7]
        "vmv.v.x        v30, t4\n\t"
        "vmv.v.x        v31, t4\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (t3)\n\t"
        "addi           t3, t3, 16\n\t"
        "vle32.v        v2, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        // pre-load pa(kernel_data)
        "lw             a0, 0(t5)\n\t"
        "lw             a1, 4(t5)\n\t"
        "lw             a2, 8(t5)\n\t"
        "lw             a3, 12(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 4f\n\t"       // if k2 == 0, jump to m8n8k1

        // m8n8k2
        "3:\n\t"
        "vle32.v        v4, (t3)\n\t"
        "addi           t3, t3, 16\n\t"
        "vle32.v        v5, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "vmaqa.vx       v17, a0, v2\n\t"
        "lw             a4, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v19, a1, v2\n\t"
        "lw             a5, 20(t5)\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "vmaqa.vx       v21, a2, v2\n\t"
        "lw             a6, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "vmaqa.vx       v23, a3, v2\n\t"
        "lw             a7, 28(t5)\n\t"
        "vmaqa.vx       v24, a4, v1\n\t"
        "vmaqa.vx       v25, a4, v2\n\t"
        "lw             a0, 32(t5)\n\t"
        "vmaqa.vx       v26, a5, v1\n\t"
        "vmaqa.vx       v27, a5, v2\n\t"
        "lw             a1, 36(t5)\n\t"
        "vmaqa.vx       v28, a6, v1\n\t"
        "vmaqa.vx       v29, a6, v2\n\t"
        "lw             a2, 40(t5)\n\t"
        "vmaqa.vx       v30, a7, v1\n\t"
        "vmaqa.vx       v31, a7, v2\n\t"
        "lw             a3, 44(t5)\n\t"  // 0

        "vle32.v        v1, (t3)\n\t"
        "addi           t3, t3, 16\n\t"
        "vle32.v        v2, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        "vmaqa.vx       v16, a0, v4\n\t"
        "vmaqa.vx       v17, a0, v5\n\t"
        "lw             a4, 48(t5)\n\t"
        "vmaqa.vx       v18, a1, v4\n\t"
        "vmaqa.vx       v19, a1, v5\n\t"
        "lw             a5, 52(t5)\n\t"
        "vmaqa.vx       v20, a2, v4\n\t"
        "vmaqa.vx       v21, a2, v5\n\t"
        "lw             a6, 56(t5)\n\t"
        "vmaqa.vx       v22, a3, v4\n\t"
        "vmaqa.vx       v23, a3, v5\n\t"
        "lw             a7, 60(t5)\n\t"
        "addi           t5, t5, 64\n\t"  // += 16 elements

        "vmaqa.vx       v24, a4, v4\n\t"
        "vmaqa.vx       v25, a4, v5\n\t"
        "lw             a0, 0(t5)\n\t"
        "vmaqa.vx       v26, a5, v4\n\t"
        "vmaqa.vx       v27, a5, v5\n\t"
        "lw             a1, 4(t5)\n\t"
        "vmaqa.vx       v28, a6, v4\n\t"
        "vmaqa.vx       v29, a6, v5\n\t"
        "lw             a2, 8(t5)\n\t"
        "vmaqa.vx       v30, a7, v4\n\t"
        "vmaqa.vx       v31, a7, v5\n\t"
        "lw             a3, 12(t5)\n\t"  // 1

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 3b\n\t"

        // m8n8k1
        "4:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 5f\n\t"       // if k1 == 0, jump to end kernel_m8n8

        "vmaqa.vx       v16, a0, v1\n\t"
        "vmaqa.vx       v17, a0, v2\n\t"
        "lw             a4, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v19, a1, v2\n\t"
        "lw             a5, 20(t5)\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "vmaqa.vx       v21, a2, v2\n\t"
        "lw             a6, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "vmaqa.vx       v23, a3, v2\n\t"
        "lw             a7, 28(t5)\n\t"
        "vmaqa.vx       v24, a4, v1\n\t"
        "vmaqa.vx       v25, a4, v2\n\t"
        "vmaqa.vx       v26, a5, v1\n\t"
        "vmaqa.vx       v27, a5, v2\n\t"
        "vmaqa.vx       v28, a6, v1\n\t"
        "vmaqa.vx       v29, a6, v2\n\t"
        "vmaqa.vx       v30, a7, v1\n\t"
        "vmaqa.vx       v31, a7, v2\n\t"

        "addi           t3, t3, 32\n\t"  // ********************

        // end kernel_m8n8
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           t3, t3, -32\n\t"  // pb -= 8

        // 后处理
        "li             t6, 8\n\t"

        "lw             a0, 0(%[mult_ptr])\n\t"
        "lw             a1, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"  // set vl = 8
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"  // set vl = 8
        "vnclip.wi	    v16, v1, 0\n\t"

        "lw             a2, 4(%[mult_ptr])\n\t"
        "lw             a3, 4(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "lw             a0, 8(%[mult_ptr])\n\t"
        "lw             a1, 8(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v20, v20, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v20, v20, a1\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v20, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v20, v1, 0\n\t"

        "lw             a2, 12(%[mult_ptr])\n\t"
        "lw             a3, 12(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v22, v22, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v22, v22, a3\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v22, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v22, v4, 0\n\t"

        "lw             a0, 16(%[mult_ptr])\n\t"
        "lw             a1, 16(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v24, v24, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v24, v24, a1\n\t"
        "vadd.vx        v24, v24, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v24, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v24, v1, 0\n\t"

        "lw             a2, 20(%[mult_ptr])\n\t"
        "lw             a3, 20(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v26, v26, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v26, v26, a3\n\t"
        "vadd.vx        v26, v26, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v26, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v26, v4, 0\n\t"

        "lw             a0, 24(%[mult_ptr])\n\t"
        "lw             a1, 24(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v28, v28, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v28, v28, a1\n\t"
        "vadd.vx        v28, v28, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v28, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v28, v1, 0\n\t"

        "lw             a2, 28(%[mult_ptr])\n\t"
        "lw             a3, 28(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v30, v30, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v30, v30, a3\n\t"
        "vadd.vx        v30, v30, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v30, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v30, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v20, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v22, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v24, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v26, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v28, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v30, (a0)\n\t"
        "addi           t2, t2, 8\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m8n4
        "6:\n\t"
        "andi           t1, %[n], 4\n\t"  // t1 = n & 4u (n4)
        "beqz           t1, 10f\n\t"      // if n4==0, jump to m8n_tail
        "li             t6, 4\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        // init out_tmp = bias
        "lw             t4, 0(%[bias_ptr])\n\t"  // bias_ptr[0]
        "vmv.v.x        v16, t4\n\t"
        "lw             t4, 4(%[bias_ptr])\n\t"  // bias_ptr[1]
        "vmv.v.x        v18, t4\n\t"
        "lw             t4, 8(%[bias_ptr])\n\t"  // bias_ptr[2]
        "vmv.v.x        v20, t4\n\t"
        "lw             t4, 12(%[bias_ptr])\n\t"  // bias_ptr[3]
        "vmv.v.x        v22, t4\n\t"
        "lw             t4, 16(%[bias_ptr])\n\t"  // bias_ptr[4]
        "vmv.v.x        v24, t4\n\t"
        "lw             t4, 20(%[bias_ptr])\n\t"  // bias_ptr[5]
        "vmv.v.x        v26, t4\n\t"
        "lw             t4, 24(%[bias_ptr])\n\t"  // bias_ptr[6]
        "vmv.v.x        v28, t4\n\t"
        "lw             t4, 28(%[bias_ptr])\n\t"  // bias_ptr[7]
        "vmv.v.x        v30, t4\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        // pre-load pa(kernel_data)
        "lw             a0, 0(t5)\n\t"
        "lw             a1, 4(t5)\n\t"
        "lw             a2, 8(t5)\n\t"
        "lw             a3, 12(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 8f\n\t"       // if k2 == 0, jump to m8n4k1

        // m8n4k2
        "7:\n\t"
        "vle32.v        v4, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lw             a4, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "lw             a5, 20(t5)\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lw             a6, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "lw             a7, 28(t5)\n\t"
        "vmaqa.vx       v24, a4, v1\n\t"
        "lw             a0, 32(t5)\n\t"
        "vmaqa.vx       v26, a5, v1\n\t"
        "lw             a1, 36(t5)\n\t"
        "vmaqa.vx       v28, a6, v1\n\t"
        "lw             a2, 40(t5)\n\t"
        "vmaqa.vx       v30, a7, v1\n\t"
        "lw             a3, 44(t5)\n\t"  // 0

        "vle32.v        v1, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        "vmaqa.vx       v16, a0, v4\n\t"
        "lw             a4, 48(t5)\n\t"
        "vmaqa.vx       v18, a1, v4\n\t"
        "lw             a5, 52(t5)\n\t"
        "vmaqa.vx       v20, a2, v4\n\t"
        "lw             a6, 56(t5)\n\t"
        "vmaqa.vx       v22, a3, v4\n\t"
        "lw             a7, 60(t5)\n\t"
        "addi           t5, t5, 64\n\t"  // += 16 elements

        "vmaqa.vx       v24, a4, v4\n\t"
        "lw             a0, 0(t5)\n\t"
        "vmaqa.vx       v26, a5, v4\n\t"
        "lw             a1, 4(t5)\n\t"
        "vmaqa.vx       v28, a6, v4\n\t"
        "lw             a2, 8(t5)\n\t"
        "vmaqa.vx       v30, a7, v4\n\t"
        "lw             a3, 12(t5)\n\t"  // 1

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 7b\n\t"

        // m8n4k1
        "8:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 9f\n\t"       // if k1 == 0, jump to end kernel_m8n4

        "vmaqa.vx       v16, a0, v1\n\t"
        "lw             a4, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "lw             a5, 20(t5)\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lw             a6, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "lw             a7, 28(t5)\n\t"
        "vmaqa.vx       v24, a4, v1\n\t"
        "vmaqa.vx       v26, a5, v1\n\t"
        "vmaqa.vx       v28, a6, v1\n\t"
        "vmaqa.vx       v30, a7, v1\n\t"

        "addi           t3, t3, 16\n\t"  // ********************

        // end kernel_m8n4
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           t3, t3, -16\n\t"  // pb -= 4

        // 后处理
        "li             t6, 4\n\t"

        "lw             a0, 0(%[mult_ptr])\n\t"
        "lw             a1, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"  // set vl = 4
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"  // set vl = 4
        "vnclip.wi	    v16, v1, 0\n\t"

        "lw             a2, 4(%[mult_ptr])\n\t"
        "lw             a3, 4(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "lw             a0, 8(%[mult_ptr])\n\t"
        "lw             a1, 8(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v20, v20, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v20, v20, a1\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v1, v20, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v20, v1, 0\n\t"

        "lw             a2, 12(%[mult_ptr])\n\t"
        "lw             a3, 12(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v22, v22, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v22, v22, a3\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v22, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v22, v4, 0\n\t"

        "lw             a0, 16(%[mult_ptr])\n\t"
        "lw             a1, 16(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v24, v24, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v24, v24, a1\n\t"
        "vadd.vx        v24, v24, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v1, v24, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v24, v1, 0\n\t"

        "lw             a2, 20(%[mult_ptr])\n\t"
        "lw             a3, 20(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v26, v26, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v26, v26, a3\n\t"
        "vadd.vx        v26, v26, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v26, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v26, v4, 0\n\t"

        "lw             a0, 24(%[mult_ptr])\n\t"
        "lw             a1, 24(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v28, v28, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v28, v28, a1\n\t"
        "vadd.vx        v28, v28, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v1, v28, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v28, v1, 0\n\t"

        "lw             a2, 28(%[mult_ptr])\n\t"
        "lw             a3, 28(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v30, v30, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v30, v30, a3\n\t"
        "vadd.vx        v30, v30, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v30, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v30, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v20, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v22, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v24, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v26, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v28, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v30, (a0)\n\t"
        "addi           t2, t2, 4\n\t"

        // m8n_tail
        "10:\n\t"
        "andi           t1, %[n], 3\n\t"        // t1 = n & 3u (n_tail)
        "beqz           t1, 14f\n\t"            // if n_tail==0, jump to end kernel_m8
        "vsetvli        zero, t1, e32, m1\n\t"  // set vl = n_tail
        "slli           t6, t1, 2\n\t"          // t6 = 4 * n_tail

        // init out_tmp = bias
        "lw             t4, 0(%[bias_ptr])\n\t"  // bias_ptr[0]
        "vmv.v.x        v16, t4\n\t"
        "lw             t4, 4(%[bias_ptr])\n\t"  // bias_ptr[1]
        "vmv.v.x        v18, t4\n\t"
        "lw             t4, 8(%[bias_ptr])\n\t"  // bias_ptr[2]
        "vmv.v.x        v20, t4\n\t"
        "lw             t4, 12(%[bias_ptr])\n\t"  // bias_ptr[3]
        "vmv.v.x        v22, t4\n\t"
        "lw             t4, 16(%[bias_ptr])\n\t"  // bias_ptr[4]
        "vmv.v.x        v24, t4\n\t"
        "lw             t4, 20(%[bias_ptr])\n\t"  // bias_ptr[5]
        "vmv.v.x        v26, t4\n\t"
        "lw             t4, 24(%[bias_ptr])\n\t"  // bias_ptr[6]
        "vmv.v.x        v28, t4\n\t"
        "lw             t4, 28(%[bias_ptr])\n\t"  // bias_ptr[7]
        "vmv.v.x        v30, t4\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (t3)\n\t"
        "add            t3, t3, t6\n\t"

        // pre-load pa(kernel_data)
        "lw             a0, 0(t5)\n\t"
        "lw             a1, 4(t5)\n\t"
        "lw             a2, 8(t5)\n\t"
        "lw             a3, 12(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 12f\n\t"      // if k2 == 0, jump to m8n_tail k1

        // m8n_tailk2
        "11:\n\t"
        "vle32.v        v4, (t3)\n\t"
        "add            t3, t3, t6\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lw             a4, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "lw             a5, 20(t5)\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lw             a6, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "lw             a7, 28(t5)\n\t"
        "vmaqa.vx       v24, a4, v1\n\t"
        "lw             a0, 32(t5)\n\t"
        "vmaqa.vx       v26, a5, v1\n\t"
        "lw             a1, 36(t5)\n\t"
        "vmaqa.vx       v28, a6, v1\n\t"
        "lw             a2, 40(t5)\n\t"
        "vmaqa.vx       v30, a7, v1\n\t"
        "lw             a3, 44(t5)\n\t"  // 0

        "vle32.v        v1, (t3)\n\t"
        "add            t3, t3, t6\n\t"

        "vmaqa.vx       v16, a0, v4\n\t"
        "lw             a4, 48(t5)\n\t"
        "vmaqa.vx       v18, a1, v4\n\t"
        "lw             a5, 52(t5)\n\t"
        "vmaqa.vx       v20, a2, v4\n\t"
        "lw             a6, 56(t5)\n\t"
        "vmaqa.vx       v22, a3, v4\n\t"
        "lw             a7, 60(t5)\n\t"
        "addi           t5, t5, 64\n\t"  // += 16 elements

        "vmaqa.vx       v24, a4, v4\n\t"
        "lw             a0, 0(t5)\n\t"
        "vmaqa.vx       v26, a5, v4\n\t"
        "lw             a1, 4(t5)\n\t"
        "vmaqa.vx       v28, a6, v4\n\t"
        "lw             a2, 8(t5)\n\t"
        "vmaqa.vx       v30, a7, v4\n\t"
        "lw             a3, 12(t5)\n\t"  // 1

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 11b\n\t"

        // m8n_tailk1
        "12:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 13f\n\t"      // if k1 == 0, jump to end kernel_m8n_tail

        "vmaqa.vx       v16, a0, v1\n\t"
        "lw             a4, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "lw             a5, 20(t5)\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lw             a6, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "lw             a7, 28(t5)\n\t"
        "vmaqa.vx       v24, a4, v1\n\t"
        "vmaqa.vx       v26, a5, v1\n\t"
        "vmaqa.vx       v28, a6, v1\n\t"
        "vmaqa.vx       v30, a7, v1\n\t"

        "add            t3, t3, t6\n\t"  // ********************

        // end kernel_m8n_tail
        "13:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "sub            t3, t3, t6\n\t"  // pb -= n_tail

        // 后处理
        "lw             a0, 0(%[mult_ptr])\n\t"
        "lw             a1, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"  // set vl = n_tail
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"  // set vl = n_tail
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"  // set vl = n_tail
        "vnclip.wi	    v16, v1, 0\n\t"

        "lw             a2, 4(%[mult_ptr])\n\t"
        "lw             a3, 4(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "lw             a0, 8(%[mult_ptr])\n\t"
        "lw             a1, 8(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v20, v20, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v20, v20, a1\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v1, v20, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v20, v1, 0\n\t"

        "lw             a2, 12(%[mult_ptr])\n\t"
        "lw             a3, 12(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v22, v22, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v22, v22, a3\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v22, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v22, v4, 0\n\t"

        "lw             a0, 16(%[mult_ptr])\n\t"
        "lw             a1, 16(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v24, v24, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v24, v24, a1\n\t"
        "vadd.vx        v24, v24, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v1, v24, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v24, v1, 0\n\t"

        "lw             a2, 20(%[mult_ptr])\n\t"
        "lw             a3, 20(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v26, v26, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v26, v26, a3\n\t"
        "vadd.vx        v26, v26, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v26, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v26, v4, 0\n\t"

        "lw             a0, 24(%[mult_ptr])\n\t"
        "lw             a1, 24(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v28, v28, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v28, v28, a1\n\t"
        "vadd.vx        v28, v28, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v1, v28, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v28, v1, 0\n\t"

        "lw             a2, 28(%[mult_ptr])\n\t"
        "lw             a3, 28(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v30, v30, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v30, v30, a3\n\t"
        "vadd.vx        v30, v30, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v30, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v30, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v20, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v22, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v24, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v26, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v28, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v30, (a0)\n\t"
        "add            t2, t2, t1\n\t"

        // end kernel_m8
        "14:\n\t"
        "addi           %[bias_ptr], %[bias_ptr], 32\n\t"    // bias_data += 8
        "addi           %[mult_ptr], %[mult_ptr], 32\n\t"    // mult_ptr += 8
        "addi           %[shift_ptr], %[shift_ptr], 32\n\t"  // shift_ptr += 8
        "slli           t6, %[k], 3\n\t"
        "add            %[kernel_ptr], %[kernel_ptr], t6\n\t"  // kernel_data += 8 * k
        "slli           t6, %[n], 3\n\t"
        "add            %[output_ptr], %[output_ptr], t6\n\t"  // output_data += 8 * n

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        // ending
        "15:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias),
        [mult_ptr] "+r"(mult), [shift_ptr] "+r"(shift)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [out_zp] "r"(out_zp)
        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v1", "v2", "v4", "v5", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
        "v25", "v26", "v27", "v28", "v29", "v30", "v31",
        // We use these general-purpose registers.
        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
}

// 如果使能xtheadc, 可用lwd指令
static inline void kernel_m8n8_int8_1(int8_t *dst, int8_t *sa, int8_t *sb, int m, int k, int n,
                                      int32_t *bias, int32_t out_zp, int32_t *mult, int32_t *shift)
{
    asm volatile(
        "srai           t0, %[m], 3\n\t"  // t0 = m8
        "beqz           t0, 15f\n\t"

        // m8
        "1:\n\t"
        "srai           t1, %[n], 3\n\t"        // t1 = n8
        "mv             t2, %[output_ptr]\n\t"  // init output addr
        "mv             t3, %[input_ptr]\n\t"   // t3 hold input data start addr

        "beqz           t1, 6f\n\t"  // if n8==0, jump to m8n4
        // m8n8
        "2:\n\t"
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v18, t5\n\t"
        "lwd            t4, t5, 8(%[bias_ptr])\n\t"  // bias_ptr[2]/[3]
        "vmv.v.x        v20, t4\n\t"
        "vmv.v.x        v22, t5\n\t"
        "lwd            t4, t5, 16(%[bias_ptr])\n\t"  // bias_ptr[4]/[5]
        "vmv.v.x        v24, t4\n\t"
        "vmv.v.x        v26, t5\n\t"
        "lwd            t4, t5, 24(%[bias_ptr])\n\t"  // bias_ptr[6]/[7]
        "vmv.v.x        v28, t4\n\t"
        "vmv.v.x        v30, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v2, (t3)\n\t"
        "addi           t3, t3, 32\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 4f\n\t"       // if k2 == 0, jump to m8n8k1

        // m8n8k2
        "3:\n\t"
        "vle32.v        v4, (t3)\n\t"
        "addi           t3, t3, 32\n\t"

        "vmaqa.vx       v16, a0, v2\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v20, a2, v2\n\t"
        "vmaqa.vx       v22, a3, v2\n\t"
        "addi           t5, t5, 32\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v24, a4, v2\n\t"
        "vmaqa.vx       v26, a5, v2\n\t"
        "vmaqa.vx       v28, a6, v2\n\t"
        "vmaqa.vx       v30, a7, v2\n\t"

        "vle32.v        v2, (t3)\n\t"
        "addi           t3, t3, 32\n\t"

        "vmaqa.vx       v16, a0, v4\n\t"
        "vmaqa.vx       v18, a1, v4\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v20, a2, v4\n\t"
        "vmaqa.vx       v22, a3, v4\n\t"
        "addi           t5, t5, 32\n\t"  // += 16 elements
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v24, a4, v4\n\t"
        "vmaqa.vx       v26, a5, v4\n\t"
        "vmaqa.vx       v28, a6, v4\n\t"
        "vmaqa.vx       v30, a7, v4\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 3b\n\t"

        // m8n8k1
        "4:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 5f\n\t"       // if k1 == 0, jump to end kernel_m8n8

        "lwd            a4, a5, 16(t5)\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v16, a0, v2\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"
        "vmaqa.vx       v20, a2, v2\n\t"
        "vmaqa.vx       v22, a3, v2\n\t"
        "vmaqa.vx       v24, a4, v2\n\t"
        "vmaqa.vx       v26, a5, v2\n\t"
        "vmaqa.vx       v28, a6, v2\n\t"
        "vmaqa.vx       v30, a7, v2\n\t"

        "addi           t3, t3, 32\n\t"  // ********************

        // end kernel_m8n8
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           t3, t3, -32\n\t"  // pb -= 8

        // 后处理
        "li             t6, 8\n\t"

        "lwd            a0, a2, 0(%[mult_ptr])\n\t"
        "lwd            a1, a3, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"  // set vl = 8
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"  // set vl = 8
        "vnclip.wi	    v16, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "lwd            a0, a2, 8(%[mult_ptr])\n\t"
        "lwd            a1, a3, 8(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v20, v20, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v20, v20, a1\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v20, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v20, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v22, v22, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v22, v22, a3\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v22, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v22, v4, 0\n\t"

        "lwd            a0, a2, 16(%[mult_ptr])\n\t"
        "lwd            a1, a3, 16(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v24, v24, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v24, v24, a1\n\t"
        "vadd.vx        v24, v24, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v24, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v24, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v26, v26, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v26, v26, a3\n\t"
        "vadd.vx        v26, v26, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v26, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v26, v4, 0\n\t"

        "lwd            a0, a2, 24(%[mult_ptr])\n\t"
        "lwd            a1, a3, 24(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v28, v28, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v28, v28, a1\n\t"
        "vadd.vx        v28, v28, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v28, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v28, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v30, v30, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v30, v30, a3\n\t"
        "vadd.vx        v30, v30, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v30, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v30, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v20, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v22, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v24, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v26, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v28, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v30, (a0)\n\t"
        "addi           t2, t2, 8\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m8n4
        "6:\n\t"
        "andi           t1, %[n], 4\n\t"  // t1 = n & 4u (n4)
        "beqz           t1, 10f\n\t"      // if n4==0, jump to m8n_tail
        "li             t6, 4\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v18, t5\n\t"
        "lwd            t4, t5, 8(%[bias_ptr])\n\t"  // bias_ptr[2]/[3]
        "vmv.v.x        v20, t4\n\t"
        "vmv.v.x        v22, t5\n\t"
        "lwd            t4, t5, 16(%[bias_ptr])\n\t"  // bias_ptr[4]/[5]
        "vmv.v.x        v24, t4\n\t"
        "vmv.v.x        v26, t5\n\t"
        "lwd            t4, t5, 24(%[bias_ptr])\n\t"  // bias_ptr[6]/[7]
        "vmv.v.x        v28, t4\n\t"
        "vmv.v.x        v30, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 8f\n\t"       // if k2 == 0, jump to m8n4k1

        // m8n4k2
        "7:\n\t"
        "vle32.v        v4, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "addi           t5, t5, 32\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "vmaqa.vx       v24, a4, v1\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "vmaqa.vx       v26, a5, v1\n\t"
        "vmaqa.vx       v28, a6, v1\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v30, a7, v1\n\t"  // 0

        "vle32.v        v1, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        "vmaqa.vx       v16, a0, v4\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v4\n\t"
        "vmaqa.vx       v20, a2, v4\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v4\n\t"
        "addi           t5, t5, 32\n\t"  // += 16 elements

        "vmaqa.vx       v24, a4, v4\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "vmaqa.vx       v26, a5, v4\n\t"
        "vmaqa.vx       v28, a6, v4\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v30, a7, v4\n\t"  // 1

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 7b\n\t"

        // m8n4k1
        "8:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 9f\n\t"       // if k1 == 0, jump to end kernel_m8n4

        "vmaqa.vx       v16, a0, v1\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "vmaqa.vx       v24, a4, v1\n\t"
        "vmaqa.vx       v26, a5, v1\n\t"
        "vmaqa.vx       v28, a6, v1\n\t"
        "vmaqa.vx       v30, a7, v1\n\t"

        "addi           t3, t3, 16\n\t"  // ********************

        // end kernel_m8n4
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           t3, t3, -16\n\t"  // pb -= 4

        // 后处理
        "li             t6, 4\n\t"

        "lwd            a0, a2, 0(%[mult_ptr])\n\t"
        "lwd            a1, a3, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"  // set vl = 4
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"  // set vl = 4
        "vnclip.wi	    v16, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "lwd            a0, a2, 8(%[mult_ptr])\n\t"
        "lwd            a1, a3, 8(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v20, v20, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v20, v20, a1\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v1, v20, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v20, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v22, v22, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v22, v22, a3\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v22, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v22, v4, 0\n\t"

        "lwd            a0, a2, 16(%[mult_ptr])\n\t"
        "lwd            a1, a3, 16(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v24, v24, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v24, v24, a1\n\t"
        "vadd.vx        v24, v24, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v1, v24, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v24, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v26, v26, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v26, v26, a3\n\t"
        "vadd.vx        v26, v26, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v26, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v26, v4, 0\n\t"

        "lwd            a0, a2, 24(%[mult_ptr])\n\t"
        "lwd            a1, a3, 24(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v28, v28, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v28, v28, a1\n\t"
        "vadd.vx        v28, v28, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v1, v28, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v28, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v30, v30, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v30, v30, a3\n\t"
        "vadd.vx        v30, v30, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v30, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v30, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v20, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v22, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v24, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v26, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v28, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v30, (a0)\n\t"
        "addi           t2, t2, 4\n\t"

        // m8n_tail
        "10:\n\t"
        "andi           t1, %[n], 3\n\t"        // t1 = n & 3u (n_tail)
        "beqz           t1, 14f\n\t"            // if n_tail==0, jump to end kernel_m8
        "vsetvli        zero, t1, e32, m1\n\t"  // set vl = n_tail
        "slli           t6, t1, 2\n\t"          // t6 = 4 * n_tail

        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v18, t5\n\t"
        "lwd            t4, t5, 8(%[bias_ptr])\n\t"  // bias_ptr[2]/[3]
        "vmv.v.x        v20, t4\n\t"
        "vmv.v.x        v22, t5\n\t"
        "lwd            t4, t5, 16(%[bias_ptr])\n\t"  // bias_ptr[4]/[5]
        "vmv.v.x        v24, t4\n\t"
        "vmv.v.x        v26, t5\n\t"
        "lwd            t4, t5, 24(%[bias_ptr])\n\t"  // bias_ptr[6]/[7]
        "vmv.v.x        v28, t4\n\t"
        "vmv.v.x        v30, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (t3)\n\t"
        "add            t3, t3, t6\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 12f\n\t"      // if k2 == 0, jump to m8n_tail k1

        // m8n_tailk2
        "11:\n\t"
        "vle32.v        v4, (t3)\n\t"
        "add            t3, t3, t6\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "addi           t5, t5, 32\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "vmaqa.vx       v24, a4, v1\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "vmaqa.vx       v26, a5, v1\n\t"
        "vmaqa.vx       v28, a6, v1\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v30, a7, v1\n\t"  // 0

        "vle32.v        v1, (t3)\n\t"
        "add            t3, t3, t6\n\t"

        "vmaqa.vx       v16, a0, v4\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v4\n\t"
        "vmaqa.vx       v20, a2, v4\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v4\n\t"
        "addi           t5, t5, 32\n\t"  // += 16 elements

        "vmaqa.vx       v24, a4, v4\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "vmaqa.vx       v26, a5, v4\n\t"
        "vmaqa.vx       v28, a6, v4\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v30, a7, v4\n\t"  // 1

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 11b\n\t"

        // m8n_tailk1
        "12:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 13f\n\t"      // if k1 == 0, jump to end kernel_m8n_tail

        "vmaqa.vx       v16, a0, v1\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "vmaqa.vx       v24, a4, v1\n\t"
        "vmaqa.vx       v26, a5, v1\n\t"
        "vmaqa.vx       v28, a6, v1\n\t"
        "vmaqa.vx       v30, a7, v1\n\t"

        "add            t3, t3, t6\n\t"  // ********************

        // end kernel_m8n_tail
        "13:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "sub            t3, t3, t6\n\t"  // pb -= n_tail

        // 后处理
        "lwd            a0, a2, 0(%[mult_ptr])\n\t"
        "lwd            a1, a3, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"  // set vl = n_tail
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"  // set vl = n_tail
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"  // set vl = n_tail
        "vnclip.wi	    v16, v1, 0\n\t"

        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "lwd            a0, a2, 8(%[mult_ptr])\n\t"
        "lwd            a1, a3, 8(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v20, v20, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v20, v20, a1\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v1, v20, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v20, v1, 0\n\t"

        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v22, v22, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v22, v22, a3\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v22, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v22, v4, 0\n\t"

        "lwd            a0, a2, 16(%[mult_ptr])\n\t"
        "lwd            a1, a3, 16(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v24, v24, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v24, v24, a1\n\t"
        "vadd.vx        v24, v24, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v1, v24, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v24, v1, 0\n\t"

        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v26, v26, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v26, v26, a3\n\t"
        "vadd.vx        v26, v26, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v26, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v26, v4, 0\n\t"

        "lwd            a0, a2, 24(%[mult_ptr])\n\t"
        "lwd            a1, a3, 24(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v28, v28, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v28, v28, a1\n\t"
        "vadd.vx        v28, v28, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v1, v28, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v28, v1, 0\n\t"

        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v30, v30, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v30, v30, a3\n\t"
        "vadd.vx        v30, v30, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v30, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v30, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v20, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v22, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v24, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v26, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v28, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v30, (a0)\n\t"
        "add            t2, t2, t1\n\t"

        // end kernel_m8
        "14:\n\t"
        "addi           %[bias_ptr], %[bias_ptr], 32\n\t"    // bias_data += 8
        "addi           %[mult_ptr], %[mult_ptr], 32\n\t"    // mult_ptr += 8
        "addi           %[shift_ptr], %[shift_ptr], 32\n\t"  // shift_ptr += 8
        "slli           t6, %[k], 3\n\t"
        "add            %[kernel_ptr], %[kernel_ptr], t6\n\t"  // kernel_data += 8 * k
        "slli           t6, %[n], 3\n\t"
        "add            %[output_ptr], %[output_ptr], t6\n\t"  // output_data += 8 * n

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        // ending
        "15:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias),
        [mult_ptr] "+r"(mult), [shift_ptr] "+r"(shift)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [out_zp] "r"(out_zp)
        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v1", "v2", "v4", "v5", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
        "v25", "v26", "v27", "v28", "v29", "v30", "v31",
        // We use these general-purpose registers.
        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
}

static inline void kernel_m4n8_int8_1(int8_t *dst, int8_t *sa, int8_t *sb, int m, int k, int n,
                                      int32_t *bias, int32_t out_zp, int32_t *mult, int32_t *shift)
{
    asm volatile(
        // m4
        "1:\n\t"
        "srai           t1, %[n], 3\n\t"        // t1 = n8
        "mv             t2, %[output_ptr]\n\t"  // init output addr

        "beqz           t1, 6f\n\t"  // if n8==0, jump to m4n4
        // m4n8
        "2:\n\t"
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v18, t5\n\t"
        "lwd            t4, t5, 8(%[bias_ptr])\n\t"  // bias_ptr[2]/[3]
        "vmv.v.x        v20, t4\n\t"
        "vmv.v.x        v22, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 4f\n\t"       // if k2 == 0, jump to m4n8k1

        // m4n8k2
        "3:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vmaqa.vx       v16, a0, v2\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v20, a2, v2\n\t"
        "vmaqa.vx       v22, a3, v2\n\t"
        "addi           t5, t5, 32\n\t"

        "vle32.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vmaqa.vx       v16, a4, v4\n\t"
        "vmaqa.vx       v18, a5, v4\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v20, a6, v4\n\t"
        "vmaqa.vx       v22, a7, v4\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 3b\n\t"

        // m4n8k1
        "4:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 5f\n\t"       // if k1 == 0, jump to end kernel_m4n8

        "vmaqa.vx       v16, a0, v2\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"
        "vmaqa.vx       v20, a2, v2\n\t"
        "vmaqa.vx       v22, a3, v2\n\t"

        "addi           %[input_ptr], %[input_ptr], 32\n\t"  // ********************

        // end kernel_m4n8
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -32\n\t"  // pb -= 8

        // 后处理
        "li             t6, 8\n\t"

        "lwd            a0, a2, 0(%[mult_ptr])\n\t"
        "lwd            a1, a3, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"  // set vl = 8
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"  // set vl = 8
        "vnclip.wi	    v16, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "lwd            a0, a2, 8(%[mult_ptr])\n\t"
        "lwd            a1, a3, 8(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v20, v20, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v20, v20, a1\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v20, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v20, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v22, v22, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v22, v22, a3\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v22, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v22, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v20, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v22, (a0)\n\t"
        "addi           t2, t2, 8\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m4n4
        "6:\n\t"
        "andi           t1, %[n], 4\n\t"  // t1 = n & 4u (n4)
        "beqz           t1, 10f\n\t"      // if n4==0, jump to m4n_tail
        "li             t6, 4\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v18, t5\n\t"
        "lwd            t4, t5, 8(%[bias_ptr])\n\t"  // bias_ptr[2]/[3]
        "vmv.v.x        v20, t4\n\t"
        "vmv.v.x        v22, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 16\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 8f\n\t"       // if k2 == 0, jump to m8n4k1

        // m8n4k2
        "7:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 16\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "addi           t5, t5, 32\n\t"

        "vle32.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 16\n\t"

        "vmaqa.vx       v16, a4, v4\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "vmaqa.vx       v18, a5, v4\n\t"
        "vmaqa.vx       v20, a6, v4\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v22, a7, v4\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 7b\n\t"

        // m4n4k1
        "8:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 9f\n\t"       // if k1 == 0, jump to end kernel_m4n4

        "vmaqa.vx       v16, a0, v1\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"

        "addi           %[input_ptr], %[input_ptr], 16\n\t"  // ********************

        // end kernel_m8n4
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -16\n\t"  // pb -= 4

        // 后处理
        "li             t6, 4\n\t"

        "lwd            a0, a2, 0(%[mult_ptr])\n\t"
        "lwd            a1, a3, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"  // set vl = 4
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"  // set vl = 4
        "vnclip.wi	    v16, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "lwd            a0, a2, 8(%[mult_ptr])\n\t"
        "lwd            a1, a3, 8(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v20, v20, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v20, v20, a1\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v1, v20, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v20, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v22, v22, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v22, v22, a3\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v22, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v22, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v20, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v22, (a0)\n\t"
        "addi           t2, t2, 4\n\t"

        // m4n_tail
        "10:\n\t"
        "andi           t1, %[n], 3\n\t"        // t1 = n & 3u (n_tail)
        "beqz           t1, 14f\n\t"            // if n_tail==0, jump to end kernel_m4
        "vsetvli        zero, t1, e32, m1\n\t"  // set vl = n_tail
        "slli           t6, t1, 2\n\t"          // t6 = 4 * n_tail

        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v18, t5\n\t"
        "lwd            t4, t5, 8(%[bias_ptr])\n\t"  // bias_ptr[2]/[3]
        "vmv.v.x        v20, t4\n\t"
        "vmv.v.x        v22, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 12f\n\t"      // if k2 == 0, jump to m8n_tail k1

        // m8n_tailk2
        "11:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "addi           t5, t5, 32\n\t"

        "vle32.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vmaqa.vx       v16, a4, v4\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "vmaqa.vx       v18, a5, v4\n\t"
        "vmaqa.vx       v20, a6, v4\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v22, a7, v4\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 11b\n\t"

        // m8n_tailk1
        "12:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 13f\n\t"      // if k1 == 0, jump to end kernel_m8n_tail

        "vmaqa.vx       v16, a0, v1\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"

        "add            %[input_ptr], %[input_ptr], t6\n\t"  // ********************

        // end kernel_m4n_tail
        "13:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "sub            %[input_ptr], %[input_ptr], t6\n\t"  // pb -= n_tail

        // 后处理
        "lwd            a0, a2, 0(%[mult_ptr])\n\t"
        "lwd            a1, a3, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"  // set vl = n_tail
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"  // set vl = n_tail
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"  // set vl = n_tail
        "vnclip.wi	    v16, v1, 0\n\t"

        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "lwd            a0, a2, 8(%[mult_ptr])\n\t"
        "lwd            a1, a3, 8(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v20, v20, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v20, v20, a1\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v1, v20, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v20, v1, 0\n\t"

        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v22, v22, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v22, v22, a3\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v22, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v22, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v20, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v22, (a0)\n\t"
        "add            t2, t2, t1\n\t"

        // ending
        "14:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias),
        [mult_ptr] "+r"(mult), [shift_ptr] "+r"(shift)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [out_zp] "r"(out_zp)
        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v1", "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        // We use these general-purpose registers.
        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t1", "t2", "t4", "t5", "t6");
}

static inline void kernel_m2n8_int8_1(int8_t *dst, int8_t *sa, int8_t *sb, int m, int k, int n,
                                      int32_t *bias, int32_t out_zp, int32_t *mult, int32_t *shift)
{
    asm volatile(
        // m4
        "1:\n\t"
        "srai           t1, %[n], 3\n\t"        // t1 = n8
        "mv             t2, %[output_ptr]\n\t"  // init output addr

        "beqz           t1, 6f\n\t"  // if n8==0, jump to m4n4
        // m4n8
        "2:\n\t"
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v18, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 4f\n\t"       // if k2 == 0, jump to m4n8k1

        // m4n8k2
        "3:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vmaqa.vx       v16, a0, v2\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "addi           t5, t5, 16\n\t"

        "vle32.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vmaqa.vx       v16, a2, v4\n\t"
        "vmaqa.vx       v18, a3, v4\n\t"
        "lwd            a0, a1, 0(t5)\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 3b\n\t"

        // m4n8k1
        "4:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 5f\n\t"       // if k1 == 0, jump to end kernel_m4n8

        "vmaqa.vx       v16, a0, v2\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"

        "addi           %[input_ptr], %[input_ptr], 32\n\t"  // ********************

        // end kernel_m4n8
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -32\n\t"  // pb -= 8

        // 后处理
        "li             t6, 8\n\t"

        "lwd            a0, a2, 0(%[mult_ptr])\n\t"
        "lwd            a1, a3, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"  // set vl = 8
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"  // set vl = 8
        "vnclip.wi	    v16, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "addi           t2, t2, 8\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m4n4
        "6:\n\t"
        "andi           t1, %[n], 4\n\t"  // t1 = n & 4u (n4)
        "beqz           t1, 10f\n\t"      // if n4==0, jump to m4n_tail
        "li             t6, 4\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v18, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 16\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 8f\n\t"       // if k2 == 0, jump to m8n4k1

        // m8n4k2
        "7:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 16\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "addi           t5, t5, 16\n\t"

        "vle32.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 16\n\t"

        "vmaqa.vx       v16, a2, v4\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "vmaqa.vx       v18, a3, v4\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 7b\n\t"

        // m4n4k1
        "8:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 9f\n\t"       // if k1 == 0, jump to end kernel_m4n4

        "vmaqa.vx       v16, a0, v1\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"

        "addi           %[input_ptr], %[input_ptr], 16\n\t"  // ********************

        // end kernel_m8n4
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -16\n\t"  // pb -= 4

        // 后处理
        "li             t6, 4\n\t"

        "lwd            a0, a2, 0(%[mult_ptr])\n\t"
        "lwd            a1, a3, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"  // set vl = 4
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"  // set vl = 4
        "vnclip.wi	    v16, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "addi           t2, t2, 4\n\t"

        // m4n_tail
        "10:\n\t"
        "andi           t1, %[n], 3\n\t"        // t1 = n & 3u (n_tail)
        "beqz           t1, 14f\n\t"            // if n_tail==0, jump to end kernel_m4
        "vsetvli        zero, t1, e32, m1\n\t"  // set vl = n_tail
        "slli           t6, t1, 2\n\t"          // t6 = 4 * n_tail

        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v18, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 12f\n\t"      // if k2 == 0, jump to m8n_tail k1

        // m8n_tailk2
        "11:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "addi           t5, t5, 16\n\t"

        "vle32.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vmaqa.vx       v16, a2, v4\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "vmaqa.vx       v18, a3, v4\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 11b\n\t"

        // m2n_tailk1
        "12:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 13f\n\t"      // if k1 == 0, jump to end kernel_m8n_tail

        "vmaqa.vx       v16, a0, v1\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"

        "add            %[input_ptr], %[input_ptr], t6\n\t"  // ********************

        // end kernel_m4n_tail
        "13:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "sub            %[input_ptr], %[input_ptr], t6\n\t"  // pb -= n_tail

        // 后处理
        "lwd            a0, a2, 0(%[mult_ptr])\n\t"
        "lwd            a1, a3, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"  // set vl = n_tail
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"  // set vl = n_tail
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"  // set vl = n_tail
        "vnclip.wi	    v16, v1, 0\n\t"

        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "add            t2, t2, t1\n\t"

        // ending
        "14:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias),
        [mult_ptr] "+r"(mult), [shift_ptr] "+r"(shift)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [out_zp] "r"(out_zp)
        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v1", "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19",
        // We use these general-purpose registers.
        "a0", "a1", "a2", "a3", "t1", "t2", "t4", "t5", "t6");
}

static inline void kernel_m1n8_int8_1(int8_t *dst, int8_t *sa, int8_t *sb, int m, int k, int n,
                                      int32_t *bias, int32_t out_zp, int32_t *mult, int32_t *shift)
{
    asm volatile(
        // m4
        "1:\n\t"
        "srai           t1, %[n], 3\n\t"        // t1 = n8
        "mv             t2, %[output_ptr]\n\t"  // init output addr

        "beqz           t1, 6f\n\t"  // if n8==0, jump to m4n4
        // m4n8
        "2:\n\t"
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        // init out_tmp = bias
        "lw             t4, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "lw             a0, 0(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 4f\n\t"       // if k2 == 0, jump to m4n8k1

        // m4n8k2
        "3:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vmaqa.vx       v16, a0, v2\n\t"
        "lw             a1, 4(t5)\n\t"
        "addi           t5, t5, 8\n\t"

        "vle32.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vmaqa.vx       v16, a1, v4\n\t"
        "lw             a0, 0(t5)\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 3b\n\t"

        // m4n8k1
        "4:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 5f\n\t"       // if k1 == 0, jump to end kernel_m4n8

        "vmaqa.vx       v16, a0, v2\n\t"

        "addi           %[input_ptr], %[input_ptr], 32\n\t"  // ********************

        // end kernel_m4n8
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -32\n\t"  // pb -= 8

        // 后处理
        "li             t6, 8\n\t"

        "lw             a0, 0(%[mult_ptr])\n\t"
        "lw             a1, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"  // set vl = 8
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"  // set vl = 8
        "vnclip.wi	    v16, v1, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "addi           t2, t2, 8\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m4n4
        "6:\n\t"
        "andi           t1, %[n], 4\n\t"  // t1 = n & 4u (n4)
        "beqz           t1, 10f\n\t"      // if n4==0, jump to m4n_tail
        "li             t6, 4\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        // init out_tmp = bias
        "lw             t4, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 16\n\t"

        // pre-load pa(kernel_data)
        "lw             a0, 0(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 8f\n\t"       // if k2 == 0, jump to m8n4k1

        // m8n4k2
        "7:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 16\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lw             a1, 4(t5)\n\t"
        "addi           t5, t5, 8\n\t"

        "vle32.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 16\n\t"

        "vmaqa.vx       v16, a1, v4\n\t"
        "lw             a0, 0(t5)\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 7b\n\t"

        // m4n4k1
        "8:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 9f\n\t"       // if k1 == 0, jump to end kernel_m4n4

        "vmaqa.vx       v16, a0, v1\n\t"

        "addi           %[input_ptr], %[input_ptr], 16\n\t"  // ********************

        // end kernel_m8n4
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -16\n\t"  // pb -= 4

        // 后处理
        "li             t6, 4\n\t"

        "lw             a0, 0(%[mult_ptr])\n\t"
        "lw             a1, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"  // set vl = 4
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"  // set vl = 4
        "vnclip.wi	    v16, v1, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "addi           t2, t2, 4\n\t"

        // m4n_tail
        "10:\n\t"
        "andi           t1, %[n], 3\n\t"        // t1 = n & 3u (n_tail)
        "beqz           t1, 14f\n\t"            // if n_tail==0, jump to end kernel_m4
        "vsetvli        zero, t1, e32, m1\n\t"  // set vl = n_tail
        "slli           t6, t1, 2\n\t"          // t6 = 4 * n_tail

        // init out_tmp = bias
        "lw             t4, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        // pre-load pa(kernel_data)
        "lw             a0, 0(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 12f\n\t"      // if k2 == 0, jump to m8n_tail k1

        // m8n_tailk2
        "11:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lw             a1, 4(t5)\n\t"
        "addi           t5, t5, 8\n\t"

        "vle32.v        v1, (%[input_ptr])\n\t"
        "add            %[input_ptr], %[input_ptr], t6\n\t"

        "vmaqa.vx       v16, a1, v4\n\t"
        "lw             a0, 0(t5)\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 11b\n\t"

        // m2n_tailk1
        "12:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 13f\n\t"      // if k1 == 0, jump to end kernel_m8n_tail

        "vmaqa.vx       v16, a0, v1\n\t"

        "add            %[input_ptr], %[input_ptr], t6\n\t"  // ********************

        // end kernel_m4n_tail
        "13:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "sub            %[input_ptr], %[input_ptr], t6\n\t"  // pb -= n_tail

        // 后处理
        "lw             a0, 0(%[mult_ptr])\n\t"
        "lw             a1, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"  // set vl = n_tail
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"  // set vl = n_tail
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"  // set vl = n_tail
        "vnclip.wi	    v16, v1, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            t2, t2, t1\n\t"

        // ending
        "14:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias),
        [mult_ptr] "+r"(mult), [shift_ptr] "+r"(shift)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [out_zp] "r"(out_zp)
        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v1", "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19",
        // We use these general-purpose registers.
        "a0", "a1", "t1", "t2", "t4", "t5", "t6");
}

// m8n8 --> m8n4 --> m8n2 --> m8n1
// 需要修改 reorder_input
static inline void kernel_m8n8_int8_2(int8_t *dst, int8_t *sa, int8_t *sb, int m, int k, int n,
                                      int32_t *bias, int32_t out_zp, int32_t *mult, int32_t *shift)
{
    asm volatile(
        "srai           t0, %[m], 3\n\t"  // t0 = m8
        "beqz           t0, 19f\n\t"

        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        "vle32.v        v8, (%[mult_ptr])\n\t"
        "vle32.v        v10, (%[shift_ptr])\n\t"
        "vxor.vi        v10, v10, -1\n\t"

        // m8
        "1:\n\t"
        "srai           t1, %[n], 3\n\t"        // t1 = n8
        "mv             t2, %[output_ptr]\n\t"  // init output addr
        "mv             t3, %[input_ptr]\n\t"   // t3 hold input data start addr

        "beqz           t1, 6f\n\t"  // if n8==0, jump to m8n4
        // m8n8
        "2:\n\t"
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        // init out_tmp = bias
        "vle32.v        v16, (%[bias_ptr])\n\t"
        "vmv.v.v        v18, v16\n\t"
        "vmv.v.v        v20, v16\n\t"
        "vmv.v.v        v22, v16\n\t"
        "vmv.v.v        v24, v16\n\t"
        "vmv.v.v        v26, v16\n\t"
        "vmv.v.v        v28, v16\n\t"
        "vmv.v.v        v30, v16\n\t"
        // "vle32.v        v18, (%[bias_ptr])\n\t"
        // "vle32.v        v20, (%[bias_ptr])\n\t"
        // "vle32.v        v22, (%[bias_ptr])\n\t"
        // "vle32.v        v24, (%[bias_ptr])\n\t"
        // "vle32.v        v26, (%[bias_ptr])\n\t"
        // "vle32.v        v28, (%[bias_ptr])\n\t"
        // "vle32.v        v30, (%[bias_ptr])\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pa(kernel_data)
        "vle32.v        v2, (t5)\n\t"
        "addi           t5, t5, 32\n\t"

        // pre-load pb (input_data)
        "lwd            a0, a1, 0(t3)\n\t"
        "lwd            a2, a3, 8(t3)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 4f\n\t"       // if k2 == 0, jump to m8n8k1

        // m8n8k2
        "3:\n\t"
        "vle32.v        v4, (t5)\n\t"
        "addi           t5, t5, 32\n\t"

        "vmaqa.vx       v16, a0, v2\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"
        "lwd            a4, a5, 16(t3)\n\t"
        "lwd            a6, a7, 24(t3)\n\t"
        "vmaqa.vx       v20, a2, v2\n\t"
        "vmaqa.vx       v22, a3, v2\n\t"
        "addi           t3, t3, 32\n\t"
        "lwd            a0, a1, 0(t3)\n\t"
        "lwd            a2, a3, 8(t3)\n\t"
        "vmaqa.vx       v24, a4, v2\n\t"
        "vmaqa.vx       v26, a5, v2\n\t"
        "vmaqa.vx       v28, a6, v2\n\t"
        "vmaqa.vx       v30, a7, v2\n\t"

        "vle32.v        v2, (t5)\n\t"
        "addi           t5, t5, 32\n\t"

        "vmaqa.vx       v16, a0, v4\n\t"
        "vmaqa.vx       v18, a1, v4\n\t"
        "lwd            a4, a5, 16(t3)\n\t"
        "lwd            a6, a7, 24(t3)\n\t"
        "vmaqa.vx       v20, a2, v4\n\t"
        "vmaqa.vx       v22, a3, v4\n\t"
        "addi           t3, t3, 32\n\t"  // += 16 elements
        "lwd            a0, a1, 0(t3)\n\t"
        "lwd            a2, a3, 8(t3)\n\t"
        "vmaqa.vx       v24, a4, v4\n\t"
        "vmaqa.vx       v26, a5, v4\n\t"
        "vmaqa.vx       v28, a6, v4\n\t"
        "vmaqa.vx       v30, a7, v4\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 3b\n\t"

        // m8n8k1
        "4:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 5f\n\t"       // if k1 == 0, jump to end kernel_m8n8

        "lwd            a4, a5, 16(t3)\n\t"
        "lwd            a6, a7, 24(t3)\n\t"
        "addi           t3, t3, 32\n\t"
        "vmaqa.vx       v16, a0, v2\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"
        "vmaqa.vx       v20, a2, v2\n\t"
        "vmaqa.vx       v22, a3, v2\n\t"
        "vmaqa.vx       v24, a4, v2\n\t"
        "vmaqa.vx       v26, a5, v2\n\t"
        "vmaqa.vx       v28, a6, v2\n\t"
        "vmaqa.vx       v30, a7, v2\n\t"

        // end kernel_m8n8
        "5:\n\t"

        // 后处理
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        "vle32.v        v8, (%[mult_ptr])\n\t"
        "vle32.v        v10, (%[shift_ptr])\n\t"
        "vxor.vi        v10, v10, -1\n\t"

        "vmulh.vv	    v16, v16, v8\n\t"
        "vssra.vv	    v16, v16, v10\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"  // set vl = 8
        "vnclip.wi	    v0, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"  // set vl = 8
        "vnclip.wi	    v16, v0, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vv	    v18, v18, v8\n\t"
        "vssra.vv	    v18, v18, v10\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v18, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v18, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vv	    v20, v20, v8\n\t"
        "vssra.vv	    v20, v20, v10\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v0, v20, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v20, v0, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vv	    v22, v22, v8\n\t"
        "vssra.vv	    v22, v22, v10\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v22, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v22, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vv	    v24, v24, v8\n\t"
        "vssra.vv	    v24, v24, v10\n\t"
        "vadd.vx        v24, v24, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v0, v24, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v24, v0, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vv	    v26, v26, v8\n\t"
        "vssra.vv	    v26, v26, v10\n\t"
        "vadd.vx        v26, v26, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v26, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v26, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vv	    v28, v28, v8\n\t"
        "vssra.vv	    v28, v28, v10\n\t"
        "vadd.vx        v28, v28, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v0, v28, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v28, v0, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vv	    v30, v30, v8\n\t"
        "vssra.vv	    v30, v30, v10\n\t"
        "vadd.vx        v30, v30, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v30, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v30, v1, 0\n\t"

        "vsse8.v        v16, (t2), %[n]\n\t"
        "addi           t2, t2, 1\n\t"
        "vsse8.v        v18, (t2), %[n]\n\t"
        "addi           t2, t2, 1\n\t"
        "vsse8.v        v20, (t2), %[n]\n\t"
        "addi           t2, t2, 1\n\t"
        "vsse8.v        v22, (t2), %[n]\n\t"
        "addi           t2, t2, 1\n\t"
        "vsse8.v        v24, (t2), %[n]\n\t"
        "addi           t2, t2, 1\n\t"
        "vsse8.v        v26, (t2), %[n]\n\t"
        "addi           t2, t2, 1\n\t"
        "vsse8.v        v28, (t2), %[n]\n\t"
        "addi           t2, t2, 1\n\t"
        "vsse8.v        v30, (t2), %[n]\n\t"
        "addi           t2, t2, 1\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m8n4
        "6:\n\t"
        "andi           t1, %[n], 4\n\t"  // t1 = n & 4u (n4)
        "beqz           t1, 10f\n\t"      // if n4==0, jump to m8n_tail
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        // init out_tmp = bias
        "vle32.v        v16, (%[bias_ptr])\n\t"
        "vmv.v.v        v18, v16\n\t"
        "vmv.v.v        v20, v16\n\t"
        "vmv.v.v        v22, v16\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pa(kernel_data)
        "vle32.v        v2, (t5)\n\t"
        "addi           t5, t5, 32\n\t"

        // pre-load pb (input_data)
        "lwd            a0, a1, 0(t3)\n\t"
        "lwd            a2, a3, 8(t3)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 8f\n\t"       // if k2 == 0, jump to m8n4k1

        // m8n4k2
        "7:\n\t"
        "vle32.v        v4, (t5)\n\t"
        "addi           t5, t5, 32\n\t"

        "vmaqa.vx       v16, a0, v2\n\t"
        "lwd            a4, a5, 16(t3)\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"
        "lwd            a6, a7, 24(t3)\n\t"
        "vmaqa.vx       v20, a2, v2\n\t"
        "vmaqa.vx       v22, a3, v2\n\t"  // 0
        "addi           t3, t3, 32\n\t"

        "vle32.v        v2, (t5)\n\t"
        "addi           t5, t5, 32\n\t"

        "vmaqa.vx       v16, a4, v4\n\t"
        "lwd            a0, a1, 0(t3)\n\t"
        "vmaqa.vx       v18, a5, v4\n\t"
        "lwd            a2, a3, 8(t3)\n\t"
        "vmaqa.vx       v20, a6, v4\n\t"
        "vmaqa.vx       v22, a7, v4\n\t"  // 1

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 7b\n\t"

        // m8n4k1
        "8:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 9f\n\t"       // if k1 == 0, jump to end kernel_m8n4

        "addi           t3, t3, 16\n\t"
        "vmaqa.vx       v16, a0, v2\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"
        "vmaqa.vx       v20, a2, v2\n\t"
        "vmaqa.vx       v22, a3, v2\n\t"

        // end kernel_m8n4
        "9:\n\t"

        // 后处理
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        "vle32.v        v8, (%[mult_ptr])\n\t"
        "vle32.v        v10, (%[shift_ptr])\n\t"
        "vxor.vi        v10, v10, -1\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vv	    v16, v16, v8\n\t"
        "vssra.vv	    v16, v16, v10\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v0, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v16, v0, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vv	    v18, v18, v8\n\t"
        "vssra.vv	    v18, v18, v10\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v18, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v18, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vv	    v20, v20, v8\n\t"
        "vssra.vv	    v20, v20, v10\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v0, v20, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v20, v0, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vv	    v22, v22, v8\n\t"
        "vssra.vv	    v22, v22, v10\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v22, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v22, v1, 0\n\t"

        "vsse8.v        v16, (t2), %[n]\n\t"
        "addi           t2, t2, 1\n\t"
        "vsse8.v        v18, (t2), %[n]\n\t"
        "addi           t2, t2, 1\n\t"
        "vsse8.v        v20, (t2), %[n]\n\t"
        "addi           t2, t2, 1\n\t"
        "vsse8.v        v22, (t2), %[n]\n\t"
        "addi           t2, t2, 1\n\t"

        // m8n2
        "10:\n\t"
        "andi           t1, %[n], 2\n\t"  // t1 = n & 2u
        "beqz           t1, 14f\n\t"      // if n2==0, jump to kernel_m8n1
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8

        // init out_tmp = bias
        "vle32.v        v16, (%[bias_ptr])\n\t"
        "vmv.v.v        v18, v16\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pa(kernel_data)
        "vle32.v        v2, (t5)\n\t"
        "addi           t5, t5, 32\n\t"

        // pre-load pb (input_data)
        "lwd            a0, a1, 0(t3)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 12f\n\t"      // if k2 == 0, jump to m8n_tail k1

        // m8n2k2
        "11:\n\t"
        "vle32.v        v4, (t5)\n\t"
        "addi           t5, t5, 32\n\t"

        "vmaqa.vx       v16, a0, v2\n\t"
        "lwd            a2, a3, 8(t3)\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"  // 0
        "addi           t3, t3, 16\n\t"

        "vle32.v        v2, (t5)\n\t"
        "addi           t5, t5, 32\n\t"

        "vmaqa.vx       v16, a2, v4\n\t"
        "lwd            a0, a1, 0(t3)\n\t"
        "vmaqa.vx       v18, a3, v4\n\t"  // 1

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 11b\n\t"

        // m8n2k1
        "12:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 13f\n\t"      // if k1 == 0, jump to end kernel_m8n_tail

        "addi           t3, t3, 8\n\t"
        "vmaqa.vx       v16, a0, v2\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"

        // end kernel_m8n2
        "13:\n\t"
        // 后处理
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        "vle32.v        v8, (%[mult_ptr])\n\t"
        "vle32.v        v10, (%[shift_ptr])\n\t"
        "vxor.vi        v10, v10, -1\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vv	    v16, v16, v8\n\t"
        "vssra.vv	    v16, v16, v10\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v0, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v16, v0, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vv	    v18, v18, v8\n\t"
        "vssra.vv	    v18, v18, v10\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v18, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v18, v1, 0\n\t"

        "vsse8.v        v16, (t2), %[n]\n\t"
        "addi           t2, t2, 1\n\t"
        "vsse8.v        v18, (t2), %[n]\n\t"
        "addi           t2, t2, 1\n\t"

        // m8n1
        "14:\n\t"
        "andi           t1, %[n], 1\n\t"  // t1 = n & 1u
        "beqz           t1, 18f\n\t"      // if n1==0, jump to kernel_m8
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8

        // init out_tmp = bias
        "vle32.v        v16, (%[bias_ptr])\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pa(kernel_data)
        "vle32.v        v2, (t5)\n\t"
        "addi           t5, t5, 32\n\t"

        // pre-load pb (input_data)
        "lw             a0, 0(t3)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 16f\n\t"      // if k2 == 0, jump to m8n_tail k1

        // m8n1k2
        "15:\n\t"
        "vle32.v        v4, (t5)\n\t"
        "addi           t5, t5, 32\n\t"

        "vmaqa.vx       v16, a0, v2\n\t"
        "lw             a1, 4(t3)\n\t"
        "addi           t3, t3, 8\n\t"

        "vle32.v        v2, (t5)\n\t"
        "addi           t5, t5, 32\n\t"

        "vmaqa.vx       v16, a1, v4\n\t"
        "lw             a0, 0(t3)\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 15b\n\t"

        // m8n1k1
        "16:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 17f\n\t"      // if k1 == 0, jump to end kernel_m8n_tail

        "addi           t3, t3, 4\n\t"
        "vmaqa.vx       v16, a0, v2\n\t"
        // end kernel_m8n1
        "17:\n\t"
        // 后处理
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        "vle32.v        v8, (%[mult_ptr])\n\t"
        "vle32.v        v10, (%[shift_ptr])\n\t"
        "vxor.vi        v10, v10, -1\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vv	    v16, v16, v8\n\t"
        "vssra.vv	    v16, v16, v10\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v0, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v16, v0, 0\n\t"

        "vsse8.v        v16, (t2), %[n]\n\t"
        // "addi           t2, t2, 1\n\t"

        // end kernel_m8
        "18:\n\t"
        "addi           %[bias_ptr], %[bias_ptr], 32\n\t"    // bias_data += 8
        "addi           %[mult_ptr], %[mult_ptr], 32\n\t"    // mult_ptr += 8
        "addi           %[shift_ptr], %[shift_ptr], 32\n\t"  // shift_ptr += 8
        "slli           t6, %[k], 3\n\t"
        "add            %[kernel_ptr], %[kernel_ptr], t6\n\t"  // kernel_data += 8 * k
        "slli           t6, %[n], 3\n\t"
        "add            %[output_ptr], %[output_ptr], t6\n\t"  // output_data += 8 * n

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        // ending
        "19:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias),
        [mult_ptr] "+r"(mult), [shift_ptr] "+r"(shift)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [out_zp] "r"(out_zp)
        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v0", "v1", "v2", "v3", "v4", "v5", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19",
        "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
        // We use these general-purpose registers.
        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
}

static inline void kernel_m8n12_int8(int8_t *dst, int8_t *sa, int8_t *sb, int m, int k, int n,
                                     int32_t *bias, int32_t out_zp, int32_t *mult, int32_t *shift)
{
    asm volatile(
        "srai           t0, %[m], 3\n\t"  // t0 = m8
        "beqz           t0, 19f\n\t"

        // m8
        "1:\n\t"
        "mv             t1, %[n]\n\t"
        "li             t6, 12\n\t"
        "mv             t2, %[output_ptr]\n\t"  // init output addr
        "mv             t3, %[input_ptr]\n\t"   // t3 hold input data start addr

        "blt            t1, t6, 6f\n\t"  // if n < 12, jump to m8n8

        // m8n12
        "2:\n\t"
        "li             t6, 4\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v8, t4\n\t"
        "vmv.v.x        v9, t4\n\t"
        "vmv.v.x        v10, t4\n\t"
        "vmv.v.x        v11, t5\n\t"
        "vmv.v.x        v12, t5\n\t"
        "vmv.v.x        v13, t5\n\t"
        "lwd            t4, t5, 8(%[bias_ptr])\n\t"  // bias_ptr[2]/[3]
        "vmv.v.x        v14, t4\n\t"
        "vmv.v.x        v15, t4\n\t"
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v17, t5\n\t"
        "vmv.v.x        v18, t5\n\t"
        "vmv.v.x        v19, t5\n\t"
        "lwd            t4, t5, 16(%[bias_ptr])\n\t"  // bias_ptr[4]/[5]
        "vmv.v.x        v20, t4\n\t"
        "vmv.v.x        v21, t4\n\t"
        "vmv.v.x        v22, t4\n\t"
        "vmv.v.x        v23, t5\n\t"
        "vmv.v.x        v24, t5\n\t"
        "vmv.v.x        v25, t5\n\t"
        "lwd            t4, t5, 24(%[bias_ptr])\n\t"  // bias_ptr[6]/[7]
        "vmv.v.x        v26, t4\n\t"
        "vmv.v.x        v27, t4\n\t"
        "vmv.v.x        v28, t4\n\t"
        "vmv.v.x        v29, t5\n\t"
        "vmv.v.x        v30, t5\n\t"
        "vmv.v.x        v31, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (t3)\n\t"
        "addi           t3, t3, 16\n\t"
        "vle32.v        v2, (t3)\n\t"
        "addi           t3, t3, 16\n\t"
        "vle32.v        v3, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 4f\n\t"       // if k2 == 0, jump to m8n12k1

        // m8n12k2
        "3:\n\t"
        "vle32.v        v4, (t3)\n\t"
        "addi           t3, t3, 16\n\t"
        "vle32.v        v5, (t3)\n\t"
        "addi           t3, t3, 16\n\t"
        "vle32.v        v6, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        "vmaqa.vx       v8, a0, v1\n\t"
        "vmaqa.vx       v9, a0, v2\n\t"
        "vmaqa.vx       v10, a0, v3\n\t"
        "vmaqa.vx       v11, a1, v1\n\t"
        "vmaqa.vx       v12, a1, v2\n\t"
        "vmaqa.vx       v13, a1, v3\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v14, a2, v1\n\t"
        "vmaqa.vx       v15, a2, v2\n\t"
        "vmaqa.vx       v16, a2, v3\n\t"
        "vmaqa.vx       v17, a3, v1\n\t"
        "vmaqa.vx       v18, a3, v2\n\t"
        "vmaqa.vx       v19, a3, v3\n\t"
        "addi           t5, t5, 32\n\t"

        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v20, a4, v1\n\t"
        "vmaqa.vx       v21, a4, v2\n\t"
        "vmaqa.vx       v22, a4, v3\n\t"
        "vmaqa.vx       v23, a5, v1\n\t"
        "vmaqa.vx       v24, a5, v2\n\t"
        "vmaqa.vx       v25, a5, v3\n\t"
        "vmaqa.vx       v26, a6, v1\n\t"
        "vmaqa.vx       v27, a6, v2\n\t"
        "vmaqa.vx       v28, a6, v3\n\t"
        "vmaqa.vx       v29, a7, v1\n\t"
        "vmaqa.vx       v30, a7, v2\n\t"
        "vmaqa.vx       v31, a7, v3\n\t"

        "vle32.v        v1, (t3)\n\t"
        "addi           t3, t3, 16\n\t"
        "vle32.v        v2, (t3)\n\t"
        "addi           t3, t3, 16\n\t"
        "vle32.v        v3, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        "vmaqa.vx       v8, a0, v4\n\t"
        "vmaqa.vx       v9, a0, v5\n\t"
        "vmaqa.vx       v10, a0, v6\n\t"
        "vmaqa.vx       v11, a1, v4\n\t"
        "vmaqa.vx       v12, a1, v5\n\t"
        "vmaqa.vx       v13, a1, v6\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v14, a2, v4\n\t"
        "vmaqa.vx       v15, a2, v5\n\t"
        "vmaqa.vx       v16, a2, v6\n\t"
        "vmaqa.vx       v17, a3, v4\n\t"
        "vmaqa.vx       v18, a3, v5\n\t"
        "vmaqa.vx       v19, a3, v6\n\t"
        "addi           t5, t5, 32\n\t"

        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v20, a4, v4\n\t"
        "vmaqa.vx       v21, a4, v5\n\t"
        "vmaqa.vx       v22, a4, v6\n\t"
        "vmaqa.vx       v23, a5, v4\n\t"
        "vmaqa.vx       v24, a5, v5\n\t"
        "vmaqa.vx       v25, a5, v6\n\t"
        "vmaqa.vx       v26, a6, v4\n\t"
        "vmaqa.vx       v27, a6, v5\n\t"
        "vmaqa.vx       v28, a6, v6\n\t"
        "vmaqa.vx       v29, a7, v4\n\t"
        "vmaqa.vx       v30, a7, v5\n\t"
        "vmaqa.vx       v31, a7, v6\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 3b\n\t"

        // m8m12k1
        "4:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 5f\n\t"       // if k1 == 0, jump to end kernel_m8n12

        "lwd            a4, a5, 16(t5)\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v8, a0, v1\n\t"
        "vmaqa.vx       v9, a0, v2\n\t"
        "vmaqa.vx       v10, a0, v3\n\t"
        "vmaqa.vx       v11, a1, v1\n\t"
        "vmaqa.vx       v12, a1, v2\n\t"
        "vmaqa.vx       v13, a1, v3\n\t"
        "vmaqa.vx       v14, a2, v1\n\t"
        "vmaqa.vx       v15, a2, v2\n\t"
        "vmaqa.vx       v16, a2, v3\n\t"
        "vmaqa.vx       v17, a3, v1\n\t"
        "vmaqa.vx       v18, a3, v2\n\t"
        "vmaqa.vx       v19, a3, v3\n\t"
        "vmaqa.vx       v20, a4, v1\n\t"
        "vmaqa.vx       v21, a4, v2\n\t"
        "vmaqa.vx       v22, a4, v3\n\t"
        "vmaqa.vx       v23, a5, v1\n\t"
        "vmaqa.vx       v24, a5, v2\n\t"
        "vmaqa.vx       v25, a5, v3\n\t"
        "vmaqa.vx       v26, a6, v1\n\t"
        "vmaqa.vx       v27, a6, v2\n\t"
        "vmaqa.vx       v28, a6, v3\n\t"
        "vmaqa.vx       v29, a7, v1\n\t"
        "vmaqa.vx       v30, a7, v2\n\t"
        "vmaqa.vx       v31, a7, v3\n\t"

        "addi           t3, t3, 48\n\t"  // ********************

        // end kernel_m8n12
        "5:\n\t"

        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           t3, t3, -48\n\t"  // pb -= 8
        // 后处理
        "li             t6, 4\n\t"
        "lwd            a0, a2, 0(%[mult_ptr])\n\t"
        "lwd            a1, a3, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        "vmulh.vx	    v8, v8, a0\n\t"
        "vmulh.vx	    v9, v9, a0\n\t"
        "vmulh.vx	    v10, v10, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v8, v8, a1\n\t"
        "vssra.vx	    v9, v9, a1\n\t"
        "vssra.vx	    v10, v10, a1\n\t"
        "vadd.vx        v8, v8, %[out_zp]\n\t"
        "vadd.vx        v9, v9, %[out_zp]\n\t"
        "vadd.vx        v10, v10, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"  // set vl = 4
        "vnclip.wi	    v1, v8, 0\n\t"
        "vnclip.wi	    v2, v9, 0\n\t"
        "vnclip.wi	    v3, v10, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"  // set vl = 4
        "vnclip.wi	    v8, v1, 0\n\t"
        "vnclip.wi	    v9, v2, 0\n\t"
        "vnclip.wi	    v10, v3, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v11, v11, a2\n\t"
        "vmulh.vx	    v12, v12, a2\n\t"
        "vmulh.vx	    v13, v13, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v11, v11, a3\n\t"
        "vssra.vx	    v12, v12, a3\n\t"
        "vssra.vx	    v13, v13, a3\n\t"
        "vadd.vx        v11, v11, %[out_zp]\n\t"
        "vadd.vx        v12, v12, %[out_zp]\n\t"
        "vadd.vx        v13, v13, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v11, 0\n\t"
        "vnclip.wi	    v5, v12, 0\n\t"
        "vnclip.wi	    v6, v13, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v11, v4, 0\n\t"
        "vnclip.wi	    v12, v5, 0\n\t"
        "vnclip.wi	    v13, v6, 0\n\t"

        "lwd            a0, a2, 8(%[mult_ptr])\n\t"
        "lwd            a1, a3, 8(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v14, v14, a0\n\t"
        "vmulh.vx	    v15, v15, a0\n\t"
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v14, v14, a1\n\t"
        "vssra.vx	    v15, v15, a1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v14, v14, %[out_zp]\n\t"
        "vadd.vx        v15, v15, %[out_zp]\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v1, v14, 0\n\t"
        "vnclip.wi	    v2, v15, 0\n\t"
        "vnclip.wi	    v3, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v14, v1, 0\n\t"
        "vnclip.wi	    v15, v2, 0\n\t"
        "vnclip.wi	    v16, v3, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v17, v17, a2\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "vmulh.vx	    v19, v19, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v17, v17, a3\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vssra.vx	    v19, v19, a3\n\t"
        "vadd.vx        v17, v17, %[out_zp]\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vadd.vx        v19, v19, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v17, 0\n\t"
        "vnclip.wi	    v5, v18, 0\n\t"
        "vnclip.wi	    v6, v19, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v17, v4, 0\n\t"
        "vnclip.wi	    v18, v5, 0\n\t"
        "vnclip.wi	    v19, v6, 0\n\t"

        "lwd            a0, a2, 16(%[mult_ptr])\n\t"
        "lwd            a1, a3, 16(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v20, v20, a0\n\t"
        "vmulh.vx	    v21, v21, a0\n\t"
        "vmulh.vx	    v22, v22, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v20, v20, a1\n\t"
        "vssra.vx	    v21, v21, a1\n\t"
        "vssra.vx	    v22, v22, a1\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vadd.vx        v21, v21, %[out_zp]\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v1, v20, 0\n\t"
        "vnclip.wi	    v2, v21, 0\n\t"
        "vnclip.wi	    v3, v22, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v20, v1, 0\n\t"
        "vnclip.wi	    v21, v2, 0\n\t"
        "vnclip.wi	    v22, v3, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v23, v23, a2\n\t"
        "vmulh.vx	    v24, v24, a2\n\t"
        "vmulh.vx	    v25, v25, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v23, v23, a3\n\t"
        "vssra.vx	    v24, v24, a3\n\t"
        "vssra.vx	    v25, v25, a3\n\t"
        "vadd.vx        v23, v23, %[out_zp]\n\t"
        "vadd.vx        v24, v24, %[out_zp]\n\t"
        "vadd.vx        v25, v25, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v23, 0\n\t"
        "vnclip.wi	    v5, v24, 0\n\t"
        "vnclip.wi	    v6, v25, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v23, v4, 0\n\t"
        "vnclip.wi	    v24, v5, 0\n\t"
        "vnclip.wi	    v25, v6, 0\n\t"

        "lwd            a0, a2, 24(%[mult_ptr])\n\t"
        "lwd            a1, a3, 24(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v26, v26, a0\n\t"
        "vmulh.vx	    v27, v27, a0\n\t"
        "vmulh.vx	    v28, v28, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v26, v26, a1\n\t"
        "vssra.vx	    v27, v27, a1\n\t"
        "vssra.vx	    v28, v28, a1\n\t"
        "vadd.vx        v26, v26, %[out_zp]\n\t"
        "vadd.vx        v27, v27, %[out_zp]\n\t"
        "vadd.vx        v28, v28, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v1, v26, 0\n\t"
        "vnclip.wi	    v2, v27, 0\n\t"
        "vnclip.wi	    v3, v28, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v26, v1, 0\n\t"
        "vnclip.wi	    v27, v2, 0\n\t"
        "vnclip.wi	    v28, v3, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v29, v29, a2\n\t"
        "vmulh.vx	    v30, v30, a2\n\t"
        "vmulh.vx	    v31, v31, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v29, v29, a3\n\t"
        "vssra.vx	    v30, v30, a3\n\t"
        "vssra.vx	    v31, v31, a3\n\t"
        "vadd.vx        v29, v29, %[out_zp]\n\t"
        "vadd.vx        v30, v30, %[out_zp]\n\t"
        "vadd.vx        v31, v31, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v29, 0\n\t"
        "vnclip.wi	    v5, v30, 0\n\t"
        "vnclip.wi	    v6, v31, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v29, v4, 0\n\t"
        "vnclip.wi	    v30, v5, 0\n\t"
        "vnclip.wi	    v31, v6, 0\n\t"

        "addi           t6, %[n], -8\n\t"
        "mv             a0, t2\n\t"
        "vse8.v         v8, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v9, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v10, (a0)\n\t"
        "add            a0, a0, t6\n\t"
        "vse8.v         v11, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v12, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v13, (a0)\n\t"
        "add            a0, a0, t6\n\t"
        "vse8.v         v14, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v15, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, t6\n\t"
        "vse8.v         v17, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v18, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v19, (a0)\n\t"
        "add            a0, a0, t6\n\t"
        "vse8.v         v20, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v21, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v22, (a0)\n\t"
        "add            a0, a0, t6\n\t"
        "vse8.v         v23, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v24, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v25, (a0)\n\t"
        "add            a0, a0, t6\n\t"
        "vse8.v         v26, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v27, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v28, (a0)\n\t"
        "add            a0, a0, t6\n\t"
        "vse8.v         v29, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v30, (a0)\n\t"
        "addi           a0, a0, 4\n\t"
        "vse8.v         v31, (a0)\n\t"

        "addi           t2, t2, 12\n\t"

        "li             t6, 12\n\t"
        "addi           t1, t1, -12\n\t"
        "bge            t1, t6, 2b\n\t"

        // m8n8
        "6:\n\t"
        "li             t6, 8\n\t"
        "blt            t1, t6, 10f\n\t"
        "addi           t1, t1, -8\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v18, t5\n\t"
        "lwd            t4, t5, 8(%[bias_ptr])\n\t"  // bias_ptr[2]/[3]
        "vmv.v.x        v20, t4\n\t"
        "vmv.v.x        v22, t5\n\t"
        "lwd            t4, t5, 16(%[bias_ptr])\n\t"  // bias_ptr[4]/[5]
        "vmv.v.x        v24, t4\n\t"
        "vmv.v.x        v26, t5\n\t"
        "lwd            t4, t5, 24(%[bias_ptr])\n\t"  // bias_ptr[6]/[7]
        "vmv.v.x        v28, t4\n\t"
        "vmv.v.x        v30, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v2, (t3)\n\t"
        "addi           t3, t3, 32\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 8f\n\t"       // if k2 == 0, jump to m8n8k1

        // m8n8k2
        "7:\n\t"
        "vle32.v        v4, (t3)\n\t"
        "addi           t3, t3, 32\n\t"

        "vmaqa.vx       v16, a0, v2\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v20, a2, v2\n\t"
        "vmaqa.vx       v22, a3, v2\n\t"
        "addi           t5, t5, 32\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v24, a4, v2\n\t"
        "vmaqa.vx       v26, a5, v2\n\t"
        "vmaqa.vx       v28, a6, v2\n\t"
        "vmaqa.vx       v30, a7, v2\n\t"

        "vle32.v        v2, (t3)\n\t"
        "addi           t3, t3, 32\n\t"

        "vmaqa.vx       v16, a0, v4\n\t"
        "vmaqa.vx       v18, a1, v4\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v20, a2, v4\n\t"
        "vmaqa.vx       v22, a3, v4\n\t"
        "addi           t5, t5, 32\n\t"  // += 16 elements
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v24, a4, v4\n\t"
        "vmaqa.vx       v26, a5, v4\n\t"
        "vmaqa.vx       v28, a6, v4\n\t"
        "vmaqa.vx       v30, a7, v4\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 7b\n\t"

        // m8n8k1
        "8:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 9f\n\t"       // if k1 == 0, jump to end kernel_m8n8

        "lwd            a4, a5, 16(t5)\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v16, a0, v2\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"
        "vmaqa.vx       v20, a2, v2\n\t"
        "vmaqa.vx       v22, a3, v2\n\t"
        "vmaqa.vx       v24, a4, v2\n\t"
        "vmaqa.vx       v26, a5, v2\n\t"
        "vmaqa.vx       v28, a6, v2\n\t"
        "vmaqa.vx       v30, a7, v2\n\t"

        "addi           t3, t3, 32\n\t"  // ********************

        // end kernel_m8n8
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           t3, t3, -32\n\t"  // pb -= 8

        // 后处理
        "li             t6, 8\n\t"

        "lwd            a0, a2, 0(%[mult_ptr])\n\t"
        "lwd            a1, a3, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"  // set vl = 8
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"  // set vl = 8
        "vnclip.wi	    v16, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "lwd            a0, a2, 8(%[mult_ptr])\n\t"
        "lwd            a1, a3, 8(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v20, v20, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v20, v20, a1\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v20, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v20, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v22, v22, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v22, v22, a3\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v22, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v22, v4, 0\n\t"

        "lwd            a0, a2, 16(%[mult_ptr])\n\t"
        "lwd            a1, a3, 16(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v24, v24, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v24, v24, a1\n\t"
        "vadd.vx        v24, v24, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v24, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v24, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v26, v26, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v26, v26, a3\n\t"
        "vadd.vx        v26, v26, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v26, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v26, v4, 0\n\t"

        "lwd            a0, a2, 24(%[mult_ptr])\n\t"
        "lwd            a1, a3, 24(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v28, v28, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v28, v28, a1\n\t"
        "vadd.vx        v28, v28, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v1, v28, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v28, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m2\n\t"
        "vmulh.vx	    v30, v30, a2\n\t"
        "not            a3, a3\n\t"
        // "addi           a3, a3, -1\n\t"
        "vssra.vx	    v30, v30, a3\n\t"
        "vadd.vx        v30, v30, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vnclip.wi	    v4, v30, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"
        "vnclip.wi	    v30, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v20, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v22, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v24, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v26, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v28, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v30, (a0)\n\t"
        "addi           t2, t2, 8\n\t"

        // m8n4
        "10:\n\t"
        "li             t6, 4\n\t"
        "blt            t1, t6, 14f\n\t"  // if n4==0, jump to m8n_tail
        "addi           t1, t1, -4\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v18, t5\n\t"
        "lwd            t4, t5, 8(%[bias_ptr])\n\t"  // bias_ptr[2]/[3]
        "vmv.v.x        v20, t4\n\t"
        "vmv.v.x        v22, t5\n\t"
        "lwd            t4, t5, 16(%[bias_ptr])\n\t"  // bias_ptr[4]/[5]
        "vmv.v.x        v24, t4\n\t"
        "vmv.v.x        v26, t5\n\t"
        "lwd            t4, t5, 24(%[bias_ptr])\n\t"  // bias_ptr[6]/[7]
        "vmv.v.x        v28, t4\n\t"
        "vmv.v.x        v30, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 12f\n\t"      // if k2 == 0, jump to m8n4k1

        // m8n4k2
        "11:\n\t"
        "vle32.v        v4, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "addi           t5, t5, 32\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "vmaqa.vx       v24, a4, v1\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "vmaqa.vx       v26, a5, v1\n\t"
        "vmaqa.vx       v28, a6, v1\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v30, a7, v1\n\t"  // 0

        "vle32.v        v1, (t3)\n\t"
        "addi           t3, t3, 16\n\t"

        "vmaqa.vx       v16, a0, v4\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v4\n\t"
        "vmaqa.vx       v20, a2, v4\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v4\n\t"
        "addi           t5, t5, 32\n\t"  // += 16 elements

        "vmaqa.vx       v24, a4, v4\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "vmaqa.vx       v26, a5, v4\n\t"
        "vmaqa.vx       v28, a6, v4\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v30, a7, v4\n\t"  // 1

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 11b\n\t"

        // m8n4k1
        "12:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 13f\n\t"      // if k1 == 0, jump to end kernel_m8n4

        "vmaqa.vx       v16, a0, v1\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "vmaqa.vx       v24, a4, v1\n\t"
        "vmaqa.vx       v26, a5, v1\n\t"
        "vmaqa.vx       v28, a6, v1\n\t"
        "vmaqa.vx       v30, a7, v1\n\t"

        "addi           t3, t3, 16\n\t"  // ********************

        // end kernel_m8n4
        "13:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           t3, t3, -16\n\t"  // pb -= 4

        // 后处理
        "li             t6, 4\n\t"

        "lwd            a0, a2, 0(%[mult_ptr])\n\t"
        "lwd            a1, a3, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"  // set vl = 4
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"  // set vl = 4
        "vnclip.wi	    v16, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "lwd            a0, a2, 8(%[mult_ptr])\n\t"
        "lwd            a1, a3, 8(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v20, v20, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v20, v20, a1\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v1, v20, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v20, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v22, v22, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v22, v22, a3\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v22, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v22, v4, 0\n\t"

        "lwd            a0, a2, 16(%[mult_ptr])\n\t"
        "lwd            a1, a3, 16(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v24, v24, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v24, v24, a1\n\t"
        "vadd.vx        v24, v24, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v1, v24, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v24, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v26, v26, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v26, v26, a3\n\t"
        "vadd.vx        v26, v26, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v26, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v26, v4, 0\n\t"

        "lwd            a0, a2, 24(%[mult_ptr])\n\t"
        "lwd            a1, a3, 24(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v28, v28, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v28, v28, a1\n\t"
        "vadd.vx        v28, v28, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v1, v28, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v28, v1, 0\n\t"

        "vsetvli        zero, t6, e32, m1\n\t"
        "vmulh.vx	    v30, v30, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v30, v30, a3\n\t"
        "vadd.vx        v30, v30, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"
        "vnclip.wi	    v4, v30, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"
        "vnclip.wi	    v30, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v20, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v22, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v24, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v26, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v28, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v30, (a0)\n\t"
        "addi           t2, t2, 4\n\t"

        // m8n_tail
        "14:\n\t"
        "beqz           t1, 18f\n\t"            // if n_tail==0, jump to end kernel_m8
        "vsetvli        zero, t1, e32, m1\n\t"  // set vl = n_tail
        "slli           t6, t1, 2\n\t"          // t6 = 4 * n_tail

        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v18, t5\n\t"
        "lwd            t4, t5, 8(%[bias_ptr])\n\t"  // bias_ptr[2]/[3]
        "vmv.v.x        v20, t4\n\t"
        "vmv.v.x        v22, t5\n\t"
        "lwd            t4, t5, 16(%[bias_ptr])\n\t"  // bias_ptr[4]/[5]
        "vmv.v.x        v24, t4\n\t"
        "vmv.v.x        v26, t5\n\t"
        "lwd            t4, t5, 24(%[bias_ptr])\n\t"  // bias_ptr[6]/[7]
        "vmv.v.x        v28, t4\n\t"
        "vmv.v.x        v30, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (t3)\n\t"
        "add            t3, t3, t6\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 16f\n\t"      // if k2 == 0, jump to m8n_tail k1

        // m8n_tailk2
        "15:\n\t"
        "vle32.v        v4, (t3)\n\t"
        "add            t3, t3, t6\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "addi           t5, t5, 32\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "vmaqa.vx       v24, a4, v1\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "vmaqa.vx       v26, a5, v1\n\t"
        "vmaqa.vx       v28, a6, v1\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v30, a7, v1\n\t"  // 0

        "vle32.v        v1, (t3)\n\t"
        "add            t3, t3, t6\n\t"

        "vmaqa.vx       v16, a0, v4\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v4\n\t"
        "vmaqa.vx       v20, a2, v4\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v4\n\t"
        "addi           t5, t5, 32\n\t"  // += 16 elements

        "vmaqa.vx       v24, a4, v4\n\t"
        "lwd            a0, a1, 0(t5)\n\t"
        "vmaqa.vx       v26, a5, v4\n\t"
        "vmaqa.vx       v28, a6, v4\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v30, a7, v4\n\t"  // 1

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 15b\n\t"

        // m8n_tailk1
        "16:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 17f\n\t"      // if k1 == 0, jump to end kernel_m8n_tail

        "vmaqa.vx       v16, a0, v1\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "vmaqa.vx       v24, a4, v1\n\t"
        "vmaqa.vx       v26, a5, v1\n\t"
        "vmaqa.vx       v28, a6, v1\n\t"
        "vmaqa.vx       v30, a7, v1\n\t"

        "add            t3, t3, t6\n\t"  // ********************

        // end kernel_m8n_tail
        "17:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "sub            t3, t3, t6\n\t"  // pb -= n_tail

        // 后处理
        "lwd            a0, a2, 0(%[mult_ptr])\n\t"
        "lwd            a1, a3, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"  // set vl = n_tail
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"  // set vl = n_tail
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"  // set vl = n_tail
        "vnclip.wi	    v16, v1, 0\n\t"

        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v18, v18, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v18, v18, a3\n\t"
        "vadd.vx        v18, v18, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v18, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v18, v4, 0\n\t"

        "lwd            a0, a2, 8(%[mult_ptr])\n\t"
        "lwd            a1, a3, 8(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v20, v20, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v20, v20, a1\n\t"
        "vadd.vx        v20, v20, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v1, v20, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v20, v1, 0\n\t"

        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v22, v22, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v22, v22, a3\n\t"
        "vadd.vx        v22, v22, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v22, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v22, v4, 0\n\t"

        "lwd            a0, a2, 16(%[mult_ptr])\n\t"
        "lwd            a1, a3, 16(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v24, v24, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v24, v24, a1\n\t"
        "vadd.vx        v24, v24, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v1, v24, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v24, v1, 0\n\t"

        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v26, v26, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v26, v26, a3\n\t"
        "vadd.vx        v26, v26, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v26, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v26, v4, 0\n\t"

        "lwd            a0, a2, 24(%[mult_ptr])\n\t"
        "lwd            a1, a3, 24(%[shift_ptr])\n\t"
        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v28, v28, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v28, v28, a1\n\t"
        "vadd.vx        v28, v28, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v1, v28, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v28, v1, 0\n\t"

        "vsetvli        zero, t1, e32, m1\n\t"
        "vmulh.vx	    v30, v30, a2\n\t"
        "not            a3, a3\n\t"
        "vssra.vx	    v30, v30, a3\n\t"
        "vadd.vx        v30, v30, %[out_zp]\n\t"
        "vsetvli        zero, t1, e16, mf2\n\t"
        "vnclip.wi	    v4, v30, 0\n\t"
        "vsetvli        zero, t1, e8, mf4\n\t"
        "vnclip.wi	    v30, v4, 0\n\t"

        "mv             a0, t2\n\t"
        "vse8.v         v16, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v18, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v20, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v22, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v24, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v26, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v28, (a0)\n\t"
        "add            a0, a0, %[n]\n\t"
        "vse8.v         v30, (a0)\n\t"
        "add            t2, t2, t1\n\t"

        // end kernel_m8
        "18:\n\t"
        "addi           %[bias_ptr], %[bias_ptr], 32\n\t"    // bias_data += 8
        "addi           %[mult_ptr], %[mult_ptr], 32\n\t"    // mult_ptr += 8
        "addi           %[shift_ptr], %[shift_ptr], 32\n\t"  // shift_ptr += 8
        "slli           t6, %[k], 3\n\t"
        "add            %[kernel_ptr], %[kernel_ptr], t6\n\t"  // kernel_data += 8 * k
        "slli           t6, %[n], 3\n\t"
        "add            %[output_ptr], %[output_ptr], t6\n\t"  // output_data += 8 * n

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        // ending
        "19:\n\t"

        :
        // Outputs.
        [kernel_ptr] "+r"(sa), [input_ptr] "+r"(sb), [output_ptr] "+r"(dst), [bias_ptr] "+r"(bias),
        [mult_ptr] "+r"(mult), [shift_ptr] "+r"(shift)
        :
        // Inputs.
        [m] "r"(m), [k] "r"(k), [n] "r"(n), [out_zp] "r"(out_zp)
        :
        // Clobbers.
        "cc", "memory",
        // We use these Vector registers.
        "v1", "v2", "v3", "v4", "v5", "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
        "v29", "v30", "v31",
        // We use these general-purpose registers.
        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
}

void shl_c908_gemm_8x8_int8(int8_t *dst, const int8_t *sa, const int8_t *sb, int32_t *bias, int m,
                            int k, int n, int ldc, int32_t out_zp, int32_t *mult, int32_t *shift)
{
    int8_t *kernel_ptr = (int8_t *)sa;
    int8_t *input_ptr = (int8_t *)sb;
    int8_t *output_ptr = dst;
    // please use fuse_zp2bias option in hhb, thus bias_data wont be NULL
    int32_t *bias_ptr = bias;

    int tail = m % 8;
    if (m > 8) {
        kernel_m8n8_int8_1(output_ptr, kernel_ptr, input_ptr, m, k, n, bias_ptr, out_zp, mult,
                           shift);
        output_ptr += (m - tail) * n;
        kernel_ptr += (m - tail) * k;
        bias_ptr += (m - tail);
        mult += (m - tail);
        shift += (m - tail);
    }
    if (tail & 4) {
        kernel_m4n8_int8_1(output_ptr, kernel_ptr, input_ptr, m, k, n, bias_ptr, out_zp, mult,
                           shift);
        output_ptr += 4 * n;
        kernel_ptr += 4 * k;
        bias_ptr += 4;
        mult += 4;
        shift += 4;
    }
    if (tail & 2) {
        kernel_m2n8_int8_1(output_ptr, kernel_ptr, input_ptr, m, k, n, bias_ptr, out_zp, mult,
                           shift);
        output_ptr += 2 * n;
        kernel_ptr += 2 * k;
        bias_ptr += 2;
        mult += 2;
        shift += 2;
    }
    if (tail & 1) {
        kernel_m1n8_int8_1(output_ptr, kernel_ptr, input_ptr, m, k, n, bias_ptr, out_zp, mult,
                           shift);
        output_ptr += 1 * n;
        kernel_ptr += 1 * k;
        bias_ptr += 1;
        mult += 1;
        shift += 1;
    }
}
