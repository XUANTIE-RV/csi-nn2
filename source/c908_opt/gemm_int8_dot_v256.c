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

#include "shl_c908.h"

/*************************************************************
 * note: VLEN = 256
 * input matrix and kernel matrix have been reordered
 *************************************************************/

// 如果使能xtheadc, 可用lwd指令
static inline void kernel_m8n16_int8_v256(int8_t *dst, int8_t *sa, int8_t *sb, int m, int k, int n,
                                          int32_t *bias, int32_t out_zp, int32_t *mult,
                                          int32_t *shift)
{
    asm volatile(
        "srai           t0, %[m], 3\n\t"  // t0 = m8
        "beqz           t0, 15f\n\t"

        // m8
        "1:\n\t"
        "srai           t1, %[n], 4\n\t"        // t1 = n16
        "mv             t2, %[output_ptr]\n\t"  // init output addr
        "mv             t3, %[input_ptr]\n\t"   // t3 hold input data start addr

        "beqz           t1, 6f\n\t"  // if n16==0, jump to m8n8
        // m8n8
        "2:\n\t"
        "li             t6, 16\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 16
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
        "addi           t3, t3, 64\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 4f\n\t"       // if k2 == 0, jump to m8n8k1

        // m8n16k2
        "3:\n\t"
        "vle32.v        v4, (t3)\n\t"
        "addi           t3, t3, 64\n\t"

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
        "addi           t3, t3, 64\n\t"

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

        // m8n16k1
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

        "addi           t3, t3, 64\n\t"  // ********************

        // end kernel_m8n8
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           t3, t3, -64\n\t"  // pb -= 8

        // 后处理
        "li             t6, 16\n\t"

        "lwd            a0, a2, 0(%[mult_ptr])\n\t"
        "lwd            a1, a3, 0(%[shift_ptr])\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 16
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        // "addi           a1, a1, -1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"  // set vl = 16
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf2\n\t"  // set vl = 16
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
        "addi           t2, t2, 16\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m8n8
        "6:\n\t"
        "andi           t1, %[n], 8\n\t"  // t1 = n & 8u (n8)
        "beqz           t1, 10f\n\t"      // if n8==0, jump to m8n_tail
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 8
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
        "addi           t3, t3, 32\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 8f\n\t"       // if k2 == 0, jump to m8n4k1

        // m8n8k2
        "7:\n\t"
        "vle32.v        v4, (t3)\n\t"
        "addi           t3, t3, 32\n\t"

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
        "addi           t3, t3, 32\n\t"

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

        // m8n8k1
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
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 8
        "vmulh.vx	    v16, v16, a0\n\t"
        "not            a1, a1\n\t"
        "vssra.vx	    v16, v16, a1\n\t"
        "vadd.vx        v16, v16, %[out_zp]\n\t"
        "vsetvli        zero, t6, e16, mf2\n\t"  // set vl = 8
        "vnclip.wi	    v1, v16, 0\n\t"
        "vsetvli        zero, t6, e8, mf4\n\t"  // set vl = 8
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
        "addi           t2, t2, 8\n\t"

        // m8n_tail
        "10:\n\t"
        "andi           t1, %[n], 7\n\t"        // t1 = n & 7u (n_tail)
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

static inline void kernel_m4n16_int8_v256(int8_t *dst, int8_t *sa, int8_t *sb, int m, int k, int n,
                                          int32_t *bias, int32_t out_zp, int32_t *mult,
                                          int32_t *shift)
{
    asm volatile(
        // m4
        "1:\n\t"
        "srai           t1, %[n], 4\n\t"        // t1 = n8
        "mv             t2, %[output_ptr]\n\t"  // init output addr

        "beqz           t1, 6f\n\t"  // if n8==0, jump to m4n4
        // m4n8
        "2:\n\t"
        "li             t6, 16\n\t"
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
        "addi           %[input_ptr], %[input_ptr], 64\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 4f\n\t"       // if k2 == 0, jump to m4n8k1

        // m4n8k2
        "3:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 64\n\t"

        "vmaqa.vx       v16, a0, v2\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v20, a2, v2\n\t"
        "vmaqa.vx       v22, a3, v2\n\t"
        "addi           t5, t5, 32\n\t"

        "vle32.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 64\n\t"

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

        "addi           %[input_ptr], %[input_ptr], 64\n\t"  // ********************

        // end kernel_m4n8
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -64\n\t"  // pb -= 8

        // 后处理
        "li             t6, 16\n\t"

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
        "addi           t2, t2, 16\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m4n4
        "6:\n\t"
        "andi           t1, %[n], 8\n\t"  // t1 = n & 4u (n4)
        "beqz           t1, 10f\n\t"      // if n4==0, jump to m4n_tail
        "li             t6, 8\n\t"
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
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"
        "lwd            a2, a3, 8(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 8f\n\t"       // if k2 == 0, jump to m8n4k1

        // m8n4k2
        "7:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lwd            a4, a5, 16(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "vmaqa.vx       v20, a2, v1\n\t"
        "lwd            a6, a7, 24(t5)\n\t"
        "vmaqa.vx       v22, a3, v1\n\t"
        "addi           t5, t5, 32\n\t"

        "vle32.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

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

        "addi           %[input_ptr], %[input_ptr], 32\n\t"  // ********************

        // end kernel_m8n4
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -32\n\t"  // pb -= 4

        // 后处理
        "li             t6, 8\n\t"

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
        "addi           t2, t2, 8\n\t"

        // m4n_tail
        "10:\n\t"
        "andi           t1, %[n], 7\n\t"        // t1 = n & 3u (n_tail)
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

static inline void kernel_m2n16_int8_v256(int8_t *dst, int8_t *sa, int8_t *sb, int m, int k, int n,
                                          int32_t *bias, int32_t out_zp, int32_t *mult,
                                          int32_t *shift)
{
    asm volatile(
        // m4
        "1:\n\t"
        "srai           t1, %[n], 4\n\t"        // t1 = n8
        "mv             t2, %[output_ptr]\n\t"  // init output addr

        "beqz           t1, 6f\n\t"  // if n8==0, jump to m4n4
        // m4n8
        "2:\n\t"
        "li             t6, 16\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v18, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 64\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 4f\n\t"       // if k2 == 0, jump to m4n8k1

        // m4n8k2
        "3:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 64\n\t"

        "vmaqa.vx       v16, a0, v2\n\t"
        "vmaqa.vx       v18, a1, v2\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "addi           t5, t5, 16\n\t"

        "vle32.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 64\n\t"

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

        "addi           %[input_ptr], %[input_ptr], 64\n\t"  // ********************

        // end kernel_m4n8
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -64\n\t"  // pb -= 8

        // 后处理
        "li             t6, 16\n\t"

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
        "addi           t2, t2, 16\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m4n4
        "6:\n\t"
        "andi           t1, %[n], 8\n\t"  // t1 = n & 4u (n4)
        "beqz           t1, 10f\n\t"      // if n4==0, jump to m4n_tail
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        // init out_tmp = bias
        "lwd            t4, t5, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"
        "vmv.v.x        v18, t5\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "lwd            a0, a1, 0(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 8f\n\t"       // if k2 == 0, jump to m8n4k1

        // m8n4k2
        "7:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lwd            a2, a3, 8(t5)\n\t"
        "vmaqa.vx       v18, a1, v1\n\t"
        "addi           t5, t5, 16\n\t"

        "vle32.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

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

        "addi           %[input_ptr], %[input_ptr], 32\n\t"  // ********************

        // end kernel_m8n4
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -32\n\t"  // pb -= 4

        // 后处理
        "li             t6, 8\n\t"

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
        "addi           t2, t2, 8\n\t"

        // m4n_tail
        "10:\n\t"
        "andi           t1, %[n], 7\n\t"        // t1 = n & 3u (n_tail)
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

static inline void kernel_m1n16_int8_v256(int8_t *dst, int8_t *sa, int8_t *sb, int m, int k, int n,
                                          int32_t *bias, int32_t out_zp, int32_t *mult,
                                          int32_t *shift)
{
    asm volatile(
        // m4
        "1:\n\t"
        "srai           t1, %[n], 4\n\t"        // t1 = n8
        "mv             t2, %[output_ptr]\n\t"  // init output addr

        "beqz           t1, 6f\n\t"  // if n8==0, jump to m4n4
        // m4n8
        "2:\n\t"
        "li             t6, 16\n\t"
        "vsetvli        zero, t6, e32, m2\n\t"  // set vl = 8
        // init out_tmp = bias
        "lw             t4, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 64\n\t"

        // pre-load pa(kernel_data)
        "lw             a0, 0(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 4f\n\t"       // if k2 == 0, jump to m4n8k1

        // m4n8k2
        "3:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 64\n\t"

        "vmaqa.vx       v16, a0, v2\n\t"
        "lw             a1, 4(t5)\n\t"
        "addi           t5, t5, 8\n\t"

        "vle32.v        v2, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 64\n\t"

        "vmaqa.vx       v16, a1, v4\n\t"
        "lw             a0, 0(t5)\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 3b\n\t"

        // m4n8k1
        "4:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 5f\n\t"       // if k1 == 0, jump to end kernel_m4n8

        "vmaqa.vx       v16, a0, v2\n\t"

        "addi           %[input_ptr], %[input_ptr], 64\n\t"  // ********************

        // end kernel_m4n8
        "5:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -64\n\t"  // pb -= 8

        // 后处理
        "li             t6, 16\n\t"

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
        "addi           t2, t2, 16\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        // m4n4
        "6:\n\t"
        "andi           t1, %[n], 8\n\t"  // t1 = n & 4u (n4)
        "beqz           t1, 10f\n\t"      // if n4==0, jump to m4n_tail
        "li             t6, 8\n\t"
        "vsetvli        zero, t6, e32, m1\n\t"  // set vl = 4
        // init out_tmp = bias
        "lw             t4, 0(%[bias_ptr])\n\t"  // bias_ptr[0]/[1]
        "vmv.v.x        v16, t4\n\t"

        "mv             t5, %[kernel_ptr]\n\t"  // s2 hold kernel 8 lines start addr

        // pre-load pb (input_data)
        "vle32.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        // pre-load pa(kernel_data)
        "lw             a0, 0(t5)\n\t"

        "srai           t4, %[k], 3\n\t"  // t4 = k8[k2]
        "beqz           t4, 8f\n\t"       // if k2 == 0, jump to m8n4k1

        // m8n4k2
        "7:\n\t"
        "vle32.v        v4, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vmaqa.vx       v16, a0, v1\n\t"
        "lw             a1, 4(t5)\n\t"
        "addi           t5, t5, 8\n\t"

        "vle32.v        v1, (%[input_ptr])\n\t"
        "addi           %[input_ptr], %[input_ptr], 32\n\t"

        "vmaqa.vx       v16, a1, v4\n\t"
        "lw             a0, 0(t5)\n\t"

        "addi           t4, t4, -1\n\t"
        "bnez           t4, 7b\n\t"

        // m4n4k1
        "8:\n\t"
        "andi           t4, %[k], 4\n\t"  // t4 = k1
        "beqz           t4, 9f\n\t"       // if k1 == 0, jump to end kernel_m4n4

        "vmaqa.vx       v16, a0, v1\n\t"

        "addi           %[input_ptr], %[input_ptr], 32\n\t"  // ********************

        // end kernel_m8n4
        "9:\n\t"
        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %[input_ptr], %[input_ptr], -32\n\t"  // pb -= 4

        // 后处理
        "li             t6, 8\n\t"

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
        "addi           t2, t2, 8\n\t"

        // m4n_tail
        "10:\n\t"
        "andi           t1, %[n], 7\n\t"        // t1 = n & 3u (n_tail)
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

void shl_c908_gemm_8x16_int8_v256_dot(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                      int32_t *bias, int m, int k, int n, int ldc, int32_t out_zp,
                                      int32_t *mult, int32_t *shift)
{
    int8_t *kernel_ptr = (int8_t *)sa;
    int8_t *input_ptr = (int8_t *)sb;
    int8_t *output_ptr = dst;
    // please use fuse_zp2bias option in hhb, thus bias_data wont be NULL
    int32_t *bias_ptr = bias;

    int tail = m % 8;
    if (m > 8) {
        kernel_m8n16_int8_v256(output_ptr, kernel_ptr, input_ptr, m, k, n, bias_ptr, out_zp, mult,
                               shift);
        output_ptr += (m - tail) * n;
        kernel_ptr += (m - tail) * k;
        bias_ptr += (m - tail);
    }
    if (tail & 4) {
        kernel_m4n16_int8_v256(output_ptr, kernel_ptr, input_ptr, m, k, n, bias_ptr, out_zp, mult,
                               shift);
        output_ptr += 4 * n;
        kernel_ptr += 4 * k;
        bias_ptr += 4;
    }
    if (tail & 2) {
        kernel_m2n16_int8_v256(output_ptr, kernel_ptr, input_ptr, m, k, n, bias_ptr, out_zp, mult,
                               shift);
        output_ptr += 2 * n;
        kernel_ptr += 2 * k;
        bias_ptr += 2;
    }
    if (tail & 1) {
        kernel_m1n16_int8_v256(output_ptr, kernel_ptr, input_ptr, m, k, n, bias_ptr, out_zp, mult,
                               shift);
        output_ptr += 1 * n;
        kernel_ptr += 1 * k;
        bias_ptr += 1;
    }
}
