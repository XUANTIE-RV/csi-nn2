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

/************************************************************************
 * reorder kernel matrix
 ***********************************************************************/
// vlen=128
void shl_c908_reorder_kernel_n8_fp32(float *src, float *dst, int m, int k, int ldc)
{
    shl_rvv_reorder_kernel_n8_fp32(src, dst, m, k, ldc);
}

void shl_c908_reorder_kernel_n8_fp16(__fp16 *src, __fp16 *dst, int m, int k, int ldc)
{
    shl_rvv_reorder_kernel_n8_fp16(src, dst, m, k, ldc);
}

void shl_c908_reorder_kernel_n8_int8_dot(int8_t *src, int8_t *dst, int m, int k, int ldc)
{
    shl_rvv_reorder_kernel_n8_int8_dot(src, dst, m, k, ldc);
}

/************************************************************************
 * reorder input matrix
 ***********************************************************************/
// vlen=128
/**************************************************************
 * input—matrix: [k, n]
 * Data arrangement: Z12 Z8 Z4 Z4_tail
 **************************************************************/
void shl_c908_reorder_input_z12_fp32(float *src, float *dst, int k, int n, int ldc)
{
    asm volatile(
        "li             a1, 12\n\t"
        "divw           t0, %[n], a1\n\t"   // t0 = n12
        "remw           t1, %[n], a1\n\t"   // t1 = n % 12
        "slli           t2, %[ldc], 2\n\t"  // t2 = ldc * 4 (line stride)

        "beqz           t0, 3f\n\t"             // if n12 == 0, jump to packn8
        "vsetvli        zero, a1, e32, m4\n\t"  // set vl = 12

        "1:\n\t"  // n12
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 48\n\t"  // src_ptr += 12
        "mv             t3, %[k]\n\t"            // k

        "2:\n\t"
        // start packn12k1
        "vle32.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse32.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 48\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 2b\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "3:\n\t"                        // n8
        "andi           t0, t1, 8\n\t"  // n & 8u
        "beqz           t0, 5f\n\t"

        "vsetvli        zero, t0, e32, m2\n\t"  // set vl = 8
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 32\n\t"  // src_ptr += 8
        "mv             t3, %[k]\n\t"            // k

        "4:\n\t"
        // start packn8k1
        "vle32.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse32.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 32\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 4b\n\t"

        "5:\n\t"                        // n4
        "andi           t0, t1, 4\n\t"  // n & 4u
        "beqz           t0, 7f\n\t"

        "vsetvli        zero, t0, e32, m1\n\t"  // set vl = 4
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 16\n\t"  // src_ptr += 4
        "mv             t3, %[k]\n\t"            // k

        "6:\n\t"
        // start packn4k1
        "vle32.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse32.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 16\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 6b\n\t"

        "7:\n\t"                        // n_tail
        "andi           t0, t1, 3\n\t"  // n & 3u
        "beqz           t0, 9f\n\t"
        "slli           t4, t0, 2\n\t"  // t4 = 4 * n_tail

        "vsetvli        zero, t0, e32, m1\n\t"  // set vl = n_tail
        "mv             a0, %[src]\n\t"
        "mv             t3, %[k]\n\t"  // k

        "8:\n\t"
        // start packn_tailk1
        "vle32.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse32.v        v4, (%[dst])\n\t"
        "add            %[dst], %[dst], t4\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 8b\n\t"

        "9:\n\t"  // ending

        : [src] "+r"(src), [dst] "+r"(dst)

        : [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)

        : "cc", "memory", "v4", "v5", "v6", "v7", "a0", "a1", "t0", "t1", "t2", "t3", "t4");
}

/**************************************************************
 * input—matrix: [k, n]
 * Data arrangement: Z24 Z16 Z8 Z8_tail
 **************************************************************/
void shl_c908_reorder_input_z24_fp16(__fp16 *src, __fp16 *dst, int k, int n, int ldc)
{
    asm volatile(
        "li             a1, 24\n\t"
        "divw           t0, %[n], a1\n\t"   // t0 = n24
        "remw           t1, %[n], a1\n\t"   // t1 = n % 24
        "slli           t2, %[ldc], 1\n\t"  // t2 = ldc * 2 (line stride)

        "beqz           t0, 3f\n\t"             // if n24 == 0, jump to packn16
        "vsetvli        zero, a1, e16, m4\n\t"  // set vl = 24

        "1:\n\t"  // n24
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 48\n\t"  // src_ptr += 24
        "mv             t3, %[k]\n\t"            // k

        "2:\n\t"
        // start packn24k1
        "vle16.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse16.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 48\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 2b\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "3:\n\t"                         // n16
        "andi           t0, t1, 16\n\t"  // n & 16u
        "beqz           t0, 5f\n\t"

        "vsetvli        zero, t0, e16, m2\n\t"  // set vl = 16
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 32\n\t"  // src_ptr += 16
        "mv             t3, %[k]\n\t"            // k

        "4:\n\t"
        // start packn16k1
        "vle16.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse16.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 32\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 4b\n\t"

        "5:\n\t"                        // n8
        "andi           t0, t1, 8\n\t"  // n & 8u
        "beqz           t0, 7f\n\t"

        "vsetvli        zero, t0, e16, m1\n\t"  // set vl = 8
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 16\n\t"  // src_ptr += 8
        "mv             t3, %[k]\n\t"            // k

        "6:\n\t"
        // start packn8k1
        "vle16.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse16.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 16\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 6b\n\t"

        "7:\n\t"                        // n_tail
        "andi           t0, t1, 7\n\t"  // n & 7u
        "beqz           t0, 9f\n\t"
        "slli           t4, t0, 1\n\t"  // t4 = 2 * n_tail

        "vsetvli        zero, t0, e16, m1\n\t"  // set vl = n_tail
        "mv             a0, %[src]\n\t"
        "mv             t3, %[k]\n\t"  // k

        "8:\n\t"
        // start packn_tailk1
        "vle16.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse16.v        v4, (%[dst])\n\t"
        "add            %[dst], %[dst], t4\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 8b\n\t"

        "9:\n\t"  // ending

        : [src] "+r"(src), [dst] "+r"(dst)

        : [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)

        : "cc", "memory", "v4", "v5", "v6", "v7", "a0", "a1", "t0", "t1", "t2", "t3", "t4");
}

/**************************************************************
 * input—matrix: [k, n]
 * Data arrangement: Z8 Z4 Z4_tail
 **************************************************************/
void shl_c908_reorder_input_z8_int8_dot(int8_t *src, int8_t *dst, int k, int n, int ldc)
{
    int vl = vsetvl_e8m1(8);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        int8_t *b0 = src + i;
        int j = 0;
        for (; j + 3 < k; j += 4) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst += 32 - 3;
        }
        // k_tail
        if (j < k) {
            int8_t *sb0 = dst;
            for (; j < k; j++) {
                vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
                b0 += n;
                vsse8_v_i8m1(sb0, 4 * sizeof(int8_t), _tmp, vl);
                sb0++;
            }
            dst += 32;
        }
    }
    for (; i + 3 < n; i += 4) {
        vl = vsetvl_e8m1(4);
        int8_t *b0 = src + i;
        int j = 0;
        for (; j + 3 < k; j += 4) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst += 13;
        }
        // k_tail
        if (j < k) {
            int8_t *sb0 = dst;
            for (; j < k; j++) {
                vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
                b0 += n;
                vsse8_v_i8m1(sb0, 4 * sizeof(int8_t), _tmp, vl);
                sb0++;
            }
            dst += 16;
        }
    }
    // n_tail
    if (i < n) {
        vl = vsetvl_e8m1(n & 3);
        int8_t *b0 = src + i;
        int j = 0;
        for (; j + 3 < k; j += 4) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst += 4 * vl - 3;
        }
        // k_tail
        if (j < k) {
            int8_t *sb0 = dst;
            for (; j < k; j++) {
                vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
                b0 += n;
                vsse8_v_i8m1(sb0, 4 * sizeof(int8_t), _tmp, vl);
                sb0++;
            }
        }
    }
}

/**************************************************************
 * input—matrix: [k, n]
 * Data arrangement: Z12 Z8 Z4 Z4_tail
 **************************************************************/
void shl_c908_reorder_input_z12_int8(int8_t *src, int8_t *dst, int k, int n, int ldc)
{
    int vl = vsetvl_e8m1(12);
    int i = 0;
    for (; i + 11 < n; i += 12) {
        int8_t *b0 = src + i;
        int j = 0;
        for (; j + 3 < k; j += 4) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst += 48 - 3;
        }
        // k_tail
        if (j < k) {
            int8_t *sb0 = dst;
            for (; j < k; j++) {
                vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
                b0 += n;
                vsse8_v_i8m1(sb0, 4 * sizeof(int8_t), _tmp, vl);
                sb0++;
            }
            dst += 48;
        }
    }
    for (; i + 7 < n; i += 8) {
        vl = vsetvl_e8m1(8);
        int8_t *b0 = src + i;
        int j = 0;
        for (; j + 3 < k; j += 4) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst += 32 - 3;
        }
        // k_tail
        if (j < k) {
            int8_t *sb0 = dst;
            for (; j < k; j++) {
                vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
                b0 += n;
                vsse8_v_i8m1(sb0, 4 * sizeof(int8_t), _tmp, vl);
                sb0++;
            }
            dst += 32;
        }
    }
    for (; i + 3 < n; i += 4) {
        vl = vsetvl_e8m1(4);
        int8_t *b0 = src + i;
        int j = 0;
        for (; j + 3 < k; j += 4) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst += 13;
        }
        // k_tail
        if (j < k) {
            int8_t *sb0 = dst;
            for (; j < k; j++) {
                vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
                b0 += n;
                vsse8_v_i8m1(sb0, 4 * sizeof(int8_t), _tmp, vl);
                sb0++;
            }
            dst += 16;
        }
    }
    // n_tail
    if (i < n) {
        vl = vsetvl_e8m1(n & 3);
        int8_t *b0 = src + i;
        int j = 0;
        for (; j + 3 < k; j += 4) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst += 4 * vl - 3;
        }
        // k_tail
        if (j < k) {
            int8_t *sb0 = dst;
            for (; j < k; j++) {
                vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
                b0 += n;
                vsse8_v_i8m1(sb0, 4 * sizeof(int8_t), _tmp, vl);
                sb0++;
            }
        }
    }
}

// vlen256
/**************************************************************
 * input—matrix: [k, n]
 * Data arrangement: Z16 Z8 Z8_tail
 **************************************************************/
void shl_c908_reorder_input_z16_fp32_v256(float *src, float *dst, int k, int n, int ldc)
{
    asm volatile(
        "li             a0, 16\n\t"
        "srai           t0, %[n], 4\n\t"    // t0 = n16
        "andi           t1, %[n], 15\n\t"   // t1 = n & 15
        "slli           t2, %[ldc], 2\n\t"  // t2 = ldc * 4 (line stride)

        "beqz           t0, 3f\n\t"             // if n16 == 0, jump to packn8
        "vsetvli        zero, a0, e32, m2\n\t"  // set vl = 16

        "1:\n\t"  // n16
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 64\n\t"  // src_ptr += 16
        "mv             t3, %[k]\n\t"            // k

        "2:\n\t"
        // start packn16k1
        "vle32.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse32.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 64\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 2b\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "3:\n\t"                        // n8
        "andi           t0, t1, 8\n\t"  // n & 8u
        "beqz           t0, 5f\n\t"

        "vsetvli        zero, t0, e32, m1\n\t"  // set vl = 8
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 32\n\t"  // src_ptr += 8
        "mv             t3, %[k]\n\t"            // k

        "4:\n\t"
        // start packn8k1
        "vle32.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse32.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 32\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 4b\n\t"

        "5:\n\t"                        // n_tail
        "andi           t0, t1, 7\n\t"  // n & 7u
        "beqz           t0, 7f\n\t"
        "slli           t4, t0, 2\n\t"  // t4 = 4 * n_tail

        "vsetvli        zero, t0, e32, m1\n\t"  // set vl = n_tail
        "mv             a0, %[src]\n\t"
        "mv             t3, %[k]\n\t"  // k

        "6:\n\t"
        // start packn8k1
        "vle32.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse32.v        v4, (%[dst])\n\t"
        "add            %[dst], %[dst], t4\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 6b\n\t"

        "7:\n\t"  // ending

        : [src] "+r"(src), [dst] "+r"(dst)

        : [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)

        : "cc", "memory", "v4", "v5", "a0", "t0", "t1", "t2", "t3", "t4");
}

/**************************************************************
 * input—matrix: [k, n]
 * Data arrangement: Z32 Z16 Z16_tail
 **************************************************************/
void shl_c908_reorder_input_z32_fp16_v256(__fp16 *src, __fp16 *dst, int k, int n, int ldc)
{
    asm volatile(
        "li             a0, 32\n\t"
        "srai           t0, %[n], 5\n\t"    // t0 = n32
        "andi           t1, %[n], 31\n\t"   // t1 = n & 31
        "slli           t2, %[ldc], 1\n\t"  // t2 = ldc * 2 (line stride)

        "beqz           t0, 3f\n\t"             // if n32 == 0, jump to packn16
        "vsetvli        zero, a0, e16, m2\n\t"  // set vl = 32

        "1:\n\t"  // n32
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 64\n\t"  // src_ptr += 32
        "mv             t3, %[k]\n\t"            // k

        "2:\n\t"
        // start packn32k1
        "vle16.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse16.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 64\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 2b\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "3:\n\t"                         // n16
        "andi           t0, t1, 16\n\t"  // n & 16u
        "beqz           t0, 5f\n\t"

        "vsetvli        zero, t0, e16, m1\n\t"  // set vl = 16
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 32\n\t"  // src_ptr += 16
        "mv             t3, %[k]\n\t"            // k

        "4:\n\t"
        // start packn16k1
        "vle16.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse16.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 32\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 4b\n\t"

        "5:\n\t"                         // n_tail
        "andi           t0, t1, 15\n\t"  // n & 15u
        "beqz           t0, 7f\n\t"
        "slli           t4, t0, 1\n\t"  // t4 = 2 * n_tail

        "vsetvli        zero, t0, e16, m1\n\t"  // set vl = n_tail
        "mv             a0, %[src]\n\t"
        "mv             t3, %[k]\n\t"  // k

        "6:\n\t"
        // start packn_tailk1
        "vle16.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse16.v        v4, (%[dst])\n\t"
        "add            %[dst], %[dst], t4\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 6b\n\t"

        "7:\n\t"  // ending

        : [src] "+r"(src), [dst] "+r"(dst)

        : [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)

        : "cc", "memory", "v4", "v5", "a0", "t0", "t1", "t2", "t3", "t4");
}

/**************************************************************
 * input—matrix: [k, n]
 * Data arrangement: Z16 Z8 Z8_tail
 **************************************************************/
void shl_c908_reorder_input_z16_int8_v256_dot(int8_t *src, int8_t *dst, int k, int n, int ldc)
{
    int vl = vsetvl_e8m1(16);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        int8_t *b0 = src + i;
        int j = 0;
        for (; j + 3 < k; j += 4) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst += 64 - 3;
        }
        // k_tail
        if (j < k) {
            int8_t *sb0 = dst;
            for (; j < k; j++) {
                vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
                b0 += n;
                vsse8_v_i8m1(sb0, 4 * sizeof(int8_t), _tmp, vl);
                sb0++;
            }
            dst += 64;
        }
    }
    for (; i + 7 < n; i += 8) {
        vl = vsetvl_e8m1(8);
        int8_t *b0 = src + i;
        int j = 0;
        for (; j + 3 < k; j += 4) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst += 32 - 3;
        }
        // k_tail
        if (j < k) {
            int8_t *sb0 = dst;
            for (; j < k; j++) {
                vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
                b0 += n;
                vsse8_v_i8m1(sb0, 4 * sizeof(int8_t), _tmp, vl);
                sb0++;
            }
            dst += 32;
        }
    }
    // n_tail
    if (i < n) {
        vl = vsetvl_e8m1(n & 7);
        int8_t *b0 = src + i;
        int j = 0;
        for (; j + 3 < k; j += 4) {
            vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst++;
            _tmp = vle8_v_i8m1(b0, vl);
            b0 += n;
            vsse8_v_i8m1(dst, 4 * sizeof(int8_t), _tmp, vl);
            dst += 4 * vl - 3;
        }
        // k_tail
        if (j < k) {
            int8_t *sb0 = dst;
            for (; j < k; j++) {
                vint8m1_t _tmp = vle8_v_i8m1(b0, vl);
                b0 += n;
                vsse8_v_i8m1(sb0, 4 * sizeof(int8_t), _tmp, vl);
                sb0++;
            }
        }
    }
}

#ifdef SHL_UNUSED_REGISTER_BLK
/**************************************************************
 * input—matrix: [k, n]
 * Data arrangement: Z8 Z4 Z4_tail
 **************************************************************/
void shl_c908_reorder_input_z8_fp32(float *src, float *dst, int k, int n, int ldc)
{
    asm volatile(
        "li             a0, 8\n\t"
        "srai           t0, %[n], 3\n\t"    // t0 = n8
        "andi           t1, %[n], 7\n\t"    // t1 = n & 7
        "slli           t2, %[ldc], 2\n\t"  // t2 = ldc * 4 (line stride)

        "beqz           t0, 3f\n\t"             // if n8 == 0, jump to packn4
        "vsetvli        zero, a0, e32, m2\n\t"  // set vl = 8

        "1:\n\t"  // n8
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 32\n\t"  // src_ptr += 8
        "mv             t3, %[k]\n\t"            // k

        "2:\n\t"
        // start packn8k1
        "vle32.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse32.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 32\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 2b\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "3:\n\t"                        // n4
        "andi           t0, t1, 4\n\t"  // n & 4u
        "beqz           t0, 5f\n\t"

        "vsetvli        zero, t0, e32, m1\n\t"  // set vl = 4
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 16\n\t"  // src_ptr += 4
        "mv             t3, %[k]\n\t"            // k

        "4:\n\t"
        // start packn4k1
        "vle32.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse32.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 16\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 4b\n\t"

        "5:\n\t"                        // n_tail
        "andi           t0, t1, 3\n\t"  // n & 3u
        "beqz           t0, 7f\n\t"
        "slli           t4, t0, 2\n\t"  // t4 = 4 * n_tail

        "vsetvli        zero, t0, e32, m1\n\t"  // set vl = n_tail
        "mv             a0, %[src]\n\t"
        "mv             t3, %[k]\n\t"  // k

        "6:\n\t"
        // start packn4k1
        "vle32.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse32.v        v4, (%[dst])\n\t"
        "add            %[dst], %[dst], t4\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 6b\n\t"

        "7:\n\t"  // ending

        : [src] "+r"(src), [dst] "+r"(dst)

        : [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)

        : "cc", "memory", "v4", "v5", "a0", "t0", "t1", "t2", "t3", "t4");
}

/**************************************************************
 * input—matrix: [k, n]
 * Data arrangement: Z16 Z8 Z8_tail
 **************************************************************/
void shl_c908_reorder_input_z16_fp16(__fp16 *src, __fp16 *dst, int k, int n, int ldc)
{
    asm volatile(
        "li             a0, 16\n\t"
        "srai           t0, %[n], 4\n\t"    // t0 = n16
        "andi           t1, %[n], 15\n\t"   // t1 = n & 15
        "slli           t2, %[ldc], 1\n\t"  // t2 = ldc * 2 (line stride)

        "beqz           t0, 3f\n\t"             // if n18 == 0, jump to packn8
        "vsetvli        zero, a0, e16, m2\n\t"  // set vl = 16

        "1:\n\t"  // n16
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 32\n\t"  // src_ptr += 16
        "mv             t3, %[k]\n\t"            // k

        "2:\n\t"
        // start packn16k1
        "vle16.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse16.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 32\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 2b\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "3:\n\t"                        // n8
        "andi           t0, t1, 8\n\t"  // n & 8u
        "beqz           t0, 5f\n\t"

        "vsetvli        zero, t0, e16, m1\n\t"  // set vl = 8
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 16\n\t"  // src_ptr += 8
        "mv             t3, %[k]\n\t"            // k

        "4:\n\t"
        // start packn8k1
        "vle16.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse16.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 16\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 4b\n\t"

        "5:\n\t"                        // n_tail
        "andi           t0, t1, 7\n\t"  // n & 7u
        "beqz           t0, 7f\n\t"
        "slli           t4, t0, 1\n\t"  // t4 = 2 * n_tail

        "vsetvli        zero, t0, e16, m1\n\t"  // set vl = n_tail
        "mv             a0, %[src]\n\t"
        "mv             t3, %[k]\n\t"  // k

        "6:\n\t"
        // start packn8k1
        "vle16.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse16.v        v4, (%[dst])\n\t"
        "add            %[dst], %[dst], t4\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 6b\n\t"

        "7:\n\t"  // ending

        : [src] "+r"(src), [dst] "+r"(dst)

        : [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)

        : "cc", "memory", "v4", "v5", "a0", "t0", "t1", "t2", "t3", "t4");
}

/**************************************************************
 * input—matrix: [k, n]
 * Data arrangement: Z24 Z16 Z8 Z8_tail
 **************************************************************/
void shl_c908_reorder_input_z24_fp32_v256(float *src, float *dst, int k, int n, int ldc)
{
    asm volatile(
        "li             a1, 12\n\t"
        "divw           t0, %[n], a1\n\t"   // t0 = n12
        "remw           t1, %[n], a1\n\t"   // t1 = n % 12
        "slli           t2, %[ldc], 2\n\t"  // t2 = ldc * 4 (line stride)

        "beqz           t0, 3f\n\t"             // if n12 == 0, jump to packn8
        "vsetvli        zero, a1, e32, m4\n\t"  // set vl = 12

        "1:\n\t"  // n12
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 48\n\t"  // src_ptr += 12
        "mv             t3, %[k]\n\t"            // k

        "2:\n\t"
        // start packn12k1
        "vle32.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse32.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 48\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 2b\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "3:\n\t"                        // n8
        "andi           t0, t1, 8\n\t"  // n & 8u
        "beqz           t0, 5f\n\t"

        "vsetvli        zero, t0, e32, m2\n\t"  // set vl = 8
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 32\n\t"  // src_ptr += 8
        "mv             t3, %[k]\n\t"            // k

        "4:\n\t"
        // start packn8k1
        "vle32.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse32.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 32\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 4b\n\t"

        "5:\n\t"                        // n4
        "andi           t0, t1, 4\n\t"  // n & 4u
        "beqz           t0, 7f\n\t"

        "vsetvli        zero, t0, e32, m1\n\t"  // set vl = 4
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 16\n\t"  // src_ptr += 4
        "mv             t3, %[k]\n\t"            // k

        "6:\n\t"
        // start packn4k1
        "vle32.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse32.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 16\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 6b\n\t"

        "7:\n\t"                        // n_tail
        "andi           t0, t1, 3\n\t"  // n & 3u
        "beqz           t0, 9f\n\t"
        "slli           t4, t0, 2\n\t"  // t4 = 4 * n_tail

        "vsetvli        zero, t0, e32, m1\n\t"  // set vl = n_tail
        "mv             a0, %[src]\n\t"
        "mv             t3, %[k]\n\t"  // k

        "8:\n\t"
        // start packn_tailk1
        "vle32.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse32.v        v4, (%[dst])\n\t"
        "add            %[dst], %[dst], t4\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 8b\n\t"

        "9:\n\t"  // ending

        : [src] "+r"(src), [dst] "+r"(dst)

        : [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)

        : "cc", "memory", "v4", "v5", "v6", "v7", "a0", "a1", "t0", "t1", "t2", "t3", "t4");
}

/**************************************************************
 * input—matrix: [k, n]
 * Data arrangement: Z48 Z32 Z16 Z16_tail
 **************************************************************/
void shl_c908_reorder_input_z48_fp16_v256(__fp16 *src, __fp16 *dst, int k, int n, int ldc)
{
    asm volatile(
        "li             a1, 24\n\t"
        "divw           t0, %[n], a1\n\t"   // t0 = n24
        "remw           t1, %[n], a1\n\t"   // t1 = n % 24
        "slli           t2, %[ldc], 1\n\t"  // t2 = ldc * 2 (line stride)

        "beqz           t0, 3f\n\t"             // if n24 == 0, jump to packn16
        "vsetvli        zero, a1, e16, m4\n\t"  // set vl = 24

        "1:\n\t"  // n24
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 48\n\t"  // src_ptr += 24
        "mv             t3, %[k]\n\t"            // k

        "2:\n\t"
        // start packn24k1
        "vle16.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse16.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 48\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 2b\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "3:\n\t"                         // n16
        "andi           t0, t1, 16\n\t"  // n & 16u
        "beqz           t0, 5f\n\t"

        "vsetvli        zero, t0, e16, m2\n\t"  // set vl = 16
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 32\n\t"  // src_ptr += 16
        "mv             t3, %[k]\n\t"            // k

        "4:\n\t"
        // start packn16k1
        "vle16.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse16.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 32\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 4b\n\t"

        "5:\n\t"                        // n8
        "andi           t0, t1, 8\n\t"  // n & 8u
        "beqz           t0, 7f\n\t"

        "vsetvli        zero, t0, e16, m1\n\t"  // set vl = 8
        "mv             a0, %[src]\n\t"
        "addi           %[src], %[src], 16\n\t"  // src_ptr += 8
        "mv             t3, %[k]\n\t"            // k

        "6:\n\t"
        // start packn8k1
        "vle16.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse16.v        v4, (%[dst])\n\t"
        "addi           %[dst], %[dst], 16\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 6b\n\t"

        "7:\n\t"                        // n_tail
        "andi           t0, t1, 7\n\t"  // n & 7u
        "beqz           t0, 9f\n\t"
        "slli           t4, t0, 1\n\t"  // t4 = 2 * n_tail

        "vsetvli        zero, t0, e16, m1\n\t"  // set vl = n_tail
        "mv             a0, %[src]\n\t"
        "mv             t3, %[k]\n\t"  // k

        "8:\n\t"
        // start packn_tailk1
        "vle16.v        v4, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse16.v        v4, (%[dst])\n\t"
        "add            %[dst], %[dst], t4\n\t"

        "addi           t3, t3, -1\n\t"
        "bnez           t3, 8b\n\t"

        "9:\n\t"  // ending

        : [src] "+r"(src), [dst] "+r"(dst)

        : [k] "r"(k), [n] "r"(n), [ldc] "r"(ldc)

        : "cc", "memory", "v4", "v5", "v6", "v7", "a0", "a1", "t0", "t1", "t2", "t3", "t4");
}
#endif
