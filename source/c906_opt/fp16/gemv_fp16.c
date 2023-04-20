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

/*
    change memory layout for matrix [k * n] by Z shape
    Z length: 8
*/
void shl_c906_reorder_matrix_z8_fp16(__fp16* src, __fp16* dst, int k, int n, int ldx)
{
    asm volatile(
        "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 8

        "slli           t2, %4, 1\n\t"  // t2 = ldx * 2 (line stride)

        "srai           t0, %3, 3\n\t"  // t0 = n8
        "beqz           t0, 3f\n\t"     // jump to packn_tail

        "1:\n\t"  // n8
        "mv             a0, %0\n\t"
        "addi           %0, %0, 16\n\t"
        "mv             t1, %2\n\t"  // k

        "2:\n\t"
        // start packn8k1
        "vle.v          v2, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "3:\n\t"                        // n_tail
        "andi           t0, %3, 7\n\t"  // n & 7u
        "beqz           t0, 8f\n\t"

        "srai           t3, %2, 3\n\t"  // k8
        "slli           t5, %4, 4\n\t"  // t5 = ldx * 8 * 2 (8 lines)
        "andi           t6, %2, 7\n\t"  // k_tail
        "slli           t4, t6, 1\n\t"  // k_tail * 2

        "4:\n\t"
        "mv             a0, %0\n\t"
        "addi           %0, %0, 2\n\t"
        "mv             t1, t3\n\t"  // t1 = k8
        "beqz           t3, 6f\n\t"

        "5:\n\t"
        "vsetvli        zero, zero, e16, m1\n\t"
        "vlse.v         v2, (a0), t2\n\t"
        "add            a0, a0, t5\n\t"
        "vse.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 5b\n\t"

        "6:\n\t"
        "vsetvli        zero, t6, e16, m1\n\t"
        "vlse.v         v2, (a0), t2\n\t"
        "vse.v          v2, (%1)\n\t"
        "add            %1, %1, t4\n\t"

        "7:\n\t"
        "addi           t0, t0, -1\n\t"
        "bnez           t0, 4b\n\t"

        "8:\n\t"  // ending

        : "=r"(src),  // %0
          "=r"(dst),  // %1
          "=r"(k),    // %2
          "=r"(n),    // %3
          "=r"(ldx)   // %4
        : "0"(src), "1"(dst), "2"(k), "3"(n), "4"(ldx)
        : "v0", "v2", "a0", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
}

void shl_c906_reorder_matrix_z16_fp16(__fp16* src, __fp16* dst, int k, int n, int ldx)
{
    asm volatile(
        "vsetvli        zero, zero, e16, m2\n\t"  // set vl = 8

        "slli           t2, %4, 1\n\t"  // t2 = ldx * 2 (line stride)

        "srai           t0, %3, 4\n\t"  // t0 = n16
        "beqz           t0, 3f\n\t"     // jump to packn_tail

        "1:\n\t"  // n8
        "mv             a0, %0\n\t"
        "addi           %0, %0, 32\n\t"
        "mv             t1, %2\n\t"  // k

        "2:\n\t"
        // start packn8k1
        "vle.v          v2, (a0)\n\t"
        "add            a0, a0, t2\n\t"
        "vse.v          v2, (%1)\n\t"
        "addi           %1, %1, 32\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "3:\n\t"                         // n_tail
        "andi           t0, %3, 15\n\t"  // n & 15u
        "beqz           t0, 8f\n\t"

        "srai           t3, %2, 4\n\t"   // k15
        "slli           t5, %4, 5\n\t"   // t5 = ldx * 16 * 2 (16 lines)
        "andi           t6, %2, 15\n\t"  // k_tail
        "slli           t4, t6, 1\n\t"   // k_tail * 2

        "4:\n\t"
        "mv             a0, %0\n\t"
        "addi           %0, %0, 2\n\t"
        "mv             t1, t3\n\t"  // t1 = k8
        "beqz           t3, 6f\n\t"

        "5:\n\t"
        "vsetvli        zero, zero, e16, m2\n\t"
        "vlse.v         v2, (a0), t2\n\t"
        "add            a0, a0, t5\n\t"
        "vse.v          v2, (%1)\n\t"
        "addi           %1, %1, 32\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 5b\n\t"

        "6:\n\t"
        "vsetvli        zero, t6, e16, m2\n\t"
        "vlse.v         v2, (a0), t2\n\t"
        "vse.v          v2, (%1)\n\t"
        "add            %1, %1, t4\n\t"

        "7:\n\t"
        "addi           t0, t0, -1\n\t"
        "bnez           t0, 4b\n\t"

        "8:\n\t"  // ending

        : "=r"(src),  // %0
          "=r"(dst),  // %1
          "=r"(k),    // %2
          "=r"(n),    // %3
          "=r"(ldx)   // %4
        : "0"(src), "1"(dst), "2"(k), "3"(n), "4"(ldx)
        : "v0", "v2", "v3", "a0", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
}

/*
    vector: 1 x k
    matrix: n x k
*/
void shl_c906_gemv_pack8_fp16(__fp16* dst, const __fp16* sa, const __fp16* sb, int k, int n,
                              int ldc, __fp16* bias)
{
}

void shl_c906_gemv_pack16_fp16(__fp16* dst, const __fp16* sa, const __fp16* sb, int k, int n,
                               int ldc, __fp16* bias)
{
}

/*
    vector: 1 x k
    matrix: k x n
*/
void shl_c906_gemv_trans_pack8_fp16(__fp16* dst, const __fp16* sa, const __fp16* sb, int k, int n,
                                    int ldc, __fp16* bias)
{
    asm volatile(
        "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 8

        "flh            ft0, (%3)\n\t"  // bias

        "srai           t4, %4, 3\n\t"  // k >> 3 (k8)
        "srai           t0, %5, 3\n\t"  // n >> 3 (n8)
        "beqz           t0, 3f\n\t"

        "1:\n\t"                      // m1n8
        "vfmv.v.f       v4, ft0\n\t"  // v4[0..n] = bias

        "mv             t1, %4\n\t"  // (k)
        "mv             t6, %1\n\t"  // vector start addr

        "2:\n\t"
        // m1n8k1
        "vle.v          v2, (%2)\n\t"
        "addi           %2, %2, 16\n\t"
        "flh            fa0, 0(t6)\n\t"
        "vfmacc.vf      v4, fa0, v2\n\t"
        "addi           t6, t6, 2\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        "vse.v          v4, (%0)\n\t"
        "addi           %0, %0, 16\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "3:\n\t"                        // n_tail
        "andi           t0, %5, 7\n\t"  // n_tail
        "beqz           t0, 8f\n\t"     // if n_tail = 0, jump to ending

        "andi           t2, %4, 7\n\t"  // k_tail
        "slli           t3, t2, 1\n\t"  // k_tail * 2

        "4:\n\t"
        "mv             t6, %1\n\t"  // init input_data addr

        "vmv.v.x        v4, zero\n\t"  // clear acc
        "vfmv.s.f       v3, ft0\n\t"   // v3[0] = bias

        "mv             t5, t4\n\t"  // t5 = k8
        "beqz           t2, 6f\n\t"

        "5:\n\t"
        // m1n1k_tail
        "vsetvli        zero, t2, e16, m1\n\t"
        "vle.v          v1, (t6)\n\t"
        "add            t6, t6, t3\n\t"
        "vle.v          v2, (%2)\n\t"
        "add            %2, %2, t3\n\t"
        "vfmacc.vv      v4, v1, v2\n\t"

        "beqz           t4, 7f\n\t"  // if k8 == 0, jump to end m1n1
        "vsetvli        zero, zero, e16, m1\n\t"

        "6:\n\t"
        // m1n1k8
        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 16\n\t"
        "vle.v          v2, (%2)\n\t"
        "addi           %2, %2, 16\n\t"
        "vfmacc.vv      v4, v1, v2\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 6b\n\t"

        "7:\n\t"                         // end m1n1
        "vfredsum.vs    v3, v4, v3\n\t"  // v3[0] = v3[0](bias) + sum(v4[0..7])
        "vfmv.f.s       fa0, v3\n\t"
        "fsh            fa0, 0(%0)\n\t"
        "addi           %0, %0, 2\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 4b\n\t"

        "8:\n\t"  // ending

        : "=r"(dst),   // %0
          "=r"(sa),    // %1
          "=r"(sb),    // %2
          "=r"(bias),  // %3
          "=r"(k),     // %4
          "=r"(n)      // %5
        : "0"(dst), "1"(sa), "2"(sb), "3"(bias), "4"(k), "5"(n)
        : "v1", "v2", "v3", "v4", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "fa0", "ft0");
}

void shl_c906_gemv_trans_pack16_fp16(__fp16* dst, const __fp16* sa, const __fp16* sb, int k, int n,
                                     int ldc, __fp16* bias)
{
    asm volatile(
        "vsetvli        zero, zero, e16, m2\n\t"  // set vl = 8

        "flh            ft0, (%3)\n\t"  // bias

        "srai           t4, %4, 4\n\t"  // k >> 4 (k16)
        "srai           t0, %5, 4\n\t"  // n >> 4 (n16)
        "beqz           t0, 3f\n\t"

        "1:\n\t"                      // m1n8
        "vfmv.v.f       v4, ft0\n\t"  // v4[0..n] = bias

        "mv             t1, %4\n\t"  // (k)
        "mv             t6, %1\n\t"  // vector start addr

        "2:\n\t"
        // m1n8k1
        "vle.v          v2, (%2)\n\t"
        "addi           %2, %2, 32\n\t"
        "flh            fa0, 0(t6)\n\t"
        "addi           t6, t6, 2\n\t"
        "vfmacc.vf      v4, fa0, v2\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        "vse.v          v4, (%0)\n\t"
        "addi           %0, %0, 32\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "3:\n\t"                         // n_tail
        "andi           t0, %5, 15\n\t"  // n_tail
        "beqz           t0, 8f\n\t"      // if n_tail = 0, jump to ending

        "andi           t2, %4, 15\n\t"  // k_tail
        "slli           t3, t2, 1\n\t"   // k_tail * 2

        "4:\n\t"
        "mv             t6, %1\n\t"  // init input_data addr

        "vmv.v.x        v4, zero\n\t"  // clear acc
        "vfmv.s.f       v8, ft0\n\t"   // v8[0] = bias

        "mv             t5, t4\n\t"  // t5 = k16
        "beqz           t2, 6f\n\t"

        "5:\n\t"
        // m1n1k_tail
        "vsetvli        zero, t2, e16, m2\n\t"
        "vle.v          v6, (t6)\n\t"
        "add            t6, t6, t3\n\t"
        "vle.v          v2, (%2)\n\t"
        "add            %2, %2, t3\n\t"
        "vfmacc.vv      v4, v2, v6\n\t"

        "beqz           t4, 7f\n\t"  // if k16 == 0, jump to end m1n1
        "vsetvli        zero, zero, e16, m2\n\t"

        "6:\n\t"
        // m1n1k16
        "vle.v          v6, (t6)\n\t"
        "addi           t6, t6, 32\n\t"
        "vle.v          v2, (%2)\n\t"
        "addi           %2, %2, 32\n\t"
        "vfmacc.vv      v4, v2, v6\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 6b\n\t"

        "7:\n\t"                         // end m1n1
        "vfredsum.vs    v8, v4, v8\n\t"  // v3[0] = v3[0](bias) + sum(v4[0..7])
        "vfmv.f.s       fa0, v8\n\t"
        "fsh            fa0, 0(%0)\n\t"
        "addi           %0, %0, 2\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 4b\n\t"

        "8:\n\t"  // ending

        : "=r"(dst),   // %0
          "=r"(sa),    // %1
          "=r"(sb),    // %2
          "=r"(bias),  // %3
          "=r"(k),     // %4
          "=r"(n)      // %5
        : "0"(dst), "1"(sa), "2"(sb), "3"(bias), "4"(k), "5"(n)
        : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "t0", "t1", "t2", "t3", "t4", "t5",
          "t6", "fa0", "ft0");
}
