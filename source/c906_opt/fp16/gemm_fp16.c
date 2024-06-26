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
    (1) Algorithm works as follows:
        change memory layout for matrix A (kernel matrix)
        memory index from  ------>  to  (col: 8 -> 4 -> 2 -> 1)
        0   1   2   3                0   8  16  24
        4   5   6   7                1   9  17  25
        8   9  10  11                2  10  18  26
       12  13  14  15                3  11  19  27
       16  17  18  19                4  12  20  28
       20  21  22  23                5  13  21  29
       24  25  26  27                6  14  22  30
       28  29  30  31                7  15  23  31
       32  33  34  35               32  36  40  44
       36  37  38  39               33  37  41  45
       40  41  42  43               34  38  42  46
       44  45  46  47               35  39  43  47
       48  49  50  51               48  50  52  54
       52  53  54  55               49  51  53  55
       56  57  58  59               56  57  58  59

    (2) register definition:
        t0:         i_n
        t1-t2:      i_k  k8, k_tail
        t3-t4:
        t5-t6:
        s2:         store stride for m8
        s3:         k8 tmp
        a0-a7:      8 rows addr for load
        v0-v14:     memcpy load / store v reg

    notice: called in the initialization function (shl_c906_conv2d_init)

*/
void shl_c906_reorder_kernel_fp16(__fp16* a, __fp16* sa, int m, int k, int ldx)
{
    asm volatile(
        "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 8

        "srai           t1, %3, 3\n\t"  // t1 = k >> 3 (k8)
        "andi           t2, %3, 7\n\t"  // t2 = k & 7 (k_tail)

        "slli           t3, %4, 1\n\t"  // t3 = ldx * 2
        "slli           t5, t2, 1\n\t"  // t5 = k_tail * 2

        "srai           t0, %2, 3\n\t"  // t0 = m >> 3 (m8)
        "beqz           t0, 5f\n\t"     // jump to packm4

        "slli           t4, %4, 4\n\t"  // t4 = 8 * ldx * 2
        "li             s2, 16\n\t"     // store_stride 8 elements

        "1:\n\t"
        // start packm8
        "mv             a0, %0\n\t"      // a0 = a
        "add            a1, a0, t3\n\t"  // a1 = a0 + 2 * ldx
        "add            a2, a1, t3\n\t"  // a2 = a1 + 2 * ldx
        "add            a3, a2, t3\n\t"  // a3 = a2 + 2 * ldx
        "add            a4, a3, t3\n\t"  // a4 = a3 + 2 * ldx
        "add            a5, a4, t3\n\t"  // a5 = a4 + 2 * ldx
        "add            a6, a5, t3\n\t"  // a6 = a5 + 2 * ldx
        "add            a7, a6, t3\n\t"  // a7 = a6 + 2 * ldx

        "beqz           t1, 3f\n\t"  // if k8 == 0, jump to subpack_m8ktail
        "mv             s3, t1\n\t"

        "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 8

        "2:\n\t"
        // start subpack_m8k8
        "vle.v          v0, (a0)\n\t"
        "addi           a0, a0, 16\n\t"  // +8 elements addr
        "vle.v          v2, (a1)\n\t"
        "addi           a1, a1, 16\n\t"
        "vle.v          v4, (a2)\n\t"
        "addi           a2, a2, 16\n\t"
        "vle.v          v6, (a3)\n\t"
        "addi           a3, a3, 16\n\t"
        "vle.v          v8, (a4)\n\t"
        "addi           a4, a4, 16\n\t"
        "vle.v          v10, (a5)\n\t"
        "addi           a5, a5, 16\n\t"
        "vle.v          v12, (a6)\n\t"
        "addi           a6, a6, 16\n\t"
        "vle.v          v14, (a7)\n\t"
        "addi           a7, a7, 16\n\t"

        "vsse.v         v0, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v2, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v4, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v6, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v8, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v10, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v12, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v14, (%1), s2\n\t"
        "addi           %1, %1, -14\n\t"
        "addi           %1, %1, 128\n\t"  // sa += 64 ele * 2

        "addi           s3, s3, -1\n\t"
        "bnez           s3, 2b\n\t"

        "3:\n\t"
        "beqz           t2, 4f\n\t"     // k_tail == 0 ?
        "slli           t6, t2, 4\n\t"  // t6 = 8 * k_tail * 2

        // start subpack_m8ktail
        "vsetvli        zero, t2, e16, m1\n\t"

        "vle.v          v0, (a0)\n\t"
        "add            a0, a0, t5\n\t"  // +k_tail elements addr
        "vle.v          v2, (a1)\n\t"
        "add            a1, a1, t5\n\t"
        "vle.v          v4, (a2)\n\t"
        "add            a2, a2, t5\n\t"
        "vle.v          v6, (a3)\n\t"
        "add            a3, a3, t5\n\t"
        "vle.v          v8, (a4)\n\t"
        "add            a4, a4, t5\n\t"
        "vle.v          v10, (a5)\n\t"
        "add            a5, a5, t5\n\t"
        "vle.v          v12, (a6)\n\t"
        "add            a6, a6, t5\n\t"
        "vle.v          v14, (a7)\n\t"
        "add            a7, a7, t5\n\t"

        "vsse.v         v0, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v2, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v4, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v6, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v8, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v10, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v12, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v14, (%1), s2\n\t"
        "addi           %1, %1, -14\n\t"
        "add            %1, %1, t6\n\t"  // sa += 8 * k_tail * 2

        "4:\n\t"

        // end packm8
        "add            %0, %0, t4\n\t"  // a += 8 * ldx * 2
        "addi           t0, t0, -1\n\t"  // m8--
        "bnez           t0, 1b\n\t"

        "5:\n\t"
        "andi           t0, %2, 7\n\t"  // m & 7
        "srai           t0, t0, 2\n\t"  // (m & 7) >> 2 (m4)
        "beqz           t0, 9f\n\t"     // jump to packm2
        // start packm4
        "mv             a0, %0\n\t"
        "add            a1, a0, t3\n\t"
        "add            a2, a1, t3\n\t"
        "add            a3, a2, t3\n\t"

        "li             s2, 8\n\t"      // store_stride 4 elements
        "slli           t4, %4, 3\n\t"  // t4 = 4 * ldx * 2

        "beqz           t1, 7f\n\t"  // if k8 == 0, jump to subpack_m4ktail
        "mv             s3, t1\n\t"

        "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 8

        "6:\n\t"
        // start subpack_m4k8
        "vle.v          v0, (a0)\n\t"
        "addi           a0, a0, 16\n\t"
        "vle.v          v2, (a1)\n\t"
        "addi           a1, a1, 16\n\t"
        "vle.v          v4, (a2)\n\t"
        "addi           a2, a2, 16\n\t"
        "vle.v          v6, (a3)\n\t"
        "addi           a3, a3, 16\n\t"

        "vsse.v         v0, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v2, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v4, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v6, (%1), s2\n\t"
        "addi           %1, %1, -6\n\t"
        "addi           %1, %1, 64\n\t"  // sa += 4 * 8 * 2

        "addi           s3, s3, -1\n\t"
        "bnez           s3, 6b\n\t"

        "7:\n\t"
        "beqz           t2, 8f\n\t"     // k_tail == 0 ?
        "slli           t6, t2, 3\n\t"  // t6 = 4 * k_tail * 2
        // start subpack_m4ktail
        "vsetvli        zero, t2, e16, m1\n\t"

        "vle.v          v0, (a0)\n\t"
        "add            a0, a0, t5\n\t"
        "vle.v          v2, (a1)\n\t"
        "add            a1, a1, t5\n\t"
        "vle.v          v4, (a2)\n\t"
        "add            a2, a2, t5\n\t"
        "vle.v          v6, (a3)\n\t"
        "add            a3, a3, t5\n\t"

        "vsse.v         v0, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v2, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v4, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v6, (%1), s2\n\t"
        "addi           %1, %1, -6\n\t"
        "add            %1, %1, t6\n\t"  // sa += 4 * k_tail * 2

        "8:\n\t"
        // end packm4
        "add            %0, %0, t4\n\t"  // a += 4 * ldx * 2

        "9:\n\t"
        "andi           t0, %2, 3\n\t"  // m & 3
        "srai           t0, t0, 1\n\t"  // (m & 3) >> 1 (m2)
        "beqz           t0, 13f\n\t"    // jump to packm1
        // start packm2
        "mv             a0, %0\n\t"
        "add            a1, a0, t3\n\t"

        "li             s2, 4\n\t"      // store_stride 2 elements
        "slli           t4, %4, 2\n\t"  // t4 = 2 * ldx * 2

        "beqz           t1, 11f\n\t"  // if k8 == 0, jump to subpack_m2ktail
        "mv             s3, t1\n\t"

        "vsetvli        zero, zero, e16, m1\n\t"

        "10:\n\t"
        // start subpack_m2k8
        "vle.v          v0, (a0)\n\t"
        "addi           a0, a0, 16\n\t"
        "vle.v          v2, (a1)\n\t"
        "addi           a1, a1, 16\n\t"

        "vsse.v         v0, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v2, (%1), s2\n\t"
        "addi           %1, %1, -2\n\t"
        "addi           %1, %1, 32\n\t"  // sa += 4 * 8 * 2

        "addi           s3, s3, -1\n\t"
        "bnez           s3, 10b\n\t"

        "11:\n\t"
        "beqz           t2, 12f\n\t"    // k_tail == 0 ?
        "slli           t6, t2, 2\n\t"  // t6 = 2 * k_tail * 2
        // start subpack_m2ktail
        "vsetvli        zero, t2, e16, m1\n\t"

        "vle.v          v0, (a0)\n\t"
        "add            a0, a0, t5\n\t"
        "vle.v          v2, (a1)\n\t"
        "add            a1, a1, t5\n\t"

        "vsse.v         v0, (%1), s2\n\t"
        "addi           %1, %1, 2\n\t"
        "vsse.v         v2, (%1), s2\n\t"
        "addi           %1, %1, -2\n\t"
        "add            %1, %1, t6\n\t"  // sa += 2 * k_tail * 2

        "12:\n\t"
        // end packm2
        "add            %0, %0, t4\n\t"  // a += 2 * ldx * 2

        "13:\n\t"
        "andi           t0, %2, 1\n\t"  // m & 1
        "beqz           t0, 16f\n\t"    // jump to ending
        // start packm1 (factually, memcpy)
        "mv             a0, %0\n\t"

        "beqz           t1, 15f\n\t"  // if k8 == 0, jump to subpack_m1ktail
        "mv             s3, t1\n\t"

        "vsetvli        zero, zero, e16, m1\n\t"

        "14:\n\t"
        // start subpack_m1k8
        "vle.v          v0, (a0)\n\t"
        "addi           a0, a0, 16\n\t"

        "vse.v          v0, (%1)\n\t"
        "addi           %1, %1, 16\n\t"

        "addi           s3, s3, -1\n\t"
        "bnez           s3, 14b\n\t"

        "15:\n\t"
        "beqz           t2, 16f\n\t"  // k_tail == 0 ?
        // start subpack_m1ktail
        "vsetvli        zero, t2, e16, m1\n\t"

        "vle.v          v0, (a0)\n\t"
        "add            a0, a0, t5\n\t"

        "vse.v          v0, (%1)\n\t"
        "add            %1, %1, t5\n\t"  // sa += k_tail * 2

        "16:\n\t"  // ending

        : "=r"(a),   // %0
          "=r"(sa),  // %1
          "=r"(m),   // %2
          "=r"(k),   // %3
          "=r"(ldx)  // %4
        : "0"(a), "1"(sa), "2"(m), "3"(k), "4"(ldx)

        : "v0", "v2", "v4", "v6", "v8", "v10", "v12", "v14", "a0", "a1", "a2", "a3", "a4", "a5",
          "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s2", "s3");
}

void shl_c906_reorder_input_fp16_1(__fp16* b, __fp16* sb, int k, int n, int ldx)
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

        : "=r"(b),   // %0
          "=r"(sb),  // %1
          "=r"(k),   // %2
          "=r"(n),   // %3
          "=r"(ldx)  // %4
        : "0"(b), "1"(sb), "2"(k), "3"(n), "4"(ldx)
        : "v0", "v2", "a0", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
}

/*
    (1) Algorithm works as follows:
        m1n8_loop:  m1n8k8_loop  -->  m1n8k4  -->  m1n8k2 -->  m1n8k1
        m1n4:       m1n4k8_loop  -->  m1n4k4  -->  m1n4k2 -->  m1n4k1
        m1n2:       m1n2k8_loop  -->  m1n2k4  -->  m1n2k2 -->  m1n2k1
        m1n1:       m1n1k8_loop  -->  m1n1k4  -->  m1n1k2 -->  m1n1k1

        n8_loop:    vfmacc.vf = kernel_data(f reg) * input_data(v reg)
        n4, n2, n1: vfmacc.vv = kernel_data(v reg - 1row) * input_data(v reg 1col) , reduce sum v
   reg, add bias

    (2) register definition:
        t0:         i_n
        t1-t5:      i_k  t1-t4:[k8, k4, k2, k1] t5:k8_tmp  because i_k for inner_loop, extract to
   the outside t6:         [ n8_loop ]: hold sa/ kernel_data 1 lines origin address a0-a7:      dst,
   sa, sb addr ft0:        [ n8_loop ] : sa / kernel_data fa0-fa3:    [ n8_loop ] : sa / kernel_data
   (sb / input_data) shadow reg  [ n4, n2 n1 ]: output res fs0:        hold 1 channels bias_data
        v1-v8:      [ n8_loop ] : sb / input_data   [ n4, n2 n1 ]: sb / kernel_data  and  sb /
   input_data v24-v31:    bias_tmp and output

    TODO: if bias == NULL
*/
static void kernel_m1_fp16(__fp16* dst, __fp16* sa, __fp16* sb, int m, int k, int n, int ldc,
                           __fp16* bias)
{
    asm volatile(
        "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 8

        "srai           t1, %4, 3\n\t"  // t1 = k >> 3 (k8)
        "andi           t2, %4, 7\n\t"  // t2 = k & 7
        "srai           t2, t2, 2\n\t"  // t2 = (k & 7) >> 2 (k4)
        "andi           t3, %4, 3\n\t"  // t3 = k & 3
        "srai           t3, t3, 1\n\t"  // t3 = (k & 3) >> 1 (k2)
        "andi           t4, %4, 1\n\t"  // t4 = k & 1 (k1)

        "flh            fs0, 0(%2)\n\t"  // load 1 bias_data for 1 out_channel

        "mv             a0, %3\n\t"  // init output addr

        "srai           t0, %5, 3\n\t"  // t0 = n >> 3 (n8)
        "beqz           t0, 7f\n\t"     // jump to m1n4

        "1:\n\t"                       // m1n8
                                       // start kernel_m1n8
        "vfmv.v.f       v24, fs0\n\t"  // init out_tmp = bias

        "mv             t6, %0\n\t"  // t6 hold kernel 1 lines start addr

        "vle.v          v1, (%1)\n\t"  // pre-load pb (input_data)
        "addi           %1, %1, 16\n\t"

        "flh            ft0, 0(t6)\n\t"  // pre-load pa (kernel_data)

        "beqz           t1, 3f\n\t"  // if k8 == 0, jump to subkernel_m1n8k4
        "mv             t5, t1\n\t"  // t5 = k8

        "2:\n\t"
        // start subkernel_m1n8k8
        "vle.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 2(t6)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"  // 0

        "vle.v          v3, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 4(t6)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"  // 1

        "vle.v          v4, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 6(t6)\n\t"
        "vfmacc.vf      v24, ft0, v3\n\t"  // 2

        "vle.v          v5, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 8(t6)\n\t"
        "vfmacc.vf      v24, fa0, v4\n\t"  // 3

        "vle.v          v6, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 10(t6)\n\t"
        "vfmacc.vf      v24, ft0, v5\n\t"  // 4

        "vle.v          v7, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 12(t6)\n\t"
        "vfmacc.vf      v24, fa0, v6\n\t"  // 5

        "vle.v          v8, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 14(t6)\n\t"
        "vfmacc.vf      v24, ft0, v7\n\t"  // 6
        "addi           t6, t6, 16\n\t"    // +8 elements, bump pa(kernel_data) to next k8 addr

        "vle.v          v1, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 0(t6)\n\t"
        "vfmacc.vf      v24, fa0, v8\n\t"  // 7

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 2b\n\t"

        "3:\n\t"
        "beqz           t2, 4f\n\t"  // if k4 == 0, jump to subkernel_m1n8k2
        // start subkernel_m1n8k4
        "vle.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 2(t6)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"  // 0

        "vle.v          v3, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 4(t6)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"  // 1

        "vle.v          v4, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 6(t6)\n\t"
        "vfmacc.vf      v24, ft0, v3\n\t"  // 2
        "addi           t6, t6, 8\n\t"     // +4 elements, bump pa to next k addr

        "vle.v          v1, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 0(t6)\n\t"
        "vfmacc.vf      v24, fa0, v4\n\t"  // 3

        "4:\n\t"
        "beqz           t3, 5f\n\t"  // if k2 == 0, jump to subkernel_m1n8k1
        // start subkernel_m1n8k2
        "vle.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 2(t6)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"  // 0
        "addi           t6, t6, 4\n\t"     // +2 elements, bump pa to next k addr

        "vle.v          v1, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 0(t6)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"  // 1

        "5:\n\t"
        "beqz           t4, 6f\n\t"  // if k1 == 0, jump to end kernel_m1n8
        // start subkernel_m1n8k1
        "vfmacc.vf      v24, ft0, v1\n\t"  // 0

        "addi           %1, %1, 16\n\t"  // ********************

        "6:\n\t"  // end kernel_m1n8

        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %1, %1, -16\n\t"  // pb -= 8

        "vse.v          v24, (a0)\n\t"
        "addi           a0, a0, 16\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "7:\n\t"  // m1n4

        // prepare for n4 n2 n1
        "andi           t2, %4, 7\n\t"  // t2 = k_tail
        "slli           t3, t2, 1\n\t"  // t3 = k_tail * 2

        "andi           t0, %5, 7\n\t"  // n & 7
        "srai           t0, t0, 2\n\t"  // (n & 7) >> 2
        "beqz           t0, 11f\n\t"    // jump to m1n2
        // start kernel_m1n4

        "vmv.v.x        v24, zero\n\t"
        "vmv.v.x        v25, zero\n\t"
        "vmv.v.x        v26, zero\n\t"
        "vmv.v.x        v27, zero\n\t"  // clear acc

        "vfmv.s.f       v28, fs0\n\t"  // v28[0] = bias
        "vfmv.s.f       v29, fs0\n\t"  // v29[0] = bias
        "vfmv.s.f       v30, fs0\n\t"  // v30[0] = bias
        "vfmv.s.f       v31, fs0\n\t"  // v31[0] = bias

        // init addr for pa, pb and pc
        "slli           t0, %4, 1\n\t"  // t_tmp = k * 2

        "mv             t6, %0\n\t"  // t6 hold pa(kernel) 1 lines start addr

        "mv             a4, %1\n\t"
        "add            a5, a4, t0\n\t"
        "add            a6, a5, t0\n\t"
        "add            a7, a6, t0\n\t"  // a4-a7 hold pb(input) 4 cols addr

        // a0 hold pc(output) addr

        "mv             t5, t1\n\t"  // t5 = k8
        "beqz           t2, 9f\n\t"  // if k_tail == 0, jump to subkernel_m1n4k8

        "8:\n\t"
        // start subkernel_m1n4k_tail
        "vsetvli        zero, t2, e16, m1\n\t"
        "vle.v          v1, (t6)\n\t"
        "add            t6, t6, t3\n\t"
        "vle.v          v2, (a4)\n\t"
        "add            a4, a4, t3\n\t"
        "vle.v          v3, (a5)\n\t"
        "add            a5, a5, t3\n\t"
        "vle.v          v4, (a6)\n\t"
        "add            a6, a6, t3\n\t"
        "vle.v          v5, (a7)\n\t"
        "add            a7, a7, t3\n\t"
        "vfmacc.vv      v24, v1, v2\n\t"
        "vfmacc.vv      v25, v1, v3\n\t"
        "vfmacc.vv      v26, v1, v4\n\t"
        "vfmacc.vv      v27, v1, v5\n\t"

        "beqz           t1, 10f\n\t"  // if k8 == 0, jump to end kernel_m1n4
        "vsetvli        zero, zero, e16, m1\n\t"

        "9:\n\t"
        // start subkernel_m1n4k8
        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 16\n\t"
        "vle.v          v2, (a4)\n\t"
        "addi           a4, a4, 16\n\t"
        "vle.v          v3, (a5)\n\t"
        "addi           a5, a5, 16\n\t"
        "vle.v          v4, (a6)\n\t"
        "addi           a6, a6, 16\n\t"
        "vle.v          v5, (a7)\n\t"
        "addi           a7, a7, 16\n\t"
        "vfmacc.vv      v24, v1, v2\n\t"
        "vfmacc.vv      v25, v1, v3\n\t"
        "vfmacc.vv      v26, v1, v4\n\t"
        "vfmacc.vv      v27, v1, v5\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 9b\n\t"

        "10:\n\t"  // end kernel_m1n4

        "vfredsum.vs    v28, v24, v28\n\t"  // v28[0] = v28[0](bias) + sum(v24[0..7])
        "vfredsum.vs    v29, v25, v29\n\t"
        "vfredsum.vs    v30, v26, v30\n\t"
        "vfredsum.vs    v31, v27, v31\n\t"
        "vfmv.f.s       fa0, v28\n\t"
        "vfmv.f.s       fa1, v29\n\t"
        "vfmv.f.s       fa2, v30\n\t"
        "vfmv.f.s       fa3, v31\n\t"
        "fsh            fa0, 0(a0)\n\t"
        "fsh            fa1, 2(a0)\n\t"
        "fsh            fa2, 4(a0)\n\t"
        "fsh            fa3, 6(a0)\n\t"

        "addi           a0, a0, 8\n\t"   // updata output start addr ( +4 cols)
        "slli           t0, %4, 3\n\t"   // t_tmp = k * 4 * 2
        "add            %1, %1, t0\n\t"  // updata pb start addr

        "11:\n\t"                       // m1n2
        "andi           t0, %5, 3\n\t"  // n & 3
        "srai           t0, t0, 1\n\t"  // (n & 3) >> 1
        "beqz           t0, 15f\n\t"    // jump to m1n1
        // start kernel_m1n2

        "vmv.v.x        v24, zero\n\t"
        "vmv.v.x        v25, zero\n\t"  // clear acc

        "vfmv.s.f       v28, fs0\n\t"  // v28[0] = bias
        "vfmv.s.f       v29, fs0\n\t"  // v29[0] = bias

        // init addr for pa, pb and pc
        "slli           t0, %4, 1\n\t"  // t_tmp = k * 2

        "mv             t6, %0\n\t"  // t6 hold pa(kernel) 8 lines start addr

        "mv             a4, %1\n\t"
        "add            a5, a4, t0\n\t"  // a4-a5 hold pb(input) 4 cols addr

        // a0 hold pc(output) addr

        "mv             t5, t1\n\t"   // t5 = k8
        "beqz           t2, 13f\n\t"  // if k_tail == 0, jump to subkernel_m1n2k8

        "12:\n\t"
        // start subkernel_m1n2k_tail
        "vsetvli        zero, t2, e16, m1\n\t"
        "vle.v          v1, (t6)\n\t"
        "add            t6, t6, t3\n\t"
        "vle.v          v2, (a4)\n\t"
        "add            a4, a4, t3\n\t"
        "vle.v          v3, (a5)\n\t"
        "add            a5, a5, t3\n\t"
        "vfmacc.vv      v24, v1, v2\n\t"
        "vfmacc.vv      v25, v1, v3\n\t"

        "beqz           t1, 14f\n\t"  // if k8 == 0, jump to end kernel_m1n2
        "vsetvli        zero, zero, e16, m1\n\t"

        "13:\n\t"
        // start subkernel_m1n2k8
        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 16\n\t"
        "vle.v          v2, (a4)\n\t"
        "addi           a4, a4, 16\n\t"
        "vle.v          v3, (a5)\n\t"
        "addi           a5, a5, 16\n\t"
        "vfmacc.vv      v24, v1, v2\n\t"
        "vfmacc.vv      v25, v1, v3\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 13b\n\t"

        "14:\n\t"  // end kernel_m1n2

        "vfredsum.vs    v28, v24, v28\n\t"  // v28[0] = v28[0](bias) + sum(v24[0..7])
        "vfredsum.vs    v29, v25, v29\n\t"
        "vfmv.f.s       fa0, v28\n\t"
        "vfmv.f.s       fa1, v29\n\t"
        "fsh            fa0, 0(a0)\n\t"
        "fsh            fa1, 2(a0)\n\t"

        "addi           a0, a0, 4\n\t"   // updata output start addr ( +2 cols)
        "slli           t0, %4, 2\n\t"   // t_tmp = k * 2 * 2
        "add            %1, %1, t0\n\t"  // updata pb start addr

        "15:\n\t"                       // m1n1
        "andi           t0, %5, 1\n\t"  // n & 1
        "beqz           t0, 19f\n\t"    // jump to ending
        // start kernel_m1n1
        "vmv.v.x        v24, zero\n\t"  // clear acc

        "vfmv.s.f       v28, fs0\n\t"  // v28[0] = bias

        // init addr for pa, pb and pc
        "mv             t6, %0\n\t"  // t6 hold pa(kernel) 8 lines start addr

        "mv             a4, %1\n\t"  // a4-a5 hold pb(input) 4 cols addr

        // a0 hold pc(output) addr

        "mv             t5, t1\n\t"   // t5 = k8
        "beqz           t2, 17f\n\t"  // if k_tail == 0, jump to subkernel_m1n1k8

        "16:\n\t"
        // start subkernel_m1n1k_tail
        "vsetvli        zero, t2, e16, m1\n\t"
        "vle.v          v1, (t6)\n\t"
        "add            t6, t6, t3\n\t"
        "vle.v          v2, (a4)\n\t"
        "add            a4, a4, t3\n\t"
        "vfmacc.vv      v24, v1, v2\n\t"

        "beqz           t1, 18f\n\t"  // if k8 == 0, jump to end kernel_m1n1
        "vsetvli        zero, zero, e16, m1\n\t"

        "17:\n\t"
        // start subkernel_m1n1k8
        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 16\n\t"
        "vle.v          v2, (a4)\n\t"
        "addi           a4, a4, 16\n\t"
        "vfmacc.vv      v24, v1, v2\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 17b\n\t"

        "18:\n\t"                           // end kernel_m1n1
        "vfredsum.vs    v28, v24, v28\n\t"  // v28[0] = v28[0](bias) + sum(v24[0..7])
        "vfmv.f.s       fa0, v28\n\t"
        "fsh            fa0, 0(a0)\n\t"

        "19:\n\t"  // ending

        : "=r"(sa),    // %0
          "=r"(sb),    // %1
          "=r"(bias),  // %2
          "=r"(dst),   // %3
          "=r"(k),     // %4
          "=r"(n),     // %5
          "=r"(ldc)    // %6
        : "0"(sa), "1"(sb), "2"(bias), "3"(dst), "4"(k), "5"(n), "6"(ldc)
        : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3",
          "t4", "t5", "t6", "fa0", "fa1", "fa2", "fa3", "ft0", "fs0");
}

/*
    (1) Algorithm works as follows:
        m2n8_loop:  m2n8k8_loop  -->  m2n8k4  -->  m2n8k2 -->  m2n8k1
        m2n4:       m2n4k8_loop  -->  m2n4k4  -->  m2n4k2 -->  m2n4k1
        m2n2:       m2n2k8_loop  -->  m2n2k4  -->  m2n2k2 -->  m2n2k1
        m2n1:       m2n1k8_loop  -->  m2n1k4  -->  m2n1k2 -->  m2n1k1

        n8_loop:    vfmacc.vf = kernel_data(f reg) * input_data(v reg)
        n4, n2, n1: vfmacc.vv = kernel_data(v reg - 1row) * input_data(v reg 1col) , reduce sum v
   reg, add bias

    (2) register definition:
        t0:         i_n
        t1-t5:      i_k  t1-t4:[k8, k4, k2, k1] t5:k8_tmp  because i_k for inner_loop, extract to
   the outside t6:         [ n8_loop ]: hold sa/ kernel_data 2 lines origin address a0-a7:      dst,
   sa, sb addr ft0-ft1:    [ n8_loop ] : sa / kernel_data fa0-fa7:    [ n8_loop ] : sa / kernel_data
   (sb / input_data) shadow reg  [ n4, n2 n1 ]: output res fs0-fs1:    hold 2 channels bias_data
        v1-v8:      [ n8_loop ] : sb / input_data   [ n4, n2 n1 ]: sb / kernel_data  and  sb /
   input_data v24-v31:    bias_tmp and output

    TODO: if bias == NULL
*/
static void kernel_m2_fp16(__fp16* dst, __fp16* sa, __fp16* sb, int m, int k, int n, int ldc,
                           __fp16* bias)
{
    asm volatile(
        "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 8

        "srai           t1, %4, 3\n\t"  // t1 = k >> 3 (k8)
        "andi           t2, %4, 7\n\t"  // t2 = k & 7
        "srai           t2, t2, 2\n\t"  // t2 = (k & 7) >> 2 (k4)
        "andi           t3, %4, 3\n\t"  // t3 = k & 3
        "srai           t3, t3, 1\n\t"  // t3 = (k & 3) >> 1 (k2)
        "andi           t4, %4, 1\n\t"  // t4 = k & 1 (k1)

        "flh            fs0, 0(%2)\n\t"
        "flh            fs1, 2(%2)\n\t"  // load 2 bias_data for 2 out_channels

        // init output addr
        "slli           t5, %6, 1\n\t"  // t5_tmp = ldx * 2
        "mv             a0, %3\n\t"
        "add            a1, a0, t5\n\t"

        "srai           t0, %5, 3\n\t"  // t0 = n >> 3 (n8)
        "beqz           t0, 7f\n\t"     // jump to m2n4

        "1:\n\t"  // m2n8
                  // start kernel_m2n8
        "vfmv.v.f       v24, fs0\n\t"
        "vfmv.v.f       v25, fs1\n\t"  // init out_tmp = bias

        "mv             t6, %0\n\t"  // t6 hold kernel 2 lines start addr

        "vle.v          v1, (%1)\n\t"  // pre-load pb (input_data)
        "addi           %1, %1, 16\n\t"

        "flh            ft0, 0(t6)\n\t"
        "flh            ft1, 2(t6)\n\t"  // pre-load pa(kernel_data)

        "beqz           t1, 3f\n\t"  // if k8 == 0, jump to subkernel_m2n8k4
        "mv             t5, t1\n\t"  // t5 = k8

        "2:\n\t"
        // start subkernel_m2n8k8
        "vle.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 4(t6)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa1, 6(t6)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"  // 0

        "vle.v          v3, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 8(t6)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"
        "flh            ft1, 10(t6)\n\t"
        "vfmacc.vf      v25, fa1, v2\n\t"  // 1

        "vle.v          v4, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 12(t6)\n\t"
        "vfmacc.vf      v24, ft0, v3\n\t"
        "flh            fa1, 14(t6)\n\t"
        "vfmacc.vf      v25, ft1, v3\n\t"  // 2

        "vle.v          v5, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 16(t6)\n\t"
        "vfmacc.vf      v24, fa0, v4\n\t"
        "flh            ft1, 18(t6)\n\t"
        "vfmacc.vf      v25, fa1, v4\n\t"  // 3

        "vle.v          v6, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 20(t6)\n\t"
        "vfmacc.vf      v24, ft0, v5\n\t"
        "flh            fa1, 22(t6)\n\t"
        "vfmacc.vf      v25, ft1, v5\n\t"  // 4

        "vle.v          v7, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 24(t6)\n\t"
        "vfmacc.vf      v24, fa0, v6\n\t"
        "flh            ft1, 26(t6)\n\t"
        "vfmacc.vf      v25, fa1, v6\n\t"  // 5

        "vle.v          v8, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 28(t6)\n\t"
        "vfmacc.vf      v24, ft0, v7\n\t"
        "flh            fa1, 30(t6)\n\t"
        "vfmacc.vf      v25, ft1, v7\n\t"  // 6
        "addi           t6, t6, 32\n\t"    // +16 elements, bump pa(kernel_data) to next k8 addr

        "vle.v          v1, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 0(t6)\n\t"
        "vfmacc.vf      v24, fa0, v8\n\t"
        "flh            ft1, 2(t6)\n\t"
        "vfmacc.vf      v25, fa1, v8\n\t"  // 7

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 2b\n\t"

        "3:\n\t"
        "beqz           t2, 4f\n\t"  // if k4 == 0, jump to subkernel_m2n8k2
        // start subkernel_m2n8k4
        "vle.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 4(t6)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa1, 6(t6)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"  // 0

        "vle.v          v3, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 8(t6)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"
        "flh            ft1, 10(t6)\n\t"
        "vfmacc.vf      v25, fa1, v2\n\t"  // 1

        "vle.v          v4, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 12(t6)\n\t"
        "vfmacc.vf      v24, ft0, v3\n\t"
        "flh            fa1, 14(t6)\n\t"
        "vfmacc.vf      v25, ft1, v3\n\t"  // 2
        "addi           t6, t6, 16\n\t"    // +8 elements, bump pa to next k addr

        "vle.v          v1, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 0(t6)\n\t"
        "vfmacc.vf      v24, fa0, v4\n\t"
        "flh            ft1, 2(t6)\n\t"
        "vfmacc.vf      v25, fa1, v4\n\t"  // 3

        "4:\n\t"
        "beqz           t3, 5f\n\t"  // if k2 == 0, jump to subkernel_m2n8k1
        // start subkernel_m2n8k2
        "vle.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 4(t6)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa1, 6(t6)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"  // 0
        "addi           t6, t6, 8\n\t"     // +4 elements, bump pa to next k addr

        "vle.v          v1, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 0(t6)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"
        "flh            ft1, 2(t6)\n\t"
        "vfmacc.vf      v25, fa1, v2\n\t"  // 1

        "5:\n\t"
        "beqz           t4, 6f\n\t"  // if k1 == 0, jump to end kernel_m2n8
        // start subkernel_m2n8k1
        "vfmacc.vf      v24, ft0, v1\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"  // 0

        "addi           %1, %1, 16\n\t"  // ********************

        "6:\n\t"  // end kernel_m2n8

        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %1, %1, -16\n\t"  // pb -= 8

        "vse.v          v24, (a0)\n\t"
        "addi           a0, a0, 16\n\t"
        "vse.v          v25, (a1)\n\t"
        "addi           a1, a1, 16\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "7:\n\t"  // m2n4

        // prepare for n4 n2 n1
        "andi           t2, %4, 7\n\t"  // t2 = k_tail
        "slli           t3, t2, 1\n\t"  // t3 = k_tail * 2
        "li             t4, 4\n\t"      // load stride for pa
        "slli           t6, t2, 2\n\t"  // t6 = k_tail * 2 lines * 2 bytes

        /////////////////////////////////////////
        "andi           t0, %5, 7\n\t"  // n & 7
        "srai           t0, t0, 2\n\t"  // (n & 7) >> 2
        "beqz           t0, 11f\n\t"    // jump to m2n2
        // start kernel_m2n4

        "vmv.v.x        v24, zero\n\t"
        "vmv.v.x        v25, zero\n\t"
        "vmv.v.x        v26, zero\n\t"
        "vmv.v.x        v27, zero\n\t"
        "vmv.v.x        v28, zero\n\t"
        "vmv.v.x        v29, zero\n\t"
        "vmv.v.x        v30, zero\n\t"
        "vmv.v.x        v31, zero\n\t"  // clear acc

        // init addr for pa, pb and pc
        "slli           t0, %4, 1\n\t"  // t_tmp = k * 2

        "mv             a2, %0\n\t"
        "addi           a3, a2, 2\n\t"  // a2-a3 hold pa(kernel) 2 lines start addr

        "mv             a4, %1\n\t"
        "add            a5, a4, t0\n\t"
        "add            a6, a5, t0\n\t"
        "add            a7, a6, t0\n\t"  // a4-a7 hold pb(input) 4 cols addr

        // a0-a1 hold pc(output) 2 rows addr

        "mv             t5, t1\n\t"  // t5 = k8
        "beqz           t2, 9f\n\t"  // if k_tail == 0, jump to subkernel_m1n4k8

        "8:\n\t"
        // start subkernel_m2n4k_tail
        "vsetvli        zero, t2, e16, m1\n\t"

        "vlse.v         v1, (a2), t4\n\t"
        "add            a2, a2, t6\n\t"
        "vlse.v         v2, (a3), t4\n\t"
        "add            a3, a3, t6\n\t"  // load pa

        "vle.v          v3, (a4)\n\t"
        "add            a4, a4, t3\n\t"
        "vle.v          v4, (a5)\n\t"
        "add            a5, a5, t3\n\t"
        "vle.v          v5, (a6)\n\t"
        "add            a6, a6, t3\n\t"
        "vle.v          v6, (a7)\n\t"
        "add            a7, a7, t3\n\t"  // load pb

        "vfmacc.vv      v24, v1, v3\n\t"  // out[0][0]
        "vfmacc.vv      v25, v1, v4\n\t"  // out[0][1]
        "vfmacc.vv      v26, v1, v5\n\t"  // out[0][2]
        "vfmacc.vv      v27, v1, v6\n\t"  // out[0][3]
        "vfmacc.vv      v28, v2, v3\n\t"  // out[1][0]
        "vfmacc.vv      v29, v2, v4\n\t"  // out[1][1]
        "vfmacc.vv      v30, v2, v5\n\t"  // out[1][2]
        "vfmacc.vv      v31, v2, v6\n\t"  // out[1][3]

        "beqz           t1, 10f\n\t"  // if k8 == 0, jump to end kernel_m2n4
        "vsetvli        zero, zero, e16, m1\n\t"

        "9:\n\t"
        // start subkernel_m2n4k8
        "vlse.v         v1, (a2), t4\n\t"
        "addi           a2, a2, 32\n\t"  // +8 * 2 * 2
        "vlse.v         v2, (a3), t4\n\t"
        "addi           a3, a2, 2\n\t"  // load pa

        "vle.v          v3, (a4)\n\t"
        "addi           a4, a4, 16\n\t"
        "vle.v          v4, (a5)\n\t"
        "addi           a5, a5, 16\n\t"
        "vle.v          v5, (a6)\n\t"
        "addi           a6, a6, 16\n\t"
        "vle.v          v6, (a7)\n\t"
        "addi           a7, a7, 16\n\t"  // load pb

        "vfmacc.vv      v24, v1, v3\n\t"  // out[0][0]
        "vfmacc.vv      v25, v1, v4\n\t"  // out[0][1]
        "vfmacc.vv      v26, v1, v5\n\t"  // out[0][2]
        "vfmacc.vv      v27, v1, v6\n\t"  // out[0][3]
        "vfmacc.vv      v28, v2, v3\n\t"  // out[1][0]
        "vfmacc.vv      v29, v2, v4\n\t"  // out[1][1]
        "vfmacc.vv      v30, v2, v5\n\t"  // out[1][2]
        "vfmacc.vv      v31, v2, v6\n\t"  // out[1][3]

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 9b\n\t"

        "10:\n\t"                     // end kernel_m2n4
        "vfmv.s.f       v1, fs0\n\t"  // init out_tmp = bias
        "vfmv.s.f       v2, fs0\n\t"
        "vfmv.s.f       v3, fs0\n\t"
        "vfmv.s.f       v4, fs0\n\t"
        "vfmv.s.f       v5, fs1\n\t"
        "vfmv.s.f       v6, fs1\n\t"
        "vfmv.s.f       v7, fs1\n\t"
        "vfmv.s.f       v8, fs1\n\t"

        "vfredsum.vs    v1, v24, v1\n\t"  // v1[0] = v1[0](bias) + sum(v24[0..7])
        "vfredsum.vs    v2, v25, v2\n\t"
        "vfredsum.vs    v3, v26, v3\n\t"
        "vfredsum.vs    v4, v27, v4\n\t"
        "vfredsum.vs    v5, v28, v5\n\t"
        "vfredsum.vs    v6, v29, v6\n\t"
        "vfredsum.vs    v7, v30, v7\n\t"
        "vfredsum.vs    v8, v31, v8\n\t"

        "vfmv.f.s       fa0, v1\n\t"  // fa0 = v1[0]
        "vfmv.f.s       fa1, v2\n\t"
        "vfmv.f.s       fa2, v3\n\t"
        "vfmv.f.s       fa3, v4\n\t"
        "vfmv.f.s       fa4, v5\n\t"
        "vfmv.f.s       fa5, v6\n\t"
        "vfmv.f.s       fa6, v7\n\t"
        "vfmv.f.s       fa7, v8\n\t"

        "fsh            fa0, 0(a0)\n\t"
        "fsh            fa1, 2(a0)\n\t"
        "fsh            fa2, 4(a0)\n\t"
        "fsh            fa3, 6(a0)\n\t"
        "fsh            fa4, 0(a1)\n\t"
        "fsh            fa5, 2(a1)\n\t"
        "fsh            fa6, 4(a1)\n\t"
        "fsh            fa7, 6(a1)\n\t"

        "addi           a0, a0, 8\n\t"
        "addi           a1, a1, 8\n\t"   // updata output start addr ( +4 cols)
        "slli           t0, %4, 3\n\t"   // t_tmp = k * 4 * 2
        "add            %1, %1, t0\n\t"  // updata pb start addr

        "11:\n\t"                       // m2n2
        "andi           t0, %5, 3\n\t"  // n & 3
        "srai           t0, t0, 1\n\t"  // (n & 3) >> 1
        "beqz           t0, 15f\n\t"    // jump to m2n1
        // start kernel_m2n2

        "vmv.v.x        v24, zero\n\t"
        "vmv.v.x        v25, zero\n\t"
        "vmv.v.x        v26, zero\n\t"
        "vmv.v.x        v27, zero\n\t"  // clear acc

        "vfmv.s.f       v28, fs0\n\t"
        "vfmv.s.f       v29, fs0\n\t"
        "vfmv.s.f       v30, fs1\n\t"
        "vfmv.s.f       v31, fs1\n\t"  // init output = bias

        // init addr for pa, pb and pc
        "slli           t0, %4, 1\n\t"  // t_tmp = k * 2

        "mv             a2, %0\n\t"
        "addi           a3, a2, 2\n\t"  // a2-a3 hold pa(kernel) 2 lines start addr

        "mv             a4, %1\n\t"
        "add            a5, a4, t0\n\t"  // a4-a5 hold pb(input) 2 cols addr

        // a0-a1 hold pc(output) 2 rows addr

        "mv             t5, t1\n\t"   // t5 = k8
        "beqz           t2, 13f\n\t"  // if k_tail == 0, jump to subkernel_m2n2k8

        "12:\n\t"
        // start subkernel_m2n2k_tail
        "vsetvli        zero, t2, e16, m1\n\t"

        "vlse.v         v1, (a2), t4\n\t"
        "add            a2, a2, t6\n\t"
        "vlse.v         v2, (a3), t4\n\t"
        "add            a3, a3, t6\n\t"  // load pa

        "vle.v          v3, (a4)\n\t"
        "add            a4, a4, t3\n\t"
        "vle.v          v4, (a5)\n\t"
        "add            a5, a5, t3\n\t"  // load pb

        "vfmacc.vv      v24, v1, v3\n\t"  // out[0][0]
        "vfmacc.vv      v25, v1, v4\n\t"  // out[0][1]
        "vfmacc.vv      v26, v2, v3\n\t"  // out[1][0]
        "vfmacc.vv      v27, v2, v4\n\t"  // out[1][1]

        "beqz           t1, 14f\n\t"  // if k8 == 0, jump to end kernel_m2n2
        "vsetvli        zero, zero, e16, m1\n\t"

        "13:\n\t"
        // start subkernel_m2n2k8
        "vlse.v         v1, (a2), t4\n\t"
        "addi           a2, a2, 32\n\t"  // +8 * 2 * 2
        "vlse.v         v2, (a3), t4\n\t"
        "addi           a3, a2, 2\n\t"  // load pa

        "vle.v          v3, (a4)\n\t"
        "addi           a4, a4, 16\n\t"
        "vle.v          v4, (a5)\n\t"
        "addi           a5, a5, 16\n\t"  // load pb

        "vfmacc.vv      v24, v1, v3\n\t"  // out[0][0]
        "vfmacc.vv      v25, v1, v4\n\t"  // out[0][1]
        "vfmacc.vv      v26, v2, v3\n\t"  // out[1][0]
        "vfmacc.vv      v27, v2, v4\n\t"  // out[1][1]

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 13b\n\t"

        "14:\n\t"                           // end kernel_m2n2
        "vfredsum.vs    v28, v24, v28\n\t"  // v28[0] = v28[0](bias) + sum(v24[0..7])
        "vfredsum.vs    v29, v25, v29\n\t"
        "vfredsum.vs    v30, v26, v30\n\t"
        "vfredsum.vs    v31, v27, v31\n\t"
        "vfmv.f.s       fa0, v28\n\t"
        "vfmv.f.s       fa1, v29\n\t"
        "vfmv.f.s       fa2, v30\n\t"
        "vfmv.f.s       fa3, v31\n\t"
        "fsh            fa0, 0(a0)\n\t"
        "fsh            fa1, 2(a0)\n\t"
        "fsh            fa2, 0(a1)\n\t"
        "fsh            fa3, 2(a1)\n\t"

        "addi           a0, a0, 4\n\t"
        "addi           a1, a1, 4\n\t"   // updata output start addr ( +2 cols)
        "slli           t0, %4, 2\n\t"   // t_tmp = k * 2 * 2
        "add            %1, %1, t0\n\t"  // updata pb start addr

        "15:\n\t"                       // m2n1
        "andi           t0, %5, 1\n\t"  // n & 1
        "beqz           t0, 19f\n\t"    // jump to ending
        // start kernel_m2n1
        "vmv.v.x        v24, zero\n\t"
        "vmv.v.x        v25, zero\n\t"  // clear acc

        "vfmv.s.f       v28, fs0\n\t"
        "vfmv.s.f       v29, fs1\n\t"  // init output = bias

        // init addr for pa, pb and pc
        "slli           t0, %4, 1\n\t"  // t_tmp = k * 2

        "mv             a2, %0\n\t"
        "addi           a3, a2, 2\n\t"  // a2-a3 hold pa(kernel) 2 lines start addr

        "mv             a4, %1\n\t"
        "add            a5, a4, t0\n\t"  // a4-a5 hold pb(input) 2 cols addr

        // a0-a1 hold pc(output) 2 rows addr

        "mv             t5, t1\n\t"   // t5 = k8
        "beqz           t2, 17f\n\t"  // if k_tail == 0, jump to subkernel_m2n1k8

        "16:\n\t"
        // start subkernel_m2n1k_tail
        "vsetvli        zero, t2, e16, m1\n\t"

        "vlse.v         v1, (a2), t4\n\t"
        "add            a2, a2, t6\n\t"
        "vlse.v         v2, (a3), t4\n\t"
        "add            a3, a3, t6\n\t"  // load pa

        "vle.v          v3, (a4)\n\t"
        "add            a4, a4, t3\n\t"  // load pb

        "vfmacc.vv      v24, v1, v3\n\t"  // out[0][0]
        "vfmacc.vv      v25, v2, v3\n\t"  // out[1][0]

        "beqz           t1, 18f\n\t"  // if k8 == 0, jump to end kernel_m2n1
        "vsetvli        zero, zero, e16, m1\n\t"

        "17:\n\t"
        // start subkernel_m2n1k8
        "vlse.v         v1, (a2), t4\n\t"
        "addi           a2, a2, 32\n\t"  // +8 * 2 * 2
        "vlse.v         v2, (a3), t4\n\t"
        "addi           a3, a2, 2\n\t"  // load pa

        "vle.v          v3, (a4)\n\t"
        "addi           a4, a4, 16\n\t"  // load pb

        "vfmacc.vv      v24, v1, v3\n\t"  // out[0][0]
        "vfmacc.vv      v25, v2, v3\n\t"  // out[1][0]

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 17b\n\t"

        "18:\n\t"                           // end kernel_m2n1
        "vfredsum.vs    v28, v24, v28\n\t"  // v28[0] = v28[0](bias) + sum(v24[0..7])
        "vfredsum.vs    v29, v25, v29\n\t"
        "vfmv.f.s       fa0, v28\n\t"
        "vfmv.f.s       fa1, v29\n\t"
        "fsh            fa0, 0(a0)\n\t"
        "fsh            fa1, 0(a1)\n\t"

        "19:\n\t"  // ending

        : "=r"(sa),    // %0
          "=r"(sb),    // %1
          "=r"(bias),  // %2
          "=r"(dst),   // %3
          "=r"(k),     // %4
          "=r"(n),     // %5
          "=r"(ldc)    // %6
        : "0"(sa), "1"(sb), "2"(bias), "3"(dst), "4"(k), "5"(n), "6"(ldc)
        : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3",
          "t4", "t5", "t6", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7", "ft0", "ft1",
          "fs0", "fs1");
}

/*
    (1) Algorithm works as follows:
        m4n8_loop:  m4n8k8_loop  -->  m4n8k4  -->  m4n8k2 -->  m4n8k1
        m4n4:       m4n4k8_loop  -->  m4n4k4  -->  m4n4k2 -->  m4n4k1
        m4n2:       m4n2k8_loop  -->  m4n2k4  -->  m4n2k2 -->  m4n2k1
        m4n1:       m4n1k8_loop  -->  m4n1k4  -->  m4n1k2 -->  m4n1k1

        n8_loop:    vfmacc.vf = kernel_data(f reg) * input_data(v reg)
        n4, n2, n1: vfmacc.vf = kernel_data(v reg) * input_data(f reg)  set vl = 4 (m = 4) , only
   half throughout capacity

    (2) register definition:
        t0:         i_n
        t1-t5:      i_k  t1-t4:[k8, k4, k2, k1] t5:k8_tmp  because i_k for inner_loop, extract to
   the outside t6:         hold sa/ kernel_data 4 lines origin address a0-a3:      dst0-dst3 addr
        ft0-ft3:    [ n8_loop ] : sa / kernel_data  [ n4, n2 n1 ]: sb / input_data
        fa0-fa3:    sa / kernel_data (sb / input_data) shadow for pipeline
        fs0-fs3:    hold 8 channels bias_data
        v1-v8:      [ n8_loop ] : sb / input_data   [ n4, n2 n1 ]: sb / kernel_data
        v24-v27:    bias_tmp and output

    TODO: if bias == NULL
*/
static void kernel_m4_fp16(__fp16* dst, __fp16* sa, __fp16* sb, int m, int k, int n, int ldc,
                           __fp16* bias)
{
    asm volatile(
        "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 8

        "srai           t1, %4, 3\n\t"  // t1 = k >> 3 (k8)
        "andi           t2, %4, 7\n\t"  // t2 = k & 7
        "srai           t2, t2, 2\n\t"  // t2 = (k & 7) >> 2 (k4)
        "andi           t3, %4, 3\n\t"  // t3 = k & 3
        "srai           t3, t3, 1\n\t"  // t3 = (k & 3) >> 1 (k2)
        "andi           t4, %4, 1\n\t"  // t4 = k & 1 (k1)

        "flh            fs0, 0(%2)\n\t"
        "flh            fs1, 2(%2)\n\t"
        "flh            fs2, 4(%2)\n\t"
        "flh            fs3, 6(%2)\n\t"  // load 4 bias_data for 4 out_channels

        // init output addr
        "slli           t5, %6, 1\n\t"  // t5_tmp = ldx * 2
        "mv             a0, %3\n\t"
        "add            a1, a0, t5\n\t"
        "add            a2, a1, t5\n\t"
        "add            a3, a2, t5\n\t"

        "srai           t0, %5, 3\n\t"  // t0 = n >> 3 (n8)
        "beqz           t0, 7f\n\t"     // jump to m4n4

        "1:\n\t"  // m4n8
                  // start kernel_m4n8
        "vfmv.v.f       v24, fs0\n\t"
        "vfmv.v.f       v25, fs1\n\t"
        "vfmv.v.f       v26, fs2\n\t"
        "vfmv.v.f       v27, fs3\n\t"  // init out_tmp = bias

        "mv             t6, %0\n\t"  // t6 hold kernel 4 lines start addr

        "vle.v          v1, (%1)\n\t"  // pre-load pb (input_data)
        "addi           %1, %1, 16\n\t"

        "flh            ft0, 0(t6)\n\t"
        "flh            ft1, 2(t6)\n\t"
        "flh            ft2, 4(t6)\n\t"
        "flh            ft3, 6(t6)\n\t"  // pre-load pa(kernel_data)

        "beqz           t1, 3f\n\t"  // if k8 == 0, jump to subkernel_m4n8k4
        "mv             t5, t1\n\t"  // t5 = k8

        "2:\n\t"
        // start subkernel_m4n8k8
        "vle.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 8(t6)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa1, 10(t6)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "flh            fa2, 12(t6)\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "flh            fa3, 14(t6)\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"  // 0

        "vle.v          v3, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 16(t6)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"
        "flh            ft1, 18(t6)\n\t"
        "vfmacc.vf      v25, fa1, v2\n\t"
        "flh            ft2, 20(t6)\n\t"
        "vfmacc.vf      v26, fa2, v2\n\t"
        "flh            ft3, 22(t6)\n\t"
        "vfmacc.vf      v27, fa3, v2\n\t"  // 1

        "vle.v          v4, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 24(t6)\n\t"
        "vfmacc.vf      v24, ft0, v3\n\t"
        "flh            fa1, 26(t6)\n\t"
        "vfmacc.vf      v25, ft1, v3\n\t"
        "flh            fa2, 28(t6)\n\t"
        "vfmacc.vf      v26, ft2, v3\n\t"
        "flh            fa3, 30(t6)\n\t"
        "vfmacc.vf      v27, ft3, v3\n\t"  // 2

        "vle.v          v5, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 32(t6)\n\t"
        "vfmacc.vf      v24, fa0, v4\n\t"
        "flh            ft1, 34(t6)\n\t"
        "vfmacc.vf      v25, fa1, v4\n\t"
        "flh            ft2, 36(t6)\n\t"
        "vfmacc.vf      v26, fa2, v4\n\t"
        "flh            ft3, 38(t6)\n\t"
        "vfmacc.vf      v27, fa3, v4\n\t"  // 3

        "vle.v          v6, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 40(t6)\n\t"
        "vfmacc.vf      v24, ft0, v5\n\t"
        "flh            fa1, 42(t6)\n\t"
        "vfmacc.vf      v25, ft1, v5\n\t"
        "flh            fa2, 44(t6)\n\t"
        "vfmacc.vf      v26, ft2, v5\n\t"
        "flh            fa3, 46(t6)\n\t"
        "vfmacc.vf      v27, ft3, v5\n\t"  // 4

        "vle.v          v7, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 48(t6)\n\t"
        "vfmacc.vf      v24, fa0, v6\n\t"
        "flh            ft1, 50(t6)\n\t"
        "vfmacc.vf      v25, fa1, v6\n\t"
        "flh            ft2, 52(t6)\n\t"
        "vfmacc.vf      v26, fa2, v6\n\t"
        "flh            ft3, 54(t6)\n\t"
        "vfmacc.vf      v27, fa3, v6\n\t"  // 5

        "vle.v          v8, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 56(t6)\n\t"
        "vfmacc.vf      v24, ft0, v7\n\t"
        "flh            fa1, 58(t6)\n\t"
        "vfmacc.vf      v25, ft1, v7\n\t"
        "flh            fa2, 60(t6)\n\t"
        "vfmacc.vf      v26, ft2, v7\n\t"
        "flh            fa3, 62(t6)\n\t"
        "vfmacc.vf      v27, ft3, v7\n\t"  // 6
        "addi           t6, t6, 64\n\t"    // +32 elements, bump pa(kernel_data) to next k8 addr

        "vle.v          v1, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 0(t6)\n\t"
        "vfmacc.vf      v24, fa0, v8\n\t"
        "flh            ft1, 2(t6)\n\t"
        "vfmacc.vf      v25, fa1, v8\n\t"
        "flh            ft2, 4(t6)\n\t"
        "vfmacc.vf      v26, fa2, v8\n\t"
        "flh            ft3, 6(t6)\n\t"
        "vfmacc.vf      v27, fa3, v8\n\t"  // 7

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 2b\n\t"

        "3:\n\t"
        "beqz           t2, 4f\n\t"  // if k4 == 0, jump to subkernel_m4n8k2
        // start subkernel_m4n8k4
        "vle.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 8(t6)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa1, 10(t6)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "flh            fa2, 12(t6)\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "flh            fa3, 14(t6)\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"  // 0

        "vle.v          v3, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 16(t6)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"
        "flh            ft1, 18(t6)\n\t"
        "vfmacc.vf      v25, fa1, v2\n\t"
        "flh            ft2, 20(t6)\n\t"
        "vfmacc.vf      v26, fa2, v2\n\t"
        "flh            ft3, 22(t6)\n\t"
        "vfmacc.vf      v27, fa3, v2\n\t"  // 1

        "vle.v          v4, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 24(t6)\n\t"
        "vfmacc.vf      v24, ft0, v3\n\t"
        "flh            fa1, 26(t6)\n\t"
        "vfmacc.vf      v25, ft1, v3\n\t"
        "flh            fa2, 28(t6)\n\t"
        "vfmacc.vf      v26, ft2, v3\n\t"
        "flh            fa3, 30(t6)\n\t"
        "vfmacc.vf      v27, ft3, v3\n\t"  // 2
        "addi           t6, t6, 32\n\t"    // +16 elements, bump pa to next k addr

        "vle.v          v1, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 0(t6)\n\t"
        "vfmacc.vf      v24, fa0, v4\n\t"
        "flh            ft1, 2(t6)\n\t"
        "vfmacc.vf      v25, fa1, v4\n\t"
        "flh            ft2, 4(t6)\n\t"
        "vfmacc.vf      v26, fa2, v4\n\t"
        "flh            ft3, 6(t6)\n\t"
        "vfmacc.vf      v27, fa3, v4\n\t"  // 3

        "4:\n\t"
        "beqz           t3, 5f\n\t"  // if k2 == 0, jump to subkernel_m4n8k1
        // start subkernel_m4n8k2
        "vle.v          v2, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 8(t6)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa1, 10(t6)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "flh            fa2, 12(t6)\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "flh            fa3, 14(t6)\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"  // 0
        "addi           t6, t6, 16\n\t"    // +8 elements, bump pa to next k addr

        "vle.v          v1, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            ft0, 0(t6)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"
        "flh            ft1, 2(t6)\n\t"
        "vfmacc.vf      v25, fa1, v2\n\t"
        "flh            ft2, 4(t6)\n\t"
        "vfmacc.vf      v26, fa2, v2\n\t"
        "flh            ft3, 6(t6)\n\t"
        "vfmacc.vf      v27, fa3, v2\n\t"  // 1

        "5:\n\t"
        "beqz           t4, 6f\n\t"  // if k1 == 0, jump to end kernel_m4n8
        // start subkernel_m4n8k1
        "vfmacc.vf      v24, ft0, v1\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"

        "addi           %1, %1, 16\n\t"  // ********************

        "6:\n\t"  // end kernel_m4n8

        // ********* bump pb to origin addr ************
        // offset pre-load
        "addi           %1, %1, -16\n\t"  // pb -= 8

        "vse.v          v24, (a0)\n\t"
        "addi           a0, a0, 16\n\t"
        "vse.v          v25, (a1)\n\t"
        "addi           a1, a1, 16\n\t"
        "vse.v          v26, (a2)\n\t"
        "addi           a2, a2, 16\n\t"
        "vse.v          v27, (a3)\n\t"
        "addi           a3, a3, 16\n\t"

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "7:\n\t"  // m4n4

        // prepare for n4, n2, n1
        "li             t0, 4\n\t"
        "vsetvli        zero, t0, e16, m1\n\t"  // set vl = 4

        "andi           t0, %5, 7\n\t"  // n & 7
        "srai           t0, t0, 2\n\t"  // (n & 7) >> 2
        "beqz           t0, 13f\n\t"    // jump to m4n2
        // start kernel_m4n4

        "vle.v          v24, (%2)\n\t"  // v24[0..3] = bias_data[0..3]
        "vle.v          v25, (%2)\n\t"
        "vle.v          v26, (%2)\n\t"
        "vle.v          v27, (%2)\n\t"  // init out_tmp = bias

        // init addr for pa, pb and pc
        "slli           t0, %4, 1\n\t"  // t0_tmp = k * 2

        "mv             t6, %0\n\t"  // t6 hold pa(kernel) 4 lines start addr

        "mv             a4, %1\n\t"
        "add            a5, a4, t0\n\t"
        "add            a6, a5, t0\n\t"
        "add            a7, a6, t0\n\t"  // a4-a7 hold pb(input_data) 4 cols addr

        "addi           a1, a0, 2\n\t"
        "addi           a2, a1, 2\n\t"
        "addi           a3, a2, 2\n\t"  // a0-a3 hold pc(output) addr

        "vle.v          v1, (t6)\n\t"   // pre-load pa(kernel_data)
        "addi           t6, t6, 8\n\t"  // m4: +4 elements

        "flh            ft0, 0(a4)\n\t"
        "flh            ft1, 0(a5)\n\t"
        "flh            ft2, 0(a6)\n\t"
        "flh            ft3, 0(a7)\n\t"  // pre-load pb(input_data)

        "beqz           t1, 9f\n\t"  // if k8 == 0, jump to subkernel_m4n4k4
        "mv             t5, t1\n\t"  // t5 = k8

        "8:\n\t"
        // start subkernel_m4n4k8
        "vle.v          v2, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 2(a4)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa1, 2(a5)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "flh            fa2, 2(a6)\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "flh            fa3, 2(a7)\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"  // 0

        "vle.v          v3, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 4(a4)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"
        "flh            ft1, 4(a5)\n\t"
        "vfmacc.vf      v25, fa1, v2\n\t"
        "flh            ft2, 4(a6)\n\t"
        "vfmacc.vf      v26, fa2, v2\n\t"
        "flh            ft3, 4(a7)\n\t"
        "vfmacc.vf      v27, fa3, v2\n\t"  // 1

        "vle.v          v4, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 6(a4)\n\t"
        "vfmacc.vf      v24, ft0, v3\n\t"
        "flh            fa1, 6(a5)\n\t"
        "vfmacc.vf      v25, ft1, v3\n\t"
        "flh            fa2, 6(a6)\n\t"
        "vfmacc.vf      v26, ft2, v3\n\t"
        "flh            fa3, 6(a7)\n\t"
        "vfmacc.vf      v27, ft3, v3\n\t"  // 2

        "vle.v          v5, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 8(a4)\n\t"
        "vfmacc.vf      v24, fa0, v4\n\t"
        "flh            ft1, 8(a5)\n\t"
        "vfmacc.vf      v25, fa1, v4\n\t"
        "flh            ft2, 8(a6)\n\t"
        "vfmacc.vf      v26, fa2, v4\n\t"
        "flh            ft3, 8(a7)\n\t"
        "vfmacc.vf      v27, fa3, v4\n\t"  // 3

        "vle.v          v6, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 10(a4)\n\t"
        "vfmacc.vf      v24, ft0, v5\n\t"
        "flh            fa1, 10(a5)\n\t"
        "vfmacc.vf      v25, ft1, v5\n\t"
        "flh            fa2, 10(a6)\n\t"
        "vfmacc.vf      v26, ft2, v5\n\t"
        "flh            fa3, 10(a7)\n\t"
        "vfmacc.vf      v27, ft3, v5\n\t"  // 4

        "vle.v          v7, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 12(a4)\n\t"
        "vfmacc.vf      v24, fa0, v6\n\t"
        "flh            ft1, 12(a5)\n\t"
        "vfmacc.vf      v25, fa1, v6\n\t"
        "flh            ft2, 12(a6)\n\t"
        "vfmacc.vf      v26, fa2, v6\n\t"
        "flh            ft3, 12(a7)\n\t"
        "vfmacc.vf      v27, fa3, v6\n\t"  // 5

        "vle.v          v8, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 14(a4)\n\t"
        "vfmacc.vf      v24, ft0, v7\n\t"
        "flh            fa1, 14(a5)\n\t"
        "vfmacc.vf      v25, ft1, v7\n\t"
        "flh            fa2, 14(a6)\n\t"
        "vfmacc.vf      v26, ft2, v7\n\t"
        "flh            fa3, 14(a7)\n\t"
        "vfmacc.vf      v27, ft3, v7\n\t"  // 6

        "addi           a4, a4, 16\n\t"
        "addi           a5, a5, 16\n\t"
        "addi           a6, a6, 16\n\t"
        "addi           a7, a7, 16\n\t"  // +8 elements, bump pb to next k8 addr

        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 0(a4)\n\t"
        "vfmacc.vf      v24, fa0, v8\n\t"
        "flh            ft1, 0(a5)\n\t"
        "vfmacc.vf      v25, fa1, v8\n\t"
        "flh            ft2, 0(a6)\n\t"
        "vfmacc.vf      v26, fa2, v8\n\t"
        "flh            ft3, 0(a7)\n\t"
        "vfmacc.vf      v27, fa3, v8\n\t"  // 7

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 8b\n\t"

        "9:\n\t"
        "beqz           t2, 10f\n\t"  // if k4 == 0, jump to subkernel_m4n4k2
        // start subkernel_m4n4k4
        "vle.v          v2, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 2(a4)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa1, 2(a5)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "flh            fa2, 2(a6)\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "flh            fa3, 2(a7)\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"  // 0

        "vle.v          v3, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 4(a4)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"
        "flh            ft1, 4(a5)\n\t"
        "vfmacc.vf      v25, fa1, v2\n\t"
        "flh            ft2, 4(a6)\n\t"
        "vfmacc.vf      v26, fa2, v2\n\t"
        "flh            ft3, 4(a7)\n\t"
        "vfmacc.vf      v27, fa3, v2\n\t"  // 1

        "vle.v          v4, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 6(a4)\n\t"
        "vfmacc.vf      v24, ft0, v3\n\t"
        "flh            fa1, 6(a5)\n\t"
        "vfmacc.vf      v25, ft1, v3\n\t"
        "flh            fa2, 6(a6)\n\t"
        "vfmacc.vf      v26, ft2, v3\n\t"
        "flh            fa3, 6(a7)\n\t"
        "vfmacc.vf      v27, ft3, v3\n\t"  // 2

        "addi           a4, a4, 8\n\t"
        "addi           a5, a5, 8\n\t"
        "addi           a6, a6, 8\n\t"
        "addi           a7, a7, 8\n\t"  // +4 elements

        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 0(a4)\n\t"
        "vfmacc.vf      v24, fa0, v4\n\t"
        "flh            ft1, 0(a5)\n\t"
        "vfmacc.vf      v25, fa1, v4\n\t"
        "flh            ft2, 0(a6)\n\t"
        "vfmacc.vf      v26, fa2, v4\n\t"
        "flh            ft3, 0(a7)\n\t"
        "vfmacc.vf      v27, fa3, v4\n\t"  // 3

        "10:\n\t"
        "beqz           t3, 11f\n\t"  // if k2 == 0, jump to subkernel_m4n4k1
        // start subkernel_m4n4k2
        "vle.v          v2, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 2(a4)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa1, 2(a5)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "flh            fa2, 2(a6)\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "flh            fa3, 2(a7)\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"  // 0

        "addi           a4, a4, 4\n\t"
        "addi           a5, a5, 4\n\t"
        "addi           a6, a6, 4\n\t"
        "addi           a7, a7, 4\n\t"  // +2 elements

        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 0(a4)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"
        "flh            ft1, 0(a5)\n\t"
        "vfmacc.vf      v25, fa1, v2\n\t"
        "flh            ft2, 0(a6)\n\t"
        "vfmacc.vf      v26, fa2, v2\n\t"
        "flh            ft3, 0(a7)\n\t"
        "vfmacc.vf      v27, fa3, v2\n\t"  // 1

        "11:\n\t"
        "beqz           t4, 12f\n\t"  // if k1 == 0, jump to end kernel_m4n4
        // start subkernl_m4n4k1
        "vfmacc.vf      v24, ft0, v1\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"
        "vfmacc.vf      v26, ft2, v1\n\t"
        "vfmacc.vf      v27, ft3, v1\n\t"

        "12:\n\t"                       // end kernel_m4n4
        "slli           t0, %6, 1\n\t"  // t0_tmp = ldx * 2 (store_stride)

        "vsse.v         v24, (a0), t0\n\t"
        "vsse.v         v25, (a1), t0\n\t"
        "vsse.v         v26, (a2), t0\n\t"
        "vsse.v         v27, (a3), t0\n\t"

        "addi           a0, a0, 8\n\t"   // updata output start addr ( +4 cols)
        "slli           t0, %4, 3\n\t"   // t_tmp = k * 4 * 2
        "add            %1, %1, t0\n\t"  // updata pb start addr

        "13:\n\t"                       // m4n2
        "andi           t0, %5, 3\n\t"  // n & 3
        "srai           t0, t0, 1\n\t"  // (n & 3) >> 1
        "beqz           t0, 19f\n\t"    // jump to m4n1
        // start kernel_m4n2

        "vle.v          v24, (%2)\n\t"  // v24[0..3] = bias_data[0..3]
        "vle.v          v25, (%2)\n\t"  // init out_tmp = bias

        // init addr for pa, pb and pc
        "slli           t0, %4, 1\n\t"  // t0_tmp = k * 2

        "mv             t6, %0\n\t"  // t6 hold pa(kernel) 4 lines start addr

        "mv             a4, %1\n\t"
        "add            a5, a4, t0\n\t"  // a4-a5 hold pb(input_data) 2 cols addr

        "addi           a1, a0, 2\n\t"  // a0-a1 hold pc(output) addr

        "vle.v          v1, (t6)\n\t"   // pre-load pa(kernel_data)
        "addi           t6, t6, 8\n\t"  // m4: +4 elements

        "flh            ft0, 0(a4)\n\t"
        "flh            ft1, 0(a5)\n\t"  // pre-load pb(input_data)

        "beqz           t1, 15f\n\t"  // if k8 == 0, jump to subkernel_m4n2k4
        "mv             t5, t1\n\t"   // t5 = k8

        "14:\n\t"
        // start subkernel_m4n2k8
        "vle.v          v2, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 2(a4)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa1, 2(a5)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"  // 0

        "vle.v          v3, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 4(a4)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"
        "flh            ft1, 4(a5)\n\t"
        "vfmacc.vf      v25, fa1, v2\n\t"  // 1

        "vle.v          v4, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 6(a4)\n\t"
        "vfmacc.vf      v24, ft0, v3\n\t"
        "flh            fa1, 6(a5)\n\t"
        "vfmacc.vf      v25, ft1, v3\n\t"  // 2

        "vle.v          v5, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 8(a4)\n\t"
        "vfmacc.vf      v24, fa0, v4\n\t"
        "flh            ft1, 8(a5)\n\t"
        "vfmacc.vf      v25, fa1, v4\n\t"  // 3

        "vle.v          v6, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 10(a4)\n\t"
        "vfmacc.vf      v24, ft0, v5\n\t"
        "flh            fa1, 10(a5)\n\t"
        "vfmacc.vf      v25, ft1, v5\n\t"  // 4

        "vle.v          v7, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 12(a4)\n\t"
        "vfmacc.vf      v24, fa0, v6\n\t"
        "flh            ft1, 12(a5)\n\t"
        "vfmacc.vf      v25, fa1, v6\n\t"  // 5

        "vle.v          v8, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 14(a4)\n\t"
        "vfmacc.vf      v24, ft0, v7\n\t"
        "flh            fa1, 14(a5)\n\t"
        "vfmacc.vf      v25, ft1, v7\n\t"  // 6

        "addi           a4, a4, 16\n\t"
        "addi           a5, a5, 16\n\t"  // +8 elements, bump pb to next k8 addr

        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 0(a4)\n\t"
        "vfmacc.vf      v24, fa0, v8\n\t"
        "flh            ft1, 0(a5)\n\t"
        "vfmacc.vf      v25, fa1, v8\n\t"  // 7

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 14b\n\t"

        "15:\n\t"
        "beqz           t2, 16f\n\t"  // if k4 == 0, jump to subkernel_m4n2k2
        // start subkernel_m4n2k4
        "vle.v          v2, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 2(a4)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa1, 2(a5)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"  // 0

        "vle.v          v3, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 4(a4)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"
        "flh            ft1, 4(a5)\n\t"
        "vfmacc.vf      v25, fa1, v2\n\t"  // 1

        "vle.v          v4, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 6(a4)\n\t"
        "vfmacc.vf      v24, ft0, v3\n\t"
        "flh            fa1, 6(a5)\n\t"
        "vfmacc.vf      v25, ft1, v3\n\t"  // 2

        "addi           a4, a4, 8\n\t"
        "addi           a5, a5, 8\n\t"  // +4 elements

        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 0(a4)\n\t"
        "vfmacc.vf      v24, fa0, v4\n\t"
        "flh            ft1, 0(a5)\n\t"
        "vfmacc.vf      v25, fa1, v4\n\t"  // 3

        "16:\n\t"
        "beqz           t3, 17f\n\t"  // if k2 == 0, jump to subkernel_m4n2k1
        // start subkernel_m4n2k2
        "vle.v          v2, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 2(a4)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"
        "flh            fa1, 2(a5)\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"  // 0

        "addi           a4, a4, 4\n\t"
        "addi           a5, a5, 4\n\t"  // +2 elements

        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 0(a4)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"
        "flh            ft1, 0(a5)\n\t"
        "vfmacc.vf      v25, fa1, v2\n\t"  // 1

        "17:\n\t"
        "beqz           t4, 18f\n\t"  // if k1 == 0, jump to end kernel_m4n2
        // start subkernel_m4n2k1
        "vfmacc.vf      v24, ft0, v1\n\t"
        "vfmacc.vf      v25, ft1, v1\n\t"  // 0

        "18:\n\t"                       // end kernel_m4n2
        "slli           t0, %6, 1\n\t"  // t0_tmp = ldx * 2 (store_stride)

        "vsse.v         v24, (a0), t0\n\t"
        "vsse.v         v25, (a1), t0\n\t"

        "addi           a0, a0, 4\n\t"   // updata output start addr ( +2 cols)
        "slli           t0, %4, 2\n\t"   // t_tmp = k * 2 * 2
        "add            %1, %1, t0\n\t"  // updata pb start addr

        "19:\n\t"                       // m4n1
        "andi           t0, %5, 1\n\t"  // n & 1
        "beqz           t0, 25f\n\t"    // jump to ending
        // start kernel_m4n1

        "vle.v          v24, (%2)\n\t"  // v24[0..3] = bias_data[0..3], init out_tmp = bias

        // init addr for pa, pb and pc
        "mv             t6, %0\n\t"  // t6 hold pa(kernel) 4 lines start addr
        "mv             a4, %1\n\t"  // a4 hold pb(input_data) 1 cols addr
                                     // a0 hold pc(output) addr

        "vle.v          v1, (t6)\n\t"   // pre-load pa(kernel_data)
        "addi           t6, t6, 8\n\t"  // m4: +4 elements

        "flh            ft0, 0(a4)\n\t"  // pre-load pb(input_data)

        "beqz           t1, 21f\n\t"  // if k8 == 0, jump to subkernel_m4n1k4
        "mv             t5, t1\n\t"   // t5 = k8

        "20:\n\t"
        // start subkernel_m4n1k8
        "vle.v          v2, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 2(a4)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"  // 0

        "vle.v          v3, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 4(a4)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"  // 1

        "vle.v          v4, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 6(a4)\n\t"
        "vfmacc.vf      v24, ft0, v3\n\t"  // 2

        "vle.v          v5, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 8(a4)\n\t"
        "vfmacc.vf      v24, fa0, v4\n\t"  // 3

        "vle.v          v6, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 10(a4)\n\t"
        "vfmacc.vf      v24, ft0, v5\n\t"  // 4

        "vle.v          v7, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 12(a4)\n\t"
        "vfmacc.vf      v24, fa0, v6\n\t"  // 5

        "vle.v          v8, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 14(a4)\n\t"
        "vfmacc.vf      v24, ft0, v7\n\t"  // 6

        "addi           a4, a4, 16\n\t"  // +8 elements, bump pb to next k8 addr

        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 0(a4)\n\t"
        "vfmacc.vf      v24, fa0, v8\n\t"  // 7

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 20b\n\t"

        "21:\n\t"
        "beqz           t2, 22f\n\t"  // if k4 == 0, jump to subkernel_m4n1k2
        // start subkernel_m4n1k4
        "vle.v          v2, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 2(a4)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"  // 0

        "vle.v          v3, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 4(a4)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"  // 1

        "vle.v          v4, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 6(a4)\n\t"
        "vfmacc.vf      v24, ft0, v3\n\t"  // 2

        "addi           a4, a4, 8\n\t"  // +4 elements

        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 0(a4)\n\t"
        "vfmacc.vf      v24, fa0, v4\n\t"  // 3

        "22:\n\t"
        "beqz           t3, 23f\n\t"  // if k2 == 0, jump to subkernel_m4n1k1
        // start subkernel_m4n1k2
        "vle.v          v2, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            fa0, 2(a4)\n\t"
        "vfmacc.vf      v24, ft0, v1\n\t"  // 0

        "addi           a4, a4, 4\n\t"  // +2 elements

        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 8\n\t"
        "flh            ft0, 0(a4)\n\t"
        "vfmacc.vf      v24, fa0, v2\n\t"  // 1

        "23:\n\t"
        "beqz           t4, 24f\n\t"  // if k1 == 0, jump to end kernel_m4n1
        // start subkernel_m4n1k1
        "vfmacc.vf      v24, ft0, v1\n\t"  // 0

        "24:\n\t"                       // end kernel_m4n1
        "slli           t0, %6, 1\n\t"  // t0_tmp = ldx * 2 (store_stride)

        "vsse.v         v24, (a0), t0\n\t"

        "25:\n\t"  // ending

        : "=r"(sa),    // %0
          "=r"(sb),    // %1
          "=r"(bias),  // %2
          "=r"(dst),   // %3
          "=r"(k),     // %4
          "=r"(n),     // %5
          "=r"(ldc)    // %6
        : "0"(sa), "1"(sb), "2"(bias), "3"(dst), "4"(k), "5"(n), "6"(ldc)
        : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v24", "v25", "v26", "v27", "a0", "a1",
          "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "fa0",
          "fa1", "fa2", "fa3", "ft0", "ft1", "ft2", "ft3", "fs0", "fs1", "fs2", "fs3");
}

static void kernel_m8_fp16_1(__fp16* dst, __fp16* sa, __fp16* sb, int m, int k, int n, int ldc,
                             __fp16* bias)
{
    asm volatile(
        "vsetvli        zero, zero, e16, m1\n\t"  // set vl = 8

        "flh            fs0, 0(%2)\n\t"
        "flh            fs1, 2(%2)\n\t"
        "flh            fs2, 4(%2)\n\t"
        "flh            fs3, 6(%2)\n\t"
        "flh            fs4, 8(%2)\n\t"
        "flh            fs5, 10(%2)\n\t"
        "flh            fs6, 12(%2)\n\t"
        "flh            fs7, 14(%2)\n\t"  // load 8 bias_data for 8 out_channels

        // init output addr
        "slli           t5, %6, 1\n\t"  // t5_tmp = ldx * 2
        "mv             a0, %3\n\t"
        "add            a1, a0, t5\n\t"
        "add            a2, a1, t5\n\t"
        "add            a3, a2, t5\n\t"
        "add            a4, a3, t5\n\t"
        "add            a5, a4, t5\n\t"
        "add            a6, a5, t5\n\t"
        "add            a7, a6, t5\n\t"

        "srai           t0, %5, 3\n\t"  // t0 = n >> 3 (n8)
        "beqz           t0, 7f\n\t"     // jump to m8n4

        "1:\n\t"  // m8n8
                  // start kernel_m8n8
        "vfmv.v.f       v24, fs0\n\t"
        "vfmv.v.f       v25, fs1\n\t"
        "vfmv.v.f       v26, fs2\n\t"
        "vfmv.v.f       v27, fs3\n\t"
        "vfmv.v.f       v28, fs4\n\t"
        "vfmv.v.f       v29, fs5\n\t"
        "vfmv.v.f       v30, fs6\n\t"
        "vfmv.v.f       v31, fs7\n\t"  // init out_tmp = bias

        "mv             t6, %0\n\t"  // t6 hold kernel 8 lines start addr
        "mv             t5, %4\n\t"  // t5 = k (k > 0)

        "2:\n\t"
        // start subkernel_m8n8k1
        "vle.v          v1, (%1)\n\t"
        "addi           %1, %1, 16\n\t"
        "flh            fa0, 0(t6)\n\t"
        "flh            fa1, 2(t6)\n\t"
        "flh            fa2, 4(t6)\n\t"
        "flh            fa3, 6(t6)\n\t"
        "flh            fa4, 8(t6)\n\t"
        "flh            fa5, 10(t6)\n\t"
        "flh            fa6, 12(t6)\n\t"
        "flh            fa7, 14(t6)\n\t"
        "addi           t6, t6, 16\n\t"

        "vfmacc.vf      v24, fa0, v1\n\t"
        "vfmacc.vf      v25, fa1, v1\n\t"
        "vfmacc.vf      v26, fa2, v1\n\t"
        "vfmacc.vf      v27, fa3, v1\n\t"
        "vfmacc.vf      v28, fa4, v1\n\t"
        "vfmacc.vf      v29, fa5, v1\n\t"
        "vfmacc.vf      v30, fa6, v1\n\t"
        "vfmacc.vf      v31, fa7, v1\n\t"  // 0

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 2b\n\t"

        "6:\n\t"  // end kernel_m8n8

        "vse.v          v24, (a0)\n\t"
        "addi           a0, a0, 16\n\t"
        "vse.v          v25, (a1)\n\t"
        "addi           a1, a1, 16\n\t"
        "vse.v          v26, (a2)\n\t"
        "addi           a2, a2, 16\n\t"
        "vse.v          v27, (a3)\n\t"
        "addi           a3, a3, 16\n\t"
        "vse.v          v28, (a4)\n\t"
        "addi           a4, a4, 16\n\t"
        "vse.v          v29, (a5)\n\t"
        "addi           a5, a5, 16\n\t"
        "vse.v          v30, (a6)\n\t"
        "addi           a6, a6, 16\n\t"
        "vse.v          v31, (a7)\n\t"
        "addi           a7, a7, 16\n\t"  // store output

        "addi           t0, t0, -1\n\t"
        "bnez           t0, 1b\n\t"

        "7:\n\t"                        // m8n4
        "andi           t0, %5, 7\n\t"  // n & 7
        "srai           t0, t0, 2\n\t"  // (n & 7) >> 2
        "beqz           t0, 13f\n\t"    // jump to m8n2
        // start kernel_m8n4

        "vle.v          v28, (%2)\n\t"  // v28[0..7] = bias_data[0..7]
        "vle.v          v29, (%2)\n\t"
        "vle.v          v30, (%2)\n\t"
        "vle.v          v31, (%2)\n\t"  // init out_tmp = bias

        // init addr for pa, pb and pc
        "slli           t0, %4, 1\n\t"  // t0_tmp = k * 2

        "mv             t6, %0\n\t"  // t6 hold pa(kernel) 8 lines start addr

        "mv             a4, %1\n\t"
        "add            a5, a4, t0\n\t"
        "add            a6, a5, t0\n\t"
        "add            a7, a6, t0\n\t"  // a4-a7 hold pb(input) 4 cols addr

        "addi           a1, a0, 2\n\t"
        "addi           a2, a1, 2\n\t"
        "addi           a3, a2, 2\n\t"  // a0-a3 hold pc(output) addr

        "mv             t5, %4\n\t"  // t5 = k

        "8:\n\t"
        // start subkernel_m8n4k1
        "vle.v          v1, (t6)\n\t"  // load pa for next
        "addi           t6, t6, 16\n\t"
        "flh            fa0, 0(a4)\n\t"
        "vfmacc.vf      v28, fa0, v1\n\t"
        "flh            fa1, 0(a5)\n\t"
        "vfmacc.vf      v29, fa1, v1\n\t"
        "flh            fa2, 0(a6)\n\t"
        "vfmacc.vf      v30, fa2, v1\n\t"
        "flh            fa3, 0(a7)\n\t"
        "vfmacc.vf      v31, fa3, v1\n\t"  // 0

        "addi           a4, a4, 2\n\t"
        "addi           a5, a5, 2\n\t"
        "addi           a6, a6, 2\n\t"
        "addi           a7, a7, 2\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 8b\n\t"

        "12:\n\t"                       // end kernel_m8n4
        "slli           t0, %6, 1\n\t"  // t0_tmp = ldx * 2 (store_stride)

        "vsse.v         v28, (a0), t0\n\t"
        "vsse.v         v29, (a1), t0\n\t"
        "vsse.v         v30, (a2), t0\n\t"
        "vsse.v         v31, (a3), t0\n\t"

        "addi           a0, a0, 8\n\t"   // updata output start addr ( +4 cols)
        "slli           t0, %4, 3\n\t"   // t_tmp = k * 4 * 2
        "add            %1, %1, t0\n\t"  // updata pb start addr

        "13:\n\t"                       // m8n2
        "andi           t0, %5, 3\n\t"  // n & 3
        "srai           t0, t0, 1\n\t"  // (n & 3) >> 1
        "beqz           t0, 19f\n\t"    // jump to m8n1
        // start kernel_m8n2

        "vle.v          v28, (%2)\n\t"  // v28[0..7] = bias[0..7]
        "vle.v          v29, (%2)\n\t"  // init out_tmp = bias

        // init addr for pa, pb and pc
        "slli           t0, %4, 1\n\t"  // t_tmp = k * 2

        "mv             t6, %0\n\t"  // t6 hold pa(kernel) 8 lines start addr

        "mv             a4, %1\n\t"
        "add            a5, a4, t0\n\t"  // a4-a5 hold pb(input) 2 cols addr

        "addi           a1, a0, 2\n\t"  // a0-a1 hold pc(output) addr

        "mv             t5, %4\n\t"  // t5 = k

        "14:\n\t"
        // start subkernel_m8n2k8
        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 16\n\t"
        "flh            fa0, 0(a4)\n\t"
        "vfmacc.vf      v28, fa0, v1\n\t"
        "flh            fa1, 0(a5)\n\t"
        "vfmacc.vf      v29, fa1, v1\n\t"  // 0

        "addi           a4, a4, 2\n\t"
        "addi           a5, a5, 2\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 14b\n\t"

        "18:\n\t"                       // end kernel_m8n2
        "slli           t0, %6, 1\n\t"  // t0_tmp = ldx * 2 (store_stride)

        "vsse.v         v28, (a0), t0\n\t"
        "vsse.v         v29, (a1), t0\n\t"

        "addi           a0, a0, 4\n\t"   // updata output start addr ( +2 cols)
        "slli           t0, %4, 2\n\t"   // t_tmp = k * 2 * 2
        "add            %1, %1, t0\n\t"  // updata pb start addr (+2 cols)

        "19:\n\t"                       // m8n1
        "andi           t0, %5, 1\n\t"  // n & 1
        "beqz           t0, 25f\n\t"    // jump to ending
        // start kernel_m8n1

        "vle.v          v28, (%2)\n\t"  // init out_tmp = bias

        // init addr for pa, pb and pc
        "mv             t6, %0\n\t"  // t6 hold pa(kernel) 8 lines start addr
        "mv             a4, %1\n\t"  // a4 hold pb(input) 1 cols addr
                                     // a0 hold pc(output) addr

        "mv             t5, %4\n\t"  // t5 = k

        "20:\n\t"
        // start subkernel_m8n1k8
        "vle.v          v1, (t6)\n\t"
        "addi           t6, t6, 16\n\t"
        "flh            fa0, 0(a4)\n\t"
        "vfmacc.vf      v28, fa0, v1\n\t"  // 0

        "addi           a4, a4, 2\n\t"

        "addi           t5, t5, -1\n\t"
        "bnez           t5, 20b\n\t"

        "24:\n\t"                       // end kernel_m8n1
        "slli           t0, %6, 1\n\t"  // t0_tmp = ldx * 2 (store_stride)

        "vsse.v         v28, (a0), t0\n\t"

        "25:\n\t"  // ending

        : "=r"(sa),    // %0
          "=r"(sb),    // %1
          "=r"(bias),  // %2
          "=r"(dst),   // %3
          "=r"(k),     // %4
          "=r"(n),     // %5
          "=r"(ldc)    // %6
        : "0"(sa), "1"(sb), "2"(bias), "3"(dst), "4"(k), "5"(n), "6"(ldc)
        : "v1", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "a0", "a1", "a2", "a3",
          "a4", "a5", "a6", "a7", "t0", "t5", "t6", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6",
          "fa7", "fs0", "fs1", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7");
}

void shl_c906_sgemm_kernel_fp16(__fp16* dst, const __fp16* sa, const __fp16* sb, int m, int k,
                                int n, int ldc, __fp16* bias)
{
    __fp16* pa = (__fp16*)sa;
    __fp16* pb = (__fp16*)sb;
    __fp16* pc = dst;

    bool flag_bias = 1;  // default: conv2d layer include bias
    if (bias == NULL) {
        flag_bias = 0;
        bias = (__fp16*)shl_mem_alloc(m * sizeof(__fp16));
    }
    __fp16* bias_tmp = bias;

    const int mm = (m >> 3) << 3;

    for (int i = 0; i < mm; i += 8) {
        kernel_m8_fp16_1(pc + i * ldc, pa + i * k, pb, m, k, n, ldc, bias_tmp + i);
    }

    pa += mm * k;
    pc += mm * ldc;
    bias_tmp += mm;

    switch (m - mm) {
        case 7:
            kernel_m4_fp16(pc, pa, pb, m, k, n, ldc, bias_tmp);
            pc += 4 * ldc;
            pa += 4 * k;
            bias_tmp += 4;
            kernel_m2_fp16(pc, pa, pb, m, k, n, ldc, bias_tmp);
            pc += 2 * ldc;
            pa += 2 * k;
            bias_tmp += 2;
            kernel_m1_fp16(pc, pa, pb, m, k, n, ldc, bias_tmp);
            break;

        case 6:
            kernel_m4_fp16(pc, pa, pb, m, k, n, ldc, bias_tmp);
            pc += 4 * ldc;
            pa += 4 * k;
            bias_tmp += 4;
            kernel_m2_fp16(pc, pa, pb, m, k, n, ldc, bias_tmp);
            break;

        case 5:
            kernel_m4_fp16(pc, pa, pb, m, k, n, ldc, bias_tmp);
            pc += 4 * ldc;
            pa += 4 * k;
            bias_tmp += 4;
            kernel_m1_fp16(pc, pa, pb, m, k, n, ldc, bias_tmp);
            break;

        case 4:
            kernel_m4_fp16(pc, pa, pb, m, k, n, ldc, bias_tmp);
            break;

        case 3:
            kernel_m2_fp16(pc, pa, pb, m, k, n, ldc, bias_tmp);
            pc += 2 * ldc;
            pa += 2 * k;
            bias_tmp += 2;
            kernel_m1_fp16(pc, pa, pb, m, k, n, ldc, bias_tmp);
            break;

        case 2:
            kernel_m2_fp16(pc, pa, pb, m, k, n, ldc, bias_tmp);
            break;

        case 1:
            kernel_m1_fp16(pc, pa, pb, m, k, n, ldc, bias_tmp);
            break;

        case 0:
            break;
        default:
            break;
    }
    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
