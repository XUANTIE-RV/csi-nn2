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

#ifndef _SGEMM_H
#define _SGEMM_H

/* The matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

#define DECOMPOSE_K                      \
    int ktmp = k;                        \
    int k8 = k >> 3;                     \
    k -= (k8 << 3);                      \
    int k4 = k >> 2;                     \
    k -= (k4 << 2);                      \
    int k2 = k >> 1;                     \
    k -= (k2 << 1);                      \
    int k1 = k;                          \
    k = ktmp;

#define DECOMPOSE_N  \
    int ntmp = n;    \
    int n4 = n >> 2; \
    n -= (n4 << 2);  \
    int n2 = n >> 1; \
    n -= (n2 << 1);  \
    int n1 = n;      \
    n = ntmp;

#define DECOMPOSE_M  \
    int mtmp = m;    \
    int m4 = m >> 2; \
    m -= (m4 << 2);  \
    int m2 = m >> 1; \
    m -= (m2 << 1);  \
    int m1 = m;      \
    m = mtmp;

/*
change memory layout for matrix A
memory index from  ------>  to
0  1  2  3                  0  4  8  12
4  5  6  7                  1  5  9  13
8  9  10 11                 2  6  10 14
12 13 14 15                 3  7  11 15
16 17 18 19                 16 18 20 22
20 21 22 23                 17 19 21 23
24 25 26 27                 24 25 26 27
*/
static void reorder_a(float *a, float *sa, int m, int k, int ldx)
{
#if __riscv_vector == 128
    DECOMPOSE_M
    DECOMPOSE_K
    /*
        Execution delay cycles: vlsw + vsw  = 6 + 1
                                vlw  + vssw = 4 + 2  ✔
    */
    if(m4 > 0) {
        float *a0 = a;
        float *a1 = a0 + ldx;
        float *a2 = a1 + ldx;
        float *a3 = a2 + ldx;
        int k_tail = k & 7;
        int store_stride = 16;
        asm volatile(
            "slli       t3, %10, 2\n\t"         // t3 = ldx * 4
            "slli       t4, t3, 2\n\t"          // t4 = 4 * ldx * 4
            "mv         t2, %5\n\t"             // t2 = m4
            "slli       t0, %7, 2\n\t"          // t0 = k_tail * 4
            "slli       t1, t0, 2\n\t"          // t1 = t0 * 4

        "1:\n\t"
            // start packm4
            "mv         %0, %9\n\t"             // a0 = a
            "add        %1, %0, t3\n\t"         // a1 = a0 + 4 * ldx
            "add        %2, %1, t3\n\t"         // a2 = a1 + 4 * ldx
            "add        %3, %2, t3\n\t"         // a3 = a2 + 4 * ldx
            "mv         t6, %6\n\t"             // t6 = k8
            "beqz       t6, 3f\n\t"             // k8 == 0 ?
            "vsetvli    zero, zero, e32, m2\n\t"

            "2:\n\t"
                // start subpack_m4k8
                "vlw.v      v0, (%0)\n\t"
                "addi       %0, %0, 32\n\t"
                "vlw.v      v2, (%1)\n\t"
                "addi       %1, %1, 32\n\t"
                "vlw.v      v4, (%2)\n\t"
                "addi       %2, %2, 32\n\t"
                "vlw.v      v6, (%3)\n\t"
                "addi       %3, %3, 32\n\t"

                "vssw.v     v0, (%4), %8\n\t"
                "addi       %4, %4, 4\n\t"
                "vssw.v     v2, (%4), %8\n\t"
                "addi       %4, %4, 4\n\t"
                "vssw.v     v4, (%4), %8\n\t"
                "addi       %4, %4, 4\n\t"
                "vssw.v     v6, (%4), %8\n\t"
                "addi       %4, %4, -12\n\t"
                "addi       %4, %4, 128\n\t"    // sa += 32 ele * 4

                "addi       t6, t6, -1\n\t"     // k8--
                "bnez       t6, 2b\n\t"

        "3:\n\t"
            "beqz       %7, 4f\n\t"      // k_tail == 0 ?
            // Processing k_tail
            "vsetvli    zero, %7, e32, m2\n\t"
            "vlw.v      v0, (%0)\n\t"
            "add        %0, %0, t0\n\t"
            "vlw.v      v2, (%1)\n\t"
            "add        %1, %1, t0\n\t"
            "vlw.v      v4, (%2)\n\t"
            "add        %2, %2, t0\n\t"
            "vlw.v      v6, (%3)\n\t"
            "add        %3, %3, t0\n\t"

            "vssw.v     v0, (%4), %8\n\t"
            "addi       %4, %4, 4\n\t"
            "vssw.v     v2, (%4), %8\n\t"
            "addi       %4, %4, 4\n\t"
            "vssw.v     v4, (%4), %8\n\t"
            "addi       %4, %4, 4\n\t"
            "vssw.v     v6, (%4), %8\n\t"
            "addi       %4, %4, -12\n\t"
            "add        %4, %4, t1\n\t"         // sa += 4 * k_tail * 4

        "4:\n\t"
            // end packm4
            "add        %9, %9, t4\n\t"         // a += 4 * ldx * 4
            "addi       t2, t2, -1\n\t"         // m4--
            "bnez       t2, 1b\n\t"

            :"=r"(a0),      // %0
            "=r"(a1),       // %1
            "=r"(a2),       // %2
            "=r"(a3),       // %3
            "=r"(sa),       // %4
            "=r"(m4),       // %5
            "=r"(k8),       // %6
            "=r"(k_tail),    // %7
            "=r"(store_stride),  // %8
            "=r"(a),         // %9
            "=r"(ldx)        // %10
            :"0"(a0),
            "1"(a1),
            "2"(a2),
            "3"(a3),
            "4"(sa),
            "5"(m4),
            "6"(k8),
            "7"(k_tail),
            "8"(store_stride),
            "9"(a),
            "10"(ldx)
            :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "t0", "t1", "t2", "t3", "t4", "t6"
        );
    }
    if(m2 > 0) {
        float *a0 = a;
        float *a1 = a0 + ldx;
        int k8 = k >> 3;
        int k_tail = k & 7;
        int store_stride = 8;

        asm volatile(
            "slli       t2, %7, 3\n\t"          // t2 = ldx * 2 * 4
            "slli       t0, %4, 2\n\t"          // t0 = k_tail * 4
            "slli       t1, t0, 1\n\t"          // t1 = t0 * 2
            "beqz       %3, 2f\n\t"    // k8 == 0 ?
            "vsetvli    zero, zero, e32, m2\n\t"

            "1:\n\t"
                // start subpack_m2k8
                "vlw.v      v0, (%0)\n\t"
                "addi       %0, %0, 32\n\t"
                "vlw.v      v2, (%1)\n\t"
                "addi       %1, %1, 32\n\t"

                "vssw.v     v0, (%2), %5\n\t"
                "addi       %2, %2, 4\n\t"
                "vssw.v     v2, (%2), %5\n\t"
                "addi       %2, %2, -4\n\t"
                "addi       %2, %2, 64\n\t"         // sa += 16 ele * 4

                "addi       %3, %3, -1\n\t"
                "bnez       %3, 1b\n\t"

        "2:\n\t"
            "beqz       %4, 3f\n\t"      // k_tail == 0 ?
            // Processing k_tail
            "vsetvli    zero, %4, e32, m2\n\t"
            "vlw.v      v0, (%0)\n\t"
            "add        %0, %0, t0\n\t"
            "vlw.v      v2, (%1)\n\t"
            "add        %1, %1, t0\n\t"

            "vssw.v     v0, (%2), %5\n\t"
            "addi       %2, %2, 4\n\t"
            "vssw.v     v2, (%2), %5\n\t"
            "addi       %2, %2, -4\n\t"
            "add        %2, %2, t1\n\t"         // sa += k_tail * 2 * 4

        "3:\n\t"
            // end packm2
            "add        %6, %6, t2\n\t"

            :"=r"(a0),      // %0
            "=r"(a1),       // %1
            "=r"(sa),       // %2
            "=r"(k8),       // %3
            "=r"(k_tail),   // %4
            "=r"(store_stride),     // %5
            "=r"(a),        // %6
            "=r"(ldx)       // %7
            :"0"(a0),
            "1"(a1),
            "2"(sa),
            "3"(k8),
            "4"(k_tail),
            "5"(store_stride),
            "6"(a),
            "7"(ldx)
            :"v0", "v1", "v2", "v3", "t0", "t1", "t2"
        );
    }
    if(m1 > 0) {
        memcpy(sa, a, sizeof(float) * ldx);
    }
#else
    int i = 0;
    for(; i + 3 < m; i += 4) {
        float *p0 = a;
        float *p1 = a + ldx;
        float *p2 = a + 2 * ldx;
        float *p3 = a + 3 * ldx;
        int j = 0;
        for(; j + 7 < k; j += 8) {
            sa[0] = p0[0];  sa[16] = p0[4];
            sa[1] = p1[0];  sa[17] = p1[4];
            sa[2] = p2[0];  sa[18] = p2[4];
            sa[3] = p3[0];  sa[19] = p3[4];

            sa[4] = p0[1];  sa[20] = p0[5];
            sa[5] = p1[1];  sa[21] = p1[5];
            sa[6] = p2[1];  sa[22] = p2[5];
            sa[7] = p3[1];  sa[23] = p3[5];

            sa[8] = p0[2];  sa[24] = p0[6];
            sa[9] = p1[2];  sa[25] = p1[6];
            sa[10] = p2[2]; sa[26] = p2[6];
            sa[11] = p3[2]; sa[27] = p3[6];

            sa[12] = p0[3]; sa[28] = p0[7];
            sa[13] = p1[3]; sa[29] = p1[7];
            sa[14] = p2[3]; sa[30] = p2[7];
            sa[15] = p3[3]; sa[31] = p3[7];

            sa += 32;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;

        }
        if(j + 3 < k) {
            j += 4;
            sa[0] = p0[0];  sa[8] = p0[2];
            sa[1] = p1[0];  sa[9] = p1[2];
            sa[2] = p2[0];  sa[10] = p2[2];
            sa[3] = p3[0];  sa[11] = p3[2];

            sa[4] = p0[1];  sa[12] = p0[3];
            sa[5] = p1[1];  sa[13] = p1[3];
            sa[6] = p2[1];  sa[14] = p2[3];
            sa[7] = p3[1];  sa[15] = p3[3];

            sa += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        if(j + 1 < k) {
            j += 2;
            sa[0] = p0[0];
            sa[1] = p1[0];
            sa[2] = p2[0];
            sa[3] = p3[0];

            sa[4] = p0[1];
            sa[5] = p1[1];
            sa[6] = p2[1];
            sa[7] = p3[1];

            sa += 8;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
        }
        if(j < k) {
            sa[0] = p0[0];
            sa[1] = p1[0];
            sa[2] = p2[0];
            sa[3] = p3[0];

            sa += 4;
        }
        a += 4 * ldx;
    }
    if(i + 1 < m) {
        i += 2;
        float *p0 = a;
        float *p1 = a + ldx;

        int j = 0;
        for(; j + 7 < k; j += 8) {
            sa[0] = p0[0];
            sa[1] = p1[0];
            sa[2] = p0[1];
            sa[3] = p1[1];
            sa[4] = p0[2];
            sa[5] = p1[2];
            sa[6] = p0[3];
            sa[7] = p1[3];
            sa[8] = p0[4];
            sa[9] = p1[4];
            sa[10] = p0[5];
            sa[11] = p1[5];
            sa[12] = p0[6];
            sa[13] = p1[6];
            sa[14] = p0[7];
            sa[15] = p1[7];

            sa += 16;
            p0 += 8;
            p1 += 8;
        }
        if(j + 3 < k) {
            j += 4;
            sa[0] = p0[0];
            sa[1] = p1[0];
            sa[2] = p0[1];
            sa[3] = p1[1];
            sa[4] = p0[2];
            sa[5] = p1[2];
            sa[6] = p0[3];
            sa[7] = p1[3];

            sa += 8;
            p0 += 4;
            p1 += 4;
        }
        if(j + 1 < k) {
            j += 2;
            sa[0] = p0[0];
            sa[1] = p1[0];
            sa[2] = p0[1];
            sa[3] = p1[1];

            sa += 4;
            p0 += 2;
            p1 += 2;
        }
        if(j < k) {
            sa[0] = p0[0];
            sa[1] = p1[0];

            sa += 2;
        }
        a += 2 * ldx;
    }
    if(i < m) {
        memcpy(sa, a, sizeof(float) * ldx);
    }
#endif // __riscv_vector
}

static void reorder_b(float *b, float *sb, int k, int n, int ldx)
{

#if __riscv_vector == 128
    DECOMPOSE_N
    DECOMPOSE_K
    if(n4 > 0) {
        float *b0 = b;
        float *b1 = b0 + 1;
        float *b2 = b1 + 1;
        float *b3 = b2 + 1;
        int k_tail = k & 7;
        int load_stride = 4 * ldx;
        int store_stride = 16;
        asm volatile(
            "slli       t0, %11, 5\n\t"         // t0 = 8 * ldx * 4
            "slli       t1, %7, 4\n\t"          // t1 = 4 * k_tail * 4

        "1:\n\t"
            // start packn4
            "mv         %0, %10\n\t"            // b0 = b
            "addi       %1, %0, 4\n\t"          // b1 = b0 + 1
            "addi       %2, %1, 4\n\t"          // b2 = b1 + 1
            "addi       %3, %2, 4\n\t"          // b3 = b2 + 1
            "mv         t6, %6\n\t"             // t6 = k8
            "beqz       t6, 3f\n\t"    // k8 == 0 ?
            "vsetvli    zero, zero, e32, m2\n\t"

            "2:\n\t"
                // start subpack_n4k8
                "vlsw.v     v0, (%0), %8\n\t"
                "vlsw.v     v2, (%1), %8\n\t"
                "vlsw.v     v4, (%2), %8\n\t"
                "vlsw.v     v6, (%3), %8\n\t"
                "add        %0, %0, t0\n\t"
                "add        %1, %1, t0\n\t"
                "add        %2, %2, t0\n\t"
                "add        %3, %3, t0\n\t"

                "vssw.v     v0, (%4), %9\n\t"
                "addi       %4, %4, 4\n\t"
                "vssw.v     v2, (%4), %9\n\t"
                "addi       %4, %4, 4\n\t"
                "vssw.v     v4, (%4), %9\n\t"
                "addi       %4, %4, 4\n\t"
                "vssw.v     v6, (%4), %9\n\t"
                "addi       %4, %4, -12\n\t"
                "addi       %4, %4, 128\n\t"    // sb += 32 * 4

                "addi       t6, t6, -1\n\t"     // k8--
                "bnez       t6, 2b\n\t"

        "3:\n\t"
            "beqz       %7, 4f\n\t"      // k_tail == 0 ?
            // Processing k_tail
            "vsetvli    zero, %7, e32, m2\n\t"
            "vlsw.v     v0, (%0), %8\n\t"
            "vlsw.v     v2, (%1), %8\n\t"
            "vlsw.v     v4, (%2), %8\n\t"
            "vlsw.v     v6, (%3), %8\n\t"

            "vssw.v     v0, (%4), %9\n\t"
            "addi       %4, %4, 4\n\t"
            "vssw.v     v2, (%4), %9\n\t"
            "addi       %4, %4, 4\n\t"
            "vssw.v     v4, (%4), %9\n\t"
            "addi       %4, %4, 4\n\t"
            "vssw.v     v6, (%4), %9\n\t"
            "addi       %4, %4, -12\n\t"
            "add        %4, %4, t1\n\t"     // sb += k_tail * 4 * 4

        "4:\n\t"
            // end packn4
            "addi %10, %10, 16\n\t"         // b += 4 * 4
            "addi %5, %5, -1\n\t"           // n4--
            "bnez %5, 1b\n\t"

            :"=r"(b0),      // %0
            "=r"(b1),       // %1
            "=r"(b2),       // %2
            "=r"(b3),       // %3
            "=r"(sb),       // %4
            "=r"(n4),       // %5
            "=r"(k8),       // %6
            "=r"(k_tail),   // %7
            "=r"(load_stride),      // %8
            "=r"(store_stride),     // %9
            "=r"(b),        // %10
            "=r"(ldx)       // %11
            :"0"(b0),
            "1"(b1),
            "2"(b2),
            "3"(b3),
            "4"(sb),
            "5"(n4),
            "6"(k8),
            "7"(k_tail),
            "8"(load_stride),
            "9"(store_stride),
            "10"(b),
            "11"(ldx)
            :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "t0", "t1", "t6"
        );
    }
    int n_tail = n & 3;
    if(n_tail > 0) {
        float *b0 = b;
        int k_tail = k & 7;
        int load_stride = 4 * ldx;
        asm volatile(
            "slli       t0, %7, 5\n\t"          // t0 = 8 * ldx * 4
            "slli       t1, %4, 2\n\t"          // t1 = k_tail * 4

        "1:\n\t"
            // pack remain n_tail cols one by one
            "mv         %0, %6\n\t"             // b0 = b
            "mv         t3, %3\n\t"             // t3 = k8
            "beqz       t3, 3f\n\t"    // k8 == 0 ?
            "vsetvli    zero, zero, e32, m2\n\t"

            "2:\n\t"
                // start subpack_n1k8
                "vlsw.v     v0, (%0), %5\n\t"
                "add        %0, %0, t0\n\t"
                "vsw.v      v0, (%1)\n\t"
                "addi       %1, %1, 32\n\t"     // sb += 8 * 4

                "addi       t3, t3, -1\n\t"     // k8--
                "bnez       t3, 2b\n\t"

        "3:\n\t"
            "beqz       %4, 4f\n\t"      // k_tail == 0 ?
            // Processing k_tail
            "vsetvli    zero, %4, e32, m2\n\t"
            "vlsw.v     v0, (%0), %5\n\t"
            "vsw.v      v0, (%1)\n\t"
            "add        %1, %1, t1\n\t"

        "4:\n\t"
            // end packn1
            "addi       %6, %6, 4\n\t"          // b += 1 * 4
            "addi       %2, %2, -1\n\t"
            "bnez       %2, 1b\n\t"

            :"=r"(b0),      // %0
            "=r"(sb),       // %1
            "=r"(n_tail),   // %2
            "=r"(k8),       // %3
            "=r"(k_tail),   // %4
            "=r"(load_stride),  // %5
            "=r"(b),         // %6
            "=r"(ldx)       // %7
            :"0"(b0),
            "1"(sb),
            "2"(n_tail),
            "3"(k8),
            "4"(k_tail),
            "5"(load_stride),
            "6"(b),
            "7"(ldx)
            :"v0", "v1", "t0", "t1", "t3"
        );
    }
#else
    int i = 0;
    for(; i + 3 < n; i += 4) {
        const float* p0 = b + i;
        const float* p1 = b + 1 * ldx + i;
        const float* p2 = b + 2 * ldx + i;
        const float* p3 = b + 3 * ldx + i;

        const float* p4 = b + 4 * ldx + i;
        const float* p5 = b + 5 * ldx + i;
        const float* p6 = b + 6 * ldx + i;
        const float* p7 = b + 7 * ldx + i;

        int j = 0;
        for(; j + 7 < k; j += 8) {
            sb[0] = p0[0];  sb[4] = p1[0];
            sb[1] = p0[1];  sb[5] = p1[1];
            sb[2] = p0[2];  sb[6] = p1[2];
            sb[3] = p0[3];  sb[7] = p1[3];

            sb[8] = p2[0];  sb[12] = p3[0];
            sb[9] = p2[1];  sb[13] = p3[1];
            sb[10] = p2[2];  sb[14] = p3[2];
            sb[11] = p2[3];  sb[15] = p3[3];

            sb[16] = p4[0];  sb[20] = p5[0];
            sb[17] = p4[1];  sb[21] = p5[1];
            sb[18] = p4[2];  sb[22] = p5[2];
            sb[19] = p4[3];  sb[23] = p5[3];

            sb[24] = p6[0];  sb[28] = p7[0];
            sb[25] = p6[1];  sb[29] = p7[1];
            sb[26] = p6[2];  sb[30] = p7[2];
            sb[27] = p6[3];  sb[31] = p7[3];

            sb += 32;
            p0 += 8 * ldx;
            p1 += 8 * ldx;
            p2 += 8 * ldx;
            p3 += 8 * ldx;
            p4 += 8 * ldx;
            p5 += 8 * ldx;
            p6 += 8 * ldx;
            p7 += 8 * ldx;
        }
        if(j + 3 < k) {
            j += 4;
            sb[0] = p0[0];
            sb[1] = p0[1];
            sb[2] = p0[2];
            sb[3] = p0[3];

            sb[4] = p1[0];
            sb[5] = p1[1];
            sb[6] = p1[2];
            sb[7] = p1[3];

            sb[8] = p2[0];
            sb[9] = p2[1];
            sb[10] = p2[2];
            sb[11] = p2[3];

            sb[12] = p3[0];
            sb[13] = p3[1];
            sb[14] = p3[2];
            sb[15] = p3[3];

            sb += 16;
            p0 += 4 * ldx;
            p1 += 4 * ldx;
            p2 += 4 * ldx;
            p3 += 4 * ldx;
        }
        if(j + 1 < k) {
            j += 2;
            sb[0] = p0[0];
            sb[1] = p0[1];
            sb[2] = p0[2];
            sb[3] = p0[3];

            sb[4] = p1[0];
            sb[5] = p1[1];
            sb[6] = p1[2];
            sb[7] = p1[3];

            sb += 8;
            p0 += 2 * ldx;
            p1 += 2 * ldx;
        }
        if(j < k) {
            sb[0] = p0[0];
            sb[1] = p0[1];
            sb[2] = p0[2];
            sb[3] = p0[3];

            sb += 4;
            p0 += ldx;
        }
    }
    while(i < n)
    {
        const float *p = b + i;
        for(int j = 0; j < k; j++) {
            *sb = *p;
            sb ++;
            p += ldx;
        }
        i++;
    }

#endif // __riscv_vector
}


static void kernel_m1_f32(float* dst, float* sa, float* sb, int m, int k, int n, int ldc, float* bias)
{
    float *pa = sa;
    float *pb = sb;
    float *pc = dst;
    DECOMPOSE_K
    DECOMPOSE_N

#if __riscv_vector == 128
    if(n4 > 0) {
        asm volatile(
            "vsetvli    zero, zero, e32, m1\n\t"
            "flw        ft0, (%8)\n\t"
        "1:\n\t"
            // start kernel_m1n4
            "vfmv.v.f   v24, ft0\n\t"    // v24[0..3] = *bias
            // "vlw.v      v24, (%8)\n\t"      // v24[0..3] = bias[0..3]
            // "addi       %8, %8, 16\n\t"

            "mv         a1, %0\n\t"         // a1 = pa
            "mv         t0, %3\n\t"         // t0 = k8
            "beqz       t0, 3f\n\t"         // k8 == 0 ?

            "2:\n\t"
                // start subkernel_m1n4k8
                "vlw.v      v1, (%1)\n\t"   // load pb
                "flw        ft1, 0(a1)\n\t" // load pa
                "vfmv.v.f   v2, ft1\n\t"
                "addi       %1, %1, 16\n\t" // pb += 4 * 4
                "vfmacc.vv  v24, v1, v2\n\t" // 0

                "vlw.v      v3, (%1)\n\t"
                "flw        ft2, 4(a1)\n\t"
                "vfmv.v.f   v4, ft2\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v3, v4\n\t" // 1

                "vlw.v      v5, (%1)\n\t"
                "flw        ft3, 8(a1)\n\t"
                "vfmv.v.f   v6, ft3\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v5, v6\n\t" // 2

                "vlw.v      v7, (%1)\n\t"
                "flw        ft4, 12(a1)\n\t"
                "vfmv.v.f   v8, ft4\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v7, v8\n\t" // 3

                "vlw.v      v9, (%1)\n\t"
                "flw        ft5, 16(a1)\n\t"
                "vfmv.v.f   v10, ft5\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v9, v10\n\t" // 4

                "vlw.v      v11, (%1)\n\t"
                "flw        ft6, 20(a1)\n\t"
                "vfmv.v.f   v12, ft6\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v11, v12\n\t" // 5

                "vlw.v      v13, (%1)\n\t"
                "flw        ft7, 24(a1)\n\t"
                "vfmv.v.f   v14, ft7\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v13, v14\n\t" // 6

                "vlw.v      v15, (%1)\n\t"
                "flw        ft8, 28(a1)\n\t"
                "vfmv.v.f   v16, ft8\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v15, v16\n\t" // 7
                "addi       a1, a1, 32\n\t"

                "addi       t0, t0, -1\n\t"
                "bnez       t0, 2b\n\t"

        "3:\n\t"
            "beqz       %4, 4f\n\t"         // k4 == 0 ?
            // start subkernel_m1n4k4
            "vlw.v      v1, (%1)\n\t"
            "flw        ft1, 0(a1)\n\t"
            "vfmv.v.f   v2, ft1\n\t"
            "addi       %1, %1, 16\n\t"
            "vfmacc.vv  v24, v1, v2\n\t" // 0

            "vlw.v      v3, (%1)\n\t"
            "flw        ft2, 4(a1)\n\t"
            "vfmv.v.f   v4, ft2\n\t"
            "addi       %1, %1, 16\n\t"
            "vfmacc.vv  v24, v3, v4\n\t" // 1

            "vlw.v      v5, (%1)\n\t"
            "flw        ft3, 8(a1)\n\t"
            "vfmv.v.f   v6, ft3\n\t"
            "addi       %1, %1, 16\n\t"
            "vfmacc.vv  v24, v5, v6\n\t" // 2

            "vlw.v      v7, (%1)\n\t"
            "flw        ft4, 12(a1)\n\t"
            "vfmv.v.f   v8, ft4\n\t"
            "addi       %1, %1, 16\n\t"
            "vfmacc.vv  v24, v7, v8\n\t" // 3
            "addi       a1, a1, 16\n\t"
 
        "4:\n\t"
            "beqz       %5, 5f\n\t"         // k2 == 0 ?
            // start subkernel_m1n4k2
            "vlw.v      v1, (%1)\n\t"
            "flw        ft1, 0(a1)\n\t"
            "vfmv.v.f   v2, ft1\n\t"
            "addi       %1, %1, 16\n\t"
            "vfmacc.vv  v24, v1, v2\n\t" // 0

            "vlw.v      v3, (%1)\n\t"
            "flw        ft2, 4(a1)\n\t"
            "vfmv.v.f   v4, ft2\n\t"
            "addi       %1, %1, 16\n\t"
            "vfmacc.vv  v24, v3, v4\n\t" // 1
            "addi       a1, a1, 8\n\t"

        "5:\n\t"
            "beqz       %6, 6f\n\t"        // k1 == 0 ?
            // start subkernel_m1n4k1
            "vlw.v      v1, (%1)\n\t"
            "flw        ft1, 0(a1)\n\t"
            "vfmv.v.f   v2, ft1\n\t"
            "addi       %1, %1, 16\n\t"
            "vfmacc.vv  v24, v1, v2\n\t" // 0
            "addi       a1, a1, 4\n\t"

        "6:\n\t"
            // end kernel_m1n4
            "vsw.v      v24, (%2)\n\t"
            "addi       %2, %2, 16\n\t"     // pc += 4 * 4

            "addi       %7, %7, -1\n\t"
            "bnez       %7, 1b\n\t"

            :"=r"(pa),   // %0
            "=r"(pb),    // %1
            "=r"(pc),    // %2
            "=r"(k8),    // %3
            "=r"(k4),    // %4
            "=r"(k2),    // %5
            "=r"(k1),    // %6
            "=r"(n4),    // %7
            "=r"(bias)   // %8
            :"0"(pa),
            "1"(pb),
            "2"(pc),
            "3"(k8),
            "4"(k4),
            "5"(k2),
            "6"(k1),
            "7"(n4),
            "8"(bias)
            :"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v24",
            "a1", "t0", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8"
        );
    }
    if(n2 > 0) {
        int k_tail = k & 7;
        float *pb0 = pb;
        float *pb1 = pb0 + k;

        asm volatile(
            "mv             t4, %4\n\t"             // t4 = k8
            "vsetvli        zero, zero, e32, m2\n\t"
            "vxor.vv        v6, v6, v6\n\t"         // clear
            "vxor.vv        v8, v8, v8\n\t"         // clear
            "flw            ft0, 0(%6)\n\t"         // ft0 = *bias
            // "flw            ft3, 4(%6)\n\t"         // ft3 = *(bias + 1)
            // "addi           %6, %6, 8\n\t"
            "vfmv.s.f       v10, ft0\n\t"           // v10[0] = ft0
            "vfmv.s.f       v12, ft0\n\t"           // v10[0] = ft0
            // "vfmv.s.f       v12, ft3\n\t"           // v12[0] = ft3

            "beqz           %5, 1f\n\t"         // k_tail == 0 ?
            // Processing k_tail
            "slli           t0, %5, 2\n\t"          // t0 = k_tail * 4
            "vsetvli        zero, %5, e32, m2\n\t"
            "vlw.v          v0, (%0)\n\t"
            "add            %0, %0, t0\n\t"
            "vlw.v          v2, (%1)\n\t"
            "add            %1, %1, t0\n\t"
            "vlw.v          v4, (%2)\n\t"
            "add            %2, %2, t0\n\t"
            "vfmacc.vv      v6, v0, v2\n\t"
            "vfmacc.vv      v8, v0, v4\n\t"
            "beqz           t4, 2f\n\t"        // k8 == 0 ?
            "vsetvli        zero, zero, e32, m2\n\t"

            "1:\n\t"
                // start subkernel_m1n2k8
                "vlw.v          v0, (%0)\n\t"
                "addi           %0, %0, 32\n\t"
                "vlw.v          v2, (%1)\n\t"
                "addi           %1, %1, 32\n\t"
                "vlw.v          v4, (%2)\n\t"
                "addi           %2, %2, 32\n\t"
                "vfmacc.vv      v6, v0, v2\n\t"
                "vfmacc.vv      v8, v0, v4\n\t"
                "addi           t4, t4, -1\n\t"
                "bnez           t4, 1b\n\t"

        "2:\n\t"
            // end kernel_m1n2
            "vfredsum.vs    v10, v6, v10\n\t"       // v10[0] = v10[0] + sum(v6[0..i])
            "vfredsum.vs    v12, v8, v12\n\t"       // v12[0] = v12[0] + sum(v8[0..i])
            "vfmv.f.s       ft1, v10\n\t"
            "fsw            ft1, 0(%3)\n\t"
            "vfmv.f.s       ft2, v12\n\t"
            "fsw            ft2, 4(%3)\n\t"

            :"=r"(pa),      // %0
            "=r"(pb0),      // %1
            "=r"(pb1),      // %2
            "=r"(pc),       // %3
            "=r"(k8),       // %4
            "=r"(k_tail),   // %5
            "=r"(bias)      // %6
            :"0"(pa),
            "1"(pb0),
            "2"(pb1),
            "3"(pc),
            "4"(k8),
            "5"(k_tail),
            "6"(bias)
            :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
            "ft0", "ft1", "ft2", "ft3", "t0", "t4"
        );
        pb += 2 * k;
        pc += 2;
    }
    if(n1 > 0) {
        pa = sa;
        int k_tail = k & 7;
        asm volatile(
            "vsetvli        zero, zero, e32, m2\n\t"
            "vxor.vv        v4, v4, v4\n\t"         // clear

            "flw            ft0, 0(%5)\n\t"         // ft0 = *bias
            "vfmv.s.f       v6, ft0\n\t"            // v6[0] = ft0

            "beqz           %4, 1f\n\t"         // k_tail == 0 ?
            // Processing k_tail
            "slli           t0, %4, 2\n\t"          // t0 = k_tail * 4
            "vsetvli        zero, %4, e32, m2\n\t"
            "vlw.v          v0, (%0)\n\t"
            "add            %0, %0, t0\n\t"
            "vlw.v          v2, (%1)\n\t"
            "add            %1, %1, t0\n\t"
            "vfmacc.vv      v4, v0, v2\n\t"
            "beqz           %3, 2f\n\t"        // k8 == 0 ?
            "vsetvli        zero, zero, e32, m2\n\t"

            "1:\n\t"
                // start subkernel_m1n1k8
                "vlw.v          v0, (%0)\n\t"
                "addi           %0, %0, 32\n\t"
                "vlw.v          v2, (%1)\n\t"
                "addi           %1, %1, 32\n\t"
                "vfmacc.vv      v4, v0, v2\n\t"
                "addi           %3, %3, -1\n\t"
                "bnez           %3, 1b\n\t"

        "2:\n\t"
            // end kernel_m1n1
            "vfredsum.vs    v6, v4, v6\n\t"         // v6[0] = v6[0] + sum(v4[0..i])
            "vfmv.f.s       ft1, v6\n\t"
            "fsw            ft1, 0(%2)\n\t"

            :"=r"(pa),      // %0
            "=r"(pb),       // %1
            "=r"(pc),       // %2
            "=r"(k8),       // %3
            "=r"(k_tail),   // %4
            "=r"(bias)      // %5
            :"0"(pa),
            "1"(pb),
            "2"(pc),
            "3"(k8),
            "4"(k_tail),
            "5"(bias)
            :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "ft0", "ft1", "t0"
        );
    }
#else
    for(int i = 0; i < n4; i++) {
        int j = 0;
        pa = sa;
        pc[0] = pc[1] = pc[2] = pc[3] = *bias;
        for(; j + 7 < k; j += 8) {
            pc[0] += pa[0] * pb[0];
            pc[1] += pa[0] * pb[1];
            pc[2] += pa[0] * pb[2];
            pc[3] += pa[0] * pb[3];

            pc[0] += pa[1] * pb[4];
            pc[1] += pa[1] * pb[5];
            pc[2] += pa[1] * pb[6];
            pc[3] += pa[1] * pb[7];

            pc[0] += pa[2] * pb[8];
            pc[1] += pa[2] * pb[9];
            pc[2] += pa[2] * pb[10];
            pc[3] += pa[2] * pb[11];

            pc[0] += pa[3] * pb[12];
            pc[1] += pa[3] * pb[13];
            pc[2] += pa[3] * pb[14];
            pc[3] += pa[3] * pb[15];

            pc[0] += pa[4] * pb[16];
            pc[1] += pa[4] * pb[17];
            pc[2] += pa[4] * pb[18];
            pc[3] += pa[4] * pb[19];

            pc[0] += pa[5] * pb[20];
            pc[1] += pa[5] * pb[21];
            pc[2] += pa[5] * pb[22];
            pc[3] += pa[5] * pb[23];

            pc[0] += pa[6] * pb[24];
            pc[1] += pa[6] * pb[25];
            pc[2] += pa[6] * pb[26];
            pc[3] += pa[6] * pb[27];

            pc[0] += pa[7] * pb[28];
            pc[1] += pa[7] * pb[29];
            pc[2] += pa[7] * pb[30];
            pc[3] += pa[7] * pb[31];

            pa += 8;
            pb += 32;
        }
        if(j + 3 < k) {
            j += 4;
            pc[0] += pa[0] * pb[0];
            pc[1] += pa[0] * pb[1];
            pc[2] += pa[0] * pb[2];
            pc[3] += pa[0] * pb[3];

            pc[0] += pa[1] * pb[4];
            pc[1] += pa[1] * pb[5];
            pc[2] += pa[1] * pb[6];
            pc[3] += pa[1] * pb[7];

            pc[0] += pa[2] * pb[8];
            pc[1] += pa[2] * pb[9];
            pc[2] += pa[2] * pb[10];
            pc[3] += pa[2] * pb[11];

            pc[0] += pa[3] * pb[12];
            pc[1] += pa[3] * pb[13];
            pc[2] += pa[3] * pb[14];
            pc[3] += pa[3] * pb[15];

            pa += 4;
            pb += 16;
        }
        if(j + 1 < k) {
            j += 2;
            pc[0] += pa[0] * pb[0];
            pc[1] += pa[0] * pb[1];
            pc[2] += pa[0] * pb[2];
            pc[3] += pa[0] * pb[3];

            pc[0] += pa[1] * pb[4];
            pc[1] += pa[1] * pb[5];
            pc[2] += pa[1] * pb[6];
            pc[3] += pa[1] * pb[7];

            pa += 2;
            pb += 8;
        }
        if(j < k) {
            pc[0] += pa[0] * pb[0];
            pc[1] += pa[0] * pb[1];
            pc[2] += pa[0] * pb[2];
            pc[3] += pa[0] * pb[3];

            pa += 1;
            pb += 4;
        }
        pc += 4;
    }
    if(n2 > 0) {
        pa = sa;
        pc[0] = pc[1] = *bias;
        float *pb0 = pb;
        float *pb1 = pb0 + k;
        int j = 0;
        for(; j + 7 < k; j += 8) {
            pc[0] += pa[0] * pb0[0];
            pc[1] += pa[0] * pb1[0];

            pc[0] += pa[1] * pb0[1];
            pc[1] += pa[1] * pb1[1];

            pc[0] += pa[2] * pb0[2];
            pc[1] += pa[2] * pb1[2];

            pc[0] += pa[3] * pb0[3];
            pc[1] += pa[3] * pb1[3];

            pc[0] += pa[4] * pb0[4];
            pc[1] += pa[4] * pb1[4];

            pc[0] += pa[5] * pb0[5];
            pc[1] += pa[5] * pb1[5];

            pc[0] += pa[6] * pb0[6];
            pc[1] += pa[6] * pb1[6];

            pc[0] += pa[7] * pb0[7];
            pc[1] += pa[7] * pb1[7];

            pa += 8;
            pb0 += 8;
            pb1 += 8;
        }
        if(j + 3 < k) {
            j += 4;
            pc[0] += pa[0] * pb0[0];
            pc[1] += pa[0] * pb1[0];

            pc[0] += pa[1] * pb0[1];
            pc[1] += pa[1] * pb1[1];

            pc[0] += pa[2] * pb0[2];
            pc[1] += pa[2] * pb1[2];

            pc[0] += pa[3] * pb0[3];
            pc[1] += pa[3] * pb1[3];

            pa += 4;
            pb0 += 4;
            pb1 += 4;
        }
        if(j + 1 < k) {
            j += 2;
            pc[0] += pa[0] * pb0[0];
            pc[1] += pa[0] * pb1[0];

            pc[0] += pa[1] * pb0[1];
            pc[1] += pa[1] * pb1[1];

            pa += 2;
            pb0 += 2;
            pb1 += 2;
        }
        if(j < k) {
            pc[0] += pa[0] * pb0[0];
            pc[1] += pa[0] * pb1[0];

            pa += 1;
            pb0 += 1;
            pb1 += 1;
        }
        pc += 2;
        pb += 2 * k;
    }
    if(n1 > 0) {
        pa = sa;
        pc[0] = *bias;
        int j = 0;
        for(; j + 7 < k; j += 8) {
            pc[0] += pa[0] * pb[0];
            pc[0] += pa[1] * pb[1];
            pc[0] += pa[2] * pb[2];
            pc[0] += pa[3] * pb[3];
            pc[0] += pa[4] * pb[4];
            pc[0] += pa[5] * pb[5];
            pc[0] += pa[6] * pb[6];
            pc[0] += pa[7] * pb[7];

            pa += 8;
            pb += 8;
        }
        if(j + 3 < k) {
            j += 4;
            pc[0] += pa[0] * pb[0];
            pc[0] += pa[1] * pb[1];
            pc[0] += pa[2] * pb[2];
            pc[0] += pa[3] * pb[3];

            pa += 4;
            pb += 4;
        }
        if(j + 1 < k) {
            j += 2;
            pc[0] += pa[0] * pb[0];
            pc[0] += pa[1] * pb[1];

            pa += 2;
            pb += 2;
        }
        if(j < k) {
            pc[0] += pa[0] * pb[0];

            pa += 1;
            pb += 1;
        }
        pc += 1;
    }
#endif // __riscv_vector
}

static void kernel_m2_f32(float* dst, float* sa, float* sb, int m, int k, int n, int ldc, float* bias)
{
    float *pa = sa;
    float *pb = sb;
    float *pc0 = dst;
    float *pc1 = pc0 + ldc;
    DECOMPOSE_K
    DECOMPOSE_N
#if __riscv_vector == 128
    if(n4 > 0) {
        asm volatile(
            "vsetvli    zero, zero, e32, m1\n\t"
            "flw        ft0, (%9)\n\t"       // ft0 = *bias
            "flw        ft10, 4(%9)\n\t"     // ft1 = *(bias + 1)

        "1:\n\t"                        // n4
            // start kernel_m2n4
            "vfmv.v.f   v24, ft0\n\t"        // v24[0..3] = ft0 = *bias
            "vfmv.v.f   v25, ft10\n\t"       // v25[0..3] = ft10 = *(bias + 1)
            // "vlw.v      v24, (%9)\n\t"          // v24[0..3] = bias[0..3]
            // "vlw.v      v25, (%9)\n\t"          // v24[0..3] = bias[0..3]
            // "addi       %9, %9, 16\n\t"

            "mv         a1, %0\n\t"             // a1 = pa
            "mv         t0, %4\n\t"             // t0 = k8
            "beqz       t0, 3f\n\t"             // k8 == 0 ?

            "2:\n\t"
                // start subkernel_m2n4k8
                "vlw.v      v1, (%1)\n\t"
                "flw        ft1, 0(a1)\n\t"
                "vfmv.v.f   v2, ft1\n\t"
                "flw        fa1, 4(a1)\n\t"
                "vfmv.v.f   v3, fa1\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v1, v2\n\t" // 0
                "vfmacc.vv  v25, v1, v3\n\t"

                "vlw.v      v4, (%1)\n\t"
                "flw        ft2, 8(a1)\n\t"
                "vfmv.v.f   v5, ft2\n\t"
                "flw        fa2, 12(a1)\n\t"
                "vfmv.v.f   v6, fa2\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v4, v5\n\t" // 1
                "vfmacc.vv  v25, v4, v6\n\t"

                "vlw.v      v7, (%1)\n\t"
                "flw        ft3, 16(a1)\n\t"
                "vfmv.v.f   v8, ft3\n\t"
                "flw        fa3, 20(a1)\n\t"
                "vfmv.v.f   v9, fa3\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v7, v8\n\t" // 2
                "vfmacc.vv  v25, v7, v9\n\t"

                "vlw.v      v10, (%1)\n\t"
                "flw        ft4, 24(a1)\n\t"
                "vfmv.v.f   v11, ft4\n\t"
                "flw        fa4, 28(a1)\n\t"
                "vfmv.v.f   v12, fa4\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v10, v11\n\t" // 3
                "vfmacc.vv  v25, v10, v12\n\t"

                "vlw.v      v13, (%1)\n\t"
                "flw        ft5, 32(a1)\n\t"
                "vfmv.v.f   v14, ft5\n\t"
                "flw        fa5, 36(a1)\n\t"
                "vfmv.v.f   v15, fa5\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v13, v14\n\t" // 4
                "vfmacc.vv  v25, v13, v15\n\t"

                "vlw.v      v16, (%1)\n\t"
                "flw        ft6, 40(a1)\n\t"
                "vfmv.v.f   v17, ft6\n\t"
                "flw        fa6, 44(a1)\n\t"
                "vfmv.v.f   v18, fa6\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v16, v17\n\t" // 5
                "vfmacc.vv  v25, v16, v18\n\t"

                "vlw.v      v19, (%1)\n\t"
                "flw        ft7, 48(a1)\n\t"
                "vfmv.v.f   v20, ft7\n\t"
                "flw        fa7, 52(a1)\n\t"
                "vfmv.v.f   v21, fa7\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v19, v20\n\t" // 6
                "vfmacc.vv  v25, v19, v21\n\t"

                "vlw.v      v28, (%1)\n\t"
                "flw        ft8, 56(a1)\n\t"
                "vfmv.v.f   v29, ft8\n\t"
                "flw        fa0, 60(a1)\n\t"
                "vfmv.v.f   v30, fa0\n\t"
                "addi       %1, %1, 16\n\t"
                "vfmacc.vv  v24, v28, v29\n\t" // 7
                "vfmacc.vv  v25, v28, v30\n\t"
                "addi       a1, a1, 64\n\t"

                "addi       t0, t0, -1\n\t"
                "bnez       t0, 2b\n\t"

        "3:\n\t"
            "beqz       %5, 4f\n\t"         // k4 == 0 ?
            // start subkernel_m2n4k4
            "vlw.v      v1, (%1)\n\t"
            "flw        ft1, 0(a1)\n\t"
            "vfmv.v.f   v2, ft1\n\t"
            "flw        fa1, 4(a1)\n\t"
            "vfmv.v.f   v3, fa1\n\t"
            "addi       %1, %1, 16\n\t"
            "vfmacc.vv  v24, v1, v2\n\t" // 0
            "vfmacc.vv  v25, v1, v3\n\t"

            "vlw.v      v4, (%1)\n\t"
            "flw        ft2, 8(a1)\n\t"
            "vfmv.v.f   v5, ft2\n\t"
            "flw        fa2, 12(a1)\n\t"
            "vfmv.v.f   v6, fa2\n\t"
            "addi       %1, %1, 16\n\t"
            "vfmacc.vv  v24, v4, v5\n\t" // 1
            "vfmacc.vv  v25, v4, v6\n\t"

            "vlw.v      v7, (%1)\n\t"
            "flw        ft3, 16(a1)\n\t"
            "vfmv.v.f   v8, ft3\n\t"
            "flw        fa3, 20(a1)\n\t"
            "vfmv.v.f   v9, fa3\n\t"
            "addi       %1, %1, 16\n\t"
            "vfmacc.vv  v24, v7, v8\n\t" // 2
            "vfmacc.vv  v25, v7, v9\n\t"

            "vlw.v      v10, (%1)\n\t"
            "flw        ft4, 24(a1)\n\t"
            "vfmv.v.f   v11, ft4\n\t"
            "flw        fa4, 28(a1)\n\t"
            "vfmv.v.f   v12, fa4\n\t"
            "addi       %1, %1, 16\n\t"
            "vfmacc.vv  v24, v10, v11\n\t" // 3
            "vfmacc.vv  v25, v10, v12\n\t"
            "addi       a1, a1, 32\n\t"

        "4:\n\t"
            "beqz       %6, 5f\n\t"         // k2 == 0 ?
            // start subkernel_m2n4k2
            "vlw.v      v1, (%1)\n\t"
            "flw        ft1, 0(a1)\n\t"
            "vfmv.v.f   v2, ft1\n\t"
            "flw        fa1, 4(a1)\n\t"
            "vfmv.v.f   v3, fa1\n\t"
            "addi       %1, %1, 16\n\t"
            "vfmacc.vv  v24, v1, v2\n\t" // 0
            "vfmacc.vv  v25, v1, v3\n\t"

            "vlw.v      v4, (%1)\n\t"
            "flw        ft2, 8(a1)\n\t"
            "vfmv.v.f   v5, ft2\n\t"
            "flw        fa2, 12(a1)\n\t"
            "vfmv.v.f   v6, fa2\n\t"
            "addi       %1, %1, 16\n\t"
            "vfmacc.vv  v24, v4, v5\n\t" // 1
            "vfmacc.vv  v25, v4, v6\n\t"
            "addi       a1, a1, 16\n\t"


        "5:\n\t"
            "beqz       %7, 6f\n\t"        // k1 == 0 ?
            // start subkernel_m2n4k1
            "vlw.v      v1, (%1)\n\t"
            "flw        ft1, 0(a1)\n\t"
            "vfmv.v.f   v2, ft1\n\t"
            "flw        fa1, 4(a1)\n\t"
            "vfmv.v.f   v3, fa1\n\t"
            "addi       %1, %1, 16\n\t"
            "vfmacc.vv  v24, v1, v2\n\t" // 0
            "vfmacc.vv  v25, v1, v3\n\t"
            "addi       a1, a1, 8\n\t"

        "6:\n\t"
            // end kernel_m2n4
            "vsw.v      v24, (%2)\n\t"      // pc0[0..3] = v24
            "addi       %2, %2, 16\n\t"
            "vsw.v      v25, (%3)\n\t"      // pc1[0..3] = v25
            "addi       %3, %3, 16\n\t"

            "addi       %8, %8, -1\n\t"
            "bnez       %8, 1b\n\t"

            :"=r"(pa),   // %0
            "=r"(pb),    // %1
            "=r"(pc0),   // %2
            "=r"(pc1),   // %3
            "=r"(k8),    // %4
            "=r"(k4),    // %5
            "=r"(k2),    // %6
            "=r"(k1),    // %7
            "=r"(n4),    // %8
            "=r"(bias)   // %9
            :"0"(pa),
            "1"(pb),
            "2"(pc0),
            "3"(pc1),
            "4"(k8),
            "5"(k4),
            "6"(k2),
            "7"(k1),
            "8"(n4),
            "9"(bias)
            : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v24", "v25", "v28", "v29", "v30",
            "a1", "t0", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7"
        );
    }
    if(n2 > 0) {
        int k_tail = k & 7;
        float *pa0 = sa;
        float *pa1 = pa0 + 1;
        float *pb0 = pb;
        float *pb1 = pb0 + k;
        int load_stride = 8;

        asm volatile(
            "mv             t6, %6\n\t"             // t6 = k8
            "vsetvli        zero, zero, e32, m2\n\t"
            "vxor.vv        v8, v8, v8\n\t"         // clear
            "vxor.vv        v10, v10, v10\n\t"      // clear
            "vxor.vv        v12, v12, v12\n\t"      // clear
            "vxor.vv        v14, v14, v14\n\t"      // clear
            "flw            ft0, 0(%8)\n\t"         // ft0 = *bias
            "flw            ft1, 4(%8)\n\t"         // ft1 = *(bias + 1)
            // "addi           %8, %8, 8\n\t"
            "vfmv.s.f       v16, ft0\n\t"           // v16[0] = ft0
            "vfmv.s.f       v18, ft0\n\t"           // v18[0] = ft0
            "vfmv.s.f       v20, ft1\n\t"           // v20[0] = ft1
            "vfmv.s.f       v22, ft1\n\t"           // v22[1] = ft1

            "beqz           %7, 1f\n\t"             // k_tail == 0 ?
            // Processing k_tail
            "slli           t0, %7, 2\n\t"          // t0 = k_tail * 4
            "slli           t1, t0, 1\n\t"          // t1 = t0 * 2
            "vsetvli        zero, %7, e32, m2\n\t"
            "vlsw.v         v0, (%0), %9\n\t"
            "add            %0, %0, t1\n\t"
            "vlsw.v         v2, (%1), %9\n\t"
            "addi           %1, %0, 4\n\t"

            "vlw.v          v4, (%2)\n\t"
            "add            %2, %2, t0\n\t"
            "vlw.v          v6, (%3)\n\t"
            "add            %3, %3, t0\n\t"

            "vfmacc.vv      v8,  v0, v4\n\t"
            "vfmacc.vv      v10, v0, v6\n\t"
            "vfmacc.vv      v12, v2, v4\n\t"
            "vfmacc.vv      v14, v2, v6\n\t"
            "beqz           t6, 2f\n\t"        // k8 == 0 ?
            "vsetvli        zero, zero, e32, m2\n\t"

            "1:\n\t"
                // start subkernel_m2n2k8
                "vlsw.v         v0, (%0), %9\n\t"
                "addi           %0, %0, 64\n\t"
                "vlsw.v         v2, (%1), %9\n\t"
                "addi           %1, %0, 4\n\t"

                "vlw.v          v4, (%2)\n\t"
                "addi           %2, %2, 32\n\t"
                "vlw.v          v6, (%3)\n\t"
                "addi           %3, %3, 32\n\t"

                "vfmacc.vv      v8,  v0, v4\n\t"
                "vfmacc.vv      v10, v0, v6\n\t"
                "vfmacc.vv      v12, v2, v4\n\t"
                "vfmacc.vv      v14, v2, v6\n\t"
                "addi           t6, t6, -1\n\t"
                "bnez           t6, 1b\n\t"

        "2:\n\t"
            // end kernel_m2n2
            "vfredsum.vs    v16, v8, v16\n\t"    // v16[0] = v16[0] + sum(v8[0..i])
            "vfredsum.vs    v18, v10, v18\n\t"   // v18[0] = v18[0] + sum(v10[0..i])
            "vfredsum.vs    v20, v12, v20\n\t"   // v20[0] = v20[0] + sum(v12[0..i])
            "vfredsum.vs    v22, v14, v22\n\t"   // v22[0] = v22[0] + sum(v14[0..i])
            "vfmv.f.s       ft2, v16\n\t"
            "fsw            ft2, 0(%4)\n\t"
            "vfmv.f.s       ft3, v18\n\t"
            "fsw            ft3, 4(%4)\n\t"
            "vfmv.f.s       ft4, v20\n\t"
            "fsw            ft4, 0(%5)\n\t"
            "vfmv.f.s       ft5, v22\n\t"
            "fsw            ft5, 4(%5)\n\t"

            :"=r"(pa0),     // %0
            "=r"(pa1),      // %1
            "=r"(pb0),      // %2
            "=r"(pb1),      // %3
            "=r"(pc0),      // %4
            "=r"(pc1),      // %5
            "=r"(k8),       // %6
            "=r"(k_tail),   // %7
            "=r"(bias),     // %8
            "=r"(load_stride) // %9
            :"0"(pa0),
            "1"(pa1),
            "2"(pb0),
            "3"(pb1),
            "4"(pc0),
            "5"(pc1),
            "6"(k8),
            "7"(k_tail),
            "8"(bias),
            "9"(load_stride)
            :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
            "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "t0", "t1", "t6"
        );
        pb += 2 * k;
        pc0 += 2;
        pc1 += 2;
    }
    if(n1 > 0) {
        float *pa0 = sa;
        float *pa1 = pa0 + 1;
        int k8 = k >> 3;
        int k_tail = k & 7;
        int load_stride = 8;
        asm volatile(
            "mv             t5, %5\n\t"             // t5 = k8
            "vsetvli        zero, zero, e32, m2\n\t"
            "vxor.vv        v6, v6, v6\n\t"         // clear
            "vxor.vv        v8, v8, v8\n\t"          // clear
            "flw            ft0, 0(%7)\n\t"         // ft0 = *bias
            "flw            ft1, 4(%7)\n\t"         // ft1 = *(bias + 1)
            "vfmv.s.f       v10, ft0\n\t"           // v10[0] = ft0
            "vfmv.s.f       v12, ft1\n\t"           // v12[0] = ft1

            "beqz           %6, 1f\n\t"         // k_tail == 0 ?
            // Processing k_tail
            "slli           t0, %6, 2\n\t"          // t0 = k_tail * 4
            "slli           t1, t0, 1\n\t"          // t1 = t0 * 2
            "vsetvli        zero, %6, e32, m2\n\t"
            "vlsw.v         v0, (%0), %8\n\t"
            "add            %0, %0, t1\n\t"
            "vlsw.v         v2, (%1), %8\n\t"
            "addi           %1, %0, 4\n\t"

            "vlw.v          v4, (%2)\n\t"
            "add            %2, %2, t0\n\t"

            "vfmacc.vv      v6, v0, v4\n\t"
            "vfmacc.vv      v8, v2, v4\n\t"
            "beqz           t5, 2f\n\t"        // k8 == 0 ?
            "vsetvli        zero, zero, e32, m2\n\t"

            "1:\n\t"
                // start subkernel_m2n1k8
                "vlsw.v         v0, (%0), %8\n\t"
                "addi           %0, %0, 64\n\t"
                "vlsw.v         v2, (%1), %8\n\t"
                "addi           %1, %0, 4\n\t"

                "vlw.v          v4, (%2)\n\t"
                "addi           %2, %2, 32\n\t"

                "vfmacc.vv      v6, v0, v4\n\t"
                "vfmacc.vv      v8, v2, v4\n\t"
                "addi           t5, t5, -1\n\t"
                "bnez           t5, 1b\n\t"

        "2:\n\t"
            // end kernel_m2n1
            "vfredsum.vs    v10, v6, v10\n\t"       // v10[0] = v10[0] + sum(v6[0..i])
            "vfredsum.vs    v12, v8, v12\n\t"       // v12[0] = v12[0] + sum(v8[0..i])
            "vfmv.f.s       ft2, v10\n\t"
            "fsw            ft2, 0(%3)\n\t"
            "vfmv.f.s       ft3, v12\n\t"
            "fsw            ft3, 0(%4)\n\t"

            :"=r"(pa0),     // %0
            "=r"(pa1),      // %1
            "=r"(pb),       // %2
            "=r"(pc0),      // %3
            "=r"(pc1),      // %4
            "=r"(k8),       // %5
            "=r"(k_tail),   // %6
            "=r"(bias),     // %7
            "=r"(load_stride) // %8
            :"0"(pa0),
            "1"(pa1),
            "2"(pb),
            "3"(pc0),
            "4"(pc1),
            "5"(k8),
            "6"(k_tail),
            "7"(bias),
            "8"(load_stride)
            :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
            "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "t0", "t1", "t5"
        );
    }
#else
    for(int i = 0; i < n4; i++) {
        pa = sa;
        pc0[0] = pc0[1] = pc0[2] = pc0[3] = *bias;
        pc1[0] = pc1[1] = pc1[2] = pc1[3] = *(bias + 1);
        int j = 0;
        for(; j + 7 < k; j += 8) {
            pc0[0] += pa[0] * pb[0];    pc1[0] += pa[1] * pb[0];
            pc0[1] += pa[0] * pb[1];    pc1[1] += pa[1] * pb[1];
            pc0[2] += pa[0] * pb[2];    pc1[2] += pa[1] * pb[2];
            pc0[3] += pa[0] * pb[3];    pc1[3] += pa[1] * pb[3];

            pc0[0] += pa[2] * pb[4];    pc1[0] += pa[3] * pb[4];
            pc0[1] += pa[2] * pb[5];    pc1[1] += pa[3] * pb[5];
            pc0[2] += pa[2] * pb[6];    pc1[2] += pa[3] * pb[6];
            pc0[3] += pa[2] * pb[7];    pc1[3] += pa[3] * pb[7];

            pc0[0] += pa[4] * pb[8];     pc1[0] += pa[5] * pb[8];
            pc0[1] += pa[4] * pb[9];     pc1[1] += pa[5] * pb[9];
            pc0[2] += pa[4] * pb[10];    pc1[2] += pa[5] * pb[10];
            pc0[3] += pa[4] * pb[11];    pc1[3] += pa[5] * pb[11];

            pc0[0] += pa[6] * pb[12];    pc1[0] += pa[7] * pb[12];
            pc0[1] += pa[6] * pb[13];    pc1[1] += pa[7] * pb[13];
            pc0[2] += pa[6] * pb[14];    pc1[2] += pa[7] * pb[14];
            pc0[3] += pa[6] * pb[15];    pc1[3] += pa[7] * pb[15];

            pc0[0] += pa[8] * pb[16];    pc1[0] += pa[9] * pb[16];
            pc0[1] += pa[8] * pb[17];    pc1[1] += pa[9] * pb[17];
            pc0[2] += pa[8] * pb[18];    pc1[2] += pa[9] * pb[18];
            pc0[3] += pa[8] * pb[19];    pc1[3] += pa[9] * pb[19];

            pc0[0] += pa[10] * pb[20];    pc1[0] += pa[11] * pb[20];
            pc0[1] += pa[10] * pb[21];    pc1[1] += pa[11] * pb[21];
            pc0[2] += pa[10] * pb[22];    pc1[2] += pa[11] * pb[22];
            pc0[3] += pa[10] * pb[23];    pc1[3] += pa[11] * pb[23];

            pc0[0] += pa[12] * pb[24];    pc1[0] += pa[13] * pb[24];
            pc0[1] += pa[12] * pb[25];    pc1[1] += pa[13] * pb[25];
            pc0[2] += pa[12] * pb[26];    pc1[2] += pa[13] * pb[26];
            pc0[3] += pa[12] * pb[27];    pc1[3] += pa[13] * pb[27];

            pc0[0] += pa[14] * pb[28];    pc1[0] += pa[15] * pb[28];
            pc0[1] += pa[14] * pb[29];    pc1[1] += pa[15] * pb[29];
            pc0[2] += pa[14] * pb[30];    pc1[2] += pa[15] * pb[30];
            pc0[3] += pa[14] * pb[31];    pc1[3] += pa[15] * pb[31];

            pa += 16;
            pb += 32;
        }
        if(j + 3 < k) {
            j += 4;
            pc0[0] += pa[0] * pb[0];    pc1[0] += pa[1] * pb[0];
            pc0[1] += pa[0] * pb[1];    pc1[1] += pa[1] * pb[1];
            pc0[2] += pa[0] * pb[2];    pc1[2] += pa[1] * pb[2];
            pc0[3] += pa[0] * pb[3];    pc1[3] += pa[1] * pb[3];

            pc0[0] += pa[2] * pb[4];    pc1[0] += pa[3] * pb[4];
            pc0[1] += pa[2] * pb[5];    pc1[1] += pa[3] * pb[5];
            pc0[2] += pa[2] * pb[6];    pc1[2] += pa[3] * pb[6];
            pc0[3] += pa[2] * pb[7];    pc1[3] += pa[3] * pb[7];

            pc0[0] += pa[4] * pb[8];    pc1[0] += pa[5] * pb[8];
            pc0[1] += pa[4] * pb[9];    pc1[1] += pa[5] * pb[9];
            pc0[2] += pa[4] * pb[10];    pc1[2] += pa[5] * pb[10];
            pc0[3] += pa[4] * pb[11];    pc1[3] += pa[5] * pb[11];

            pc0[0] += pa[6] * pb[12];    pc1[0] += pa[7] * pb[12];
            pc0[1] += pa[6] * pb[13];    pc1[1] += pa[7] * pb[13];
            pc0[2] += pa[6] * pb[14];    pc1[2] += pa[7] * pb[14];
            pc0[3] += pa[6] * pb[15];    pc1[3] += pa[7] * pb[15];

            pa += 8;
            pb += 16;
        }
        if(j + 1 < k) {
            j += 2;
            pc0[0] += pa[0] * pb[0];    pc1[0] += pa[1] * pb[0];
            pc0[1] += pa[0] * pb[1];    pc1[1] += pa[1] * pb[1];
            pc0[2] += pa[0] * pb[2];    pc1[2] += pa[1] * pb[2];
            pc0[3] += pa[0] * pb[3];    pc1[3] += pa[1] * pb[3];

            pc0[0] += pa[2] * pb[4];    pc1[0] += pa[3] * pb[4];
            pc0[1] += pa[2] * pb[5];    pc1[1] += pa[3] * pb[5];
            pc0[2] += pa[2] * pb[6];    pc1[2] += pa[3] * pb[6];
            pc0[3] += pa[2] * pb[7];    pc1[3] += pa[3] * pb[7];

            pa += 4;
            pb += 8;
        }
        if(j < k) {
            pc0[0] += pa[0] * pb[0];    pc1[0] += pa[1] * pb[0];
            pc0[1] += pa[0] * pb[1];    pc1[1] += pa[1] * pb[1];
            pc0[2] += pa[0] * pb[2];    pc1[2] += pa[1] * pb[2];
            pc0[3] += pa[0] * pb[3];    pc1[3] += pa[1] * pb[3];

            pa += 2;
            pb += 4;
        }
        pc0 += 4;
        pc1 += 4;
    }
    if(n2 > 0) {
        pa = sa;
        pc0[0] = pc0[1] = *bias;
        pc1[0] = pc1[1] = *(bias + 1);
        float *pb0 = pb;
        float *pb1 = pb0 + k;
        int j = 0;
        for(; j + 7 < k; j += 8) {
            pc0[0] += pa[0] * pb0[0];    pc1[0] += pa[1] * pb0[0];
            pc0[1] += pa[0] * pb1[0];    pc1[1] += pa[1] * pb1[0];

            pc0[0] += pa[2] * pb0[1];    pc1[0] += pa[3] * pb0[1];
            pc0[1] += pa[2] * pb1[1];    pc1[1] += pa[3] * pb1[1];

            pc0[0] += pa[4] * pb0[2];    pc1[0] += pa[5] * pb0[2];
            pc0[1] += pa[4] * pb1[2];    pc1[1] += pa[5] * pb1[2];

            pc0[0] += pa[6] * pb0[3];    pc1[0] += pa[7] * pb0[3];
            pc0[1] += pa[6] * pb1[3];    pc1[1] += pa[7] * pb1[3];

            pc0[0] += pa[8] * pb0[4];    pc1[0] += pa[9] * pb0[4];
            pc0[1] += pa[8] * pb1[4];    pc1[1] += pa[9] * pb1[4];

            pc0[0] += pa[10] * pb0[5];    pc1[0] += pa[11] * pb0[5];
            pc0[1] += pa[10] * pb1[5];    pc1[1] += pa[11] * pb1[5];

            pc0[0] += pa[12] * pb0[6];    pc1[0] += pa[13] * pb0[6];
            pc0[1] += pa[12] * pb1[6];    pc1[1] += pa[13] * pb1[6];

            pc0[0] += pa[14] * pb0[7];    pc1[0] += pa[15] * pb0[7];
            pc0[1] += pa[14] * pb1[7];    pc1[1] += pa[15] * pb1[7];

            pa += 16;
            pb0 += 8;
            pb1 += 8;
        }
        if(j + 3 < k) {
            j += 4;
            pc0[0] += pa[0] * pb0[0];    pc1[0] += pa[1] * pb0[0];
            pc0[1] += pa[0] * pb1[0];    pc1[1] += pa[1] * pb1[0];

            pc0[0] += pa[2] * pb0[1];    pc1[0] += pa[3] * pb0[1];
            pc0[1] += pa[2] * pb1[1];    pc1[1] += pa[3] * pb1[1];

            pc0[0] += pa[4] * pb0[2];    pc1[0] += pa[5] * pb0[2];
            pc0[1] += pa[4] * pb1[2];    pc1[1] += pa[5] * pb1[2];

            pc0[0] += pa[6] * pb0[3];    pc1[0] += pa[7] * pb0[3];
            pc0[1] += pa[6] * pb1[3];    pc1[1] += pa[7] * pb1[3];

            pa += 8;
            pb0 += 4;
            pb1 += 4;
        }
        if(j + 1 < k) {
            j += 2;
            pc0[0] += pa[0] * pb0[0];    pc1[0] += pa[1] * pb0[0];
            pc0[1] += pa[0] * pb1[0];    pc1[1] += pa[1] * pb1[0];

            pc0[0] += pa[2] * pb0[1];    pc1[0] += pa[3] * pb0[1];
            pc0[1] += pa[2] * pb1[1];    pc1[1] += pa[3] * pb1[1];

            pa += 4;
            pb0 += 2;
            pb1 += 2;
        }
        if(j < k) {
            pc0[0] += pa[0] * pb0[0];    pc1[0] += pa[1] * pb0[0];
            pc0[1] += pa[0] * pb1[0];    pc1[1] += pa[1] * pb1[0];

            pa += 2;
            pb0 += 1;
            pb1 += 1;
        }
        pc0 += 2;
        pc1 += 2;
        pb += 2 * k;
    }
    if(n1 > 0) {
        pa = sa;
        pc0[0] = *bias;
        pc1[0] = *(bias + 1);
        int j = 0;
        for(; j + 7 < k; j += 8) {
            pc0[0] += pa[0] * pb[0];    pc1[0] += pa[1] * pb[0];

            pc0[0] += pa[2] * pb[1];    pc1[0] += pa[3] * pb[1];

            pc0[0] += pa[4] * pb[2];    pc1[0] += pa[5] * pb[2];

            pc0[0] += pa[6] * pb[3];    pc1[0] += pa[7] * pb[3];

            pc0[0] += pa[8] * pb[4];    pc1[0] += pa[9] * pb[4];

            pc0[0] += pa[10] * pb[5];    pc1[0] += pa[11] * pb[5];

            pc0[0] += pa[12] * pb[6];    pc1[0] += pa[13] * pb[6];

            pc0[0] += pa[14] * pb[7];    pc1[0] += pa[15] * pb[7];

            pa += 16;
            pb += 8;
        }
        if(j + 3 < k) {
            j += 4;
            pc0[0] += pa[0] * pb[0];    pc1[0] += pa[1] * pb[0];

            pc0[0] += pa[2] * pb[1];    pc1[0] += pa[3] * pb[1];

            pc0[0] += pa[4] * pb[2];    pc1[0] += pa[5] * pb[2];

            pc0[0] += pa[6] * pb[3];    pc1[0] += pa[7] * pb[3];

            pa += 8;
            pb += 4;
        }
        if(j + 1 < k) {
            j += 2;
            pc0[0] += pa[0] * pb[0];    pc1[0] += pa[1] * pb[0];

            pc0[0] += pa[2] * pb[1];    pc1[0] += pa[3] * pb[1];

            pa += 4;
            pb += 2;
        }
        if(j < k) {
            pc0[0] += pa[0] * pb[0];    pc1[0] += pa[1] * pb[0];

            pa += 2;
            pb += 1;
        }
        pc0 += 1;
        pc1 += 1;
    }
#endif // __riscv_vector
}

static void kernel_m4_f32(float* dst, float* sa, float* sb, int m, int k, int n, int ldc, float* bias)
{
    float *pa = sa;
    float *pb = sb;
    float *pc0 = dst;
    float *pc1 = pc0 + ldc;
    float *pc2 = pc1 + ldc;
    float *pc3 = pc2 + ldc;
    DECOMPOSE_K
    DECOMPOSE_N

#if __riscv_vector == 128
    if(n4 > 0) {
        asm volatile(
            "vsetvli        zero, zero, e32, m1\n\t"
            "flw            ft0, (%11)\n\t"
            "flw            ft1, 4(%11)\n\t"
            "flw            ft2, 8(%11)\n\t"
            "flw            ft3, 12(%11)\n\t"
        "1:\n\t"                        // n4
            // start kernel_m4n4
            "vfmv.v.f       v24, ft0\n\t"    // v24[0..3] = *bias
            "vfmv.v.f       v25, ft1\n\t"    // v25[0..3] = *(bias + 1)
            "vfmv.v.f       v26, ft2\n\t"    // v26[0..3] = *(bias + 2)
            "vfmv.v.f       v27, ft3\n\t"    // v27[0..3] = *(bias + 3)
            // "vlw.v          v24, (%11)\n\t"     // v24[0..3] = bias[0..3]
            // "vlw.v          v25, (%11)\n\t"     // v25[0..3] = bias[0..3]
            // "vlw.v          v26, (%11)\n\t"     // v26[0..3] = bias[0..3]
            // "vlw.v          v27, (%11)\n\t"     // v27[0..3] = bias[0..3]
            // "addi           %11, %11, 16\n\t"   // bias += 4 * 4

            "mv             a1, %0\n\t"         // a1 = pa
            "mv             t0, %6\n\t"         // t0 = k8
            "beqz           t0, 3f\n\t"         // k8 == 0 ?

            "2:\n\t"
                // start subkernel_m4n4k8
                "vlw.v          v0, (a1)\n\t"       // load pa
                "addi           a1, a1, 16\n\t"
                "vlw.v          v1, (%1)\n\t"       // load pb
                "addi           %1, %1, 16\n\t"
                "vrgather.vi    v2, v0, 0\n\t"
                "vrgather.vi    v3, v0, 1\n\t"
                "vrgather.vi    v4, v0, 2\n\t"
                "vrgather.vi    v5, v0, 3\n\t"
                "vfmacc.vv      v24, v1, v2\n\t"
                "vfmacc.vv      v25, v1, v3\n\t"
                "vfmacc.vv      v26, v1, v4\n\t"
                "vfmacc.vv      v27, v1, v5\n\t" // 0

                "vlw.v          v6, (a1)\n\t"
                "addi           a1, a1, 16\n\t"
                "vlw.v          v7, (%1)\n\t"
                "addi           %1, %1, 16\n\t"
                "vrgather.vi    v8, v6, 0\n\t"
                "vrgather.vi    v9, v6, 1\n\t"
                "vrgather.vi    v10, v6, 2\n\t"
                "vrgather.vi    v11, v6, 3\n\t"
                "vfmacc.vv      v24, v7, v8\n\t"
                "vfmacc.vv      v25, v7, v9\n\t"
                "vfmacc.vv      v26, v7, v10\n\t"
                "vfmacc.vv      v27, v7, v11\n\t" // 1

                "vlw.v          v12, (a1)\n\t"
                "addi           a1, a1, 16\n\t"
                "vlw.v          v13, (%1)\n\t"
                "addi           %1, %1, 16\n\t"
                "vrgather.vi    v14, v12, 0\n\t"
                "vrgather.vi    v15, v12, 1\n\t"
                "vrgather.vi    v16, v12, 2\n\t"
                "vrgather.vi    v17, v12, 3\n\t"
                "vfmacc.vv      v24, v13, v14\n\t"
                "vfmacc.vv      v25, v13, v15\n\t"
                "vfmacc.vv      v26, v13, v16\n\t"
                "vfmacc.vv      v27, v13, v17\n\t" // 2

                "vlw.v          v18, (a1)\n\t"
                "addi           a1, a1, 16\n\t"
                "vlw.v          v19, (%1)\n\t"
                "addi           %1, %1, 16\n\t"
                "vrgather.vi    v20, v18, 0\n\t"
                "vrgather.vi    v21, v18, 1\n\t"
                "vrgather.vi    v22, v18, 2\n\t"
                "vrgather.vi    v23, v18, 3\n\t"
                "vfmacc.vv      v24, v19, v20\n\t"
                "vfmacc.vv      v25, v19, v21\n\t"
                "vfmacc.vv      v26, v19, v22\n\t"
                "vfmacc.vv      v27, v19, v23\n\t" // 3

                "vlw.v          v0, (a1)\n\t"
                "addi           a1, a1, 16\n\t"
                "vlw.v          v1, (%1)\n\t"
                "addi           %1, %1, 16\n\t"
                "vrgather.vi    v2, v0, 0\n\t"
                "vrgather.vi    v3, v0, 1\n\t"
                "vrgather.vi    v4, v0, 2\n\t"
                "vrgather.vi    v5, v0, 3\n\t"
                "vfmacc.vv      v24, v1, v2\n\t"
                "vfmacc.vv      v25, v1, v3\n\t"
                "vfmacc.vv      v26, v1, v4\n\t"
                "vfmacc.vv      v27, v1, v5\n\t" // 4

                "vlw.v          v6, (a1)\n\t"
                "addi           a1, a1, 16\n\t"
                "vlw.v          v7, (%1)\n\t"
                "addi           %1, %1, 16\n\t"
                "vrgather.vi    v8, v6, 0\n\t"
                "vrgather.vi    v9, v6, 1\n\t"
                "vrgather.vi    v10, v6, 2\n\t"
                "vrgather.vi    v11, v6, 3\n\t"
                "vfmacc.vv      v24, v7, v8\n\t"
                "vfmacc.vv      v25, v7, v9\n\t"
                "vfmacc.vv      v26, v7, v10\n\t"
                "vfmacc.vv      v27, v7, v11\n\t" // 5

                "vlw.v          v12, (a1)\n\t"
                "addi           a1, a1, 16\n\t"
                "vlw.v          v13, (%1)\n\t"
                "addi           %1, %1, 16\n\t"
                "vrgather.vi    v14, v12, 0\n\t"
                "vrgather.vi    v15, v12, 1\n\t"
                "vrgather.vi    v16, v12, 2\n\t"
                "vrgather.vi    v17, v12, 3\n\t"
                "vfmacc.vv      v24, v13, v14\n\t"
                "vfmacc.vv      v25, v13, v15\n\t"
                "vfmacc.vv      v26, v13, v16\n\t"
                "vfmacc.vv      v27, v13, v17\n\t" // 6

                "vlw.v          v18, (a1)\n\t"
                "addi           a1, a1, 16\n\t"
                "vlw.v          v19, (%1)\n\t"
                "addi           %1, %1, 16\n\t"
                "vrgather.vi    v20, v18, 0\n\t"
                "vrgather.vi    v21, v18, 1\n\t"
                "vrgather.vi    v22, v18, 2\n\t"
                "vrgather.vi    v23, v18, 3\n\t"
                "vfmacc.vv      v24, v19, v20\n\t"
                "vfmacc.vv      v25, v19, v21\n\t"
                "vfmacc.vv      v26, v19, v22\n\t"
                "vfmacc.vv      v27, v19, v23\n\t" // 7

                "addi           t0, t0, -1\n\t"
                "bnez           t0, 2b\n\t"

        "3:\n\t"
            "beqz           %7, 4f\n\t"     // k4 == 0 ?
            // start subkernel_m4n4k4
            "vlw.v          v0, (a1)\n\t"
            "addi           a1, a1, 16\n\t"
            "vlw.v          v1, (%1)\n\t"
            "addi           %1, %1, 16\n\t"
            "vrgather.vi    v2, v0, 0\n\t"
            "vrgather.vi    v3, v0, 1\n\t"
            "vrgather.vi    v4, v0, 2\n\t"
            "vrgather.vi    v5, v0, 3\n\t"
            "vfmacc.vv      v24, v1, v2\n\t"
            "vfmacc.vv      v25, v1, v3\n\t"
            "vfmacc.vv      v26, v1, v4\n\t"
            "vfmacc.vv      v27, v1, v5\n\t" // 0

            "vlw.v          v6, (a1)\n\t"
            "addi           a1, a1, 16\n\t"
            "vlw.v          v7, (%1)\n\t"
            "addi           %1, %1, 16\n\t"
            "vrgather.vi    v8, v6, 0\n\t"
            "vrgather.vi    v9, v6, 1\n\t"
            "vrgather.vi    v10, v6, 2\n\t"
            "vrgather.vi    v11, v6, 3\n\t"
            "vfmacc.vv      v24, v7, v8\n\t"
            "vfmacc.vv      v25, v7, v9\n\t"
            "vfmacc.vv      v26, v7, v10\n\t"
            "vfmacc.vv      v27, v7, v11\n\t" // 1

            "vlw.v          v12, (a1)\n\t"
            "addi           a1, a1, 16\n\t"
            "vlw.v          v13, (%1)\n\t"
            "addi           %1, %1, 16\n\t"
            "vrgather.vi    v14, v12, 0\n\t"
            "vrgather.vi    v15, v12, 1\n\t"
            "vrgather.vi    v16, v12, 2\n\t"
            "vrgather.vi    v17, v12, 3\n\t"
            "vfmacc.vv      v24, v13, v14\n\t"
            "vfmacc.vv      v25, v13, v15\n\t"
            "vfmacc.vv      v26, v13, v16\n\t"
            "vfmacc.vv      v27, v13, v17\n\t" // 2

            "vlw.v          v18, (a1)\n\t"
            "addi           a1, a1, 16\n\t"
            "vlw.v          v19, (%1)\n\t"
            "addi           %1, %1, 16\n\t"
            "vrgather.vi    v20, v18, 0\n\t"
            "vrgather.vi    v21, v18, 1\n\t"
            "vrgather.vi    v22, v18, 2\n\t"
            "vrgather.vi    v23, v18, 3\n\t"
            "vfmacc.vv      v24, v19, v20\n\t"
            "vfmacc.vv      v25, v19, v21\n\t"
            "vfmacc.vv      v26, v19, v22\n\t"
            "vfmacc.vv      v27, v19, v23\n\t" // 3

        "4:\n\t"
            "beqz           %8, 5f\n\t"     // k2 == 0 ?
            // start subkernel_m4n4k2
            "vlw.v          v0, (a1)\n\t"
            "addi           a1, a1, 16\n\t"
            "vlw.v          v1, (%1)\n\t"
            "addi           %1, %1, 16\n\t"
            "vrgather.vi    v2, v0, 0\n\t"
            "vrgather.vi    v3, v0, 1\n\t"
            "vrgather.vi    v4, v0, 2\n\t"
            "vrgather.vi    v5, v0, 3\n\t"
            "vfmacc.vv      v24, v1, v2\n\t"
            "vfmacc.vv      v25, v1, v3\n\t"
            "vfmacc.vv      v26, v1, v4\n\t"
            "vfmacc.vv      v27, v1, v5\n\t" // 0

            "vlw.v          v6, (a1)\n\t"
            "addi           a1, a1, 16\n\t"
            "vlw.v          v7, (%1)\n\t"
            "addi           %1, %1, 16\n\t"
            "vrgather.vi    v8, v6, 0\n\t"
            "vrgather.vi    v9, v6, 1\n\t"
            "vrgather.vi    v10, v6, 2\n\t"
            "vrgather.vi    v11, v6, 3\n\t"
            "vfmacc.vv      v24, v7, v8\n\t"
            "vfmacc.vv      v25, v7, v9\n\t"
            "vfmacc.vv      v26, v7, v10\n\t"
            "vfmacc.vv      v27, v7, v11\n\t" // 1

        "5:\n\t"
            "beqz           %9, 6f\n\t"    // k1 == 0 ?
            // start subkernel_m4n4k1
            "vlw.v          v0, (a1)\n\t"
            "addi           a1, a1, 16\n\t"
            "vlw.v          v1, (%1)\n\t"
            "addi           %1, %1, 16\n\t"
            "vrgather.vi    v2, v0, 0\n\t"
            "vrgather.vi    v3, v0, 1\n\t"
            "vrgather.vi    v4, v0, 2\n\t"
            "vrgather.vi    v5, v0, 3\n\t"
            "vfmacc.vv      v24, v1, v2\n\t"
            "vfmacc.vv      v25, v1, v3\n\t"
            "vfmacc.vv      v26, v1, v4\n\t"
            "vfmacc.vv      v27, v1, v5\n\t" // 0

        "6:\n\t"
            // end kernel_m4n4
            "vsw.v          v24, (%2)\n\t"
            "addi           %2, %2, 16\n\t"
            "vsw.v          v25, (%3)\n\t"
            "addi           %3, %3, 16\n\t"
            "vsw.v          v26, (%4)\n\t"
            "addi           %4, %4, 16\n\t"
            "vsw.v          v27, (%5)\n\t"
            "addi           %5, %5, 16\n\t"

            "addi           %10, %10, -1\n\t"
            "bnez           %10, 1b\n\t"

            :"=r"(pa),   // %0
            "=r"(pb),    // %1
            "=r"(pc0),   // %2
            "=r"(pc1),   // %3
            "=r"(pc2),   // %4
            "=r"(pc3),   // %5
            "=r"(k8),    // %6
            "=r"(k4),    // %7
            "=r"(k2),    // %8
            "=r"(k1),    // %9
            "=r"(n4),    // %10
            "=r"(bias)   // %11
            :"0"(pa),
            "1"(pb),
            "2"(pc0),
            "3"(pc1),
            "4"(pc2),
            "5"(pc3),
            "6"(k8),
            "7"(k4),
            "8"(k2),
            "9"(k1),
            "10"(n4),
            "11"(bias)
            :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
            "a1", "t0", "ft0", "ft1", "ft2", "ft3"
        );
    }
    if(n2 > 0) {
        float *pa = sa;
        float *pb0 = pb;
        float *pb1 = pb0 + k;
        float *pc00 = pc0;
        float *pc11 = pc00 + 1;
        asm volatile(
            "slli           t1, %10, 2\n\t"
            "vsetvli        zero, zero, e32, m1\n\t"
            // "flw            ft8, (%9)\n\t"
            // "flw            ft9, 4(%9)\n\t"
            // "addi           %9, %9, 8\n\t"

            "vlw.v          v24, (%9)\n\t"   // v24[0..3] = bias[0]..bias[3]
            "vlw.v          v25, (%9)\n\t"   // v25[0..3] = bias[0]..bias[3]
            // "vfmv.v.f       v24, ft8\n\t"       // v24[0..3] = bias[0];
            // "vfmv.v.f       v25, ft9\n\t"       // v25[0..3] = bias[1];

            "mv             t0, %5\n\t"         // t0 = k8
            "beqz           t0, 2f\n\t"         // k8 == 0 ?

            "1:\n\t"
                // start subkernel_m4n2k8
                "vlw.v          v0, (%0)\n\t"       // load pa
                "addi           %0, %0, 16\n\t"
                "flw            ft0, (%1)\n\t"      // load pb0
                "addi           %1, %1, 4\n\t"
                "flw            fa0, (%2)\n\t"      // load pb1
                "addi           %2, %2, 4\n\t"
                "vfmv.v.f       v1, ft0\n\t"        // or vfmacc.vf ???
                "vfmv.v.f       v2, fa0\n\t"
                "vfmacc.vv      v24, v0, v1\n\t"
                "vfmacc.vv      v25, v0, v2\n\t"  // 0

                "vlw.v          v3, (%0)\n\t"
                "addi           %0, %0, 16\n\t"
                "flw            ft1, (%1)\n\t"
                "addi           %1, %1, 4\n\t"
                "flw            fa1, (%2)\n\t"
                "addi           %2, %2, 4\n\t"
                "vfmv.v.f       v4, ft1\n\t"
                "vfmv.v.f       v5, fa1\n\t"
                "vfmacc.vv      v24, v3, v4\n\t"
                "vfmacc.vv      v25, v3, v5\n\t"  // 1

                "vlw.v          v6, (%0)\n\t"
                "addi           %0, %0, 16\n\t"
                "flw            ft2, (%1)\n\t"
                "addi           %1, %1, 4\n\t"
                "flw            fa2, (%2)\n\t"
                "addi           %2, %2, 4\n\t"
                "vfmv.v.f       v7, ft2\n\t"
                "vfmv.v.f       v8, fa2\n\t"
                "vfmacc.vv      v24, v6, v7\n\t"
                "vfmacc.vv      v25, v6, v8\n\t"  // 2

                "vlw.v          v9, (%0)\n\t"
                "addi           %0, %0, 16\n\t"
                "flw            ft3, (%1)\n\t"
                "addi           %1, %1, 4\n\t"
                "flw            fa3, (%2)\n\t"
                "addi           %2, %2, 4\n\t"
                "vfmv.v.f       v10, ft3\n\t"
                "vfmv.v.f       v11, fa3\n\t"
                "vfmacc.vv      v24, v9, v10\n\t"
                "vfmacc.vv      v25, v9, v11\n\t"  // 3

                "vlw.v          v12, (%0)\n\t"
                "addi           %0, %0, 16\n\t"
                "flw            ft4, (%1)\n\t"
                "addi           %1, %1, 4\n\t"
                "flw            fa4, (%2)\n\t"
                "addi           %2, %2, 4\n\t"
                "vfmv.v.f       v13, ft4\n\t"
                "vfmv.v.f       v14, fa4\n\t"
                "vfmacc.vv      v24, v12, v13\n\t"
                "vfmacc.vv      v25, v12, v14\n\t"  // 4

                "vlw.v          v15, (%0)\n\t"
                "addi           %0, %0, 16\n\t"
                "flw            ft5, (%1)\n\t"
                "addi           %1, %1, 4\n\t"
                "flw            fa5, (%2)\n\t"
                "addi           %2, %2, 4\n\t"
                "vfmv.v.f       v16, ft5\n\t"
                "vfmv.v.f       v17, fa5\n\t"
                "vfmacc.vv      v24, v15, v16\n\t"
                "vfmacc.vv      v25, v15, v17\n\t"  // 5

                "vlw.v          v18, (%0)\n\t"
                "addi           %0, %0, 16\n\t"
                "flw            ft6, (%1)\n\t"
                "addi           %1, %1, 4\n\t"
                "flw            fa6, (%2)\n\t"
                "addi           %2, %2, 4\n\t"
                "vfmv.v.f       v19, ft6\n\t"
                "vfmv.v.f       v20, fa6\n\t"
                "vfmacc.vv      v24, v18, v19\n\t"
                "vfmacc.vv      v25, v18, v20\n\t"  // 6

                "vlw.v          v21, (%0)\n\t"
                "addi           %0, %0, 16\n\t"
                "flw            ft7, (%1)\n\t"
                "addi           %1, %1, 4\n\t"
                "flw            fa7, (%2)\n\t"
                "addi           %2, %2, 4\n\t"
                "vfmv.v.f       v22, ft7\n\t"
                "vfmv.v.f       v23, fa7\n\t"
                "vfmacc.vv      v24, v21, v22\n\t"
                "vfmacc.vv      v25, v21, v23\n\t"  // 7

                "addi           t0, t0, -1\n\t"
                "bnez           t0, 1b\n\t"

        "2:\n\t"
            "beqz           %6, 3f\n\t"     // k4 == 0 ?
            // start subkernel_m4n2k4
            "vlw.v          v0, (%0)\n\t"
            "addi           %0, %0, 16\n\t"
            "flw            ft0, (%1)\n\t"
            "addi           %1, %1, 4\n\t"
            "flw            fa0, (%2)\n\t"
            "addi           %2, %2, 4\n\t"
            "vfmv.v.f       v1, ft0\n\t"
            "vfmv.v.f       v2, fa0\n\t"
            "vfmacc.vv      v24, v0, v1\n\t"
            "vfmacc.vv      v25, v0, v2\n\t"  // 0

            "vlw.v          v3, (%0)\n\t"
            "addi           %0, %0, 16\n\t"
            "flw            ft1, (%1)\n\t"
            "addi           %1, %1, 4\n\t"
            "flw            fa1, (%2)\n\t"
            "addi           %2, %2, 4\n\t"
            "vfmv.v.f       v4, ft1\n\t"
            "vfmv.v.f       v5, fa1\n\t"
            "vfmacc.vv      v24, v3, v4\n\t"
            "vfmacc.vv      v25, v3, v5\n\t"  // 1

            "vlw.v          v6, (%0)\n\t"
            "addi           %0, %0, 16\n\t"
            "flw            ft2, (%1)\n\t"
            "addi           %1, %1, 4\n\t"
            "flw            fa2, (%2)\n\t"
            "addi           %2, %2, 4\n\t"
            "vfmv.v.f       v7, ft2\n\t"
            "vfmv.v.f       v8, fa2\n\t"
            "vfmacc.vv      v24, v6, v7\n\t"
            "vfmacc.vv      v25, v6, v8\n\t"  // 2

            "vlw.v          v9, (%0)\n\t"
            "addi           %0, %0, 16\n\t"
            "flw            ft3, (%1)\n\t"
            "addi           %1, %1, 4\n\t"
            "flw            fa3, (%2)\n\t"
            "addi           %2, %2, 4\n\t"
            "vfmv.v.f       v10, ft3\n\t"
            "vfmv.v.f       v11, fa3\n\t"
            "vfmacc.vv      v24, v9, v10\n\t"
            "vfmacc.vv      v25, v9, v11\n\t"  // 3

        "3:\n\t"
            "beqz           %7, 4f\n\t"     // k2 == 0 ?
            // start subkernel_m4n2k2
            "vlw.v          v0, (%0)\n\t"
            "addi           %0, %0, 16\n\t"
            "flw            ft0, (%1)\n\t"
            "addi           %1, %1, 4\n\t"
            "flw            fa0, (%2)\n\t"
            "addi           %2, %2, 4\n\t"
            "vfmv.v.f       v1, ft0\n\t"
            "vfmv.v.f       v2, fa0\n\t"
            "vfmacc.vv      v24, v0, v1\n\t"
            "vfmacc.vv      v25, v0, v2\n\t"  // 0

            "vlw.v          v3, (%0)\n\t"
            "addi           %0, %0, 16\n\t"
            "flw            ft1, (%1)\n\t"
            "addi           %1, %1, 4\n\t"
            "flw            fa1, (%2)\n\t"
            "addi           %2, %2, 4\n\t"
            "vfmv.v.f       v4, ft1\n\t"
            "vfmv.v.f       v5, fa1\n\t"
            "vfmacc.vv      v24, v3, v4\n\t"
            "vfmacc.vv      v25, v3, v5\n\t"  // 1

        "4:\n\t"
            "beqz           %8, 5f\n\t"    // k1 == 0 ?
            // start subkernel_m4n2k1
            "vlw.v          v0, (%0)\n\t"
            "addi           %0, %0, 16\n\t"
            "flw            ft0, (%1)\n\t"
            "addi           %1, %1, 4\n\t"
            "flw            fa0, (%2)\n\t"
            "addi           %2, %2, 4\n\t"
            "vfmv.v.f       v1, ft0\n\t"
            "vfmv.v.f       v2, fa0\n\t"
            "vfmacc.vv      v24, v0, v1\n\t"
            "vfmacc.vv      v25, v0, v2\n\t"  // 0

        "5:\n\t"
            "vssw.v v24, (%3), t1\n\t"
            "vssw.v v25, (%4), t1\n\t"

            :"=r"(pa),      // %0
            "=r"(pb0),      // %1
            "=r"(pb1),      // %2
            "=r"(pc00),     // %3
            "=r"(pc11),     // %4
            "=r"(k8),       // %5
            "=r"(k4),       // %6
            "=r"(k2),       // %7
            "=r"(k1),       // %8
            "=r"(bias),     // %9
            "=r"(ldc)       // %10
            :"0"(pa),
            "1"(pb0),
            "2"(pb1),
            "3"(pc00),
            "4"(pc11),
            "5"(k8),
            "6"(k4),
            "7"(k2),
            "8"(k1),
            "9"(bias),
            "10"(ldc)
            :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
            "t0", "t1", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7"
        );
        pb += 2 * k;
        pc0 += 2;
        pc1 += 2;
        pc2 += 2;
        pc3 += 2;
    }
    if(n1 > 0) {
        pa = sa;
        float *pc00 = pc0;
        asm volatile(
            "slli           t1, %8, 2\n\t"          // t1 = ldc * 4
            "vsetvli        zero, zero, e32, m1\n\t"
            // "flw            ft8, 0(%7)\n\t"
            // "vfmv.v.f       v16, ft8\n\t"
            "vlw.v          v16, (%7)\n\t"       // v24[0..3] = bias[0]..bias[3]
            "beqz           %3, 2f\n\t"             // k8 == 0 ?

            "1:\n\t"
                // start subkernel_m4n1k8
                "vlw.v          v0, (%0)\n\t"       // load pa
                "addi           %0, %0, 16\n\t"
                "flw            ft0, (%1)\n\t"      // load pb
                "addi           %1, %1, 4\n\t"
                "vfmv.v.f       v1, ft0\n\t"        // or vfmacc.vf ???
                "vfmacc.vv      v16, v0, v1\n\t"    // 0

                "vlw.v          v2, (%0)\n\t"
                "addi           %0, %0, 16\n\t"
                "flw            ft1, (%1)\n\t"
                "addi           %1, %1, 4\n\t"
                "vfmv.v.f       v3, ft1\n\t"
                "vfmacc.vv      v16, v2, v3\n\t"  // 1

                "vlw.v          v4, (%0)\n\t"
                "addi           %0, %0, 16\n\t"
                "flw            ft2, (%1)\n\t"
                "addi           %1, %1, 4\n\t"
                "vfmv.v.f       v5, ft2\n\t"
                "vfmacc.vv      v16, v4, v5\n\t"  // 2

                "vlw.v          v6, (%0)\n\t"
                "addi           %0, %0, 16\n\t"
                "flw            ft3, (%1)\n\t"
                "addi           %1, %1, 4\n\t"
                "vfmv.v.f       v7, ft3\n\t"
                "vfmacc.vv      v16, v6, v7\n\t"  // 3

                "vlw.v          v8, (%0)\n\t"
                "addi           %0, %0, 16\n\t"
                "flw            ft4, (%1)\n\t"
                "addi           %1, %1, 4\n\t"
                "vfmv.v.f       v9, ft4\n\t"
                "vfmacc.vv      v16, v8, v9\n\t"  // 4

                "vlw.v          v10, (%0)\n\t"
                "addi           %0, %0, 16\n\t"
                "flw            ft5, (%1)\n\t"
                "addi           %1, %1, 4\n\t"
                "vfmv.v.f       v11, ft5\n\t"
                "vfmacc.vv      v16 , v10, v11\n\t"  // 5

                "vlw.v          v12, (%0)\n\t"
                "addi           %0, %0, 16\n\t"
                "flw            ft6, (%1)\n\t"
                "addi           %1, %1, 4\n\t"
                "vfmv.v.f       v13, ft6\n\t"
                "vfmacc.vv      v16 , v12, v13\n\t"  // 6

                "vlw.v          v14, (%0)\n\t"
                "addi           %0, %0, 16\n\t"
                "flw            ft7, (%1)\n\t"
                "addi           %1, %1, 4\n\t"
                "vfmv.v.f       v15, ft7\n\t"
                "vfmacc.vv      v16 , v14, v15\n\t"  // 7

                "addi           %3, %3, -1\n\t"
                "bnez           %3, 1b\n\t"

        "2:\n\t"
            "beqz           %4, 3f\n\t"         // k4 == 0 ?
            // start subkernel_m4n1k4
            "vlw.v          v0, (%0)\n\t"
            "addi           %0, %0, 16\n\t"
            "flw            ft0, (%1)\n\t"
            "addi           %1, %1, 4\n\t"
            "vfmv.v.f       v1, ft0\n\t"
            "vfmacc.vv      v16, v0, v1\n\t"    // 0

            "vlw.v          v2, (%0)\n\t"
            "addi           %0, %0, 16\n\t"
            "flw            ft1, (%1)\n\t"
            "addi           %1, %1, 4\n\t"
            "vfmv.v.f       v3, ft1\n\t"
            "vfmacc.vv      v16, v2, v3\n\t"    // 1

            "vlw.v          v4, (%0)\n\t"
            "addi           %0, %0, 16\n\t"
            "flw            ft2, (%1)\n\t"
            "addi           %1, %1, 4\n\t"
            "vfmv.v.f       v5, ft2\n\t"
            "vfmacc.vv      v16, v4, v5\n\t"    // 2

            "vlw.v          v6, (%0)\n\t"
            "addi           %0, %0, 16\n\t"
            "flw            ft3, (%1)\n\t"
            "addi           %1, %1, 4\n\t"
            "vfmv.v.f       v7, ft3\n\t"
            "vfmacc.vv      v16, v6, v7\n\t"    // 3

        "3:\n\t"
            "beqz           %5, 4f\n\t"         // k2 == 0 ?
            // start subkernel_m4n1k2
            "vlw.v          v0, (%0)\n\t"
            "addi           %0, %0, 16\n\t"
            "flw            ft0, (%1)\n\t"
            "addi           %1, %1, 4\n\t"
            "vfmv.v.f       v1, ft0\n\t"
            "vfmacc.vv      v16, v0, v1\n\t"    // 0

            "vlw.v          v2, (%0)\n\t"
            "addi           %0, %0, 16\n\t"
            "flw            ft1, (%1)\n\t"
            "addi           %1, %1, 4\n\t"
            "vfmv.v.f       v3, ft1\n\t"
            "vfmacc.vv      v16, v2, v3\n\t"    // 1

        "4:\n\t"
            "beqz           %6, 5f\n\t"        // k1 == 0 ?
            // start subkernel_m4n2k1
            "vlw.v          v0, (%0)\n\t"
            "addi           %0, %0, 16\n\t"
            "flw            ft0, (%1)\n\t"
            "addi           %1, %1, 4\n\t"
            "vfmv.v.f       v1, ft0\n\t"
            "vfmacc.vv      v16, v0, v1\n\t"    // 0

        "5:\n\t"
            "vssw.v v16, (%2), t1\n\t"

            :"=r"(pa),      // %0
            "=r"(pb),       // %1
            "=r"(pc00),     // %2
            "=r"(k8),       // %3
            "=r"(k4),       // %4
            "=r"(k2),       // %5
            "=r"(k1),       // %6
            "=r"(bias),     // %7
            "=r"(ldc)       // %8
            :"0"(pa),
            "1"(pb),
            "2"(pc00),
            "3"(k8),
            "4"(k4),
            "5"(k2),
            "6"(k1),
            "7"(bias),
            "8"(ldc)
            :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
            "t0", "t1", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8"
        );
    }
#else
    for(int i = 0; i < n4; i++) {
        pa = sa;
        pc0[0] = pc0[1] = pc0[2] = pc0[3] = *bias;
        pc1[0] = pc1[1] = pc1[2] = pc1[3] = *(bias + 1);
        pc2[0] = pc2[1] = pc2[2] = pc2[3] = *(bias + 2);
        pc3[0] = pc3[1] = pc3[2] = pc3[3] = *(bias + 3);
        int j = 0;
        for(; j + 7 < k; j += 8) {
            pc0[0] += pa[0] * pb[0];      pc1[0] += pa[1] * pb[0];      pc2[0] += pa[2] * pb[0];      pc3[0] += pa[3] * pb[0];
            pc0[1] += pa[0] * pb[1];      pc1[1] += pa[1] * pb[1];      pc2[1] += pa[2] * pb[1];      pc3[1] += pa[3] * pb[1];
            pc0[2] += pa[0] * pb[2];      pc1[2] += pa[1] * pb[2];      pc2[2] += pa[2] * pb[2];      pc3[2] += pa[3] * pb[2];
            pc0[3] += pa[0] * pb[3];      pc1[3] += pa[1] * pb[3];      pc2[3] += pa[2] * pb[3];      pc3[3] += pa[3] * pb[3];

            pc0[0] += pa[4] * pb[4];      pc1[0] += pa[5] * pb[4];      pc2[0] += pa[6] * pb[4];      pc3[0] += pa[7] * pb[4];
            pc0[1] += pa[4] * pb[5];      pc1[1] += pa[5] * pb[5];      pc2[1] += pa[6] * pb[5];      pc3[1] += pa[7] * pb[5];
            pc0[2] += pa[4] * pb[6];      pc1[2] += pa[5] * pb[6];      pc2[2] += pa[6] * pb[6];      pc3[2] += pa[7] * pb[6];
            pc0[3] += pa[4] * pb[7];      pc1[3] += pa[5] * pb[7];      pc2[3] += pa[6] * pb[7];      pc3[3] += pa[7] * pb[7];

            pc0[0] += pa[8] * pb[8];      pc1[0] += pa[9] * pb[8];      pc2[0] += pa[10] * pb[8];     pc3[0] += pa[11] * pb[8];
            pc0[1] += pa[8] * pb[9];      pc1[1] += pa[9] * pb[9];      pc2[1] += pa[10] * pb[9];     pc3[1] += pa[11] * pb[9];
            pc0[2] += pa[8] * pb[10];     pc1[2] += pa[9] * pb[10];     pc2[2] += pa[10] * pb[10];    pc3[2] += pa[11] * pb[10];
            pc0[3] += pa[8] * pb[11];     pc1[3] += pa[9] * pb[11];     pc2[3] += pa[10] * pb[11];    pc3[3] += pa[11] * pb[11];

            pc0[0] += pa[12] * pb[12];    pc1[0] += pa[13] * pb[12];    pc2[0] += pa[14] * pb[12];    pc3[0] += pa[15] * pb[12];
            pc0[1] += pa[12] * pb[13];    pc1[1] += pa[13] * pb[13];    pc2[1] += pa[14] * pb[13];    pc3[1] += pa[15] * pb[13];
            pc0[2] += pa[12] * pb[14];    pc1[2] += pa[13] * pb[14];    pc2[2] += pa[14] * pb[14];    pc3[2] += pa[15] * pb[14];
            pc0[3] += pa[12] * pb[15];    pc1[3] += pa[13] * pb[15];    pc2[3] += pa[14] * pb[15];    pc3[3] += pa[15] * pb[15];

            pc0[0] += pa[16] * pb[16];    pc1[0] += pa[17] * pb[16];    pc2[0] += pa[18] * pb[16];    pc3[0] += pa[19] * pb[16];
            pc0[1] += pa[16] * pb[17];    pc1[1] += pa[17] * pb[17];    pc2[1] += pa[18] * pb[17];    pc3[1] += pa[19] * pb[17];
            pc0[2] += pa[16] * pb[18];    pc1[2] += pa[17] * pb[18];    pc2[2] += pa[18] * pb[18];    pc3[2] += pa[19] * pb[18];
            pc0[3] += pa[16] * pb[19];    pc1[3] += pa[17] * pb[19];    pc2[3] += pa[18] * pb[19];    pc3[3] += pa[19] * pb[19];

            pc0[0] += pa[20] * pb[20];    pc1[0] += pa[21] * pb[20];    pc2[0] += pa[22] * pb[20];    pc3[0] += pa[23] * pb[20];
            pc0[1] += pa[20] * pb[21];    pc1[1] += pa[21] * pb[21];    pc2[1] += pa[22] * pb[21];    pc3[1] += pa[23] * pb[21];
            pc0[2] += pa[20] * pb[22];    pc1[2] += pa[21] * pb[22];    pc2[2] += pa[22] * pb[22];    pc3[2] += pa[23] * pb[22];
            pc0[3] += pa[20] * pb[23];    pc1[3] += pa[21] * pb[23];    pc2[3] += pa[22] * pb[23];    pc3[3] += pa[23] * pb[23];

            pc0[0] += pa[24] * pb[24];    pc1[0] += pa[25] * pb[24];    pc2[0] += pa[26] * pb[24];    pc3[0] += pa[27] * pb[24];
            pc0[1] += pa[24] * pb[25];    pc1[1] += pa[25] * pb[25];    pc2[1] += pa[26] * pb[25];    pc3[1] += pa[27] * pb[25];
            pc0[2] += pa[24] * pb[26];    pc1[2] += pa[25] * pb[26];    pc2[2] += pa[26] * pb[26];    pc3[2] += pa[27] * pb[26];
            pc0[3] += pa[24] * pb[27];    pc1[3] += pa[25] * pb[27];    pc2[3] += pa[26] * pb[27];    pc3[3] += pa[27] * pb[27];

            pc0[0] += pa[28] * pb[28];    pc1[0] += pa[29] * pb[28];    pc2[0] += pa[30] * pb[28];    pc3[0] += pa[31] * pb[28];
            pc0[1] += pa[28] * pb[29];    pc1[1] += pa[29] * pb[29];    pc2[1] += pa[30] * pb[29];    pc3[1] += pa[31] * pb[29];
            pc0[2] += pa[28] * pb[30];    pc1[2] += pa[29] * pb[30];    pc2[2] += pa[30] * pb[30];    pc3[2] += pa[31] * pb[30];
            pc0[3] += pa[28] * pb[31];    pc1[3] += pa[29] * pb[31];    pc2[3] += pa[30] * pb[31];    pc3[3] += pa[31] * pb[31];

            pa += 32;
            pb += 32;
        }
        if(j + 3 < k) {
            j += 4;
            pc0[0] += pa[0] * pb[0];      pc1[0] += pa[1] * pb[0];      pc2[0] += pa[2] * pb[0];      pc3[0] += pa[3] * pb[0];
            pc0[1] += pa[0] * pb[1];      pc1[1] += pa[1] * pb[1];      pc2[1] += pa[2] * pb[1];      pc3[1] += pa[3] * pb[1];
            pc0[2] += pa[0] * pb[2];      pc1[2] += pa[1] * pb[2];      pc2[2] += pa[2] * pb[2];      pc3[2] += pa[3] * pb[2];
            pc0[3] += pa[0] * pb[3];      pc1[3] += pa[1] * pb[3];      pc2[3] += pa[2] * pb[3];      pc3[3] += pa[3] * pb[3];

            pc0[0] += pa[4] * pb[4];      pc1[0] += pa[5] * pb[4];      pc2[0] += pa[6] * pb[4];      pc3[0] += pa[7] * pb[4];
            pc0[1] += pa[4] * pb[5];      pc1[1] += pa[5] * pb[5];      pc2[1] += pa[6] * pb[5];      pc3[1] += pa[7] * pb[5];
            pc0[2] += pa[4] * pb[6];      pc1[2] += pa[5] * pb[6];      pc2[2] += pa[6] * pb[6];      pc3[2] += pa[7] * pb[6];
            pc0[3] += pa[4] * pb[7];      pc1[3] += pa[5] * pb[7];      pc2[3] += pa[6] * pb[7];      pc3[3] += pa[7] * pb[7];

            pc0[0] += pa[8] * pb[8];      pc1[0] += pa[9] * pb[8];      pc2[0] += pa[10] * pb[8];     pc3[0] += pa[11] * pb[8];
            pc0[1] += pa[8] * pb[9];      pc1[1] += pa[9] * pb[9];      pc2[1] += pa[10] * pb[9];     pc3[1] += pa[11] * pb[9];
            pc0[2] += pa[8] * pb[10];     pc1[2] += pa[9] * pb[10];     pc2[2] += pa[10] * pb[10];    pc3[2] += pa[11] * pb[10];
            pc0[3] += pa[8] * pb[11];     pc1[3] += pa[9] * pb[11];     pc2[3] += pa[10] * pb[11];    pc3[3] += pa[11] * pb[11];

            pc0[0] += pa[12] * pb[12];    pc1[0] += pa[13] * pb[12];    pc2[0] += pa[14] * pb[12];    pc3[0] += pa[15] * pb[12];
            pc0[1] += pa[12] * pb[13];    pc1[1] += pa[13] * pb[13];    pc2[1] += pa[14] * pb[13];    pc3[1] += pa[15] * pb[13];
            pc0[2] += pa[12] * pb[14];    pc1[2] += pa[13] * pb[14];    pc2[2] += pa[14] * pb[14];    pc3[2] += pa[15] * pb[14];
            pc0[3] += pa[12] * pb[15];    pc1[3] += pa[13] * pb[15];    pc2[3] += pa[14] * pb[15];    pc3[3] += pa[15] * pb[15];

            pa += 16;
            pb += 16;
        }
        if(j + 1 < k) {
            j += 2;
            pc0[0] += pa[0] * pb[0];      pc1[0] += pa[1] * pb[0];      pc2[0] += pa[2] * pb[0];      pc3[0] += pa[3] * pb[0];
            pc0[1] += pa[0] * pb[1];      pc1[1] += pa[1] * pb[1];      pc2[1] += pa[2] * pb[1];      pc3[1] += pa[3] * pb[1];
            pc0[2] += pa[0] * pb[2];      pc1[2] += pa[1] * pb[2];      pc2[2] += pa[2] * pb[2];      pc3[2] += pa[3] * pb[2];
            pc0[3] += pa[0] * pb[3];      pc1[3] += pa[1] * pb[3];      pc2[3] += pa[2] * pb[3];      pc3[3] += pa[3] * pb[3];

            pc0[0] += pa[4] * pb[4];      pc1[0] += pa[5] * pb[4];      pc2[0] += pa[6] * pb[4];      pc3[0] += pa[7] * pb[4];
            pc0[1] += pa[4] * pb[5];      pc1[1] += pa[5] * pb[5];      pc2[1] += pa[6] * pb[5];      pc3[1] += pa[7] * pb[5];
            pc0[2] += pa[4] * pb[6];      pc1[2] += pa[5] * pb[6];      pc2[2] += pa[6] * pb[6];      pc3[2] += pa[7] * pb[6];
            pc0[3] += pa[4] * pb[7];      pc1[3] += pa[5] * pb[7];      pc2[3] += pa[6] * pb[7];      pc3[3] += pa[7] * pb[7];

            pa += 8;
            pb += 8;
        }
        if(j < k) {
            pc0[0] += pa[0] * pb[0];      pc1[0] += pa[1] * pb[0];      pc2[0] += pa[2] * pb[0];      pc3[0] += pa[3] * pb[0];
            pc0[1] += pa[0] * pb[1];      pc1[1] += pa[1] * pb[1];      pc2[1] += pa[2] * pb[1];      pc3[1] += pa[3] * pb[1];
            pc0[2] += pa[0] * pb[2];      pc1[2] += pa[1] * pb[2];      pc2[2] += pa[2] * pb[2];      pc3[2] += pa[3] * pb[2];
            pc0[3] += pa[0] * pb[3];      pc1[3] += pa[1] * pb[3];      pc2[3] += pa[2] * pb[3];      pc3[3] += pa[3] * pb[3];

            pa += 4;
            pb += 4;
        }
        pc0 += 4;
        pc1 += 4;
        pc2 += 4;
        pc3 += 4;
    }
    if(n2 > 0) {
        pa = sa;
        pc0[0] = pc0[1] = *bias;
        pc1[0] = pc1[1] = *(bias + 1);
        pc2[0] = pc2[1] = *(bias + 2);
        pc3[0] = pc3[1] = *(bias + 3);
        float *pb0 = pb;
        float *pb1 = pb0 + k;
        int j = 0;
        for(; j + 7 < k; j += 8) {
            pc0[0] += pa[0] * pb0[0];      pc1[0] += pa[1] * pb0[0];      pc2[0] += pa[2] * pb0[0];      pc3[0] += pa[3] * pb0[0];
            pc0[1] += pa[0] * pb1[0];      pc1[1] += pa[1] * pb1[0];      pc2[1] += pa[2] * pb1[0];      pc3[1] += pa[3] * pb1[0];

            pc0[0] += pa[4] * pb0[1];      pc1[0] += pa[5] * pb0[1];      pc2[0] += pa[6] * pb0[1];      pc3[0] += pa[7] * pb0[1];
            pc0[1] += pa[4] * pb1[1];      pc1[1] += pa[5] * pb1[1];      pc2[1] += pa[6] * pb1[1];      pc3[1] += pa[7] * pb1[1];

            pc0[0] += pa[8] * pb0[2];      pc1[0] += pa[9] * pb0[2];      pc2[0] += pa[10] * pb0[2];     pc3[0] += pa[11] * pb0[2];
            pc0[1] += pa[8] * pb1[2];      pc1[1] += pa[9] * pb1[2];      pc2[1] += pa[10] * pb1[2];     pc3[1] += pa[11] * pb1[2];

            pc0[0] += pa[12] * pb0[3];     pc1[0] += pa[13] * pb0[3];     pc2[0] += pa[14] * pb0[3];     pc3[0] += pa[15] * pb0[3];
            pc0[1] += pa[12] * pb1[3];     pc1[1] += pa[13] * pb1[3];     pc2[1] += pa[14] * pb1[3];     pc3[1] += pa[15] * pb1[3];

            pc0[0] += pa[16] * pb0[4];     pc1[0] += pa[17] * pb0[4];     pc2[0] += pa[18] * pb0[4];     pc3[0] += pa[19] * pb0[4];
            pc0[1] += pa[16] * pb1[4];     pc1[1] += pa[17] * pb1[4];     pc2[1] += pa[18] * pb1[4];     pc3[1] += pa[19] * pb1[4];

            pc0[0] += pa[20] * pb0[5];     pc1[0] += pa[21] * pb0[5];     pc2[0] += pa[22] * pb0[5];    pc3[0] += pa[23] * pb0[5];
            pc0[1] += pa[20] * pb1[5];     pc1[1] += pa[21] * pb1[5];     pc2[1] += pa[22] * pb1[5];    pc3[1] += pa[23] * pb1[5];

            pc0[0] += pa[24] * pb0[6];     pc1[0] += pa[25] * pb0[6];     pc2[0] += pa[26] * pb0[6];    pc3[0] += pa[27] * pb0[6];
            pc0[1] += pa[24] * pb1[6];     pc1[1] += pa[25] * pb1[6];     pc2[1] += pa[26] * pb1[6];    pc3[1] += pa[27] * pb1[6];

            pc0[0] += pa[28] * pb0[7];     pc1[0] += pa[29] * pb0[7];     pc2[0] += pa[30] * pb0[7];    pc3[0] += pa[31] * pb0[7];
            pc0[1] += pa[28] * pb1[7];     pc1[1] += pa[29] * pb1[7];     pc2[1] += pa[30] * pb1[7];    pc3[1] += pa[31] * pb1[7];

            pa += 32;
            pb0 += 8;
            pb1 += 8;
        }
        if(j + 3 < k) {
            j += 4;
            pc0[0] += pa[0] * pb0[0];      pc1[0] += pa[1] * pb0[0];      pc2[0] += pa[2] * pb0[0];      pc3[0] += pa[3] * pb0[0];
            pc0[1] += pa[0] * pb1[0];      pc1[1] += pa[1] * pb1[0];      pc2[1] += pa[2] * pb1[0];      pc3[1] += pa[3] * pb1[0];

            pc0[0] += pa[4] * pb0[1];      pc1[0] += pa[5] * pb0[1];      pc2[0] += pa[6] * pb0[1];      pc3[0] += pa[7] * pb0[1];
            pc0[1] += pa[4] * pb1[1];      pc1[1] += pa[5] * pb1[1];      pc2[1] += pa[6] * pb1[1];      pc3[1] += pa[7] * pb1[1];

            pc0[0] += pa[8] * pb0[2];      pc1[0] += pa[9] * pb0[2];      pc2[0] += pa[10] * pb0[2];     pc3[0] += pa[11] * pb0[2];
            pc0[1] += pa[8] * pb1[2];      pc1[1] += pa[9] * pb1[2];      pc2[1] += pa[10] * pb1[2];     pc3[1] += pa[11] * pb1[2];

            pc0[0] += pa[12] * pb0[3];     pc1[0] += pa[13] * pb0[3];     pc2[0] += pa[14] * pb0[3];     pc3[0] += pa[15] * pb0[3];
            pc0[1] += pa[12] * pb1[3];     pc1[1] += pa[13] * pb1[3];     pc2[1] += pa[14] * pb1[3];     pc3[1] += pa[15] * pb1[3];

            pa += 16;
            pb0 += 4;
            pb1 += 4;
        }
        if(j + 1 < k) {
            j += 2;
            pc0[0] += pa[0] * pb0[0];      pc1[0] += pa[1] * pb0[0];      pc2[0] += pa[2] * pb0[0];      pc3[0] += pa[3] * pb0[0];
            pc0[1] += pa[0] * pb1[0];      pc1[1] += pa[1] * pb1[0];      pc2[1] += pa[2] * pb1[0];      pc3[1] += pa[3] * pb1[0];

            pc0[0] += pa[4] * pb0[1];      pc1[0] += pa[5] * pb0[1];      pc2[0] += pa[6] * pb0[1];      pc3[0] += pa[7] * pb0[1];
            pc0[1] += pa[4] * pb1[1];      pc1[1] += pa[5] * pb1[1];      pc2[1] += pa[6] * pb1[1];      pc3[1] += pa[7] * pb1[1];

            pa += 8;
            pb0 += 2;
            pb1 += 2;
        }
        if(j < k) {
            pc0[0] += pa[0] * pb0[0];      pc1[0] += pa[1] * pb0[0];      pc2[0] += pa[2] * pb0[0];      pc3[0] += pa[3] * pb0[0];
            pc0[1] += pa[0] * pb1[0];      pc1[1] += pa[1] * pb1[0];      pc2[1] += pa[2] * pb1[0];      pc3[1] += pa[3] * pb1[0];

            pa += 4;
            pb0 += 1;
            pb1 += 1;
        }
        pc0 += 2;
        pc1 += 2;
        pc2 += 2;
        pc3 += 2;
        pb += 2 * k;
    }
    if(n1 > 0) {
        pa = sa;
        pc0[0] = *bias;
        pc1[0] = *(bias + 1);
        pc2[0] = *(bias + 2);
        pc3[0] = *(bias + 3);
        int j = 0;
        for(; j + 7 < k; j += 8) {
            pc0[0] += pa[0] * pb[0];      pc1[0] += pa[1] * pb[0];      pc2[0] += pa[2] * pb[0];      pc3[0] += pa[3] * pb[0];

            pc0[0] += pa[4] * pb[1];      pc1[0] += pa[5] * pb[1];      pc2[0] += pa[6] * pb[1];      pc3[0] += pa[7] * pb[1];

            pc0[0] += pa[8] * pb[2];      pc1[0] += pa[9] * pb[2];      pc2[0] += pa[10] * pb[2];     pc3[0] += pa[11] * pb[2];

            pc0[0] += pa[12] * pb[3];     pc1[0] += pa[13] * pb[3];     pc2[0] += pa[14] * pb[3];     pc3[0] += pa[15] * pb[3];

            pc0[0] += pa[16] * pb[4];     pc1[0] += pa[17] * pb[4];     pc2[0] += pa[18] * pb[4];     pc3[0] += pa[19] * pb[4];

            pc0[0] += pa[20] * pb[5];     pc1[0] += pa[21] * pb[5];     pc2[0] += pa[22] * pb[5];     pc3[0] += pa[23] * pb[5];

            pc0[0] += pa[24] * pb[6];     pc1[0] += pa[25] * pb[6];     pc2[0] += pa[26] * pb[6];     pc3[0] += pa[27] * pb[6];

            pc0[0] += pa[28] * pb[7];     pc1[0] += pa[29] * pb[7];     pc2[0] += pa[30] * pb[7];     pc3[0] += pa[31] * pb[7];

            pa += 32;
            pb += 8;

        }
        if(j + 3 < k) {
            j += 4;
            pc0[0] += pa[0] * pb[0];      pc1[0] += pa[1] * pb[0];      pc2[0] += pa[2] * pb[0];      pc3[0] += pa[3] * pb[0];

            pc0[0] += pa[4] * pb[1];      pc1[0] += pa[5] * pb[1];      pc2[0] += pa[6] * pb[1];      pc3[0] += pa[7] * pb[1];

            pc0[0] += pa[8] * pb[2];      pc1[0] += pa[9] * pb[2];      pc2[0] += pa[10] * pb[2];     pc3[0] += pa[11] * pb[2];

            pc0[0] += pa[12] * pb[3];     pc1[0] += pa[13] * pb[3];     pc2[0] += pa[14] * pb[3];     pc3[0] += pa[15] * pb[3];

            pa += 16;
            pb += 4;
        }
        if(j + 1 < k) {
            j += 2;
            pc0[0] += pa[0] * pb[0];      pc1[0] += pa[1] * pb[0];      pc2[0] += pa[2] * pb[0];      pc3[0] += pa[3] * pb[0];

            pc0[0] += pa[4] * pb[1];      pc1[0] += pa[5] * pb[1];      pc2[0] += pa[6] * pb[1];      pc3[0] += pa[7] * pb[1];

            pa += 8;
            pb += 2;
        }
        if(j < k) {
            pc0[0] += pa[0] * pb[0];      pc1[0] += pa[1] * pb[0];      pc2[0] += pa[2] * pb[0];      pc3[0] += pa[3] * pb[0];

            pa += 4;
            pb += 1;
        }
        pc0 += 1;
        pc1 += 1;
        pc2 += 1;
        pc3 += 1;
    }
#endif // __riscv_vector
}

static void sgemm_kernel_f32(float* dst, const float* sa, const float* sb, int m, int k, int n, int ldc, float* bias)
{
    float* pa = (float *)sa;
    float* pb = (float *)sb;
    float* pc = dst;
    const int mm = (m >> 2) << 2;

    for (int i = 0; i < mm; i += 4) {
        kernel_m4_f32(pc + i * ldc, pa + i * k, pb, m, k, n, ldc, bias + i);
    }

    pa += mm * k;
    pc += mm * ldc;
    bias += mm;

    switch (m - mm) {
        case 3:
            kernel_m2_f32(pc, pa, pb, m, k, n, ldc, bias);
            pc += 2 * ldc;
            pa += 2 * k;
            bias += 2;
            kernel_m1_f32(pc, pa, pb, m, k, n, ldc, bias);
            break;
        case 2:
            kernel_m2_f32(pc, pa, pb, m, k, n, ldc, bias);
            break;
        case 1:
            kernel_m1_f32(pc, pa, pb, m, k, n, ldc, bias);
            break;
        case 0:
            break;
        default:
            break;
    }
}

#endif
