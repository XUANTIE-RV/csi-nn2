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

#include "rvm/rvm.h"

/*************************************************************
 * mrows = rlenb / 4
 * m2rows = mrows * 2
 * mcols = rlenb / sizeof(__fp16)
 * msize_m: m2rows, mrows, m_tail
 * msize_n: m2rows, mrows, n_tail
 * msize_k: mcols, k_tail
 *
 * dst - output: [M, N]
 * sa - input:   [M, K]
 * sb - weights: [N/msize_n, K/msize_k, msize_n, msize_k]
 * bias:         [N]
 ************************************************************/
void shl_rvm_gemm_a0b1_fp16(__fp16 *dst, __fp16 *sa, __fp16 *sb, __fp16 *bias, int M, int K, int N)
{
    int mrows = mread_csr(RVM_XRLENB) / 4;
    int m2rows = mrows * 2;
    int mcols = m2rows;

    int flag_bias = 1;
    if (bias == NULL) {
        flag_bias = 0;
        bias = (__fp16 *)shl_mem_alloc(N * sizeof(__fp16));
    }

    int stride_a = K * sizeof(__fp16);
    int stride_c = N * sizeof(__fp16);

    int i = 0;
    // m = m2rows
    uint8_t msize_m = mrows;
    for (; i + m2rows - 1 < M; i += m2rows) {
        __fp16 *sa_ptr = sa + i * K;

        int j = 0;
        // n = m2rows
        uint8_t msize_n = m2rows;
        mcfgn(msize_n);
        for (; j + m2rows - 1 < N; j += m2rows) {
            __fp16 *a0_ptr = sa_ptr;
            __fp16 *a1_ptr = a0_ptr + mrows * K;
            __fp16 *b_ptr = sb + j * K;
            __fp16 *c0_ptr = dst + i * N + j;
            __fp16 *c1_ptr = c0_ptr + mrows * N;

            mcfgm(1);
            mcfgk(msize_n * sizeof(__fp16));
            mfloat16_t m0 = mld_f16(bias + j, stride_c);
            mcfgm(msize_m);
            mfloat16_t m4 = mmov_mf16v(m0, 0);
            mfloat16_t m5 = mmov_mf16v(m0, 0);

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(__fp16));
                mfloat16_t m0 = mld_f16(a0_ptr, stride_a);  // a00
                mfloat16_t m1 = mld_f16(a1_ptr, stride_a);  // a10
                a0_ptr += msize_k;
                a1_ptr += msize_k;

                // mcfgm(mrows);  // msizem == mrows
                mfloat16_t m2_0 = msld_f16(b_ptr, msize_k * sizeof(__fp16));  // b00
                mfloat16_t m2_1 =
                    msld_f16(b_ptr + mrows * msize_k, msize_k * sizeof(__fp16));  // b00 (half)
                mfloat16x2_t m2 = mmov_mf16x2(m2_0, m2_1);
                b_ptr += msize_n * msize_k;

                // mcfgm(msize_m);  // msizem == mrows
                m4 = fmmacc_mf16x2_mf16(m4, m2, m0);  // c00 += a00 * b00
                m5 = fmmacc_mf16x2_mf16(m5, m2, m1);  // c10 += a10 * b00
                c += msize_k;
            }

            mcfgk(msize_n * sizeof(__fp16));
            msst_f16_mf16(c0_ptr, stride_c, m4);
            msst_f16_mf16(c1_ptr, stride_c, m5);
        }

        // n = mrows/n_tail
        while (j < N) {
            uint8_t msize_n = (N - j >= mrows) ? mrows : (N - j);
            mcfgn(msize_n);
            __fp16 *a0_ptr = sa_ptr;
            __fp16 *a1_ptr = a0_ptr + mrows * K;
            __fp16 *b_ptr = sb + j * K;
            __fp16 *c0_ptr = dst + i * N + j;
            __fp16 *c1_ptr = c0_ptr + mrows * N;

            mcfgm(1);
            mcfgk(msize_n * sizeof(__fp16));
            mfloat16_t m0 = mld_f16(bias + j, stride_c);
            mcfgm(msize_m);
            mfloat16_t m4 = mmov_mf16v(m0, 0);
            mfloat16_t m5 = mmov_mf16v(m0, 0);

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(__fp16));
                mfloat16_t m0 = mld_f16(a0_ptr, stride_a);  // a00
                mfloat16_t m1 = mld_f16(a1_ptr, stride_a);  // a10
                a0_ptr += msize_k;
                a1_ptr += msize_k;

                mcfgm(msize_n);
                mfloat16_t m2_0 = msld_f16(b_ptr, msize_k * sizeof(__fp16));  // b00
                mfloat16x2_t m2 = mmov_mf16x2(m2_0, mundefined_mf16());
                b_ptr += msize_n * msize_k;

                mcfgm(msize_m);
                m4 = fmmacc_mf16x2_mf16(m4, m2, m0);  // c00 += a00 * b00
                m5 = fmmacc_mf16x2_mf16(m5, m2, m1);  // c10 += a10 * b00
                c += msize_k;
            }

            mcfgk(msize_n * sizeof(__fp16));
            msst_f16_mf16(c0_ptr, stride_c, m4);
            msst_f16_mf16(c1_ptr, stride_c, m5);
            j += msize_n;
        }
    }

    // m = mrows/m_tail
    while (i < M) {
        uint8_t msize_m = (M - i >= mrows) ? mrows : (M - i);
        __fp16 *sa_ptr = sa + i * K;

        int j = 0;
        // n = m2rows
        uint8_t msize_n = m2rows;
        mcfgn(msize_n);
        for (; j + m2rows - 1 < N; j += m2rows) {
            __fp16 *a0_ptr = sa_ptr;
            __fp16 *b_ptr = sb + j * K;
            __fp16 *c0_ptr = dst + i * N + j;

            mcfgm(1);
            mcfgk(msize_n * sizeof(__fp16));
            mfloat16_t m0 = mld_f16(bias + j, stride_c);
            mcfgm(msize_m);
            mfloat16_t m4 = mmov_mf16v(m0, 0);

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(__fp16));
                mfloat16_t m0 = mld_f16(a0_ptr, stride_a);  // a00
                a0_ptr += msize_k;

                mcfgm(mrows);
                mfloat16_t m2_0 = msld_f16(b_ptr, msize_k * sizeof(__fp16));  // b00
                mfloat16_t m2_1 =
                    msld_f16(b_ptr + mrows * msize_k, msize_k * sizeof(__fp16));  // b00 (half)
                mfloat16x2_t m2 = mmov_mf16x2(m2_0, m2_1);
                b_ptr += msize_n * msize_k;

                mcfgm(msize_m);
                m4 = fmmacc_mf16x2_mf16(m4, m2, m0);  // c00 += a00 * b00
                c += msize_k;
            }

            mcfgk(msize_n * sizeof(__fp16));
            msst_f16_mf16(c0_ptr, stride_c, m4);
        }

        // n = mrows/n_tail
        while (j < N) {
            uint8_t msize_n = (N - j >= mrows) ? mrows : (N - j);
            mcfgn(msize_n);
            __fp16 *a0_ptr = sa_ptr;
            __fp16 *b_ptr = sb + j * K;
            __fp16 *c0_ptr = dst + i * N + j;

            mcfgm(1);
            mcfgk(msize_n * sizeof(__fp16));
            mfloat16_t m0 = mld_f16(bias + j, stride_c);
            mcfgm(msize_m);
            mfloat16_t m4 = mmov_mf16v(m0, 0);

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(__fp16));
                mfloat16_t m0 = mld_f16(a0_ptr, stride_a);  // a00
                a0_ptr += msize_k;

                mcfgm(msize_n);
                mfloat16_t m2_0 = msld_f16(b_ptr, msize_k * sizeof(__fp16));  // b00
                mfloat16x2_t m2 = mmov_mf16x2(m2_0, mundefined_mf16());
                b_ptr += msize_n * msize_k;

                mcfgm(msize_m);
                m4 = fmmacc_mf16x2_mf16(m4, m2, m0);  // c00 += a00 * b00
                c += msize_k;
            }

            mcfgk(msize_n * sizeof(__fp16));
            msst_f16_mf16(c0_ptr, stride_c, m4);
            j += msize_n;
        }
        i += msize_m;
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
