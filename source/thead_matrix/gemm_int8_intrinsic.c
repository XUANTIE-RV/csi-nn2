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

#ifdef MATRIX_PW_I32
/*************************************************************
 * mrows = rlenb / 4
 * m2rows = mrows * 2
 * mcols = rlenb / sizeof(int8_t)
 * msize_m: m2rows, mrows, m_tail
 * msize_n: m2rows, mrows, n_tail
 * msize_k: mcols, k_tail
 *
 * dst - output: [M, N]
 * sa - input:   [M, K]
 * sb - weights: [N/msize_n, K/msize_k, msize_n, msize_k]
 * bias:         [N]
 ************************************************************/
void shl_rvm_gemm_a0b1_int8_pw_i32(int8_t *dst, int8_t *sa, int8_t *sb, int32_t *bias, int M, int K,
                                   int N, int32_t out_zp, int32_t *mult, int32_t *shift)
{
    int mcols = csrr_xrlenb();
    int mrows = mcols / 4;
    int m2rows = mrows * 2;

    int flag_bias = 1;
    if (bias == NULL) {
        flag_bias = 0;
        bias = (int32_t *)shl_mem_alloc(N * sizeof(int32_t));
    }

    int stride_a = K * sizeof(int8_t);
    int stride_c = N * sizeof(int8_t);

    int i = 0;
    // m = m2rows
    uint8_t msize_m = mrows;
    for (; i + m2rows - 1 < M; i += m2rows) {
        int8_t *sa_ptr = sa + i * K;

        int j = 0;
        // n = m2rows
        uint8_t msize_n = mrows;
        mcfgn(msize_n);
        for (; j + m2rows - 1 < N; j += m2rows) {
            int8_t *a0_ptr = sa_ptr;
            int8_t *a1_ptr = a0_ptr + mrows * K;
            int8_t *k_ptr = sb + j * K;
            int8_t *c0_ptr = dst + i * N + j;
            int8_t *c1_ptr = c0_ptr + mrows * N;

            mcfgm(2);
            mcfgk(msize_n * sizeof(int32_t));
            mint32_t m0 = mld_i32(bias + j, msize_n * sizeof(int32_t));
            mcfgm(msize_m);
            mint32_t m4 = mmov_mi32v(m0, 0);
            mint32_t m5 = mmov_mi32v(m0, 1);
            mint32_t m6 = mmov_mi32v(m0, 0);
            mint32_t m7 = mmov_mi32v(m0, 1);

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(int8_t));
                mint8_t m0 = mld_i8(a0_ptr, stride_a);  // a00
                mint8_t m1 = mld_i8(a1_ptr, stride_a);  // a10
                a0_ptr += msize_k;
                a1_ptr += msize_k;

                // mcfgm(mrows);  // msizem == mrows
                mint8_t m2 = msld_i8(k_ptr, msize_k * sizeof(int8_t));                    // b00
                mint8_t m3 = msld_i8(k_ptr + mrows * msize_k, msize_k * sizeof(int8_t));  // b10
                k_ptr += 2 * mrows * msize_k;

                // mcfgm(msize_m);  // msizem == mrows
                m4 = mmaqa_mi8(m4, m2, m0);  // c00 += a00 * b00
                m5 = mmaqa_mi8(m5, m3, m0);  // c01 += a00 * b10
                m6 = mmaqa_mi8(m6, m2, m1);  // c10 += a10 * b00
                m7 = mmaqa_mi8(m7, m3, m1);  // c11 += a10 * b10
                c += msize_k;
            }

            // requantize
            mcfgk(msize_n * sizeof(int32_t));
            mcfgm(2);
            mint32_t m_mult = mld_i32(mult + j, msize_n * sizeof(int32_t));
            muint32_t m_shift = mld_ui32(shift + j, msize_n * sizeof(int32_t));

            mcfgm(msize_m);
            m4 = mmulh_mi32_mi32v(m4, m_mult, 0);
            m5 = mmulh_mi32_mi32v(m5, m_mult, 1);
            m6 = mmulh_mi32_mi32v(m6, m_mult, 0);
            m7 = mmulh_mi32_mi32v(m7, m_mult, 1);

            m4 = msra_mi32_mui32v(m4, m_shift, 0);
            m5 = msra_mi32_mui32v(m5, m_shift, 1);
            m6 = msra_mi32_mui32v(m6, m_shift, 0);
            m7 = msra_mi32_mui32v(m7, m_shift, 1);

            m4 = madd_mi32_i32(m4, out_zp);
            m5 = madd_mi32_i32(m5, out_zp);
            m6 = madd_mi32_i32(m6, out_zp);
            m7 = madd_mi32_i32(m7, out_zp);

            mint8_t res00 = mn4clip_mi32_ui32(m4, 0);
            mint8_t res01 = mn4clip_mi32_ui32(m5, 0);
            mint8_t res10 = mn4clip_mi32_ui32(m6, 0);
            mint8_t res11 = mn4clip_mi32_ui32(m7, 0);

            mcfgk(msize_n * sizeof(int8_t));
            msst_i8_mi8(c0_ptr, stride_c, res00);
            msst_i8_mi8(c0_ptr + mrows, stride_c, res01);
            msst_i8_mi8(c1_ptr, stride_c, res10);
            msst_i8_mi8(c1_ptr + mrows, stride_c, res11);
        }

        // n = mrows/n_tail
        while (j < N) {
            uint8_t msize_n = (N - j >= mrows) ? mrows : (N - j);
            int8_t *a0_ptr = sa_ptr;
            int8_t *a1_ptr = a0_ptr + mrows * K;
            int8_t *k_ptr = sb + j * K;
            int8_t *c0_ptr = dst + i * N + j;
            int8_t *c1_ptr = c0_ptr + mrows * N;

            mcfgm(1);
            mcfgk(msize_n * sizeof(int32_t));
            mint32_t m0 = mld_i32(bias + j, msize_n * sizeof(int32_t));
            mcfgm(msize_m);
            mint32_t m4 = mmov_mi32v(m0, 0);
            mint32_t m6 = mmov_mi32v(m0, 0);

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(int8_t));
                mint8_t m0 = mld_i8(a0_ptr, stride_a);  // a00
                mint8_t m1 = mld_i8(a1_ptr, stride_a);  // a10
                a0_ptr += msize_k;
                a1_ptr += msize_k;

                mcfgm(msize_n);
                mint8_t m2 = msld_i8(k_ptr, msize_k * sizeof(int8_t));  // b00
                k_ptr += msize_n * msize_k;

                mcfgm(msize_m);
                m4 = mmaqa_mi8(m4, m2, m0);  // c00 += a00 * b00
                m6 = mmaqa_mi8(m6, m2, m1);  // c10 += a10 * b00
                c += msize_k;
            }

            // requantize
            mcfgk(msize_n * sizeof(int32_t));
            mcfgm(1);
            mint32_t m_mult = mld_i32(mult + j, msize_n * sizeof(int32_t));
            muint32_t m_shift = mld_ui32(shift + j, msize_n * sizeof(int32_t));

            mcfgm(msize_m);
            m4 = mmulh_mi32_mi32v(m4, m_mult, 0);
            m6 = mmulh_mi32_mi32v(m6, m_mult, 0);

            m4 = msra_mi32_mui32v(m4, m_shift, 0);
            m6 = msra_mi32_mui32v(m6, m_shift, 0);

            m4 = madd_mi32_i32(m4, out_zp);
            m6 = madd_mi32_i32(m6, out_zp);

            mint8_t res00 = mn4clip_mi32_ui32(m4, 0);
            mint8_t res10 = mn4clip_mi32_ui32(m6, 0);

            mcfgk(msize_n * sizeof(int8_t));
            msst_i8_mi8(c0_ptr, stride_c, res00);
            msst_i8_mi8(c1_ptr, stride_c, res10);
            j += msize_n;
        }
    }

    // m = mrows/m_tail
    while (i < M) {
        uint8_t msize_m = (M - i >= mrows) ? mrows : (M - i);
        int8_t *sa_ptr = sa + i * K;

        int j = 0;
        // n = m2rows
        uint8_t msize_n = mrows;
        mcfgn(msize_n);
        for (; j + m2rows - 1 < N; j += m2rows) {
            int8_t *a0_ptr = sa_ptr;
            int8_t *k_ptr = sb + j * K;
            int8_t *c0_ptr = dst + i * N + j;

            mcfgm(2);
            mcfgk(msize_n * sizeof(int32_t));
            mint32_t m0 = mld_i32(bias + j, msize_n * sizeof(int32_t));
            mcfgm(msize_m);
            mint32_t m4 = mmov_mi32v(m0, 0);
            mint32_t m5 = mmov_mi32v(m0, 1);

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(int8_t));
                mint8_t m0 = mld_i8(a0_ptr, stride_a);  // a00
                a0_ptr += msize_k;

                mcfgm(mrows);
                mint8_t m2 = msld_i8(k_ptr, msize_k * sizeof(int8_t));                    // b00
                mint8_t m3 = msld_i8(k_ptr + mrows * msize_k, msize_k * sizeof(int8_t));  // b10
                k_ptr += 2 * mrows * msize_k;

                mcfgm(msize_m);
                m4 = mmaqa_mi8(m4, m2, m0);  // c00 += a00 * b00
                m5 = mmaqa_mi8(m5, m3, m0);  // c01 += a00 * b10
                c += msize_k;
            }

            // requantize
            mcfgk(msize_n * sizeof(int32_t));
            mcfgm(2);
            mint32_t m_mult = mld_i32(mult + j, msize_n * sizeof(int32_t));
            muint32_t m_shift = mld_ui32(shift + j, msize_n * sizeof(int32_t));

            mcfgm(msize_m);
            m4 = mmulh_mi32_mi32v(m4, m_mult, 0);
            m5 = mmulh_mi32_mi32v(m5, m_mult, 1);

            m4 = msra_mi32_mui32v(m4, m_shift, 0);
            m5 = msra_mi32_mui32v(m5, m_shift, 1);

            m4 = madd_mi32_i32(m4, out_zp);
            m5 = madd_mi32_i32(m5, out_zp);

            mint8_t res00 = mn4clip_mi32_ui32(m4, 0);
            mint8_t res01 = mn4clip_mi32_ui32(m5, 0);

            mcfgk(msize_n * sizeof(int8_t));
            msst_i8_mi8(c0_ptr, stride_c, res00);
            msst_i8_mi8(c0_ptr + mrows, stride_c, res01);
        }

        // n = mrows/n_tail
        while (j < N) {
            uint8_t msize_n = (N - j >= mrows) ? mrows : (N - j);
            mcfgn(msize_n);
            int8_t *a0_ptr = sa_ptr;
            int8_t *k_ptr = sb + j * K;
            int8_t *c0_ptr = dst + i * N + j;

            mcfgm(1);
            mcfgk(msize_n * sizeof(int32_t));
            mint32_t bias_m0 = mld_i32(bias + j, msize_n * sizeof(int32_t));
            mcfgm(msize_m);
            mint32_t m4 = mmov_mi32v(bias_m0, 0);

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(int8_t));
                mint8_t m0 = mld_i8(a0_ptr, stride_a);  // a00
                a0_ptr += msize_k;

                mcfgm(msize_n);
                mint8_t m2 = msld_i8(k_ptr, msize_k * sizeof(int8_t));  // b00
                k_ptr += msize_n * msize_k;

                mcfgm(msize_m);
                m4 = mmaqa_mi8(m4, m2, m0);  // c00 += a00 * b00
                c += msize_k;
            }

            // requantize
            mcfgk(msize_n * sizeof(int32_t));
            mcfgm(1);
            mint32_t m_mult = mld_i32(mult + j, msize_n * sizeof(int32_t));
            muint32_t m_shift = mld_ui32(shift + j, msize_n * sizeof(int32_t));
            mcfgm(msize_m);
            m4 = mmulh_mi32_mi32v(m4, m_mult, 0);

            m4 = msra_mi32_mui32v(m4, m_shift, 0);

            m4 = madd_mi32_i32(m4, out_zp);

            mint8_t res00 = mn4clip_mi32_ui32(m4, 0);

            mcfgk(msize_n * sizeof(int8_t));
            msst_i8_mi8(c0_ptr, stride_c, res00);
            j += msize_n;
        }
        i += msize_m;
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
#else

static void requantize_m4(int8_t *dst, int32_t *src, int row, int col, int32_t out_zp,
                          int32_t *mult, int32_t *shift)
{
    for (int i = 0; i < row; i++) {
        int32_t *in_ptr = src + i * col;
        int8_t *out_ptr = dst + i * col;
        int c = 0;
        while (c < col) {
            int vl = vsetvl_e32m4(col - c);
            vint32m4_t _input = vle32_v_i32m4(in_ptr, vl);
            vint32m4_t _mult = vle32_v_i32m4(mult + c, vl);
            vint32m4_t _shift = vle32_v_i32m4(shift + c, vl);
            vint32m4_t _mulh = vmulh_vv_i32m4(_input, _mult, vl);
            _mulh = vssra_vv_i32m4(_mulh, vreinterpret_v_i32m4_u32m4(_shift), vl);
            _mulh = vadd_vx_i32m4(_mulh, out_zp, vl);
            vint16m2_t _tmp1 = vnclip_wx_i16m2(_mulh, 0, vl);
            vint8m1_t _res = vnclip_wx_i8m1(_tmp1, 0, vl);
            vse8_v_i8m1(out_ptr, _res, vl);
            in_ptr += vl;
            out_ptr += vl;
            c += vl;
        }
    }
}

/*************************************************************
 * mrows = rlenb / 4
 * m2rows = mrows * 2
 * mcols = rlenb / sizeof(int8_t)
 * msize_m: m2rows, mrows, m_tail
 * msize_n: m2rows, mrows, n_tail
 * msize_k: mcols, k_tail
 *
 * dst - output: [M, N]
 * sa - input:   [M, K]
 * sb - kernel:  [N/msize_n, K/msize_k, msize_n, msize_k]
 * bias:         [N]
 ************************************************************/
void shl_rvm_gemm_a0b1_int8_to_int32(int8_t *dst, int8_t *sa, int8_t *sb, int32_t *bias, int M,
                                     int K, int N, int32_t out_zp, int32_t *mult, int32_t *shift)
{
    int mcols = csrr_xrlenb();
    int mrows = mcols / 4;
    int m2rows = mrows * 2;

    int flag_bias = 1;
    if (bias == NULL) {
        flag_bias = 0;
        bias = (int32_t *)shl_mem_alloc(N * sizeof(int32_t));
    }

    int stride_a = K * sizeof(int8_t);
    int stride_c = N * sizeof(int32_t);

    int32_t *out_i32 = (int32_t *)shl_mem_alloc(m2rows * N * sizeof(int32_t));

    int i = 0;
    // m = m2rows
    uint8_t msize_m = mrows;
    for (; i + m2rows - 1 < M; i += m2rows) {
        int8_t *sa_ptr = sa + i * K;

        int j = 0;
        // n = m2rows
        uint8_t msize_n = mrows;
        mcfgn(msize_n);
        for (; j + m2rows - 1 < N; j += m2rows) {
            int8_t *a0_ptr = sa_ptr;
            int8_t *a1_ptr = a0_ptr + mrows * K;
            int8_t *k_ptr = sb + j * K;
            int32_t *c0_ptr = out_i32 + j;
            int32_t *c1_ptr = c0_ptr + mrows * N;

            mcfgm(2);
            mcfgk(msize_n * sizeof(int32_t));
            mint32_t m0 = mld_i32(bias + j, msize_n * sizeof(int32_t));
            mcfgm(msize_m);
            mint32_t m4 = mmov_mi32v(m0, 0);
            mint32_t m5 = mmov_mi32v(m0, 1);
            mint32_t m6 = mmov_mi32v(m0, 0);
            mint32_t m7 = mmov_mi32v(m0, 1);

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(int8_t));
                mint8_t m0 = mld_i8(a0_ptr, stride_a);  // a00
                mint8_t m1 = mld_i8(a1_ptr, stride_a);  // a10
                a0_ptr += msize_k;
                a1_ptr += msize_k;

                // mcfgm(mrows);  // msizem == mrows
                mint8_t m2 = msld_i8(k_ptr, msize_k * sizeof(int8_t));                    // b00
                mint8_t m3 = msld_i8(k_ptr + mrows * msize_k, msize_k * sizeof(int8_t));  // b10
                k_ptr += 2 * mrows * msize_k;

                // mcfgm(msize_m);  // msizem == mrows
                m4 = mmaqa_mi8(m4, m2, m0);  // c00 += a00 * b00
                m5 = mmaqa_mi8(m5, m3, m0);  // c01 += a00 * b10
                m6 = mmaqa_mi8(m6, m2, m1);  // c10 += a10 * b00
                m7 = mmaqa_mi8(m7, m3, m1);  // c11 += a10 * b10
                c += msize_k;
            }

            mcfgk(msize_n * sizeof(int32_t));
            msst_i32_mi32(c0_ptr, stride_c, m4);
            msst_i32_mi32(c0_ptr + mrows, stride_c, m5);
            msst_i32_mi32(c1_ptr, stride_c, m6);
            msst_i32_mi32(c1_ptr + mrows, stride_c, m7);
        }

        // n = mrows/n_tail
        while (j < N) {
            uint8_t msize_n = (N - j >= mrows) ? mrows : (N - j);
            int8_t *a0_ptr = sa_ptr;
            int8_t *a1_ptr = a0_ptr + mrows * K;
            int8_t *k_ptr = sb + j * K;
            int32_t *c0_ptr = out_i32 + j;
            int32_t *c1_ptr = c0_ptr + mrows * N;

            mcfgm(1);
            mcfgk(msize_n * sizeof(int32_t));
            mint32_t m0 = mld_i32(bias + j, msize_n * sizeof(int32_t));
            mcfgm(msize_m);
            mint32_t m4 = mmov_mi32v(m0, 0);
            mint32_t m6 = mmov_mi32v(m0, 0);

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(int8_t));
                mint8_t m0 = mld_i8(a0_ptr, stride_a);  // a00
                mint8_t m1 = mld_i8(a1_ptr, stride_a);  // a10
                a0_ptr += msize_k;
                a1_ptr += msize_k;

                mcfgm(msize_n);
                mint8_t m2 = msld_i8(k_ptr, msize_k * sizeof(int8_t));  // b00
                k_ptr += msize_n * msize_k;

                mcfgm(msize_m);
                m4 = mmaqa_mi8(m4, m2, m0);  // c00 += a00 * b00
                m6 = mmaqa_mi8(m6, m2, m1);  // c10 += a10 * b00
                c += msize_k;
            }

            mcfgk(msize_n * sizeof(int32_t));
            msst_i32_mi32(c0_ptr, stride_c, m4);
            msst_i32_mi32(c1_ptr, stride_c, m6);
            j += msize_n;
        }

        // requantize
        requantize_m4(dst + i * N, out_i32, m2rows, N, out_zp, mult, shift);
    }

    // m = mrows/m_tail
    while (i < M) {
        uint8_t msize_m = (M - i >= mrows) ? mrows : (M - i);
        int8_t *sa_ptr = sa + i * K;

        int j = 0;
        // n = m2rows
        uint8_t msize_n = mrows;
        mcfgn(msize_n);
        for (; j + m2rows - 1 < N; j += m2rows) {
            int8_t *a0_ptr = sa_ptr;
            int8_t *k_ptr = sb + j * K;
            int32_t *c0_ptr = out_i32 + j;

            mcfgm(2);
            mcfgk(msize_n * sizeof(int32_t));
            mint32_t m0 = mld_i32(bias + j, msize_n * sizeof(int32_t));
            mcfgm(msize_m);
            mint32_t m4 = mmov_mi32v(m0, 0);
            mint32_t m5 = mmov_mi32v(m0, 1);

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(int8_t));
                mint8_t m0 = mld_i8(a0_ptr, stride_a);  // a00
                a0_ptr += msize_k;

                mcfgm(mrows);
                mint8_t m2 = msld_i8(k_ptr, msize_k * sizeof(int8_t));                    // b00
                mint8_t m3 = msld_i8(k_ptr + mrows * msize_k, msize_k * sizeof(int8_t));  // b10
                k_ptr += 2 * mrows * msize_k;

                mcfgm(msize_m);
                m4 = mmaqa_mi8(m4, m2, m0);  // c00 += a00 * b00
                m5 = mmaqa_mi8(m5, m3, m0);  // c01 += a00 * b10
                c += msize_k;
            }

            mcfgk(msize_n * sizeof(int32_t));
            msst_i32_mi32(c0_ptr, stride_c, m4);
            msst_i32_mi32(c0_ptr + mrows, stride_c, m5);
        }

        // n = mrows/n_tail
        while (j < N) {
            uint8_t msize_n = (N - j >= mrows) ? mrows : (N - j);
            int8_t *a0_ptr = sa_ptr;
            int8_t *k_ptr = sb + j * K;
            int32_t *c0_ptr = out_i32 + j;

            mcfgm(1);
            mcfgk(msize_n * sizeof(int32_t));
            mint32_t m0 = mld_i32(bias + j, msize_n * sizeof(int32_t));
            mcfgm(msize_m);
            mint32_t m4 = mmov_mi32v(m0, 0);

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(int8_t));
                mint8_t m0 = mld_i8(a0_ptr, stride_a);  // a00
                a0_ptr += msize_k;

                mcfgm(msize_n);
                mint8_t m2 = msld_i8(k_ptr, msize_k * sizeof(int8_t));  // b00
                k_ptr += msize_n * msize_k;

                mcfgm(msize_m);
                m4 = mmaqa_mi8(m4, m2, m0);  // c00 += a00 * b00
                c += msize_k;
            }

            mcfgk(msize_n * sizeof(int32_t));
            msst_i32_mi32(c0_ptr, stride_c, m4);
            j += msize_n;
        }

        // requantize
        requantize_m4(dst + i * N, out_i32, msize_m, N, out_zp, mult, shift);
        i += msize_m;
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
    shl_mem_free(out_i32);
}
#endif  // MATRIX_PW_I32
