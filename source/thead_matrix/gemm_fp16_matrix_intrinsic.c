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

#include "rvm/rvm.h"
#if 0
static void gemm_fp16_nhwc_matrix_2rowxn(__fp16 *output, const __fp16 *kernel, const __fp16 *input,
                                         const __fp16 *bias, int mrows, int K, int N)
{
    int mcols = mrows * 2;
    int input_stride = K * sizeof(__fp16);
    int output_stride = N * sizeof(__fp16);

    uint8_t msize_m = mrows;
    uint8_t msize_n = mcols;
    uint16_t msize_k = mcols * sizeof(__fp16);
    mcfgn(msize_n);
    mcfgk(msize_k);

    __fp16 *out0 = (__fp16 *)output;
    __fp16 *out1 = (__fp16 *)out0 + mrows * N;

    __fp16 *k0 = (__fp16 *)kernel;
    __fp16 *b_ptr = (__fp16 *)bias;

    int n = 0;
    for (; n + mcols - 1 < N; n += mcols) {
        __fp16 *in0 = (__fp16 *)input;
        __fp16 *in1 = (__fp16 *)in0 + mrows * K;

        mcfgm(1);
        mfloat16_t m0 = mld_f16(b_ptr, output_stride);
        b_ptr += mcols;

        mcfgm(msize_m);
        mfloat16_t m4 = mmov_mf16v(m0, 0);
        mfloat16_t m5 = mmov_mf16v(m0, 0);

        for (int k = 0; k < K; k += mcols) {
            mfloat16_t m0 = mld_f16(in0, input_stride);
            mfloat16_t m1 = mld_f16(in1, input_stride);
            in0 += mcols;
            in1 += mcols;

            mfloat16x2_t m2 = msld_f16x2(k0, k0 + mrows * mcols, mcols * sizeof(__fp16));
            k0 += 2 * mrows * mcols;

            m4 = fmmacc_mf16x2_mf16(m4, m2, m0);
            m5 = fmmacc_mf16x2_mf16(m5, m2, m1);
        }

        mst_f16_mf16(out0, output_stride, m4);
        mst_f16_mf16(out1, output_stride, m5);
        out0 += mcols;
        out1 += mcols;
    }
    if (n < N) {
        int n_tail = (N - n) * sizeof(__fp16);

        __fp16 *in0 = (__fp16 *)input;
        __fp16 *in1 = (__fp16 *)in0 + mrows * K;

        mcfgm(1);
        mcfgk(n_tail);
        mfloat16_t m0 = mld_f16(b_ptr, output_stride);

        mcfgm(msize_m);
        mfloat16_t m4 = mmov_mf16v(m0, 0);
        mfloat16_t m5 = mmov_mf16v(m0, 0);
        mcfgk(msize_k);

        for (int k = 0; k < K; k += mcols) {
            mfloat16_t m0 = mld_f16(in0, input_stride);
            mfloat16_t m1 = mld_f16(in1, input_stride);
            in0 += mcols;
            in1 += mcols;

            mfloat16x2_t m2 = msld_f16x2(k0, k0 + mrows * mcols, mcols * sizeof(__fp16));
            k0 += 2 * mrows * mcols;

            m4 = fmmacc_mf16x2_mf16(m4, m2, m0);
            m5 = fmmacc_mf16x2_mf16(m5, m2, m1);
        }
        mcfgk(n_tail);
        mst_f16_mf16(out0, output_stride, m4);
        mst_f16_mf16(out1, output_stride, m5);
    }
}

static void gemm_fp16_nhwc_matrix_rowxn(__fp16 *output, const __fp16 *kernel, const __fp16 *input,
                                        const __fp16 *bias, int mrows, int K, int N)
{
    int mcols = mrows * 2;
    int input_stride = K * sizeof(__fp16);
    int output_stride = N * sizeof(__fp16);

    uint8_t msize_m = mrows;
    uint8_t msize_n = mcols;
    uint16_t msize_k = mcols * sizeof(__fp16);
    mcfgn(msize_n);
    mcfgk(msize_k);

    __fp16 *out0 = (__fp16 *)output;
    __fp16 *k0 = (__fp16 *)kernel;
    __fp16 *b_ptr = (__fp16 *)bias;

    int n = 0;
    for (; n + mcols - 1 < N; n += mcols) {
        __fp16 *in0 = (__fp16 *)input;

        mcfgm(1);
        mfloat16_t m0 = mld_f16(b_ptr, output_stride);
        b_ptr += mcols;

        mcfgm(msize_m);
        mfloat16_t m4 = mmov_mf16v(m0, 0);

        for (int k = 0; k < K; k += mcols) {
            mfloat16_t m0 = mld_f16(in0, input_stride);
            in0 += mcols;

            mfloat16x2_t m2 = msld_f16x2(k0, k0 + mrows * mcols, mcols * sizeof(__fp16));
            k0 += 2 * mrows * mcols;

            m4 = fmmacc_mf16x2_mf16(m4, m2, m0);
        }

        mst_f16_mf16(out0, output_stride, m4);
        out0 += mcols;
    }
    if (n < N) {
        int n_tail = (N - n) * sizeof(__fp16);

        __fp16 *in0 = (__fp16 *)input;

        mcfgm(1);
        mcfgk(n_tail);
        mfloat16_t m0 = mld_f16(b_ptr, output_stride);

        mcfgm(msize_m);
        mfloat16_t m4 = mmov_mf16v(m0, 0);
        mcfgk(msize_k);

        for (int k = 0; k < K; k += mcols) {
            mfloat16_t m0 = mld_f16(in0, input_stride);
            in0 += mcols;

            mfloat16x2_t m2 = msld_f16x2(k0, k0 + mrows * mcols, mcols * sizeof(__fp16));
            k0 += 2 * mrows * mcols;

            m4 = fmmacc_mf16x2_mf16(m4, m2, m0);
        }
        mcfgk(n_tail);
        mst_f16_mf16(out0, output_stride, m4);
    }
}

static void gemm_fp16_nhwc_matrix_row_tailxn(__fp16 *output, const __fp16 *kernel,
                                             const __fp16 *input, const __fp16 *bias, int row,
                                             int K, int N)
{
    int mrows = mread_csr(RVM_XRLENB) / 4;
    int mcols = mrows * 2;
    int input_stride = K * sizeof(__fp16);
    int output_stride = N * sizeof(__fp16);

    uint8_t msize_m = row;
    uint8_t msize_n = mcols;
    uint16_t msize_k = mcols * sizeof(__fp16);
    mcfgn(msize_n);
    mcfgk(msize_k);

    __fp16 *out0 = (__fp16 *)output;
    __fp16 *k0 = (__fp16 *)kernel;
    __fp16 *b_ptr = (__fp16 *)bias;

    int n = 0;
    for (; n + mcols - 1 < N; n += mcols) {
        __fp16 *in0 = (__fp16 *)input;

        mcfgm(1);
        mfloat16_t m0 = mld_f16(b_ptr, output_stride);
        b_ptr += mcols;

        mcfgm(msize_m);
        mfloat16_t m4 = mmov_mf16v(m0, 0);

        for (int k = 0; k < K; k += mcols) {
            mcfgm(mrows);
            mfloat16x2_t m2 = msld_f16x2(k0, k0 + mrows * mcols, mcols * sizeof(__fp16));
            k0 += 2 * mrows * mcols;

            mcfgm(msize_m);
            mfloat16_t m0 = mld_f16(in0, input_stride);
            in0 += mcols;

            m4 = fmmacc_mf16x2_mf16(m4, m2, m0);
        }
        mst_f16_mf16(out0, output_stride, m4);
        out0 += mcols;
    }
    if (n < N) {
        int n_tail = (N - n) * sizeof(__fp16);
        __fp16 *in0 = (__fp16 *)input;

        mcfgm(1);
        mcfgk(n_tail);
        mfloat16_t m0 = mld_f16(b_ptr, output_stride);
        b_ptr += mcols;

        mcfgm(msize_m);
        mfloat16_t m4 = mmov_mf16v(m0, 0);
        mcfgk(msize_k);

        for (int k = 0; k < K; k += mcols) {
            mcfgm(mrows);
            mfloat16x2_t m2 = msld_f16x2(k0, k0 + mrows * mcols, mcols * sizeof(__fp16));
            k0 += 2 * mrows * mcols;

            mcfgm(msize_m);
            mfloat16_t m0 = mld_f16(in0, input_stride);
            in0 += mcols;

            m4 = fmmacc_mf16x2_mf16(m4, m2, m0);
        }

        mcfgk(n_tail);
        mst_f16_mf16(out0, output_stride, m4);
        out0 += mcols;
    }
}

void shl_rvm_nhwc_gemm_fp16_intrinsic(__fp16 *dst, const __fp16 *sa, const __fp16 *sb,
                                      const __fp16 *bias, int m, int k, int n)
{
    int mrows = mread_csr(RVM_XRLENB) / 4;
    int m2rows = mrows * 2;

    __fp16 *bias_shadow = NULL;
    if (bias == NULL) {
        bias_shadow = (__fp16 *)shl_mem_alloc(n * sizeof(__fp16));
        bias = bias_shadow;
    }
    int hw = 0;
    for (; hw + m2rows - 1 < m; hw += m2rows) {
        gemm_fp16_nhwc_matrix_2rowxn(dst, sa, sb, bias, mrows, k, n);
        sb += m2rows * k;
        dst += m2rows * n;
    }
    for (; hw + mrows - 1 < m; hw += mrows) {
        gemm_fp16_nhwc_matrix_rowxn(dst, sa, sb, bias, mrows, k, n);
        sb += mrows * k;
        dst += mrows * n;
    }
    if (hw < m) {
        gemm_fp16_nhwc_matrix_row_tailxn(dst, sa, sb, bias, m - hw, k, n);
    }
    if (bias_shadow) {
        shl_mem_free(bias_shadow);
        bias_shadow = NULL;
    }
}
#endif