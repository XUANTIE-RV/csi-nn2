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

#include "c920/c920.h"

/*************************************************************
 * packn = vlenb / sizeof(__fp16)
 * m_blk: 8/4/2/1
 * n_blk: pack2n/packn/n_tail
 *
 * dst - output: [M, N]
 * sa - input:   [M/m_blk, K, m_blk]
 * sb - weights: [N/n_blk, K, n_blk]
 * bias:         [N]
 ************************************************************/
void shl_c920_gemm_a0b1_8xpack2n_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, __fp16 *bias,
                                      int M, int K, int N)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int pack2n = packn * 2;

    int flag_bias = 1;
    if (bias == NULL) {
        flag_bias = 0;
        bias = (__fp16 *)shl_mem_alloc(N * sizeof(__fp16));
    }

    int i = 0;
    for (; i + 7 < M; i += 8) {
        const __fp16 *sa_ptr = sa + i * K;
        int j = 0;
        int vl = vsetvl_e16m1(packn);
        for (; j + pack2n - 1 < N; j += pack2n) {
            const __fp16 *a_ptr = sa_ptr;
            const __fp16 *b0_ptr = sb + j * K;
            const __fp16 *b1_ptr = b0_ptr + packn;
            __fp16 *c0_ptr = dst + i * N + j;
            __fp16 *c1_ptr = c0_ptr + packn;

            // [n, 0]
            vfloat16m1_t _acc00 = vle16_v_f16m1(bias + j, vl);
            vfloat16m1_t _acc10 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc20 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc30 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc40 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc50 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc60 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc70 = vmv_v_v_f16m1(_acc00, vl);
            // [n, 1]
            vfloat16m1_t _acc01 = vle16_v_f16m1(bias + j + packn, vl);
            vfloat16m1_t _acc11 = vmv_v_v_f16m1(_acc01, vl);
            vfloat16m1_t _acc21 = vmv_v_v_f16m1(_acc01, vl);
            vfloat16m1_t _acc31 = vmv_v_v_f16m1(_acc01, vl);
            vfloat16m1_t _acc41 = vmv_v_v_f16m1(_acc01, vl);
            vfloat16m1_t _acc51 = vmv_v_v_f16m1(_acc01, vl);
            vfloat16m1_t _acc61 = vmv_v_v_f16m1(_acc01, vl);
            vfloat16m1_t _acc71 = vmv_v_v_f16m1(_acc01, vl);

            for (int c = 0; c < K; c++) {
                vfloat16m1_t _a0 = vfmv_v_f_f16m1(a_ptr[0], vl);
                vfloat16m1_t _a1 = vfmv_v_f_f16m1(a_ptr[1], vl);
                vfloat16m1_t _a2 = vfmv_v_f_f16m1(a_ptr[2], vl);
                vfloat16m1_t _a3 = vfmv_v_f_f16m1(a_ptr[3], vl);
                vfloat16m1_t _a4 = vfmv_v_f_f16m1(a_ptr[4], vl);
                vfloat16m1_t _a5 = vfmv_v_f_f16m1(a_ptr[5], vl);
                vfloat16m1_t _a6 = vfmv_v_f_f16m1(a_ptr[6], vl);
                vfloat16m1_t _a7 = vfmv_v_f_f16m1(a_ptr[7], vl);
                a_ptr += 8;
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr, vl);
                vfloat16m1_t _b1 = vle16_v_f16m1(b1_ptr, vl);
                b0_ptr += pack2n;
                b1_ptr += pack2n;

                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _a1, _b0, vl);
                _acc20 = vfmacc_vv_f16m1(_acc20, _a2, _b0, vl);
                _acc30 = vfmacc_vv_f16m1(_acc30, _a3, _b0, vl);
                _acc40 = vfmacc_vv_f16m1(_acc40, _a4, _b0, vl);
                _acc50 = vfmacc_vv_f16m1(_acc50, _a5, _b0, vl);
                _acc60 = vfmacc_vv_f16m1(_acc60, _a6, _b0, vl);
                _acc70 = vfmacc_vv_f16m1(_acc70, _a7, _b0, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _a0, _b1, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _a1, _b1, vl);
                _acc21 = vfmacc_vv_f16m1(_acc21, _a2, _b1, vl);
                _acc31 = vfmacc_vv_f16m1(_acc31, _a3, _b1, vl);
                _acc41 = vfmacc_vv_f16m1(_acc41, _a4, _b1, vl);
                _acc51 = vfmacc_vv_f16m1(_acc51, _a5, _b1, vl);
                _acc61 = vfmacc_vv_f16m1(_acc61, _a6, _b1, vl);
                _acc71 = vfmacc_vv_f16m1(_acc71, _a7, _b1, vl);
            }

            vse16_v_f16m1(c0_ptr, _acc00, vl);
            vse16_v_f16m1(c0_ptr + N, _acc10, vl);
            vse16_v_f16m1(c0_ptr + N * 2, _acc20, vl);
            vse16_v_f16m1(c0_ptr + N * 3, _acc30, vl);
            vse16_v_f16m1(c0_ptr + N * 4, _acc40, vl);
            vse16_v_f16m1(c0_ptr + N * 5, _acc50, vl);
            vse16_v_f16m1(c0_ptr + N * 6, _acc60, vl);
            vse16_v_f16m1(c0_ptr + N * 7, _acc70, vl);
            vse16_v_f16m1(c1_ptr, _acc01, vl);
            vse16_v_f16m1(c1_ptr + N, _acc11, vl);
            vse16_v_f16m1(c1_ptr + N * 2, _acc21, vl);
            vse16_v_f16m1(c1_ptr + N * 3, _acc31, vl);
            vse16_v_f16m1(c1_ptr + N * 4, _acc41, vl);
            vse16_v_f16m1(c1_ptr + N * 5, _acc51, vl);
            vse16_v_f16m1(c1_ptr + N * 6, _acc61, vl);
            vse16_v_f16m1(c1_ptr + N * 7, _acc71, vl);
        }
        while (j < N) {
            int vl = vsetvl_e16m1(N - j);
            const __fp16 *a_ptr = sa_ptr;
            const __fp16 *b0_ptr = sb + j * K;
            __fp16 *c0_ptr = dst + i * N + j;

            vfloat16m1_t _acc00 = vle16_v_f16m1(bias + j, vl);
            vfloat16m1_t _acc10 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc20 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc30 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc40 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc50 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc60 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc70 = vmv_v_v_f16m1(_acc00, vl);

            for (int c = 0; c < K; c++) {
                vfloat16m1_t _a0 = vfmv_v_f_f16m1(a_ptr[0], vl);
                vfloat16m1_t _a1 = vfmv_v_f_f16m1(a_ptr[1], vl);
                vfloat16m1_t _a2 = vfmv_v_f_f16m1(a_ptr[2], vl);
                vfloat16m1_t _a3 = vfmv_v_f_f16m1(a_ptr[3], vl);
                vfloat16m1_t _a4 = vfmv_v_f_f16m1(a_ptr[4], vl);
                vfloat16m1_t _a5 = vfmv_v_f_f16m1(a_ptr[5], vl);
                vfloat16m1_t _a6 = vfmv_v_f_f16m1(a_ptr[6], vl);
                vfloat16m1_t _a7 = vfmv_v_f_f16m1(a_ptr[7], vl);
                a_ptr += 8;
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr, vl);
                b0_ptr += vl;

                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _a1, _b0, vl);
                _acc20 = vfmacc_vv_f16m1(_acc20, _a2, _b0, vl);
                _acc30 = vfmacc_vv_f16m1(_acc30, _a3, _b0, vl);
                _acc40 = vfmacc_vv_f16m1(_acc40, _a4, _b0, vl);
                _acc50 = vfmacc_vv_f16m1(_acc50, _a5, _b0, vl);
                _acc60 = vfmacc_vv_f16m1(_acc60, _a6, _b0, vl);
                _acc70 = vfmacc_vv_f16m1(_acc70, _a7, _b0, vl);
            }

            vse16_v_f16m1(c0_ptr, _acc00, vl);
            vse16_v_f16m1(c0_ptr + N, _acc10, vl);
            vse16_v_f16m1(c0_ptr + N * 2, _acc20, vl);
            vse16_v_f16m1(c0_ptr + N * 3, _acc30, vl);
            vse16_v_f16m1(c0_ptr + N * 4, _acc40, vl);
            vse16_v_f16m1(c0_ptr + N * 5, _acc50, vl);
            vse16_v_f16m1(c0_ptr + N * 6, _acc60, vl);
            vse16_v_f16m1(c0_ptr + N * 7, _acc70, vl);
            j += vl;
        }
    }
    for (; i + 3 < M; i += 4) {
        const __fp16 *sa_ptr = sa + i * K;
        int j = 0;
        int vl = vsetvl_e16m1(packn);
        for (; j + pack2n - 1 < N; j += pack2n) {
            const __fp16 *a_ptr = sa_ptr;
            const __fp16 *b0_ptr = sb + j * K;
            const __fp16 *b1_ptr = b0_ptr + packn;
            __fp16 *c0_ptr = dst + i * N + j;
            __fp16 *c1_ptr = c0_ptr + packn;

            vfloat16m1_t _acc00 = vle16_v_f16m1(bias + j, vl);
            vfloat16m1_t _acc10 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc20 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc30 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc01 = vle16_v_f16m1(bias + j + packn, vl);
            vfloat16m1_t _acc11 = vmv_v_v_f16m1(_acc01, vl);
            vfloat16m1_t _acc21 = vmv_v_v_f16m1(_acc01, vl);
            vfloat16m1_t _acc31 = vmv_v_v_f16m1(_acc01, vl);

            for (int c = 0; c < K; c++) {
                vfloat16m1_t _a0 = vfmv_v_f_f16m1(a_ptr[0], vl);
                vfloat16m1_t _a1 = vfmv_v_f_f16m1(a_ptr[1], vl);
                vfloat16m1_t _a2 = vfmv_v_f_f16m1(a_ptr[2], vl);
                vfloat16m1_t _a3 = vfmv_v_f_f16m1(a_ptr[3], vl);
                a_ptr += 4;
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr, vl);
                vfloat16m1_t _b1 = vle16_v_f16m1(b1_ptr, vl);
                b0_ptr += pack2n;
                b1_ptr += pack2n;

                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _a1, _b0, vl);
                _acc20 = vfmacc_vv_f16m1(_acc20, _a2, _b0, vl);
                _acc30 = vfmacc_vv_f16m1(_acc30, _a3, _b0, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _a0, _b1, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _a1, _b1, vl);
                _acc21 = vfmacc_vv_f16m1(_acc21, _a2, _b1, vl);
                _acc31 = vfmacc_vv_f16m1(_acc31, _a3, _b1, vl);
            }

            vse16_v_f16m1(c0_ptr, _acc00, vl);
            vse16_v_f16m1(c0_ptr + N, _acc10, vl);
            vse16_v_f16m1(c0_ptr + N * 2, _acc20, vl);
            vse16_v_f16m1(c0_ptr + N * 3, _acc30, vl);
            vse16_v_f16m1(c1_ptr, _acc01, vl);
            vse16_v_f16m1(c1_ptr + N, _acc11, vl);
            vse16_v_f16m1(c1_ptr + N * 2, _acc21, vl);
            vse16_v_f16m1(c1_ptr + N * 3, _acc31, vl);
        }
        while (j < N) {
            int vl = vsetvl_e16m1(N - j);
            const __fp16 *a_ptr = sa_ptr;
            const __fp16 *b0_ptr = sb + j * K;
            __fp16 *c0_ptr = dst + i * N + j;

            vfloat16m1_t _acc00 = vle16_v_f16m1(bias + j, vl);
            vfloat16m1_t _acc10 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc20 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc30 = vmv_v_v_f16m1(_acc00, vl);

            for (int c = 0; c < K; c++) {
                vfloat16m1_t _a0 = vfmv_v_f_f16m1(a_ptr[0], vl);
                vfloat16m1_t _a1 = vfmv_v_f_f16m1(a_ptr[1], vl);
                vfloat16m1_t _a2 = vfmv_v_f_f16m1(a_ptr[2], vl);
                vfloat16m1_t _a3 = vfmv_v_f_f16m1(a_ptr[3], vl);
                a_ptr += 4;
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr, vl);
                b0_ptr += vl;

                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _a1, _b0, vl);
                _acc20 = vfmacc_vv_f16m1(_acc20, _a2, _b0, vl);
                _acc30 = vfmacc_vv_f16m1(_acc30, _a3, _b0, vl);
            }

            vse16_v_f16m1(c0_ptr, _acc00, vl);
            vse16_v_f16m1(c0_ptr + N, _acc10, vl);
            vse16_v_f16m1(c0_ptr + N * 2, _acc20, vl);
            vse16_v_f16m1(c0_ptr + N * 3, _acc30, vl);
            j += vl;
        }
    }
    for (; i + 1 < M; i += 2) {
        const __fp16 *sa_ptr = sa + i * K;
        int j = 0;
        int vl = vsetvl_e16m1(packn);
        for (; j + pack2n - 1 < N; j += pack2n) {
            const __fp16 *a_ptr = sa_ptr;
            const __fp16 *b0_ptr = sb + j * K;
            const __fp16 *b1_ptr = b0_ptr + packn;
            __fp16 *c0_ptr = dst + i * N + j;
            __fp16 *c1_ptr = c0_ptr + packn;

            vfloat16m1_t _acc00 = vle16_v_f16m1(bias + j, vl);
            vfloat16m1_t _acc10 = vmv_v_v_f16m1(_acc00, vl);
            vfloat16m1_t _acc01 = vle16_v_f16m1(bias + j + packn, vl);
            vfloat16m1_t _acc11 = vmv_v_v_f16m1(_acc01, vl);

            for (int c = 0; c < K; c++) {
                vfloat16m1_t _a0 = vfmv_v_f_f16m1(a_ptr[0], vl);
                vfloat16m1_t _a1 = vfmv_v_f_f16m1(a_ptr[1], vl);
                a_ptr += 2;
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr, vl);
                vfloat16m1_t _b1 = vle16_v_f16m1(b1_ptr, vl);
                b0_ptr += pack2n;
                b1_ptr += pack2n;

                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _a1, _b0, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _a0, _b1, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _a1, _b1, vl);
            }

            vse16_v_f16m1(c0_ptr, _acc00, vl);
            vse16_v_f16m1(c0_ptr + N, _acc10, vl);
            vse16_v_f16m1(c1_ptr, _acc01, vl);
            vse16_v_f16m1(c1_ptr + N, _acc11, vl);
        }
        while (j < N) {
            int vl = vsetvl_e16m1(N - j);
            const __fp16 *a_ptr = sa_ptr;
            const __fp16 *b0_ptr = sb + j * K;
            __fp16 *c0_ptr = dst + i * N + j;

            vfloat16m1_t _acc00 = vle16_v_f16m1(bias + j, vl);
            vfloat16m1_t _acc10 = vmv_v_v_f16m1(_acc00, vl);

            for (int c = 0; c < K; c++) {
                vfloat16m1_t _a0 = vfmv_v_f_f16m1(a_ptr[0], vl);
                vfloat16m1_t _a1 = vfmv_v_f_f16m1(a_ptr[1], vl);
                a_ptr += 2;
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr, vl);
                b0_ptr += vl;

                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _a1, _b0, vl);
            }

            vse16_v_f16m1(c0_ptr, _acc00, vl);
            vse16_v_f16m1(c0_ptr + N, _acc10, vl);
            j += vl;
        }
    }
    for (; i < M; i++) {
        const __fp16 *sa_ptr = sa + i * K;
        int j = 0;
        int vl = vsetvl_e16m1(packn);
        for (; j + pack2n - 1 < N; j += pack2n) {
            const __fp16 *a_ptr = sa_ptr;
            const __fp16 *b0_ptr = sb + j * K;
            const __fp16 *b1_ptr = b0_ptr + packn;
            __fp16 *c0_ptr = dst + i * N + j;
            __fp16 *c1_ptr = c0_ptr + packn;

            vfloat16m1_t _acc00 = vle16_v_f16m1(bias + j, vl);
            vfloat16m1_t _acc01 = vle16_v_f16m1(bias + j + packn, vl);

            for (int c = 0; c < K; c++) {
                vfloat16m1_t _a0 = vfmv_v_f_f16m1(a_ptr[0], vl);
                a_ptr += 1;
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr, vl);
                vfloat16m1_t _b1 = vle16_v_f16m1(b1_ptr, vl);
                b0_ptr += pack2n;
                b1_ptr += pack2n;

                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _a0, _b1, vl);
            }

            vse16_v_f16m1(c0_ptr, _acc00, vl);
            vse16_v_f16m1(c1_ptr, _acc01, vl);
        }
        while (j < N) {
            int vl = vsetvl_e16m1(N - j);
            const __fp16 *a_ptr = sa_ptr;
            const __fp16 *b0_ptr = sb + j * K;
            __fp16 *c0_ptr = dst + i * N + j;

            vfloat16m1_t _acc00 = vle16_v_f16m1(bias + j, vl);

            for (int c = 0; c < K; c++) {
                vfloat16m1_t _a0 = vfmv_v_f_f16m1(a_ptr[0], vl);
                a_ptr += 1;
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr, vl);
                b0_ptr += vl;

                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vl);
            }

            vse16_v_f16m1(c0_ptr, _acc00, vl);
            j += vl;
        }
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
