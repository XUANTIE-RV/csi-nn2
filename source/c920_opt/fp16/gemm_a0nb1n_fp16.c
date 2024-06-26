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

static inline void c920_omp_get_mn_partition(int M, int N, int *M_start, int *M_end, int *N_start,
                                             int *N_end)
{
#ifdef _OPENMP
    int rank = omp_get_thread_num();
    int threads = omp_get_num_threads();

    if (M > 2 * N) {
        int q = M / threads;
        int r = M % threads;
        *M_start = rank < r ? rank * (q + 1) : rank * q + r;
        *M_end = rank < r ? (rank + 1) * (q + 1) : (rank + 1) * q + r;
    } else if (N > 2 * M) {
        int q = N / threads;
        int r = N % threads;
        *N_start = rank < r ? rank * (q + 1) : rank * q + r;
        *N_end = rank < r ? (rank + 1) * (q + 1) : (rank + 1) * q + r;
    } else {
        // TODO: support any number of threads
        float _s = sqrt(threads);
        assert(floor(_s + 0.5) == _s);
        int t_sqrt = (int)_s;

        int r_rank = rank / t_sqrt;
        int c_rank = rank % (int)t_sqrt;

        int M_q = M / t_sqrt;
        int M_r = M % t_sqrt;
        *M_start = r_rank < M_r ? r_rank * (M_q + 1) : r_rank * M_q + M_r;
        *M_end = r_rank < M_r ? (r_rank + 1) * (M_q + 1) : (r_rank + 1) * M_q + M_r;

        int N_q = N / t_sqrt;
        int N_r = N % t_sqrt;
        *N_start = c_rank < N_r ? c_rank * (N_q + 1) : c_rank * N_q + N_r;
        *N_end = c_rank < N_r ? (c_rank + 1) * (N_q + 1) : (c_rank + 1) * N_q + N_r;
    }
#endif
}

static inline vfloat16m4_t vdeq_vf_f16m4(vint8m2_t _src, __fp16 scale, int vl)
{
    vint16m4_t _i16 = vwadd_vx_i16m4(_src, 0, vl);
    vfloat16m4_t _f16 = vfcvt_f_x_v_f16m4(_i16, vl);
    _f16 = vfmul_vf_f16m4(_f16, scale, vl);
    return _f16;
}

/*************************************************************
 * constrain: vlen = 128, and K % 32 == 0
 ************************************************************/
static inline void gemm_dot_1x1_fp16_q8(__fp16 *dst, const __fp16 *sa, const int8_t *sb,
                                        const __fp16 *scale, __fp16 *bias, int M, int K, int N,
                                        int lda, int ldb, int ldc, int k_idx)
{
    int block_size = 32;
    int i = 0;
    for (; i < M; i++) {
        const __fp16 *sa_ptr = sa + i * lda;
        int j = 0;
        for (; j < N; j++) {
            const __fp16 *a0_ptr = sa_ptr;
            const int8_t *b0_ptr = sb + j * ldb;
            const __fp16 *s0_ptr = scale + j * ldb / block_size;

            // vlen128 e16m4=32
            int vl = vsetvl_e16m4(block_size);
            // dst[0, 0]
            vfloat16m4_t _acc00 = vfmv_v_f_f16m4(0.0f, vl);

            int c = 0;
            for (; c + block_size - 1 < K; c += block_size) {
                vfloat16m4_t _a0 = vle16_v_f16m4(a0_ptr + c, vl);
                vint8m2_t _b0_i8 = vle8_v_i8m2(b0_ptr + c, vl);
                vfloat16m4_t _b0_f32 = vdeq_vf_f16m4(_b0_i8, s0_ptr[0], vl);
                s0_ptr += 1;
                _acc00 = vfmacc_vv_f16m4(_acc00, _a0, _b0_f32, vl);
            }

            int idx00 = (i + 0) * ldc + (j + 0);
            vfloat16m1_t _sum00;
            if (k_idx == 0) {
                _sum00 = vfmv_v_f_f16m1(bias[j + 0], 1);
            } else {
                _sum00 = vfmv_v_f_f16m1(dst[idx00], 1);
            }

            _sum00 = vfredosum_vs_f16m4_f16m1(vundefined_f16m1(), _acc00, _sum00, vl);
            dst[idx00] = vfmv_f_s_f16m1_f16(_sum00);
        }
    }
}

static void gemm_dot_1x1_fp16_q8_omp(__fp16 *dst, const __fp16 *sa, const int8_t *sb,
                                     const __fp16 *scale, __fp16 *bias, int M, int K, int N,
                                     int lda, int ldb, int ldc, int k_idx)
{
    if (shl_multithread_is_enable()) {
#pragma omp parallel
        {
            int M_start = 0, M_end = M;
            int N_start = 0, N_end = N;
            c920_omp_get_mn_partition(M, N, &M_start, &M_end, &N_start, &N_end);

            __fp16 *thread_dst = dst + M_start * ldc + N_start;
            const __fp16 *thread_sa = sa + M_start * lda;
            const int8_t *thread_sb = sb + N_start * ldb;
            const __fp16 *thread_scale = scale + N_start * ldb / 32;
            __fp16 *thread_bias = bias + N_start;
            int thread_M = M_end - M_start;
            int thread_N = N_end - N_start;
            gemm_dot_1x1_fp16_q8(thread_dst, thread_sa, thread_sb, thread_scale, thread_bias,
                                 thread_M, K, thread_N, lda, ldb, ldc, k_idx);
        }
    } else {
        gemm_dot_1x1_fp16_q8(dst, sa, sb, scale, bias, M, K, N, lda, ldb, ldc, k_idx);
    }
}

/*************************************************************
 * constrain: vlen = 128, and K % 32 == 0
 ************************************************************/
static inline void gemm_dot_1x1_fp16_q4(__fp16 *dst, const __fp16 *sa, const int8_t *sb,
                                        const __fp16 *scale, __fp16 *bias, int M, int K, int N,
                                        int lda, int ldb, int ldc, int k_idx)
{
    int block_size = 32;
    int half_block = block_size / 2;
    int i = 0;
    for (; i < M; i++) {
        const __fp16 *sa_ptr = sa + i * lda;
        int j = 0;
        for (; j < N; j++) {
            const __fp16 *a0_ptr = sa_ptr;
            const int8_t *b0_ptr = sb + j * ldb / 2;
            const __fp16 *s0_ptr = scale + j * ldb / block_size;

            // vlen128 e16m2=16
            int vl = vsetvl_e16m2(half_block);
            // dst[0, 0]
            vfloat16m2_t _acc00 = vfmv_v_f_f16m2(0.0f, vl);

            int c = 0;
            for (; c + block_size - 1 < K; c += block_size) {
                vfloat16m2_t _a00 = vle16_v_f16m2(a0_ptr + c, vl);
                vfloat16m2_t _a01 = vle16_v_f16m2(a0_ptr + c + half_block, vl);

                vint8m1_t _b0_i8 = vle8_v_i8m1(b0_ptr, vl);
                b0_ptr += half_block;

                vint8m1_t _low_i8 = vand_vx_i8m1(_b0_i8, 0x0f, vl);
                vint8m1_t _high_i8 = vsra_vx_i8m1(_b0_i8, 4, vl);
                _high_i8 = vand_vx_i8m1(_high_i8, 0x0f, vl);
                vint16m2_t _low_i16 = vwsub_vx_i16m2(_low_i8, 8, vl);
                vint16m2_t _high_i16 = vwsub_vx_i16m2(_high_i8, 8, vl);
                vfloat16m2_t _low_f16 = vfcvt_f_x_v_f16m2(_low_i16, vl);
                vfloat16m2_t _high_f16 = vfcvt_f_x_v_f16m2(_high_i16, vl);
                _low_f16 = vfmul_vf_f16m2(_low_f16, s0_ptr[0], vl);
                _high_f16 = vfmul_vf_f16m2(_high_f16, s0_ptr[0], vl);
                s0_ptr += 1;

                _acc00 = vfmacc_vv_f16m2(_acc00, _a00, _low_f16, vl);
                _acc00 = vfmacc_vv_f16m2(_acc00, _a01, _high_f16, vl);
            }

            int idx00 = (i + 0) * ldc + (j + 0);
            vfloat16m1_t _sum00;
            if (k_idx == 0) {
                _sum00 = vfmv_v_f_f16m1(bias[j + 0], 1);
            } else {
                _sum00 = vfmv_v_f_f16m1(dst[idx00], 1);
            }

            _sum00 = vfredosum_vs_f16m2_f16m1(vundefined_f16m1(), _acc00, _sum00, vl);
            dst[idx00] = vfmv_f_s_f16m1_f16(_sum00);
        }
    }
}

static void gemm_dot_1x1_fp16_q4_omp(__fp16 *dst, const __fp16 *sa, const int8_t *sb,
                                     const __fp16 *scale, __fp16 *bias, int M, int K, int N,
                                     int lda, int ldb, int ldc, int k_idx)
{
    if (shl_multithread_is_enable()) {
#pragma omp parallel
        {
            int M_start = 0, M_end = M;
            int N_start = 0, N_end = N;
            c920_omp_get_mn_partition(M, N, &M_start, &M_end, &N_start, &N_end);

            __fp16 *thread_dst = dst + M_start * ldc + N_start;
            const __fp16 *thread_sa = sa + M_start * lda;
            const int8_t *thread_sb = sb + N_start * ldb / 2;
            const __fp16 *thread_scale = scale + N_start * ldb / 32;
            __fp16 *thread_bias = bias + N_start;
            int thread_M = M_end - M_start;
            int thread_N = N_end - N_start;
            gemm_dot_1x1_fp16_q4(thread_dst, thread_sa, thread_sb, thread_scale, thread_bias,
                                 thread_M, K, thread_N, lda, ldb, ldc, k_idx);
        }
    } else {
        gemm_dot_1x1_fp16_q4(dst, sa, sb, scale, bias, M, K, N, lda, ldb, ldc, k_idx);
    }
}

/* q4 ****************************************************************************/

static inline void gemm_dot_4x4_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, __fp16 *bias,
                                     int M, int K, int N, int lda, int ldb, int ldc, int k_idx)
{
    int i = 0;
    for (; i + 3 < M; i += 4) {
        const __fp16 *sa_ptr = sa + i * lda;
        int j = 0;
        for (; j + 3 < N; j += 4) {
            const __fp16 *a0_ptr = sa_ptr;
            const __fp16 *a1_ptr = sa_ptr + 1 * lda;
            const __fp16 *a2_ptr = sa_ptr + 2 * lda;
            const __fp16 *a3_ptr = sa_ptr + 3 * lda;
            const __fp16 *b0_ptr = sb + j * ldb;
            const __fp16 *b1_ptr = b0_ptr + 1 * ldb;
            const __fp16 *b2_ptr = b0_ptr + 2 * ldb;
            const __fp16 *b3_ptr = b0_ptr + 3 * ldb;

            int vlmax = vsetvl_e16m1(csrr_vlenb() / sizeof(__fp16));
            // dst[m, 0]
            vfloat16m1_t _acc00 = vfmv_v_f_f16m1(0.0f, vlmax);
            vfloat16m1_t _acc10 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc20 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc30 = vmv_v_v_f16m1(_acc00, vlmax);
            // dst[m, 1]
            vfloat16m1_t _acc01 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc11 = vmv_v_v_f16m1(_acc01, vlmax);
            vfloat16m1_t _acc21 = vmv_v_v_f16m1(_acc01, vlmax);
            vfloat16m1_t _acc31 = vmv_v_v_f16m1(_acc01, vlmax);
            // dst[m, 2]
            vfloat16m1_t _acc02 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc12 = vmv_v_v_f16m1(_acc02, vlmax);
            vfloat16m1_t _acc22 = vmv_v_v_f16m1(_acc02, vlmax);
            vfloat16m1_t _acc32 = vmv_v_v_f16m1(_acc02, vlmax);
            // dst[m, 3]
            vfloat16m1_t _acc03 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc13 = vmv_v_v_f16m1(_acc03, vlmax);
            vfloat16m1_t _acc23 = vmv_v_v_f16m1(_acc03, vlmax);
            vfloat16m1_t _acc33 = vmv_v_v_f16m1(_acc03, vlmax);

            int c = 0;
            while (c < K) {
                int vl = vsetvl_e16m1(K - c);
                vfloat16m1_t _a0 = vle16_v_f16m1(a0_ptr + c, vl);
                vfloat16m1_t _a1 = vle16_v_f16m1(a1_ptr + c, vl);
                vfloat16m1_t _a2 = vle16_v_f16m1(a2_ptr + c, vl);
                vfloat16m1_t _a3 = vle16_v_f16m1(a3_ptr + c, vl);
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr + c, vl);
                vfloat16m1_t _b1 = vle16_v_f16m1(b1_ptr + c, vl);
                vfloat16m1_t _b2 = vle16_v_f16m1(b2_ptr + c, vl);
                vfloat16m1_t _b3 = vle16_v_f16m1(b3_ptr + c, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vlmax);
                _acc10 = vfmacc_vv_f16m1(_acc10, _a1, _b0, vlmax);
                _acc20 = vfmacc_vv_f16m1(_acc20, _a2, _b0, vlmax);
                _acc30 = vfmacc_vv_f16m1(_acc30, _a3, _b0, vlmax);

                _acc01 = vfmacc_vv_f16m1(_acc01, _a0, _b1, vlmax);
                _acc11 = vfmacc_vv_f16m1(_acc11, _a1, _b1, vlmax);
                _acc21 = vfmacc_vv_f16m1(_acc21, _a2, _b1, vlmax);
                _acc31 = vfmacc_vv_f16m1(_acc31, _a3, _b1, vlmax);

                _acc02 = vfmacc_vv_f16m1(_acc02, _a0, _b2, vlmax);
                _acc12 = vfmacc_vv_f16m1(_acc12, _a1, _b2, vlmax);
                _acc22 = vfmacc_vv_f16m1(_acc22, _a2, _b2, vlmax);
                _acc32 = vfmacc_vv_f16m1(_acc32, _a3, _b2, vlmax);

                _acc03 = vfmacc_vv_f16m1(_acc03, _a0, _b3, vlmax);
                _acc13 = vfmacc_vv_f16m1(_acc13, _a1, _b3, vlmax);
                _acc23 = vfmacc_vv_f16m1(_acc23, _a2, _b3, vlmax);
                _acc33 = vfmacc_vv_f16m1(_acc33, _a3, _b3, vlmax);
                c += vl;
            }

            int idx00 = (i + 0) * ldc + (j + 0);
            int idx10 = (i + 1) * ldc + (j + 0);
            int idx20 = (i + 2) * ldc + (j + 0);
            int idx30 = (i + 3) * ldc + (j + 0);

            int idx01 = (i + 0) * ldc + (j + 1);
            int idx11 = (i + 1) * ldc + (j + 1);
            int idx21 = (i + 2) * ldc + (j + 1);
            int idx31 = (i + 3) * ldc + (j + 1);

            int idx02 = (i + 0) * ldc + (j + 2);
            int idx12 = (i + 1) * ldc + (j + 2);
            int idx22 = (i + 2) * ldc + (j + 2);
            int idx32 = (i + 3) * ldc + (j + 2);

            int idx03 = (i + 0) * ldc + (j + 3);
            int idx13 = (i + 1) * ldc + (j + 3);
            int idx23 = (i + 2) * ldc + (j + 3);
            int idx33 = (i + 3) * ldc + (j + 3);

            // dst[m, 0]
            vfloat16m1_t _sum00;
            vfloat16m1_t _sum10;
            vfloat16m1_t _sum20;
            vfloat16m1_t _sum30;
            // dst[m, 1]
            vfloat16m1_t _sum01;
            vfloat16m1_t _sum11;
            vfloat16m1_t _sum21;
            vfloat16m1_t _sum31;
            // dst[m, 2]
            vfloat16m1_t _sum02;
            vfloat16m1_t _sum12;
            vfloat16m1_t _sum22;
            vfloat16m1_t _sum32;
            // dst[m, 3]
            vfloat16m1_t _sum03;
            vfloat16m1_t _sum13;
            vfloat16m1_t _sum23;
            vfloat16m1_t _sum33;
            if (k_idx == 0) {
                _sum00 = vfmv_v_f_f16m1(bias[j + 0], 1);
                _sum10 = vmv_v_v_f16m1(_sum00, 1);
                _sum20 = vmv_v_v_f16m1(_sum00, 1);
                _sum30 = vmv_v_v_f16m1(_sum00, 1);

                _sum01 = vfmv_v_f_f16m1(bias[j + 1], 1);
                _sum11 = vmv_v_v_f16m1(_sum01, 1);
                _sum21 = vmv_v_v_f16m1(_sum01, 1);
                _sum31 = vmv_v_v_f16m1(_sum01, 1);

                _sum02 = vfmv_v_f_f16m1(bias[j + 2], 1);
                _sum12 = vmv_v_v_f16m1(_sum02, 1);
                _sum22 = vmv_v_v_f16m1(_sum02, 1);
                _sum32 = vmv_v_v_f16m1(_sum02, 1);

                _sum03 = vfmv_v_f_f16m1(bias[j + 3], 1);
                _sum13 = vmv_v_v_f16m1(_sum03, 1);
                _sum23 = vmv_v_v_f16m1(_sum03, 1);
                _sum33 = vmv_v_v_f16m1(_sum03, 1);
            } else {
                _sum00 = vfmv_v_f_f16m1(dst[idx00], 1);
                _sum10 = vfmv_v_f_f16m1(dst[idx10], 1);
                _sum20 = vfmv_v_f_f16m1(dst[idx20], 1);
                _sum30 = vfmv_v_f_f16m1(dst[idx30], 1);

                _sum01 = vfmv_v_f_f16m1(dst[idx01], 1);
                _sum11 = vfmv_v_f_f16m1(dst[idx11], 1);
                _sum21 = vfmv_v_f_f16m1(dst[idx21], 1);
                _sum31 = vfmv_v_f_f16m1(dst[idx31], 1);

                _sum02 = vfmv_v_f_f16m1(dst[idx02], 1);
                _sum12 = vfmv_v_f_f16m1(dst[idx12], 1);
                _sum22 = vfmv_v_f_f16m1(dst[idx22], 1);
                _sum32 = vfmv_v_f_f16m1(dst[idx32], 1);

                _sum03 = vfmv_v_f_f16m1(dst[idx03], 1);
                _sum13 = vfmv_v_f_f16m1(dst[idx13], 1);
                _sum23 = vfmv_v_f_f16m1(dst[idx23], 1);
                _sum33 = vfmv_v_f_f16m1(dst[idx33], 1);
            }

            _sum00 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc00, _sum00, vlmax);
            _sum10 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc10, _sum10, vlmax);
            _sum20 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc20, _sum20, vlmax);
            _sum30 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc30, _sum30, vlmax);

            _sum01 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc01, _sum01, vlmax);
            _sum11 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc11, _sum11, vlmax);
            _sum21 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc21, _sum21, vlmax);
            _sum31 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc31, _sum31, vlmax);

            _sum02 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc02, _sum02, vlmax);
            _sum12 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc12, _sum12, vlmax);
            _sum22 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc22, _sum22, vlmax);
            _sum32 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc32, _sum32, vlmax);

            _sum03 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc03, _sum03, vlmax);
            _sum13 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc13, _sum13, vlmax);
            _sum23 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc23, _sum23, vlmax);
            _sum33 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc33, _sum33, vlmax);

            dst[idx00] = vfmv_f_s_f16m1_f16(_sum00);
            dst[idx10] = vfmv_f_s_f16m1_f16(_sum10);
            dst[idx20] = vfmv_f_s_f16m1_f16(_sum20);
            dst[idx30] = vfmv_f_s_f16m1_f16(_sum30);

            dst[idx01] = vfmv_f_s_f16m1_f16(_sum01);
            dst[idx11] = vfmv_f_s_f16m1_f16(_sum11);
            dst[idx21] = vfmv_f_s_f16m1_f16(_sum21);
            dst[idx31] = vfmv_f_s_f16m1_f16(_sum31);

            dst[idx02] = vfmv_f_s_f16m1_f16(_sum02);
            dst[idx12] = vfmv_f_s_f16m1_f16(_sum12);
            dst[idx22] = vfmv_f_s_f16m1_f16(_sum22);
            dst[idx32] = vfmv_f_s_f16m1_f16(_sum32);

            dst[idx03] = vfmv_f_s_f16m1_f16(_sum03);
            dst[idx13] = vfmv_f_s_f16m1_f16(_sum13);
            dst[idx23] = vfmv_f_s_f16m1_f16(_sum23);
            dst[idx33] = vfmv_f_s_f16m1_f16(_sum33);
        }
        for (; j < N; j++) {
            const __fp16 *a0_ptr = sa_ptr;
            const __fp16 *a1_ptr = sa_ptr + 1 * lda;
            const __fp16 *a2_ptr = sa_ptr + 2 * lda;
            const __fp16 *a3_ptr = sa_ptr + 3 * lda;
            const __fp16 *b0_ptr = sb + j * ldb;

            int vlmax = vsetvl_e16m1(csrr_vlenb() / sizeof(__fp16));
            // dst[m, 0]
            vfloat16m1_t _acc00 = vfmv_v_f_f16m1(0.0f, vlmax);
            vfloat16m1_t _acc10 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc20 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc30 = vmv_v_v_f16m1(_acc00, vlmax);

            int c = 0;
            while (c < K) {
                int vl = vsetvl_e16m1(K - c);
                vfloat16m1_t _a0 = vle16_v_f16m1(a0_ptr + c, vl);
                vfloat16m1_t _a1 = vle16_v_f16m1(a1_ptr + c, vl);
                vfloat16m1_t _a2 = vle16_v_f16m1(a2_ptr + c, vl);
                vfloat16m1_t _a3 = vle16_v_f16m1(a3_ptr + c, vl);
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr + c, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vlmax);
                _acc10 = vfmacc_vv_f16m1(_acc10, _a1, _b0, vlmax);
                _acc20 = vfmacc_vv_f16m1(_acc20, _a2, _b0, vlmax);
                _acc30 = vfmacc_vv_f16m1(_acc30, _a3, _b0, vlmax);
                c += vl;
            }

            int idx00 = (i + 0) * ldc + (j + 0);
            int idx10 = (i + 1) * ldc + (j + 0);
            int idx20 = (i + 2) * ldc + (j + 0);
            int idx30 = (i + 3) * ldc + (j + 0);

            // dst[m, 0]
            vfloat16m1_t _sum00;
            vfloat16m1_t _sum10;
            vfloat16m1_t _sum20;
            vfloat16m1_t _sum30;
            if (k_idx == 0) {
                _sum00 = vfmv_v_f_f16m1(bias[j + 0], 1);
                _sum10 = vmv_v_v_f16m1(_sum00, 1);
                _sum20 = vmv_v_v_f16m1(_sum00, 1);
                _sum30 = vmv_v_v_f16m1(_sum00, 1);
            } else {
                _sum00 = vfmv_v_f_f16m1(dst[idx00], 1);
                _sum10 = vfmv_v_f_f16m1(dst[idx10], 1);
                _sum20 = vfmv_v_f_f16m1(dst[idx20], 1);
                _sum30 = vfmv_v_f_f16m1(dst[idx30], 1);
            }

            _sum00 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc00, _sum00, vlmax);
            _sum10 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc10, _sum10, vlmax);
            _sum20 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc20, _sum20, vlmax);
            _sum30 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc30, _sum30, vlmax);

            dst[idx00] = vfmv_f_s_f16m1_f16(_sum00);
            dst[idx10] = vfmv_f_s_f16m1_f16(_sum10);
            dst[idx20] = vfmv_f_s_f16m1_f16(_sum20);
            dst[idx30] = vfmv_f_s_f16m1_f16(_sum30);
        }
    }
    for (; i < M; i += 1) {
        const __fp16 *sa_ptr = sa + i * lda;
        int j = 0;
        for (; j + 3 < N; j += 4) {
            const __fp16 *a0_ptr = sa_ptr;
            const __fp16 *b0_ptr = sb + j * ldb;
            const __fp16 *b1_ptr = b0_ptr + 1 * ldb;
            const __fp16 *b2_ptr = b0_ptr + 2 * ldb;
            const __fp16 *b3_ptr = b0_ptr + 3 * ldb;

            int vlmax = vsetvl_e16m1(csrr_vlenb() / sizeof(__fp16));
            // dst[0, n]
            vfloat16m1_t _acc00 = vfmv_v_f_f16m1(0.0f, vlmax);
            vfloat16m1_t _acc01 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc02 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc03 = vmv_v_v_f16m1(_acc00, vlmax);

            int c = 0;
            while (c < K) {
                int vl = vsetvl_e16m1(K - c);
                vfloat16m1_t _a0 = vle16_v_f16m1(a0_ptr + c, vl);
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr + c, vl);
                vfloat16m1_t _b1 = vle16_v_f16m1(b1_ptr + c, vl);
                vfloat16m1_t _b2 = vle16_v_f16m1(b2_ptr + c, vl);
                vfloat16m1_t _b3 = vle16_v_f16m1(b3_ptr + c, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vlmax);
                _acc01 = vfmacc_vv_f16m1(_acc01, _a0, _b1, vlmax);
                _acc02 = vfmacc_vv_f16m1(_acc02, _a0, _b2, vlmax);
                _acc03 = vfmacc_vv_f16m1(_acc03, _a0, _b3, vlmax);
                c += vl;
            }

            int idx00 = (i + 0) * ldc + (j + 0);
            int idx01 = (i + 0) * ldc + (j + 1);
            int idx02 = (i + 0) * ldc + (j + 2);
            int idx03 = (i + 0) * ldc + (j + 3);

            // dst[0, n]
            vfloat16m1_t _sum00;
            vfloat16m1_t _sum01;
            vfloat16m1_t _sum02;
            vfloat16m1_t _sum03;
            if (k_idx == 0) {
                _sum00 = vfmv_v_f_f16m1(bias[j + 0], 1);
                _sum01 = vfmv_v_f_f16m1(bias[j + 1], 1);
                _sum02 = vfmv_v_f_f16m1(bias[j + 2], 1);
                _sum03 = vfmv_v_f_f16m1(bias[j + 3], 1);
            } else {
                _sum00 = vfmv_v_f_f16m1(dst[idx00], 1);
                _sum01 = vfmv_v_f_f16m1(dst[idx01], 1);
                _sum02 = vfmv_v_f_f16m1(dst[idx02], 1);
                _sum03 = vfmv_v_f_f16m1(dst[idx03], 1);
            }

            _sum00 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc00, _sum00, vlmax);
            _sum01 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc01, _sum01, vlmax);
            _sum02 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc02, _sum02, vlmax);
            _sum03 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc03, _sum03, vlmax);

            dst[idx00] = vfmv_f_s_f16m1_f16(_sum00);
            dst[idx01] = vfmv_f_s_f16m1_f16(_sum01);
            dst[idx02] = vfmv_f_s_f16m1_f16(_sum02);
            dst[idx03] = vfmv_f_s_f16m1_f16(_sum03);
        }
        for (; j < N; j++) {
            const __fp16 *a0_ptr = sa_ptr;
            const __fp16 *b0_ptr = sb + j * ldb;

            int vlmax = vsetvl_e16m1(csrr_vlenb() / sizeof(__fp16));
            // dst[0, 0]
            vfloat16m1_t _acc00 = vfmv_v_f_f16m1(0.0f, vlmax);

            int c = 0;
            while (c < K) {
                int vl = vsetvl_e16m1(K - c);
                vfloat16m1_t _a0 = vle16_v_f16m1(a0_ptr + c, vl);
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr + c, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vlmax);
                c += vl;
            }

            int idx00 = (i + 0) * ldc + (j + 0);

            // dst[0, 0]
            vfloat16m1_t _sum00;
            if (k_idx == 0) {
                _sum00 = vfmv_v_f_f16m1(bias[j + 0], 1);
            } else {
                _sum00 = vfmv_v_f_f16m1(dst[idx00], 1);
            }

            _sum00 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc00, _sum00, vlmax);
            dst[idx00] = vfmv_f_s_f16m1_f16(_sum00);
        }
    }
}

static void gemm_dot_4x4_fp16_omp(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, __fp16 *bias,
                                  int M, int K, int N, int lda, int ldb, int ldc, int k_idx)
{
    if (shl_multithread_is_enable()) {
#pragma omp parallel
        {
            int M_start = 0, M_end = M;
            int N_start = 0, N_end = N;
            c920_omp_get_mn_partition(M, N, &M_start, &M_end, &N_start, &N_end);

            __fp16 *thread_dst = dst + M_start * ldc + N_start;
            const __fp16 *thread_sa = sa + M_start * lda;
            const __fp16 *thread_sb = sb + N_start * ldb;
            __fp16 *thread_bias = bias + N_start;
            int thread_M = M_end - M_start;
            int thread_N = N_end - N_start;
            gemm_dot_4x4_fp16(thread_dst, thread_sa, thread_sb, thread_bias, thread_M, K, thread_N,
                              lda, ldb, ldc, k_idx);
        }
    } else {
        gemm_dot_4x4_fp16(dst, sa, sb, bias, M, K, N, lda, ldb, ldc, k_idx);
    }
}

#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define CAL_LAST(SIZE, x, y) ((SIZE - x * y) / (x + y))

static inline void c920_get_blk_size(int M, int N, int K, int *m_blk, int *n_blk, int *k_blk)
{
    const int M_BLK = 256;
    const int N_BLK = 256;
    const int K_BLK = 370;
    const int CACHE_SIZE = 1024 * 1024 / sizeof(__fp16) * 0.75;

    if (M <= M_BLK && N <= N_BLK && K <= K_BLK) {
        *m_blk = M;
        *n_blk = N;
        *k_blk = K;
    } else if (M > M_BLK && N > N_BLK && K > K_BLK) {
        *m_blk = M_BLK;
        *n_blk = N_BLK;
        *k_blk = K_BLK;
    } else {
        if (M <= M_BLK && N <= N_BLK && K > K_BLK) {
            *m_blk = M;
            *n_blk = N;
            *k_blk = MIN(CAL_LAST(CACHE_SIZE, *m_blk, *n_blk), K);
        } else if (M <= M_BLK && N > N_BLK && K <= K_BLK) {
            *m_blk = M;
            *k_blk = K;
            *n_blk = MIN(CAL_LAST(CACHE_SIZE, *m_blk, *k_blk), N);
        } else if (M > M_BLK && N <= N_BLK && K <= K_BLK) {
            *n_blk = N;
            *k_blk = K;
            *m_blk = MIN(CAL_LAST(CACHE_SIZE, *n_blk, *k_blk), M);
        } else if (M > M_BLK && N > N_BLK && K <= K_BLK) {
            *k_blk = K;
            int tmp_m = M_BLK;
            *n_blk = MIN(CAL_LAST(CACHE_SIZE, tmp_m, *k_blk), N);
            *m_blk = MIN(CAL_LAST(CACHE_SIZE, *n_blk, *k_blk), M);
        } else if (M > M_BLK && N <= N_BLK && K > K_BLK) {
            *n_blk = N;
            int tmp_m = M_BLK;
            *k_blk = MIN(CAL_LAST(CACHE_SIZE, tmp_m, *n_blk), K);
            *m_blk = MIN(CAL_LAST(CACHE_SIZE, *n_blk, *k_blk), M);
        } else if (M <= M_BLK && N > N_BLK && K > K_BLK) {
            *m_blk = M;
            int tmp_n = N_BLK;
            *k_blk = MIN(CAL_LAST(CACHE_SIZE, tmp_n, *m_blk), K);
            *n_blk = MIN(CAL_LAST(CACHE_SIZE, *m_blk, *k_blk), N);
        }
    }

    int tmp_n = *n_blk;
    if (tmp_n < N && tmp_n % 4 != 0) {
        *n_blk = (tmp_n / 4) * 4;
    }

    int tmp_k = *k_blk;
    const int block_size = 32;
    if (tmp_k < K && tmp_k % block_size != 0) {
        *k_blk = (tmp_k / block_size) * block_size;
    }
}

/*************************************************************
 * constrain: vlen >= 128, and K % 32 == 0
 ************************************************************/
static void dequantize_block_q8_to_f16(const int8_t *src, const __fp16 *scale, __fp16 *dst,
                                       int n_blk, int k_blk, int ld_src, int ld_dst)
{
    int block_size = 32;
    int vl = vsetvl_e8m2(block_size);
    for (int i = 0; i < n_blk; i++) {
        const int8_t *s_ptr = src + i * ld_src;
        const __fp16 *scale_ptr = scale + i * ld_src / block_size;
        __fp16 *d_ptr = dst + i * ld_dst;
        for (int j = 0; j + block_size - 1 < k_blk; j += block_size) {
            vint8m2_t _i8 = vle8_v_i8m2(s_ptr + j, vl);
            vint16m4_t _i16 = vwadd_vx_i16m4(_i8, 0, vl);
            vfloat16m4_t _f16 = vfcvt_f_x_v_f16m4(_i16, vl);
            _f16 = vfmul_vf_f16m4(_f16, scale_ptr[0], vl);
            scale_ptr += 1;
            vse16_v_f16m4(d_ptr + j, _f16, vl);
        }
    }
}

void shl_c920_gemm_a0nb1n_dot_fp16_q8(__fp16 *dst, const __fp16 *sa, const int8_t *sb, __fp16 *bias,
                                      int M, int K, int N, const __fp16 *scale)
{
    int flag_bias = 1;
    if (bias == NULL) {
        flag_bias = 0;
        bias = (__fp16 *)shl_mem_alloc(N * sizeof(__fp16));
    }

    if (M > 1) {
        int M_BLK, N_BLK, K_BLK;
        c920_get_blk_size(M, N, K, &M_BLK, &N_BLK, &K_BLK);

        __fp16 *b_fp16 = (__fp16 *)shl_mem_alloc(N_BLK * K_BLK * sizeof(__fp16));
        int lda = K;
        int ldb = K_BLK;  // after dequantize
        int ldc = N;

        int m_block = M_BLK;
        int m_idx = 0;
        while (m_idx < M) {
            if (M - m_idx < m_block) {
                m_block = M - m_idx;
            }
            int n_block = N_BLK;
            int n_idx = 0;
            while (n_idx < N) {
                if (N - n_idx < n_block) {
                    n_block = N - n_idx;
                }
                int k_block = K_BLK;
                int k_idx = 0;
                while (k_idx < K) {
                    if (K - k_idx < k_block) {
                        k_block = K - k_idx;
                    }
                    __fp16 *c_ptr = dst + m_idx * N + n_idx;
                    const __fp16 *a_ptr = sa + m_idx * K + k_idx;
                    const int8_t *b_ptr = sb + n_idx * K + k_idx;
                    const __fp16 *scale_ptr = scale + n_idx * (K / 32) + k_idx / 32;

                    // dequantize before gemm
                    dequantize_block_q8_to_f16(b_ptr, scale_ptr, b_fp16, n_block, k_block, K,
                                               K_BLK);
                    gemm_dot_4x4_fp16_omp(c_ptr, a_ptr, b_fp16, bias + n_idx, m_block, k_block,
                                          n_block, lda, ldb, ldc, k_idx);

                    k_idx += k_block;
                }
                n_idx += n_block;
            }
            m_idx += m_block;
        }

        shl_mem_free(b_fp16);
    } else {
        int lda = K;
        int ldb = K;
        int ldc = N;
        // dequantize in gemm
        gemm_dot_1x1_fp16_q8_omp(dst, sa, sb, scale, bias, M, K, N, lda, ldb, ldc, 0);
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}

/*************************************************************
 * constrain: vlen >= 128, and K % 32 == 0
 ************************************************************/
static void dequantize_block_q4_to_f16(const int8_t *src, const __fp16 *scale, __fp16 *dst,
                                       int n_blk, int k_blk, int ld_src, int ld_dst)
{
    int block_size = 32;
    int half_block = block_size / 2;
    int vl = vsetvl_e8m1(half_block);
    for (int i = 0; i < n_blk; i++) {
        const int8_t *s_ptr = src + i * ld_src / 2;
        const __fp16 *scale_ptr = scale + i * ld_src / block_size;
        __fp16 *d_ptr = dst + i * ld_dst;
        for (int j = 0; j + block_size - 1 < k_blk; j += block_size) {
            vint8m1_t _in = vle8_v_i8m1(s_ptr, vl);
            s_ptr += half_block;
            vint8m1_t _low_i8 = vand_vx_i8m1(_in, 0x0f, vl);
            vint8m1_t _high_i8 = vsra_vx_i8m1(_in, 4, vl);
            _high_i8 = vand_vx_i8m1(_high_i8, 0x0f, vl);
            vint16m2_t _low_i16 = vwsub_vx_i16m2(_low_i8, 8, vl);
            vint16m2_t _high_i16 = vwsub_vx_i16m2(_high_i8, 8, vl);
            vfloat16m2_t _low_f16 = vfcvt_f_x_v_f16m2(_low_i16, vl);
            vfloat16m2_t _high_f16 = vfcvt_f_x_v_f16m2(_high_i16, vl);
            _low_f16 = vfmul_vf_f16m2(_low_f16, scale_ptr[0], vl);
            _high_f16 = vfmul_vf_f16m2(_high_f16, scale_ptr[0], vl);
            scale_ptr += 1;
            vse16_v_f16m2(d_ptr, _low_f16, vl);
            vse16_v_f16m2(d_ptr + half_block, _high_f16, vl);
            d_ptr += block_size;
        }
    }
}

void shl_c920_gemm_a0nb1n_dot_fp16_q4(__fp16 *dst, const __fp16 *sa, const int8_t *sb, __fp16 *bias,
                                      int M, int K, int N, const __fp16 *scale)
{
    int flag_bias = 1;
    if (bias == NULL) {
        flag_bias = 0;
        bias = (__fp16 *)shl_mem_alloc(N * sizeof(__fp16));
    }

    if (M > 1) {
        int M_BLK, N_BLK, K_BLK;
        c920_get_blk_size(M, N, K, &M_BLK, &N_BLK, &K_BLK);

        __fp16 *b_fp16 = (__fp16 *)shl_mem_alloc(N_BLK * K_BLK * sizeof(__fp16));
        int lda = K;
        int ldb = K_BLK;  // after dequantize
        int ldc = N;

        int m_block = M_BLK;
        int m_idx = 0;
        while (m_idx < M) {
            if (M - m_idx < m_block) {
                m_block = M - m_idx;
            }

            int n_block = N_BLK;
            int n_idx = 0;
            while (n_idx < N) {
                if (N - n_idx < n_block) {
                    n_block = N - n_idx;
                }

                int k_block = K_BLK;
                int k_idx = 0;
                while (k_idx < K) {
                    if (K - k_idx < k_block) {
                        k_block = K - k_idx;
                    }

                    __fp16 *c_ptr = dst + m_idx * N + n_idx;
                    const __fp16 *a_ptr = sa + m_idx * K + k_idx;
                    const int8_t *b_ptr = sb + n_idx * K / 2 + k_idx / 2;
                    const __fp16 *scale_ptr = scale + n_idx * (K / 32) + k_idx / 32;

                    // dequantize before gemm
                    dequantize_block_q4_to_f16(b_ptr, scale_ptr, b_fp16, n_block, k_block, K,
                                               K_BLK);
                    gemm_dot_4x4_fp16_omp(c_ptr, a_ptr, b_fp16, bias + n_idx, m_block, k_block,
                                          n_block, lda, ldb, ldc, k_idx);

                    k_idx += k_block;
                }

                n_idx += n_block;
            }

            m_idx += m_block;
        }
        shl_mem_free(b_fp16);
    } else {
        int lda = K;
        int ldb = K;
        int ldc = N;
        // dequantize in gemm
        gemm_dot_1x1_fp16_q4_omp(dst, sa, sb, scale, bias, M, K, N, lda, ldb, ldc, 0);
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
