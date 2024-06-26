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

#include "rvv/rvv.h"

#ifdef SHL_USE_DOT_INT8
static vint8mf2_t requantize_m2(vint32m2_t _src, int32_t *mult, int32_t *shift, int32_t out_zp,
                                int vl)
{
    vint32m2_t _mult = vle32_v_i32m2(mult, vl);
    vint32m2_t _shift = vle32_v_i32m2(shift, vl);
    vint32m2_t _mulh = vmulh_vv_i32m2(_src, _mult, vl);
    _mulh = vssra_vv_i32m2(_mulh, vreinterpret_v_i32m2_u32m2(_shift), vl);
    _mulh = vadd_vx_i32m2(_mulh, out_zp, vl);
    vint16m1_t _res0 = vnclip_wx_i16m1(_mulh, 0, vl);
    vint8mf2_t _res1 = vnclip_wx_i8mf2(_res0, 0, vl);
    return _res1;
}

/*************************************************************
 * mf2 = vlenb / sizeof(int8_t) / 2
 * dst - output: [M, N]
 * sa - mat0
 *   K % 4 == 0: [M/8, K/4, 8, 4]
 *   k_tail    : [M/8, k_tail, 8]
 * sb - mat1
 *   K % 4 == 0: [N/mf2, K/4, mf2, 4]
 *   k_tail    : [N/mf2, k_tail, mf2]
 *************************************************************/
void shl_rvv_gemm_a0b1_8xmf2_int8_dot(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                      const int32_t *bias, int M, int K, int N, int32_t out_zp,
                                      int32_t *mult, int32_t *shift)
{
    const int m2 = csrr_vlenb() / sizeof(int8_t) * 2;

    int i = 0;
    for (; i + 7 < M; i += 8) {
        const int8_t *sa_ptr = sa + i * K;
        int j = 0;
        while (j < N) {
            int vl = vsetvl_e8mf2(N - j);
            const int32_t *a32_ptr = (int32_t *)sa_ptr;
            const int8_t *b_ptr = sb + j * K;
            int8_t *c_ptr = dst + i * N + j;

            vint32m2_t _acc0 = vle32_v_i32m2(bias + j, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc2 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc3 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc4 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc5 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc6 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc7 = vmv_v_v_i32m2(_acc0, vl);

            int c = 0;
            for (; c + 3 < K; c += 4) {
                vint8m2_t _b = vle8_v_i8m2(b_ptr, vl * 4);
                b_ptr += vl * 4;

                _acc0 = vmaqa_vx_i32m2(_acc0, a32_ptr[0], _b, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, a32_ptr[1], _b, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, a32_ptr[2], _b, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, a32_ptr[3], _b, vl);
                _acc4 = vmaqa_vx_i32m2(_acc4, a32_ptr[4], _b, vl);
                _acc5 = vmaqa_vx_i32m2(_acc5, a32_ptr[5], _b, vl);
                _acc6 = vmaqa_vx_i32m2(_acc6, a32_ptr[6], _b, vl);
                _acc7 = vmaqa_vx_i32m2(_acc7, a32_ptr[7], _b, vl);
                a32_ptr += 8;
            }

            const int8_t *a_ptr = sa_ptr + 8 * c;
            for (; c < K; c++) {
                vint8mf2_t _b = vle8_v_i8mf2(b_ptr, vl);
                b_ptr += vl;

                vint16m1_t _mul0 = vwmul_vx_i16m1(_b, a_ptr[0], vl);
                vint16m1_t _mul1 = vwmul_vx_i16m1(_b, a_ptr[1], vl);
                vint16m1_t _mul2 = vwmul_vx_i16m1(_b, a_ptr[2], vl);
                vint16m1_t _mul3 = vwmul_vx_i16m1(_b, a_ptr[3], vl);
                vint16m1_t _mul4 = vwmul_vx_i16m1(_b, a_ptr[4], vl);
                vint16m1_t _mul5 = vwmul_vx_i16m1(_b, a_ptr[5], vl);
                vint16m1_t _mul6 = vwmul_vx_i16m1(_b, a_ptr[6], vl);
                vint16m1_t _mul7 = vwmul_vx_i16m1(_b, a_ptr[7], vl);
                a_ptr += 8;

                _acc0 = vwmacc_vx_i32m2(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m2(_acc1, 1, _mul1, vl);
                _acc2 = vwmacc_vx_i32m2(_acc2, 1, _mul2, vl);
                _acc3 = vwmacc_vx_i32m2(_acc3, 1, _mul3, vl);
                _acc4 = vwmacc_vx_i32m2(_acc4, 1, _mul4, vl);
                _acc5 = vwmacc_vx_i32m2(_acc5, 1, _mul5, vl);
                _acc6 = vwmacc_vx_i32m2(_acc6, 1, _mul6, vl);
                _acc7 = vwmacc_vx_i32m2(_acc7, 1, _mul7, vl);
            }

            vint8mf2_t _res0 = requantize_m2(_acc0, mult + j, shift + j, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2(_acc1, mult + j, shift + j, out_zp, vl);
            vint8mf2_t _res2 = requantize_m2(_acc2, mult + j, shift + j, out_zp, vl);
            vint8mf2_t _res3 = requantize_m2(_acc3, mult + j, shift + j, out_zp, vl);
            vint8mf2_t _res4 = requantize_m2(_acc4, mult + j, shift + j, out_zp, vl);
            vint8mf2_t _res5 = requantize_m2(_acc5, mult + j, shift + j, out_zp, vl);
            vint8mf2_t _res6 = requantize_m2(_acc6, mult + j, shift + j, out_zp, vl);
            vint8mf2_t _res7 = requantize_m2(_acc7, mult + j, shift + j, out_zp, vl);

            vse8_v_i8mf2(c_ptr, _res0, vl);
            vse8_v_i8mf2(c_ptr + N, _res1, vl);
            vse8_v_i8mf2(c_ptr + N * 2, _res2, vl);
            vse8_v_i8mf2(c_ptr + N * 3, _res3, vl);
            vse8_v_i8mf2(c_ptr + N * 4, _res4, vl);
            vse8_v_i8mf2(c_ptr + N * 5, _res5, vl);
            vse8_v_i8mf2(c_ptr + N * 6, _res6, vl);
            vse8_v_i8mf2(c_ptr + N * 7, _res7, vl);
            j += vl;
        }
    }
    for (; i + 3 < M; i += 4) {
        const int8_t *sa_ptr = sa + i * K;
        int j = 0;
        while (j < N) {
            int vl = vsetvl_e8mf2(N - j);
            const int32_t *a32_ptr = (int32_t *)sa_ptr;
            const int8_t *b_ptr = sb + j * K;
            int8_t *c_ptr = dst + i * N + j;

            vint32m2_t _acc0 = vle32_v_i32m2(bias + j, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc2 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc3 = vmv_v_v_i32m2(_acc0, vl);

            int c = 0;
            for (; c + 3 < K; c += 4) {
                vint8m2_t _b = vle8_v_i8m2(b_ptr, vl * 4);
                b_ptr += vl * 4;

                _acc0 = vmaqa_vx_i32m2(_acc0, a32_ptr[0], _b, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, a32_ptr[1], _b, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, a32_ptr[2], _b, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, a32_ptr[3], _b, vl);
                a32_ptr += 4;
            }

            const int8_t *a_ptr = sa_ptr + 4 * c;
            for (; c < K; c++) {
                vint8mf2_t _b = vle8_v_i8mf2(b_ptr, vl);
                b_ptr += vl;

                vint16m1_t _mul0 = vwmul_vx_i16m1(_b, a_ptr[0], vl);
                vint16m1_t _mul1 = vwmul_vx_i16m1(_b, a_ptr[1], vl);
                vint16m1_t _mul2 = vwmul_vx_i16m1(_b, a_ptr[2], vl);
                vint16m1_t _mul3 = vwmul_vx_i16m1(_b, a_ptr[3], vl);
                a_ptr += 4;

                _acc0 = vwmacc_vx_i32m2(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m2(_acc1, 1, _mul1, vl);
                _acc2 = vwmacc_vx_i32m2(_acc2, 1, _mul2, vl);
                _acc3 = vwmacc_vx_i32m2(_acc3, 1, _mul3, vl);
            }

            vint8mf2_t _res0 = requantize_m2(_acc0, mult + j, shift + j, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2(_acc1, mult + j, shift + j, out_zp, vl);
            vint8mf2_t _res2 = requantize_m2(_acc2, mult + j, shift + j, out_zp, vl);
            vint8mf2_t _res3 = requantize_m2(_acc3, mult + j, shift + j, out_zp, vl);

            vse8_v_i8mf2(c_ptr, _res0, vl);
            vse8_v_i8mf2(c_ptr + N, _res1, vl);
            vse8_v_i8mf2(c_ptr + N * 2, _res2, vl);
            vse8_v_i8mf2(c_ptr + N * 3, _res3, vl);
            j += vl;
        }
    }
    for (; i + 1 < M; i += 2) {
        const int8_t *sa_ptr = sa + i * K;
        int j = 0;
        while (j < N) {
            int vl = vsetvl_e8mf2(N - j);
            const int32_t *a32_ptr = (int32_t *)sa_ptr;
            const int8_t *b_ptr = sb + j * K;
            int8_t *c_ptr = dst + i * N + j;

            vint32m2_t _acc0 = vle32_v_i32m2(bias + j, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);

            int c = 0;
            for (; c + 3 < K; c += 4) {
                vint8m2_t _b = vle8_v_i8m2(b_ptr, vl * 4);
                b_ptr += vl * 4;

                _acc0 = vmaqa_vx_i32m2(_acc0, a32_ptr[0], _b, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, a32_ptr[1], _b, vl);
                a32_ptr += 2;
            }

            const int8_t *a_ptr = sa_ptr + 2 * c;
            for (; c < K; c++) {
                vint8mf2_t _b = vle8_v_i8mf2(b_ptr, vl);
                b_ptr += vl;

                vint16m1_t _mul0 = vwmul_vx_i16m1(_b, a_ptr[0], vl);
                vint16m1_t _mul1 = vwmul_vx_i16m1(_b, a_ptr[1], vl);
                a_ptr += 2;

                _acc0 = vwmacc_vx_i32m2(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m2(_acc1, 1, _mul1, vl);
            }

            vint8mf2_t _res0 = requantize_m2(_acc0, mult + j, shift + j, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2(_acc1, mult + j, shift + j, out_zp, vl);

            vse8_v_i8mf2(c_ptr, _res0, vl);
            vse8_v_i8mf2(c_ptr + N, _res1, vl);
            j += vl;
        }
    }
    for (; i < M; i++) {
        const int8_t *sa_ptr = sa + i * K;
        int j = 0;
        while (j < N) {
            int vl = vsetvl_e8mf2(N - j);
            const int32_t *a32_ptr = (int32_t *)sa_ptr;
            const int8_t *b_ptr = sb + j * K;
            int8_t *c_ptr = dst + i * N + j;

            vint32m2_t _acc0 = vle32_v_i32m2(bias + j, vl);

            int c = 0;
            for (; c + 3 < K; c += 4) {
                vint8m2_t _b = vle8_v_i8m2(b_ptr, vl * 4);
                b_ptr += vl * 4;

                _acc0 = vmaqa_vx_i32m2(_acc0, a32_ptr[0], _b, vl);
                a32_ptr += 1;
            }

            const int8_t *a_ptr = sa_ptr + 1 * c;
            for (; c < K; c++) {
                vint8mf2_t _b = vle8_v_i8mf2(b_ptr, vl);
                b_ptr += vl;

                vint16m1_t _mul0 = vwmul_vx_i16m1(_b, a_ptr[0], vl);
                a_ptr += 1;

                _acc0 = vwmacc_vx_i32m2(_acc0, 1, _mul0, vl);
            }

            vint8mf2_t _res0 = requantize_m2(_acc0, mult + j, shift + j, out_zp, vl);

            vse8_v_i8mf2(c_ptr, _res0, vl);
            j += vl;
        }
    }
}
#else

static vint8m1_t requantize_m4(vint32m4_t _src, int32_t *mult, int32_t *shift, int32_t out_zp,
                               int vl)
{
    vint32m4_t _mult = vle32_v_i32m4(mult, vl);
    vint32m4_t _shift = vle32_v_i32m4(shift, vl);
    vint32m4_t _mulh = vmulh_vv_i32m4(_src, _mult, vl);
    _mulh = vssra_vv_i32m4(_mulh, vreinterpret_v_i32m4_u32m4(_shift), vl);
    _mulh = vadd_vx_i32m4(_mulh, out_zp, vl);
    vint16m2_t _res0 = vnclip_wx_i16m2(_mulh, 0, vl);
    vint8m1_t _res1 = vnclip_wx_i8m1(_res0, 0, vl);
    return _res1;
}

/*************************************************************
 * packn = vlenb / sizeof(int8_t)
 * m_blk: 4/2/1
 * n_blk: packn/n_tail
 *
 * dst - output: [M, N]
 * sa - input:   [M/m_blk, K, m_blk]
 * sb - weights: [N/n_blk, K, n_blk]
 * bias:         [N]
 *************************************************************/
void shl_rvv_gemm_a0b1_4xpackn_int8(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                    const int32_t *bias, int M, int K, int N, int32_t out_zp,
                                    int32_t *mult, int32_t *shift)
{
    int i = 0;
    for (; i + 3 < M; i += 4) {
        const int8_t *sa_ptr = sa + i * K;
        int j = 0;
        while (j < N) {
            int vl = vsetvl_e8m1(N - j);
            const int8_t *a_ptr = sa_ptr;
            const int8_t *b_ptr = sb + j * K;
            int8_t *c_ptr = dst + i * N + j;

            vint32m4_t _acc0 = vle32_v_i32m4(bias + j, vl);
            vint32m4_t _acc1 = vmv_v_v_i32m4(_acc0, vl);
            vint32m4_t _acc2 = vmv_v_v_i32m4(_acc0, vl);
            vint32m4_t _acc3 = vmv_v_v_i32m4(_acc0, vl);

            for (int k = 0; k < K; k++) {
                vint8m1_t _b = vle8_v_i8m1(b_ptr, vl);
                b_ptr += vl;

                vint16m2_t _mul0 = vwmul_vx_i16m2(_b, a_ptr[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_b, a_ptr[1], vl);
                vint16m2_t _mul2 = vwmul_vx_i16m2(_b, a_ptr[2], vl);
                vint16m2_t _mul3 = vwmul_vx_i16m2(_b, a_ptr[3], vl);
                a_ptr += 4;

                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                _acc2 = vwmacc_vx_i32m4(_acc2, 1, _mul2, vl);
                _acc3 = vwmacc_vx_i32m4(_acc3, 1, _mul3, vl);
            }

            vint8m1_t _res0 = requantize_m4(_acc0, mult + j, shift + j, out_zp, vl);
            vint8m1_t _res1 = requantize_m4(_acc1, mult + j, shift + j, out_zp, vl);
            vint8m1_t _res2 = requantize_m4(_acc2, mult + j, shift + j, out_zp, vl);
            vint8m1_t _res3 = requantize_m4(_acc3, mult + j, shift + j, out_zp, vl);
            vse8_v_i8m1(c_ptr, _res0, vl);
            vse8_v_i8m1(c_ptr + N, _res1, vl);
            vse8_v_i8m1(c_ptr + N * 2, _res2, vl);
            vse8_v_i8m1(c_ptr + N * 3, _res3, vl);
            j += vl;
        }
    }
    for (; i + 1 < M; i += 2) {
        const int8_t *sa_ptr = sa + i * K;
        int j = 0;
        while (j < N) {
            int vl = vsetvl_e8m1(N - j);
            const int8_t *a_ptr = sa_ptr;
            const int8_t *b_ptr = sb + j * K;
            int8_t *c_ptr = dst + i * N + j;

            vint32m4_t _acc0 = vle32_v_i32m4(bias + j, vl);
            vint32m4_t _acc1 = vmv_v_v_i32m4(_acc0, vl);

            for (int k = 0; k < K; k++) {
                vint8m1_t _b = vle8_v_i8m1(b_ptr, vl);
                b_ptr += vl;

                vint16m2_t _mul0 = vwmul_vx_i16m2(_b, a_ptr[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_b, a_ptr[1], vl);
                a_ptr += 2;

                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
            }

            vint8m1_t _res0 = requantize_m4(_acc0, mult + j, shift + j, out_zp, vl);
            vint8m1_t _res1 = requantize_m4(_acc1, mult + j, shift + j, out_zp, vl);
            vse8_v_i8m1(c_ptr, _res0, vl);
            vse8_v_i8m1(c_ptr + N, _res1, vl);
            j += vl;
        }
    }
    for (; i < M; i += 1) {
        const int8_t *sa_ptr = sa + i * K;
        int j = 0;
        while (j < N) {
            int vl = vsetvl_e8m1(N - j);
            const int8_t *a_ptr = sa_ptr;
            const int8_t *b_ptr = sb + j * K;
            int8_t *c_ptr = dst + i * N + j;

            vint32m4_t _acc0 = vle32_v_i32m4(bias + j, vl);

            for (int k = 0; k < K; k++) {
                vint8m1_t _b = vle8_v_i8m1(b_ptr, vl);
                b_ptr += vl;

                vint16m2_t _mul0 = vwmul_vx_i16m2(_b, a_ptr[0], vl);
                a_ptr += 1;

                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
            }

            vint8m1_t _res0 = requantize_m4(_acc0, mult + j, shift + j, out_zp, vl);
            vse8_v_i8m1(c_ptr, _res0, vl);
            j += vl;
        }
    }
}
#endif  // SHL_USE_DOT_INT8
