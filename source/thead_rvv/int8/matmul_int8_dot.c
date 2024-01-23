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

#include "rvv/rvv.h"

#ifdef SHL_USE_DOT_INT8
/*************************************************************
 * src: [m, k]
 * dst:
 *   k % 4 == 0: [m/8, k/4, 8, 4]
 *   k_tail    : [m/8, k_tail, 8]
 ************************************************************/
void shl_rvv_matmul_reorder_mat0_n8z4_int8_dot(int8_t *src, int8_t *dst, int m, int k, int lda)
{
    int i = 0;
    for (; i + 7 < m; i += 8) {
        int j = 0;
        for (; j + 3 < k; j += 4) {
            int8_t *s_ptr = src + j;
            for (int c = 0; c < 8; c++) {
                vint8m1_t _src = vle8_v_i8m1(s_ptr, 4);
                s_ptr += lda;
                vse8_v_i8m1(dst, _src, 4);
                dst += 4;
            }
        }
        // k_tail
        if (j < k) {
            int8_t *s_ptr = src + j;
            int stride = 8 * sizeof(int8_t);
            int vl = vsetvl_e8m1(k - j);
            vint8m1_t _s0 = vle8_v_i8m1(s_ptr, vl);
            vint8m1_t _s1 = vle8_v_i8m1(s_ptr + lda, vl);
            vint8m1_t _s2 = vle8_v_i8m1(s_ptr + lda * 2, vl);
            vint8m1_t _s3 = vle8_v_i8m1(s_ptr + lda * 3, vl);
            vint8m1_t _s4 = vle8_v_i8m1(s_ptr + lda * 4, vl);
            vint8m1_t _s5 = vle8_v_i8m1(s_ptr + lda * 5, vl);
            vint8m1_t _s6 = vle8_v_i8m1(s_ptr + lda * 6, vl);
            vint8m1_t _s7 = vle8_v_i8m1(s_ptr + lda * 7, vl);
            vsse8_v_i8m1(dst, stride, _s0, vl);
            vsse8_v_i8m1(dst + 1, stride, _s1, vl);
            vsse8_v_i8m1(dst + 2, stride, _s2, vl);
            vsse8_v_i8m1(dst + 3, stride, _s3, vl);
            vsse8_v_i8m1(dst + 4, stride, _s4, vl);
            vsse8_v_i8m1(dst + 5, stride, _s5, vl);
            vsse8_v_i8m1(dst + 6, stride, _s6, vl);
            vsse8_v_i8m1(dst + 7, stride, _s7, vl);
            s_ptr += vl;
            dst += vl * 8;
            j += vl;
        }
        src += 8 * k;
    }
    for (; i + 3 < m; i += 4) {
        int j = 0;
        for (; j + 3 < k; j += 4) {
            int8_t *s_ptr = src + j;
            for (int c = 0; c < 4; c++) {
                vint8m1_t _src = vle8_v_i8m1(s_ptr, 4);
                s_ptr += lda;
                vse8_v_i8m1(dst, _src, 4);
                dst += 4;
            }
        }
        if (j < k) {
            int8_t *s_ptr = src + j;
            int stride = 4 * sizeof(int8_t);
            int vl = vsetvl_e8m1(k - j);
            vint8m1_t _s0 = vle8_v_i8m1(s_ptr, vl);
            vint8m1_t _s1 = vle8_v_i8m1(s_ptr + lda, vl);
            vint8m1_t _s2 = vle8_v_i8m1(s_ptr + lda * 2, vl);
            vint8m1_t _s3 = vle8_v_i8m1(s_ptr + lda * 3, vl);
            vsse8_v_i8m1(dst, stride, _s0, vl);
            vsse8_v_i8m1(dst + 1, stride, _s1, vl);
            vsse8_v_i8m1(dst + 2, stride, _s2, vl);
            vsse8_v_i8m1(dst + 3, stride, _s3, vl);
            s_ptr += vl;
            dst += vl * 4;
            j += vl;
        }
        src += 4 * k;
    }
    for (; i + 1 < m; i += 2) {
        int j = 0;
        for (; j + 3 < k; j += 4) {
            int8_t *s_ptr = src + j;
            for (int c = 0; c < 2; c++) {
                vint8m1_t _src = vle8_v_i8m1(s_ptr, 4);
                s_ptr += lda;
                vse8_v_i8m1(dst, _src, 4);
                dst += 4;
            }
        }
        if (j < k) {
            int8_t *s_ptr = src + j;
            int stride = 2 * sizeof(int8_t);
            int vl = vsetvl_e8m1(k - j);
            vint8m1_t _s0 = vle8_v_i8m1(s_ptr, vl);
            vint8m1_t _s1 = vle8_v_i8m1(s_ptr + lda, vl);
            vsse8_v_i8m1(dst, stride, _s0, vl);
            vsse8_v_i8m1(dst + 1, stride, _s1, vl);
            s_ptr += vl;
            dst += vl * 2;
            j += vl;
        }
        src += 2 * k;
    }
    for (; i < m; i++) {
        memcpy(dst, src, k * sizeof(int8_t));
    }
}

/*************************************************************
 * mf2 = vlenb / sizeof(int8_t) / 2
 * src: [k, n]
 * dst:
 *   k % 4 == 0: [n/mf2, k/4, mf2, 4]
 *   k_tail    : [n/mf2, k_tail, mf2]
 ************************************************************/
void shl_rvv_matmul_reorder_mat1_zmf2n4_int8_dot(int8_t *src, int8_t *dst, int k, int n, int ldb)
{
    int j = 0;
    while (j < n) {
        int vl = vsetvl_e8mf2(n - j);
        int8_t *s_ptr = src + j;
        int c = 0;
        for (; c + 3 < k; c += 4) {
            for (int i = 0; i < 4; i++) {
                vint8mf2_t _src = vle8_v_i8mf2(s_ptr, vl);
                s_ptr += ldb;
                vsse8_v_i8mf2(dst + i, 4 * sizeof(int8_t), _src, vl);
            }
            dst += 4 * vl;
        }
        // k_tail
        for (; c < k; c++) {
            vint8m1_t _src = vle8_v_i8m1(s_ptr, vl);
            vse8_v_i8m1(dst, _src, vl);
            s_ptr += ldb;
            dst += vl;
        }
        j += vl;
    }
}

static vint8mf2_t requantize_m2(vint32m2_t _src, int32_t multiplier, int32_t shift, int32_t out_zp,
                                int vl)
{
    vint32m2_t _mulh = vmulh_vx_i32m2(_src, multiplier, vl);
    _mulh = vssra_vx_i32m2(_mulh, -shift - 1, vl);
    _mulh = vadd_vx_i32m2(_mulh, out_zp, vl);
    vint16m1_t _tmp1 = vnclip_wx_i16m1(_mulh, 0, vl);
    vint8mf2_t _tmp2 = vnclip_wx_i8mf2(_tmp1, 0, vl);
    return _tmp2;
}

/*************************************************************
 * mf2 = vlenb / sizeof(int8_t) / 2
 * dst - output: [m, n]
 * sa - mat0
 *   k % 4 == 0: [m/8, k/4, 8, 4]
 *   k_tail    : [m/8, k_tail, 8]
 * sb - mat1
 *   k % 4 == 0: [n/mf2, k/4, mf2, 4]
 *   k_tail    : [n/mf2, k_tail, mf2]
 ************************************************************/
void shl_rvv_matmul_8xmf2_int8_dot(int8_t *dst, const int8_t *sa, const int8_t *sb, int m, int k,
                                   int n, int ldc, int32_t z1, int32_t z2, int32_t z3, int32_t mult,
                                   int32_t shift)
{
    const int8_t *kernel_data = sa;
    const int8_t *input_data = sb;
    int8_t *output_data = dst;

    const int m2 = csrr_vlenb() / sizeof(int8_t) * 2;
    int8_t z1_i8 = (int8_t)-z1;
    int8_t z1_i8_4[4] = {z1_i8, z1_i8, z1_i8, z1_i8};
    int32_t *z1_i32 = (int32_t *)z1_i8_4;
    vint8m2_t _z2_i8 = vmv_v_x_i8m2((int8_t)-z2, m2);
    int32_t z1z2 = z1 * z2;

    int i = 0;
    for (; i + 7 < m; i += 8) {
        const int8_t *kernel_ptr = kernel_data + i * k;
        int j = 0;
        while (j < n) {
            int vl = vsetvl_e8mf2(n - j);
            const int32_t *k32_ptr = (int32_t *)kernel_ptr;
            const int8_t *in_ptr = input_data + j * k;
            int8_t *out_ptr = output_data + i * ldc + j;

            vint32m2_t _acc0 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _acc1 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _acc2 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _acc3 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _acc4 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _acc5 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _acc6 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _acc7 = vmv_v_x_i32m2(0, vl);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _in = vle8_v_i8m2(in_ptr, vl * 4);
                in_ptr += vl * 4;

                // q1 * q2
                _acc0 = vmaqa_vx_i32m2(_acc0, k32_ptr[0], _in, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, k32_ptr[1], _in, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, k32_ptr[2], _in, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, k32_ptr[3], _in, vl);
                _acc4 = vmaqa_vx_i32m2(_acc4, k32_ptr[4], _in, vl);
                _acc5 = vmaqa_vx_i32m2(_acc5, k32_ptr[5], _in, vl);
                _acc6 = vmaqa_vx_i32m2(_acc6, k32_ptr[6], _in, vl);
                _acc7 = vmaqa_vx_i32m2(_acc7, k32_ptr[7], _in, vl);
                // - z1 * q2
                _acc0 = vmaqa_vx_i32m2(_acc0, z1_i32[0], _in, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, z1_i32[0], _in, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, z1_i32[0], _in, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, z1_i32[0], _in, vl);
                _acc4 = vmaqa_vx_i32m2(_acc4, z1_i32[0], _in, vl);
                _acc5 = vmaqa_vx_i32m2(_acc5, z1_i32[0], _in, vl);
                _acc6 = vmaqa_vx_i32m2(_acc6, z1_i32[0], _in, vl);
                _acc7 = vmaqa_vx_i32m2(_acc7, z1_i32[0], _in, vl);
                // - z2 * q1
                _acc0 = vmaqa_vx_i32m2(_acc0, k32_ptr[0], _z2_i8, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, k32_ptr[1], _z2_i8, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, k32_ptr[2], _z2_i8, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, k32_ptr[3], _z2_i8, vl);
                _acc4 = vmaqa_vx_i32m2(_acc4, k32_ptr[4], _z2_i8, vl);
                _acc5 = vmaqa_vx_i32m2(_acc5, k32_ptr[5], _z2_i8, vl);
                _acc6 = vmaqa_vx_i32m2(_acc6, k32_ptr[6], _z2_i8, vl);
                _acc7 = vmaqa_vx_i32m2(_acc7, k32_ptr[7], _z2_i8, vl);
                k32_ptr += 8;
            }
            // + z1 * z2
            int32_t acc_z1z2 = c * z1z2;
            _acc0 = vadd_vx_i32m2(_acc0, acc_z1z2, vl);
            _acc1 = vadd_vx_i32m2(_acc1, acc_z1z2, vl);
            _acc2 = vadd_vx_i32m2(_acc2, acc_z1z2, vl);
            _acc3 = vadd_vx_i32m2(_acc3, acc_z1z2, vl);
            _acc4 = vadd_vx_i32m2(_acc4, acc_z1z2, vl);
            _acc5 = vadd_vx_i32m2(_acc5, acc_z1z2, vl);
            _acc6 = vadd_vx_i32m2(_acc6, acc_z1z2, vl);
            _acc7 = vadd_vx_i32m2(_acc7, acc_z1z2, vl);

            const int8_t *k_ptr = kernel_ptr + 8 * c;
            for (; c < k; c++) {
                vint8mf2_t _in = vle8_v_i8mf2(in_ptr, vl);
                vint16m1_t _in_w = vwsub_vx_i16m1(_in, z2, vl);
                in_ptr += vl;

                _acc0 = vwmacc_vx_i32m2(_acc0, k_ptr[0] - z1, _in_w, vl);
                _acc1 = vwmacc_vx_i32m2(_acc1, k_ptr[1] - z1, _in_w, vl);
                _acc2 = vwmacc_vx_i32m2(_acc2, k_ptr[2] - z1, _in_w, vl);
                _acc3 = vwmacc_vx_i32m2(_acc3, k_ptr[3] - z1, _in_w, vl);
                _acc4 = vwmacc_vx_i32m2(_acc4, k_ptr[4] - z1, _in_w, vl);
                _acc5 = vwmacc_vx_i32m2(_acc5, k_ptr[5] - z1, _in_w, vl);
                _acc6 = vwmacc_vx_i32m2(_acc6, k_ptr[6] - z1, _in_w, vl);
                _acc7 = vwmacc_vx_i32m2(_acc7, k_ptr[7] - z1, _in_w, vl);
                k_ptr += 8;
            }

            vint8mf2_t _res0 = requantize_m2(_acc0, mult, shift, z3, vl);
            vint8mf2_t _res1 = requantize_m2(_acc1, mult, shift, z3, vl);
            vint8mf2_t _res2 = requantize_m2(_acc2, mult, shift, z3, vl);
            vint8mf2_t _res3 = requantize_m2(_acc3, mult, shift, z3, vl);
            vint8mf2_t _res4 = requantize_m2(_acc4, mult, shift, z3, vl);
            vint8mf2_t _res5 = requantize_m2(_acc5, mult, shift, z3, vl);
            vint8mf2_t _res6 = requantize_m2(_acc6, mult, shift, z3, vl);
            vint8mf2_t _res7 = requantize_m2(_acc7, mult, shift, z3, vl);

            vse8_v_i8mf2(out_ptr, _res0, vl);
            vse8_v_i8mf2(out_ptr + ldc, _res1, vl);
            vse8_v_i8mf2(out_ptr + ldc * 2, _res2, vl);
            vse8_v_i8mf2(out_ptr + ldc * 3, _res3, vl);
            vse8_v_i8mf2(out_ptr + ldc * 4, _res4, vl);
            vse8_v_i8mf2(out_ptr + ldc * 5, _res5, vl);
            vse8_v_i8mf2(out_ptr + ldc * 6, _res6, vl);
            vse8_v_i8mf2(out_ptr + ldc * 7, _res7, vl);
            j += vl;
        }
    }
    for (; i + 3 < m; i += 4) {
        const int8_t *kernel_ptr = kernel_data + i * k;
        int j = 0;
        while (j < n) {
            int vl = vsetvl_e8mf2(n - j);
            const int32_t *k32_ptr = (int32_t *)kernel_ptr;
            const int8_t *in_ptr = input_data + j * k;
            int8_t *out_ptr = output_data + i * ldc + j;

            vint32m2_t _acc0 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _acc1 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _acc2 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _acc3 = vmv_v_x_i32m2(0, vl);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _in = vle8_v_i8m2(in_ptr, vl * 4);
                in_ptr += vl * 4;

                // q1 * q2
                _acc0 = vmaqa_vx_i32m2(_acc0, k32_ptr[0], _in, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, k32_ptr[1], _in, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, k32_ptr[2], _in, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, k32_ptr[3], _in, vl);
                // - z1 * q2
                _acc0 = vmaqa_vx_i32m2(_acc0, z1_i32[0], _in, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, z1_i32[0], _in, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, z1_i32[0], _in, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, z1_i32[0], _in, vl);
                // - z2 * q1
                _acc0 = vmaqa_vx_i32m2(_acc0, k32_ptr[0], _z2_i8, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, k32_ptr[1], _z2_i8, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, k32_ptr[2], _z2_i8, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, k32_ptr[3], _z2_i8, vl);
                k32_ptr += 4;
            }
            // + z1 * z2
            int32_t acc_z1z2 = c * z1z2;
            _acc0 = vadd_vx_i32m2(_acc0, acc_z1z2, vl);
            _acc1 = vadd_vx_i32m2(_acc1, acc_z1z2, vl);
            _acc2 = vadd_vx_i32m2(_acc2, acc_z1z2, vl);
            _acc3 = vadd_vx_i32m2(_acc3, acc_z1z2, vl);

            const int8_t *k_ptr = kernel_ptr + 4 * c;
            for (; c < k; c++) {
                vint8mf2_t _in = vle8_v_i8mf2(in_ptr, vl);
                vint16m1_t _in_w = vwsub_vx_i16m1(_in, z2, vl);
                in_ptr += vl;

                _acc0 = vwmacc_vx_i32m2(_acc0, k_ptr[0] - z1, _in_w, vl);
                _acc1 = vwmacc_vx_i32m2(_acc1, k_ptr[1] - z1, _in_w, vl);
                _acc2 = vwmacc_vx_i32m2(_acc2, k_ptr[2] - z1, _in_w, vl);
                _acc3 = vwmacc_vx_i32m2(_acc3, k_ptr[3] - z1, _in_w, vl);
                k_ptr += 4;
            }

            vint8mf2_t _res0 = requantize_m2(_acc0, mult, shift, z3, vl);
            vint8mf2_t _res1 = requantize_m2(_acc1, mult, shift, z3, vl);
            vint8mf2_t _res2 = requantize_m2(_acc2, mult, shift, z3, vl);
            vint8mf2_t _res3 = requantize_m2(_acc3, mult, shift, z3, vl);

            vse8_v_i8mf2(out_ptr, _res0, vl);
            vse8_v_i8mf2(out_ptr + ldc, _res1, vl);
            vse8_v_i8mf2(out_ptr + ldc * 2, _res2, vl);
            vse8_v_i8mf2(out_ptr + ldc * 3, _res3, vl);
            j += vl;
        }
    }
    for (; i + 1 < m; i += 2) {
        const int8_t *kernel_ptr = kernel_data + i * k;
        int j = 0;
        while (j < n) {
            int vl = vsetvl_e8mf2(n - j);
            const int32_t *k32_ptr = (int32_t *)kernel_ptr;
            const int8_t *in_ptr = input_data + j * k;
            int8_t *out_ptr = output_data + i * ldc + j;

            vint32m2_t _acc0 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _acc1 = vmv_v_x_i32m2(0, vl);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _in = vle8_v_i8m2(in_ptr, vl * 4);
                in_ptr += vl * 4;

                // q1 * q2
                _acc0 = vmaqa_vx_i32m2(_acc0, k32_ptr[0], _in, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, k32_ptr[1], _in, vl);
                // - z1 * q2
                _acc0 = vmaqa_vx_i32m2(_acc0, z1_i32[0], _in, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, z1_i32[0], _in, vl);
                // z1_i8_4[3]);
                // - z2 * q1
                _acc0 = vmaqa_vx_i32m2(_acc0, k32_ptr[0], _z2_i8, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, k32_ptr[1], _z2_i8, vl);
                k32_ptr += 2;
            }
            // + z1 * z2
            int32_t acc_z1z2 = c * z1z2;
            _acc0 = vadd_vx_i32m2(_acc0, acc_z1z2, vl);
            _acc1 = vadd_vx_i32m2(_acc1, acc_z1z2, vl);

            const int8_t *k_ptr = kernel_ptr + 2 * c;
            for (; c < k; c++) {
                vint8mf2_t _in = vle8_v_i8mf2(in_ptr, vl);
                vint16m1_t _in_w = vwsub_vx_i16m1(_in, z2, vl);
                in_ptr += vl;

                _acc0 = vwmacc_vx_i32m2(_acc0, k_ptr[0] - z1, _in_w, vl);
                _acc1 = vwmacc_vx_i32m2(_acc1, k_ptr[1] - z1, _in_w, vl);
                k_ptr += 2;
            }

            vint8mf2_t _res0 = requantize_m2(_acc0, mult, shift, z3, vl);
            vint8mf2_t _res1 = requantize_m2(_acc1, mult, shift, z3, vl);

            vse8_v_i8mf2(out_ptr, _res0, vl);
            vse8_v_i8mf2(out_ptr + ldc, _res1, vl);
            j += vl;
        }
    }
    for (; i < m; i++) {
        const int8_t *kernel_ptr = kernel_data + i * k;
        int j = 0;
        while (j < n) {
            int vl = vsetvl_e8mf2(n - j);
            const int32_t *k32_ptr = (int32_t *)kernel_ptr;
            const int8_t *in_ptr = input_data + j * k;
            int8_t *out_ptr = output_data + i * ldc + j;

            vint32m2_t _acc0 = vmv_v_x_i32m2(0, vl);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _in = vle8_v_i8m2(in_ptr, vl * 4);
                in_ptr += vl * 4;

                // q1 * q2
                _acc0 = vmaqa_vx_i32m2(_acc0, k32_ptr[0], _in, vl);
                // - z1 * q2
                _acc0 = vmaqa_vx_i32m2(_acc0, z1_i32[0], _in, vl);
                // - z2 * q1
                _acc0 = vmaqa_vx_i32m2(_acc0, k32_ptr[0], _z2_i8, vl);
                k32_ptr += 1;
            }
            // + z1 * z2
            int32_t acc_z1z2 = c * z1z2;
            _acc0 = vadd_vx_i32m2(_acc0, acc_z1z2, vl);

            const int8_t *k_ptr = kernel_ptr + 1 * c;
            for (; c < k; c++) {
                vint8mf2_t _in = vle8_v_i8mf2(in_ptr, vl);
                vint16m1_t _in_w = vwsub_vx_i16m1(_in, z2, vl);
                in_ptr += vl;

                _acc0 = vwmacc_vx_i32m2(_acc0, k_ptr[0] - z1, _in_w, vl);
                k_ptr += 1;
            }

            vint8mf2_t _res0 = requantize_m2(_acc0, mult, shift, z3, vl);

            vse8_v_i8mf2(out_ptr, _res0, vl);
            j += vl;
        }
    }
}
#endif
