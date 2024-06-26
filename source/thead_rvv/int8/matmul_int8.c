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

/*************************************************************
 * src: [m, k]
 * dst: [m/4, k, 4]
 ************************************************************/
void shl_rvv_matmul_reorder_mat0_n4_int8(int8_t *src, int8_t *dst, int m, int k, int lda)
{
    int i = 0;
    for (; i + 3 < m; i += 4) {
        int8_t *s_ptr = src + i * lda;
        int8_t *d_ptr = dst + i * k;
        int stride = 4 * sizeof(int8_t);
        int c = 0;
        while (c < k) {
            int vl = vsetvl_e8m4(k - c);
            vint8m4_t _s0 = vle8_v_i8m4(s_ptr, vl);
            vint8m4_t _s1 = vle8_v_i8m4(s_ptr + lda, vl);
            vint8m4_t _s2 = vle8_v_i8m4(s_ptr + lda * 2, vl);
            vint8m4_t _s3 = vle8_v_i8m4(s_ptr + lda * 3, vl);
            vsse8_v_i8m4(d_ptr, stride, _s0, vl);
            vsse8_v_i8m4(d_ptr + 1, stride, _s1, vl);
            vsse8_v_i8m4(d_ptr + 2, stride, _s2, vl);
            vsse8_v_i8m4(d_ptr + 3, stride, _s3, vl);
            s_ptr += vl;
            d_ptr += vl * 4;
            c += vl;
        }
    }
    for (; i + 1 < m; i += 2) {
        int8_t *s_ptr = src + i * lda;
        int8_t *d_ptr = dst + i * k;
        int stride = 2 * sizeof(int8_t);
        int c = 0;
        while (c < k) {
            int vl = vsetvl_e8m4(k - c);
            vint8m4_t _s0 = vle8_v_i8m4(s_ptr, vl);
            vint8m4_t _s1 = vle8_v_i8m4(s_ptr + lda, vl);
            vsse8_v_i8m4(d_ptr, stride, _s0, vl);
            vsse8_v_i8m4(d_ptr + 1, stride, _s1, vl);
            s_ptr += vl;
            d_ptr += vl * 2;
            c += vl;
        }
    }
    for (; i < m; i++) {
        int8_t *s_ptr = src + i * lda;
        int8_t *d_ptr = dst + i * k;
        int c = 0;
        while (c < k) {
            int vl = vsetvl_e8m4(k - c);
            vint8m4_t _src = vle8_v_i8m4(s_ptr, vl);
            vse8_v_i8m4(d_ptr, _src, vl);
            s_ptr += vl;
            d_ptr += vl;
            c += vl;
        }
    }
}

/*************************************************************
 * src: [k, n]
 * dst: [n/packn, k, packn]
 ************************************************************/
void shl_rvv_matmul_reorder_mat1_zpackn_int8(int8_t *src, int8_t *dst, int k, int n, int ldb)
{
    int j = 0;
    while (j < n) {
        int vl = vsetvl_e8m1(n - j);
        int8_t *s_ptr = src + j;
        for (int c = 0; c < k; c++) {
            vint8m1_t _src = vle8_v_i8m1(s_ptr, vl);
            vse8_v_i8m1(dst, _src, vl);
            s_ptr += ldb;
            dst += vl;
        }
        j += vl;
    }
}

static vint8m1_t requantize_m4(vint32m4_t _src, int32_t multiplier, int32_t shift, int32_t out_zp,
                               int vl)
{
    vint32m4_t _mulh = vmulh_vx_i32m4(_src, multiplier, vl);
    _mulh = vssra_vx_i32m4(_mulh, -shift - 1, vl);
    _mulh = vadd_vx_i32m4(_mulh, out_zp, vl);
    vint16m2_t _tmp1 = vnclip_wx_i16m2(_mulh, 0, vl);
    vint8m1_t _tmp2 = vnclip_wx_i8m1(_tmp1, 0, vl);
    return _tmp2;
}

/*************************************************************
 * packn = vlenb / sizeof(int8_t)
 * dst - output: [m, n]
 * sa - mat0:    [m/4, k, 4]
 * sb - mat1:    [n/packn, k, packn]
 ************************************************************/
void shl_rvv_matmul_4xpackn_int8(int8_t *dst, const int8_t *sa, const int8_t *sb, int m, int k,
                                 int n, int ldc, int32_t z1, int32_t z2, int32_t z3, int32_t mult,
                                 int32_t shift)
{
    const int8_t *kernel_data = sa;
    const int8_t *input_data = sb;
    int8_t *output_data = dst;

    const int packn = csrr_vlenb() / sizeof(int8_t);
    int vl = vsetvl_e8m1(packn);

    int i = 0;
    for (; i + 3 < m; i += 4) {
        const int8_t *kernel_ptr = kernel_data + i * k;
        int j = 0;
        while (j < n) {
            vl = vsetvl_e8m1(n - j);
            const int8_t *k_ptr = kernel_ptr;
            const int8_t *in_ptr = input_data + j * k;
            int8_t *out_ptr = output_data + i * ldc + j;

            vint32m4_t _acc0 = vmv_v_x_i32m4(0, vl);
            vint32m4_t _acc1 = vmv_v_x_i32m4(0, vl);
            vint32m4_t _acc2 = vmv_v_x_i32m4(0, vl);
            vint32m4_t _acc3 = vmv_v_x_i32m4(0, vl);

            for (int c = 0; c < k; c++) {
                vint8m1_t _in = vle8_v_i8m1(in_ptr, vl);
                vint16m2_t _in_w = vwsub_vx_i16m2(_in, z2, vl);
                in_ptr += vl;

                _acc0 = vwmacc_vx_i32m4(_acc0, k_ptr[0] - z1, _in_w, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, k_ptr[1] - z1, _in_w, vl);
                _acc2 = vwmacc_vx_i32m4(_acc2, k_ptr[2] - z1, _in_w, vl);
                _acc3 = vwmacc_vx_i32m4(_acc3, k_ptr[3] - z1, _in_w, vl);
                k_ptr += 4;
            }

            vint8m1_t _res0 = requantize_m4(_acc0, mult, shift, z3, vl);
            vint8m1_t _res1 = requantize_m4(_acc1, mult, shift, z3, vl);
            vint8m1_t _res2 = requantize_m4(_acc2, mult, shift, z3, vl);
            vint8m1_t _res3 = requantize_m4(_acc3, mult, shift, z3, vl);
            vse8_v_i8m1(out_ptr, _res0, vl);
            vse8_v_i8m1(out_ptr + ldc, _res1, vl);
            vse8_v_i8m1(out_ptr + ldc * 2, _res2, vl);
            vse8_v_i8m1(out_ptr + ldc * 3, _res3, vl);
            j += vl;
        }
    }
    for (; i + 1 < m; i += 2) {
        const int8_t *kernel_ptr = kernel_data + i * k;
        int j = 0;
        while (j < n) {
            vl = vsetvl_e8m1(n - j);
            const int8_t *k_ptr = kernel_ptr;
            const int8_t *in_ptr = input_data + j * k;
            int8_t *out_ptr = output_data + i * ldc + j;

            vint32m4_t _acc0 = vmv_v_x_i32m4(0, vl);
            vint32m4_t _acc1 = vmv_v_x_i32m4(0, vl);

            for (int c = 0; c < k; c++) {
                vint8m1_t _in = vle8_v_i8m1(in_ptr, vl);
                vint16m2_t _in_w = vwsub_vx_i16m2(_in, z2, vl);
                in_ptr += vl;

                _acc0 = vwmacc_vx_i32m4(_acc0, k_ptr[0] - z1, _in_w, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, k_ptr[1] - z1, _in_w, vl);
                k_ptr += 2;
            }

            vint8m1_t _res0 = requantize_m4(_acc0, mult, shift, z3, vl);
            vint8m1_t _res1 = requantize_m4(_acc1, mult, shift, z3, vl);
            vse8_v_i8m1(out_ptr, _res0, vl);
            vse8_v_i8m1(out_ptr + ldc, _res1, vl);
            j += vl;
        }
    }
    for (; i < m; i++) {
        const int8_t *kernel_ptr = kernel_data + i * k;
        int j = 0;
        while (j < n) {
            vl = vsetvl_e8m1(n - j);
            const int8_t *k_ptr = kernel_ptr;
            const int8_t *in_ptr = input_data + j * k;
            int8_t *out_ptr = output_data + i * ldc + j;

            vint32m4_t _acc0 = vmv_v_x_i32m4(0, vl);

            for (int c = 0; c < k; c++) {
                vint8m1_t _in = vle8_v_i8m1(in_ptr, vl);
                vint16m2_t _in_w = vwsub_vx_i16m2(_in, z2, vl);
                in_ptr += vl;

                _acc0 = vwmacc_vx_i32m4(_acc0, k_ptr[0] - z1, _in_w, vl);
                k_ptr += 1;
            }

            vint8m1_t _res0 = requantize_m4(_acc0, mult, shift, z3, vl);
            vse8_v_i8m1(out_ptr, _res0, vl);
            j += vl;
        }
    }
}
