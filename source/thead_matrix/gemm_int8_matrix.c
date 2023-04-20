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

/* SHL version 2.1.x */

#include "shl_thead_rvm.h"

#ifndef MATRIX_PW_I32
static void requantize_m4_nhwc(int8_t *dst, int32_t *src, int row, int col, int32_t out_zp,
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
            // _shift = vrsub_vx_i32m4(_shift, -1, vl);
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

void gemm_int8_to_int32_nhwc_2rowxn_matrix(int32_t *output, const int8_t *kernel,
                                           const int8_t *input, const int32_t *bias, int m, int k,
                                           int n);
void gemm_int8_to_int32_nhwc_rowxn_matrix(int32_t *output, const int8_t *kernel,
                                          const int8_t *input, const int32_t *bias, int m, int k,
                                          int n);
void gemm_int8_to_int32_nhwc_row_tailxn_matrix(int32_t *output, const int8_t *kernel,
                                               const int8_t *input, const int32_t *bias, int m,
                                               int k, int n);
#else
void gemm_int8_nhwc_2rowxn_matrix(int8_t *output, const int8_t *kernel, const int8_t *input,
                                  const int32_t *bias, int m, int k, int n, int32_t out_zp,
                                  int32_t *mult, int32_t *shift);
void gemm_int8_nhwc_rowxn_matrix(int8_t *output, const int8_t *kernel, const int8_t *input,
                                 const int32_t *bias, int m, int k, int n, int32_t out_zp,
                                 int32_t *mult, int32_t *shift);
void gemm_int8_nhwc_row_tailxn_matrix(int8_t *output, const int8_t *kernel, const int8_t *input,
                                      const int32_t *bias, int m, int k, int n, int32_t out_zp,
                                      int32_t *mult, int32_t *shift);
#endif  // MATRIX_PW_I32

void shl_rvm_nhwc_gemm_int8(int8_t *dst, const int8_t *sa, const int8_t *sb, const int32_t *bias,
                            int m, int k, int n, int32_t out_zp, int32_t *mult, int32_t *shift)
{
    const int mlenb = csrr_xrlenb();
    const int mregrows = mlenb / 4;

#ifndef MATRIX_PW_I32
    int32_t *out_ptr = (int32_t *)shl_mem_alloc(2 * mregrows * n * sizeof(int32_t));
    int hw = 0;
    for (; hw + 2 * mregrows - 1 < m; hw += 2 * mregrows) {
        gemm_int8_to_int32_nhwc_2rowxn_matrix(out_ptr, sa, sb, bias, mregrows, k, n);
        requantize_m4_nhwc(dst, out_ptr, 2 * mregrows, n, out_zp, mult, shift);
        sb += 2 * mregrows * k;
        dst += 2 * mregrows * n;
    }
    for (; hw + mregrows - 1 < m; hw += mregrows) {
        gemm_int8_to_int32_nhwc_rowxn_matrix(out_ptr, sa, sb, bias, mregrows, k, n);
        requantize_m4_nhwc(dst, out_ptr, mregrows, n, out_zp, mult, shift);
        sb += mregrows * k;
        dst += mregrows * n;
    }
    if (hw < m) {
        gemm_int8_to_int32_nhwc_row_tailxn_matrix(out_ptr, sa, sb, bias, m - hw, k, n);
        requantize_m4_nhwc(dst, out_ptr, m - hw, n, out_zp, mult, shift);
    }
    shl_mem_free(out_ptr);
#else
    int hw = 0;
    for (; hw + 2 * mregrows - 1 < m; hw += 2 * mregrows) {
        gemm_int8_nhwc_2rowxn_matrix(dst, sa, sb, bias, mregrows, k, n, out_zp, mult, shift);
        sb += 2 * mregrows * k;
        dst += 2 * mregrows * n;
    }
    for (; hw + mregrows - 1 < m; hw += mregrows) {
        gemm_int8_nhwc_rowxn_matrix(dst, sa, sb, bias, mregrows, k, n, out_zp, mult, shift);
        sb += mregrows * k;
        dst += mregrows * n;
    }
    if (hw < m) {
        gemm_int8_nhwc_row_tailxn_matrix(dst, sa, sb, bias, m - hw, k, n, out_zp, mult, shift);
    }
#endif  // MATRIX_PW_I32
}
