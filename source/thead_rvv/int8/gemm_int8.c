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

/**************************************************************
 * dst - output: [m, n]
 * sa - kernel:  [m, k]
 * sb - input:   [k, n]
 **************************************************************/
void shl_rvv_gemm_4x16_int8_v128(int8_t *dst, const int8_t *sa, const int8_t *sb, int32_t *bias,
                                 int m, int k, int n, int ldc, int32_t out_zp, int32_t *mult,
                                 int32_t *shift)
{
    int8_t *kernel_data = (int8_t *)sa;
    int8_t *input_data = (int8_t *)sb;
    int8_t *output_data = dst;
    int32_t *bias_data = bias;

    int vl = vsetvl_e8m1(16);
    int oc = 0;
    for (; oc + 3 < m; oc += 4) {
        int8_t *in_ptr = input_data;
        int8_t *out_ptr0 = output_data;
        int8_t *out_ptr1 = out_ptr0 + ldc;
        int8_t *out_ptr2 = out_ptr1 + ldc;
        int8_t *out_ptr3 = out_ptr2 + ldc;

        vl = vsetvl_e8m1(16);
        int j = 0;
        for (; j + 15 < n; j += 16) {
            int8_t *ker_ptr = kernel_data;
            vint32m4_t _acc0 = vmv_v_x_i32m4(bias_data[0], vl);
            vint32m4_t _acc1 = vmv_v_x_i32m4(bias_data[1], vl);
            vint32m4_t _acc2 = vmv_v_x_i32m4(bias_data[2], vl);
            vint32m4_t _acc3 = vmv_v_x_i32m4(bias_data[3], vl);

            for (int c = 0; c < k; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, vl);
                vint16m2_t _mul0 = vwmul_vx_i16m2(_input, ker_ptr[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_input, ker_ptr[1], vl);
                vint16m2_t _mul2 = vwmul_vx_i16m2(_input, ker_ptr[2], vl);
                vint16m2_t _mul3 = vwmul_vx_i16m2(_input, ker_ptr[3], vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                _acc2 = vwmacc_vx_i32m4(_acc2, 1, _mul2, vl);
                _acc3 = vwmacc_vx_i32m4(_acc3, 1, _mul3, vl);
                in_ptr += vl;
                ker_ptr += 4;
            }
            // requantization
            vint8m1_t _res0 = requantize_m4(_acc0, mult[0], shift[0], out_zp, vl);
            vint8m1_t _res1 = requantize_m4(_acc1, mult[1], shift[1], out_zp, vl);
            vint8m1_t _res2 = requantize_m4(_acc2, mult[2], shift[2], out_zp, vl);
            vint8m1_t _res3 = requantize_m4(_acc3, mult[3], shift[3], out_zp, vl);

            vse8_v_i8m1(out_ptr0, _res0, vl);
            vse8_v_i8m1(out_ptr1, _res1, vl);
            vse8_v_i8m1(out_ptr2, _res2, vl);
            vse8_v_i8m1(out_ptr3, _res3, vl);
            out_ptr0 += 16;
            out_ptr1 += 16;
            out_ptr2 += 16;
            out_ptr3 += 16;
        }
        for (; j + 7 < n; j += 8) {
            vl = vsetvl_e8m1(8);
            int8_t *ker_ptr = kernel_data;
            vint32m4_t _acc0 = vmv_v_x_i32m4(bias_data[0], vl);
            vint32m4_t _acc1 = vmv_v_x_i32m4(bias_data[1], vl);
            vint32m4_t _acc2 = vmv_v_x_i32m4(bias_data[2], vl);
            vint32m4_t _acc3 = vmv_v_x_i32m4(bias_data[3], vl);

            for (int c = 0; c < k; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, vl);
                vint16m2_t _mul0 = vwmul_vx_i16m2(_input, ker_ptr[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_input, ker_ptr[1], vl);
                vint16m2_t _mul2 = vwmul_vx_i16m2(_input, ker_ptr[2], vl);
                vint16m2_t _mul3 = vwmul_vx_i16m2(_input, ker_ptr[3], vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                _acc2 = vwmacc_vx_i32m4(_acc2, 1, _mul2, vl);
                _acc3 = vwmacc_vx_i32m4(_acc3, 1, _mul3, vl);
                in_ptr += vl;
                ker_ptr += 4;
            }
            // requantization
            vint8m1_t _res0 = requantize_m4(_acc0, mult[0], shift[0], out_zp, vl);
            vint8m1_t _res1 = requantize_m4(_acc1, mult[1], shift[1], out_zp, vl);
            vint8m1_t _res2 = requantize_m4(_acc2, mult[2], shift[2], out_zp, vl);
            vint8m1_t _res3 = requantize_m4(_acc3, mult[3], shift[3], out_zp, vl);
            vse8_v_i8m1(out_ptr0, _res0, vl);
            vse8_v_i8m1(out_ptr1, _res1, vl);
            vse8_v_i8m1(out_ptr2, _res2, vl);
            vse8_v_i8m1(out_ptr3, _res3, vl);
            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
        }
        // n_tail
        if (j < n) {
            vl = vsetvl_e8m1(n - j);
            int8_t *ker_ptr = kernel_data;
            vint32m4_t _acc0 = vmv_v_x_i32m4(bias_data[0], vl);
            vint32m4_t _acc1 = vmv_v_x_i32m4(bias_data[1], vl);
            vint32m4_t _acc2 = vmv_v_x_i32m4(bias_data[2], vl);
            vint32m4_t _acc3 = vmv_v_x_i32m4(bias_data[3], vl);

            for (int c = 0; c < k; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, vl);
                vint16m2_t _mul0 = vwmul_vx_i16m2(_input, ker_ptr[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_input, ker_ptr[1], vl);
                vint16m2_t _mul2 = vwmul_vx_i16m2(_input, ker_ptr[2], vl);
                vint16m2_t _mul3 = vwmul_vx_i16m2(_input, ker_ptr[3], vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                _acc2 = vwmacc_vx_i32m4(_acc2, 1, _mul2, vl);
                _acc3 = vwmacc_vx_i32m4(_acc3, 1, _mul3, vl);
                in_ptr += vl;
                ker_ptr += 4;
            }
            // requantization
            vint8m1_t _res0 = requantize_m4(_acc0, mult[0], shift[0], out_zp, vl);
            vint8m1_t _res1 = requantize_m4(_acc1, mult[1], shift[1], out_zp, vl);
            vint8m1_t _res2 = requantize_m4(_acc2, mult[2], shift[2], out_zp, vl);
            vint8m1_t _res3 = requantize_m4(_acc3, mult[3], shift[3], out_zp, vl);

            vse8_v_i8m1(out_ptr0, _res0, vl);
            vse8_v_i8m1(out_ptr1, _res1, vl);
            vse8_v_i8m1(out_ptr2, _res2, vl);
            vse8_v_i8m1(out_ptr3, _res3, vl);
        }
        kernel_data += 4 * k;
        output_data += 4 * n;
        bias_data += 4;
        mult += 4;
        shift += 4;
    }
    for (; oc + 1 < m; oc += 2) {
        int8_t *in_ptr = input_data;
        int8_t *out_ptr0 = output_data;
        int8_t *out_ptr1 = out_ptr0 + ldc;

        vl = vsetvl_e8m1(16);
        int j = 0;
        for (; j + 15 < n; j += 16) {
            int8_t *ker_ptr = kernel_data;
            vint32m4_t _acc0 = vmv_v_x_i32m4(bias_data[0], vl);
            vint32m4_t _acc1 = vmv_v_x_i32m4(bias_data[1], vl);

            for (int c = 0; c < k; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, vl);
                vint16m2_t _mul0 = vwmul_vx_i16m2(_input, ker_ptr[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_input, ker_ptr[1], vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                in_ptr += vl;
                ker_ptr += 2;
            }
            // requantization
            vint8m1_t _res0 = requantize_m4(_acc0, mult[0], shift[0], out_zp, vl);
            vint8m1_t _res1 = requantize_m4(_acc1, mult[1], shift[1], out_zp, vl);
            vse8_v_i8m1(out_ptr0, _res0, vl);
            vse8_v_i8m1(out_ptr1, _res1, vl);
            out_ptr0 += 16;
            out_ptr1 += 16;
        }
        for (; j + 7 < n; j += 8) {
            vl = vsetvl_e8m1(8);
            int8_t *ker_ptr = kernel_data;
            vint32m4_t _acc0 = vmv_v_x_i32m4(bias_data[0], vl);
            vint32m4_t _acc1 = vmv_v_x_i32m4(bias_data[1], vl);

            for (int c = 0; c < k; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, vl);
                vint16m2_t _mul0 = vwmul_vx_i16m2(_input, ker_ptr[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_input, ker_ptr[1], vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                in_ptr += vl;
                ker_ptr += 2;
            }
            // requantization
            vint8m1_t _res0 = requantize_m4(_acc0, mult[0], shift[0], out_zp, vl);
            vint8m1_t _res1 = requantize_m4(_acc1, mult[1], shift[1], out_zp, vl);
            vse8_v_i8m1(out_ptr0, _res0, vl);
            vse8_v_i8m1(out_ptr1, _res1, vl);
            out_ptr0 += 8;
            out_ptr1 += 8;
        }
        // n_tail
        if (j < n) {
            vl = vsetvl_e8m1(n - j);
            int8_t *ker_ptr = kernel_data;
            vint32m4_t _acc0 = vmv_v_x_i32m4(bias_data[0], vl);
            vint32m4_t _acc1 = vmv_v_x_i32m4(bias_data[1], vl);

            for (int c = 0; c < k; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, vl);
                vint16m2_t _mul0 = vwmul_vx_i16m2(_input, ker_ptr[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_input, ker_ptr[1], vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                in_ptr += vl;
                ker_ptr += 2;
            }
            // requantization
            vint8m1_t _res0 = requantize_m4(_acc0, mult[0], shift[0], out_zp, vl);
            vint8m1_t _res1 = requantize_m4(_acc1, mult[1], shift[1], out_zp, vl);
            vse8_v_i8m1(out_ptr0, _res0, vl);
            vse8_v_i8m1(out_ptr1, _res1, vl);
        }
        kernel_data += 2 * k;
        output_data += 2 * n;
        bias_data += 2;
        mult += 2;
        shift += 2;
    }
    for (; oc < m; oc += 1) {
        int8_t *in_ptr = input_data;
        int8_t *out_ptr0 = output_data;

        vl = vsetvl_e8m1(16);
        int j = 0;
        for (; j + 15 < n; j += 16) {
            int8_t *ker_ptr = kernel_data;
            vint32m4_t _acc0 = vmv_v_x_i32m4(bias_data[0], vl);

            for (int c = 0; c < k; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, vl);
                vint16m2_t _mul0 = vwmul_vx_i16m2(_input, ker_ptr[0], vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                in_ptr += vl;
                ker_ptr += 1;
            }
            // requantization
            vint8m1_t _res0 = requantize_m4(_acc0, mult[0], shift[0], out_zp, vl);
            vse8_v_i8m1(out_ptr0, _res0, vl);
            out_ptr0 += 16;
        }
        for (; j + 7 < n; j += 8) {
            vl = vsetvl_e8m1(8);
            int8_t *ker_ptr = kernel_data;
            vint32m4_t _acc0 = vmv_v_x_i32m4(bias_data[0], vl);

            for (int c = 0; c < k; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, vl);
                vint16m2_t _mul0 = vwmul_vx_i16m2(_input, ker_ptr[0], vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                in_ptr += vl;
                ker_ptr += 1;
            }
            // requantization
            vint8m1_t _res0 = requantize_m4(_acc0, mult[0], shift[0], out_zp, vl);
            vse8_v_i8m1(out_ptr0, _res0, vl);
            out_ptr0 += 8;
        }
        // n_tail
        if (j < n) {
            vl = vsetvl_e8m1(n - j);
            int8_t *ker_ptr = kernel_data;
            vint32m4_t _acc0 = vmv_v_x_i32m4(bias_data[0], vl);

            for (int c = 0; c < k; c++) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, vl);
                vint16m2_t _mul0 = vwmul_vx_i16m2(_input, ker_ptr[0], vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                in_ptr += vl;
                ker_ptr += 1;
            }
            // requantization
            vint8m1_t _res0 = requantize_m4(_acc0, mult[0], shift[0], out_zp, vl);
            vse8_v_i8m1(out_ptr0, _res0, vl);
        }
    }
}
