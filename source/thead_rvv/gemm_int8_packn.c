/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

#include "shl_thead_rvv.h"

/*************************************************************
 * note: VLEN = 128/256 ... flexible vlen
 * input matrix and kernel matrix have been reordered
 *************************************************************/

// shift 已经处理
static vint8m1_t requantize_m4_s(vint32m4_t _src, vint32m4_t _multiplier, vint32m4_t _shift,
                                 int32_t out_zp, int vl)
{
    vint32m4_t _mulh = vmulh_vv_i32m4(_src, _multiplier, vl);
    _mulh = vssra_vv_i32m4(_mulh, vreinterpret_v_i32m4_u32m4(_shift), vl);
    _mulh = vadd_vx_i32m4(_mulh, out_zp, vl);
    vint16m2_t _tmp1 = vnclip_wx_i16m2(_mulh, 0, vl);
    vint8m1_t _tmp2 = vnclip_wx_i8m1(_tmp1, 0, vl);
    return _tmp2;
}

/**************************************************************
 * dst - output: [m/packn, n, packn]
 * sa - kernel:  [m/packn, k, packn]
 * sb - input:   [n/4, k, 4]
 **************************************************************/
void shl_rvv_ncxhwx_gemm_4xpack2n_int8(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                       int32_t *bias, int m, int k, int n, int ldc, int32_t out_zp,
                                       int32_t *mult, int32_t *shift)
{
    int8_t *kernel_data = (int8_t *)sa;
    int8_t *input_data = (int8_t *)sb;
    int8_t *output_data = dst;
    // please use fuse_zp2bias option in hhb, thus bias_data wont be NULL
    int32_t *bias_data = bias;

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int pack2n = packn * 2;
    int vl = vsetvl_e8m1(pack2n);

    int oc = 0;
    for (; oc + pack2n - 1 < m; oc += pack2n) {
        vint32m4_t _mult = vle32_v_i32m4(mult + oc, vl);
        vint32m4_t _shift = vle32_v_i32m4(shift + oc, vl);
        _shift = vrsub_vx_i32m4(_shift, -1, vl);

        int8_t *output0 = output_data + oc * n;
        int8_t *output1 = output0 + packn * n;
        const int8_t *img0 = input_data;
        const int32_t *b0 = bias_data + oc;

        int t = 0;
        for (; t + 3 < n; t += 4) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m4_t _acc0 = vle32_v_i32m4(b0, vl);
            vint32m4_t _acc1 = vmv_v_v_i32m4(_acc0, vl);
            vint32m4_t _acc2 = vmv_v_v_i32m4(_acc0, vl);
            vint32m4_t _acc3 = vmv_v_v_i32m4(_acc0, vl);

            for (int c = 0; c < k; c += 1) {
                vint8m1_t _kernel0 = vle8_v_i8m1(k0, vl);
                k0 += vl;
                vint16m2_t _mul0 = vwmul_vx_i16m2(_kernel0, img0[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_kernel0, img0[1], vl);
                vint16m2_t _mul2 = vwmul_vx_i16m2(_kernel0, img0[2], vl);
                vint16m2_t _mul3 = vwmul_vx_i16m2(_kernel0, img0[3], vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                _acc2 = vwmacc_vx_i32m4(_acc2, 1, _mul2, vl);
                _acc3 = vwmacc_vx_i32m4(_acc3, 1, _mul3, vl);
                img0 += 4;
            }
            vint8m1_t _res0 = requantize_m4_s(_acc0, _mult, _shift, out_zp, vl);
            vint8m1_t _res1 = requantize_m4_s(_acc1, _mult, _shift, out_zp, vl);
            vint8m1_t _res2 = requantize_m4_s(_acc2, _mult, _shift, out_zp, vl);
            vint8m1_t _res3 = requantize_m4_s(_acc3, _mult, _shift, out_zp, vl);

            vse8_v_i8m1(output0, _res0, vl / 2);
            vse8_v_i8m1(output0 + packn * 1, _res1, vl / 2);
            vse8_v_i8m1(output0 + packn * 2, _res2, vl / 2);
            vse8_v_i8m1(output0 + packn * 3, _res3, vl / 2);
            output0 += packn * 4;

            _res0 = vslidedown_vx_i8m1(_res0, _res0, packn, vl);
            _res1 = vslidedown_vx_i8m1(_res1, _res1, packn, vl);
            _res2 = vslidedown_vx_i8m1(_res2, _res2, packn, vl);
            _res3 = vslidedown_vx_i8m1(_res3, _res3, packn, vl);
            vse8_v_i8m1(output1, _res0, vl / 2);
            vse8_v_i8m1(output1 + packn * 1, _res1, vl / 2);
            vse8_v_i8m1(output1 + packn * 2, _res2, vl / 2);
            vse8_v_i8m1(output1 + packn * 3, _res3, vl / 2);
            output1 += packn * 4;
        }
        for (; t + 1 < n; t += 2) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m4_t _acc0 = vle32_v_i32m4(b0, vl);
            vint32m4_t _acc1 = vmv_v_v_i32m4(_acc0, vl);

            for (int c = 0; c < k; c += 1) {
                vint8m1_t _kernel0 = vle8_v_i8m1(k0, vl);
                k0 += vl;
                vint16m2_t _mul0 = vwmul_vx_i16m2(_kernel0, img0[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_kernel0, img0[1], vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                img0 += 2;
            }
            vint8m1_t _res0 = requantize_m4_s(_acc0, _mult, _shift, out_zp, vl);
            vint8m1_t _res1 = requantize_m4_s(_acc1, _mult, _shift, out_zp, vl);

            vse8_v_i8m1(output0, _res0, vl / 2);
            vse8_v_i8m1(output0 + packn * 1, _res1, vl / 2);
            output0 += packn * 2;

            _res0 = vslidedown_vx_i8m1(_res0, _res0, packn, vl);
            _res1 = vslidedown_vx_i8m1(_res1, _res1, packn, vl);
            vse8_v_i8m1(output1, _res0, vl / 2);
            vse8_v_i8m1(output1 + packn * 1, _res1, vl / 2);
            output1 += packn * 2;
        }
        for (; t < n; t += 1) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m4_t _acc0 = vle32_v_i32m4(b0, vl);

            for (int c = 0; c < k; c += 1) {
                vint8m1_t _kernel0 = vle8_v_i8m1(k0, vl);
                k0 += vl;
                vint16m2_t _mul0 = vwmul_vx_i16m2(_kernel0, img0[0], vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                img0 += 1;
            }
            vint8m1_t _res0 = requantize_m4_s(_acc0, _mult, _shift, out_zp, vl);
            vse8_v_i8m1(output0, _res0, vl / 2);
            output0 += packn * 1;

            _res0 = vslidedown_vx_i8m1(_res0, _res0, packn, vl);
            vse8_v_i8m1(output1, _res0, vl / 2);
            output1 += packn * 1;
        }
    }

    for (; oc + packn - 1 < m; oc += packn) {
        vl = vsetvl_e8m1(packn);
        vint32m4_t _mult = vle32_v_i32m4(mult + oc, vl);
        vint32m4_t _shift = vle32_v_i32m4(shift + oc, vl);
        _shift = vrsub_vx_i32m4(_shift, -1, vl);

        int8_t *output0 = output_data + oc * n;
        const int8_t *img0 = input_data;
        const int32_t *b0 = bias_data + oc;

        int t = 0;
        for (; t + 3 < n; t += 4) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m4_t _acc0 = vle32_v_i32m4(b0, vl);
            vint32m4_t _acc1 = vmv_v_v_i32m4(_acc0, vl);
            vint32m4_t _acc2 = vmv_v_v_i32m4(_acc0, vl);
            vint32m4_t _acc3 = vmv_v_v_i32m4(_acc0, vl);

            for (int c = 0; c < k; c += 1) {
                vint8m1_t _kernel0 = vle8_v_i8m1(k0, vl);
                k0 += vl;
                vint16m2_t _mul0 = vwmul_vx_i16m2(_kernel0, img0[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_kernel0, img0[1], vl);
                vint16m2_t _mul2 = vwmul_vx_i16m2(_kernel0, img0[2], vl);
                vint16m2_t _mul3 = vwmul_vx_i16m2(_kernel0, img0[3], vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                _acc2 = vwmacc_vx_i32m4(_acc2, 1, _mul2, vl);
                _acc3 = vwmacc_vx_i32m4(_acc3, 1, _mul3, vl);
                img0 += 4;
            }
            vint8m1_t _res0 = requantize_m4_s(_acc0, _mult, _shift, out_zp, vl);
            vint8m1_t _res1 = requantize_m4_s(_acc1, _mult, _shift, out_zp, vl);
            vint8m1_t _res2 = requantize_m4_s(_acc2, _mult, _shift, out_zp, vl);
            vint8m1_t _res3 = requantize_m4_s(_acc3, _mult, _shift, out_zp, vl);

            vse8_v_i8m1(output0, _res0, vl);
            vse8_v_i8m1(output0 + packn * 1, _res1, vl);
            vse8_v_i8m1(output0 + packn * 2, _res2, vl);
            vse8_v_i8m1(output0 + packn * 3, _res3, vl);

            output0 += packn * 4;
        }
        for (; t + 1 < n; t += 2) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m4_t _acc0 = vle32_v_i32m4(b0, vl);
            vint32m4_t _acc1 = vmv_v_v_i32m4(_acc0, vl);

            for (int c = 0; c < k; c += 1) {
                vint8m1_t _kernel0 = vle8_v_i8m1(k0, vl);
                k0 += vl;
                vint16m2_t _mul0 = vwmul_vx_i16m2(_kernel0, img0[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_kernel0, img0[1], vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                img0 += 2;
            }
            vint8m1_t _res0 = requantize_m4_s(_acc0, _mult, _shift, out_zp, vl);
            vint8m1_t _res1 = requantize_m4_s(_acc1, _mult, _shift, out_zp, vl);

            vse8_v_i8m1(output0, _res0, vl);
            vse8_v_i8m1(output0 + packn * 1, _res1, vl);

            output0 += packn * 2;
        }
        for (; t < n; t += 1) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m4_t _acc0 = vle32_v_i32m4(b0, vl);

            for (int c = 0; c < k; c += 1) {
                vint8m1_t _kernel0 = vle8_v_i8m1(k0, vl);
                k0 += vl;
                vint16m2_t _mul0 = vwmul_vx_i16m2(_kernel0, img0[0], vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                img0 += 1;
            }
            vint8m1_t _res0 = requantize_m4_s(_acc0, _mult, _shift, out_zp, vl);
            vse8_v_i8m1(output0, _res0, vl);
            output0 += packn * 1;
        }
    }

    /* tail output_channel */
    if (oc < m) {
        vl = vsetvl_e8m1(m - oc);
        vint32m4_t _mult = vle32_v_i32m4(mult + oc, vl);
        vint32m4_t _shift = vle32_v_i32m4(shift + oc, vl);
        _shift = vrsub_vx_i32m4(_shift, -1, vl);

        int8_t *output0 = output_data + oc * n;
        const int8_t *img0 = input_data;
        const int32_t *b0 = bias_data + oc;

        int t = 0;
        for (; t + 3 < n; t += 4) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m4_t _acc0 = vle32_v_i32m4(b0, vl);
            vint32m4_t _acc1 = vmv_v_v_i32m4(_acc0, vl);
            vint32m4_t _acc2 = vmv_v_v_i32m4(_acc0, vl);
            vint32m4_t _acc3 = vmv_v_v_i32m4(_acc0, vl);

            for (int c = 0; c < k; c += 1) {
                vint8m1_t _kernel0 = vle8_v_i8m1(k0, vl);
                k0 += vl;
                vint16m2_t _mul0 = vwmul_vx_i16m2(_kernel0, img0[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_kernel0, img0[1], vl);
                vint16m2_t _mul2 = vwmul_vx_i16m2(_kernel0, img0[2], vl);
                vint16m2_t _mul3 = vwmul_vx_i16m2(_kernel0, img0[3], vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                _acc2 = vwmacc_vx_i32m4(_acc2, 1, _mul2, vl);
                _acc3 = vwmacc_vx_i32m4(_acc3, 1, _mul3, vl);
                img0 += 4;
            }
            vint8m1_t _res0 = requantize_m4_s(_acc0, _mult, _shift, out_zp, vl);
            vint8m1_t _res1 = requantize_m4_s(_acc1, _mult, _shift, out_zp, vl);
            vint8m1_t _res2 = requantize_m4_s(_acc2, _mult, _shift, out_zp, vl);
            vint8m1_t _res3 = requantize_m4_s(_acc3, _mult, _shift, out_zp, vl);

            vse8_v_i8m1(output0, _res0, vl);
            vse8_v_i8m1(output0 + vl * 1, _res1, vl);
            vse8_v_i8m1(output0 + vl * 2, _res2, vl);
            vse8_v_i8m1(output0 + vl * 3, _res3, vl);

            output0 += vl * 4;
        }
        for (; t + 1 < n; t += 2) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m4_t _acc0 = vle32_v_i32m4(b0, vl);
            vint32m4_t _acc1 = vmv_v_v_i32m4(_acc0, vl);

            for (int c = 0; c < k; c += 1) {
                vint8m1_t _kernel0 = vle8_v_i8m1(k0, vl);
                k0 += vl;
                vint16m2_t _mul0 = vwmul_vx_i16m2(_kernel0, img0[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_kernel0, img0[1], vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                img0 += 2;
            }
            vint8m1_t _res0 = requantize_m4_s(_acc0, _mult, _shift, out_zp, vl);
            vint8m1_t _res1 = requantize_m4_s(_acc1, _mult, _shift, out_zp, vl);

            vse8_v_i8m1(output0, _res0, vl);
            vse8_v_i8m1(output0 + vl * 1, _res1, vl);

            output0 += vl * 2;
        }
        for (; t < n; t += 1) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m4_t _acc0 = vle32_v_i32m4(b0, vl);

            for (int c = 0; c < k; c += 1) {
                vint8m1_t _kernel0 = vle8_v_i8m1(k0, vl);
                k0 += vl;
                vint16m2_t _mul0 = vwmul_vx_i16m2(_kernel0, img0[0], vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                img0 += 1;
            }
            vint8m1_t _res0 = requantize_m4_s(_acc0, _mult, _shift, out_zp, vl);
            vse8_v_i8m1(output0, _res0, vl);
            output0 += vl * 1;
        }
    }
}
