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

/* CSI-NN2 version 2.0.x */

#include "shl_thead_rvv.h"
#ifdef XTHEADV
/*************************************************************
 * note: VLEN = 128/256 ... flexible vlen
 * input matrix and kernel matrix have been reordered
 *************************************************************/

// shift 已经处理
static vint8mf2_t requantize_m2_s(vint32m2_t _src, vint32m2_t _multiplier, vint32m2_t _shift,
                                  int32_t out_zp, int vl)
{
    vint32m2_t _mulh = vmulh_vv_i32m2(_src, _multiplier, vl);
    _mulh = vssra_vv_i32m2(_mulh, vreinterpret_v_i32m2_u32m2(_shift), vl);
    _mulh = vadd_vx_i32m2(_mulh, out_zp, vl);
    vint16m1_t _tmp1 = vnclip_wx_i16m1(_mulh, 0, vl);
    vint8mf2_t _tmp2 = vnclip_wx_i8mf2(_tmp1, 0, vl);
    return _tmp2;
}

/**************************************************************
 * dst - output: [m/packn, n, packn]
 * sa - kernel:  [m/packn, k, packn]
 * sb - input:   [n/12, k, 12]
 **************************************************************/
void shl_rvv_ncxhwx_gemm_12xpackn_int8(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                       int32_t *bias, int m, int k, int n, int ldc, int32_t out_zp,
                                       int32_t *mult, int32_t *shift)
{
    int8_t *kernel_data = (int8_t *)sa;
    int8_t *input_data = (int8_t *)sb;
    int8_t *output_data = dst;
    // please use fuse_zp2bias option in hhb, thus bias_data wont be NULL
    int32_t *bias_data = bias;

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    int vl = vsetvl_e32m2(packn);

    int oc = 0;
    for (; oc + packn - 1 < m; oc += packn) {
        vint32m2_t _mult = vle32_v_i32m2(mult + oc, vl);
        vint32m2_t _shift = vle32_v_i32m2(shift + oc, vl);
        _shift = vrsub_vx_i32m2(_shift, -1, vl);

        int8_t *output0 = output_data + oc * n;
        const int32_t *img0 = (const int32_t *)input_data;
        const int32_t *b0 = bias_data + oc;

        int t = 0;
        for (; t + 11 < n; t += 12) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc2 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc3 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc4 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc5 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc6 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc7 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc8 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc9 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acca = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _accb = vmv_v_v_i32m2(_acc0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, img0[1], _kernel0, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, img0[2], _kernel0, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, img0[3], _kernel0, vl);
                _acc4 = vmaqa_vx_i32m2(_acc4, img0[4], _kernel0, vl);
                _acc5 = vmaqa_vx_i32m2(_acc5, img0[5], _kernel0, vl);
                _acc6 = vmaqa_vx_i32m2(_acc6, img0[6], _kernel0, vl);
                _acc7 = vmaqa_vx_i32m2(_acc7, img0[7], _kernel0, vl);
                _acc8 = vmaqa_vx_i32m2(_acc8, img0[8], _kernel0, vl);
                _acc9 = vmaqa_vx_i32m2(_acc9, img0[9], _kernel0, vl);
                _acca = vmaqa_vx_i32m2(_acca, img0[10], _kernel0, vl);
                _accb = vmaqa_vx_i32m2(_accb, img0[11], _kernel0, vl);

                img0 += 12;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, _mult, _shift, out_zp, vl);
            vint8mf2_t _res2 = requantize_m2_s(_acc2, _mult, _shift, out_zp, vl);
            vint8mf2_t _res3 = requantize_m2_s(_acc3, _mult, _shift, out_zp, vl);
            vint8mf2_t _res4 = requantize_m2_s(_acc4, _mult, _shift, out_zp, vl);
            vint8mf2_t _res5 = requantize_m2_s(_acc5, _mult, _shift, out_zp, vl);
            vint8mf2_t _res6 = requantize_m2_s(_acc6, _mult, _shift, out_zp, vl);
            vint8mf2_t _res7 = requantize_m2_s(_acc7, _mult, _shift, out_zp, vl);
            vint8mf2_t _res8 = requantize_m2_s(_acc8, _mult, _shift, out_zp, vl);
            vint8mf2_t _res9 = requantize_m2_s(_acc9, _mult, _shift, out_zp, vl);
            vint8mf2_t _resa = requantize_m2_s(_acca, _mult, _shift, out_zp, vl);
            vint8mf2_t _resb = requantize_m2_s(_accb, _mult, _shift, out_zp, vl);

            vse8_v_i8mf2(output0, _res0, vl);
            vse8_v_i8mf2(output0 + packn * 1, _res1, vl);
            vse8_v_i8mf2(output0 + packn * 2, _res2, vl);
            vse8_v_i8mf2(output0 + packn * 3, _res3, vl);
            vse8_v_i8mf2(output0 + packn * 4, _res4, vl);
            vse8_v_i8mf2(output0 + packn * 5, _res5, vl);
            vse8_v_i8mf2(output0 + packn * 6, _res6, vl);
            vse8_v_i8mf2(output0 + packn * 7, _res7, vl);
            vse8_v_i8mf2(output0 + packn * 8, _res8, vl);
            vse8_v_i8mf2(output0 + packn * 9, _res9, vl);
            vse8_v_i8mf2(output0 + packn * 10, _resa, vl);
            vse8_v_i8mf2(output0 + packn * 11, _resb, vl);

            output0 += packn * 12;
        }
        for (; t + 7 < n; t += 8) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc2 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc3 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc4 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc5 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc6 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc7 = vmv_v_v_i32m2(_acc0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, img0[1], _kernel0, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, img0[2], _kernel0, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, img0[3], _kernel0, vl);
                _acc4 = vmaqa_vx_i32m2(_acc4, img0[4], _kernel0, vl);
                _acc5 = vmaqa_vx_i32m2(_acc5, img0[5], _kernel0, vl);
                _acc6 = vmaqa_vx_i32m2(_acc6, img0[6], _kernel0, vl);
                _acc7 = vmaqa_vx_i32m2(_acc7, img0[7], _kernel0, vl);

                img0 += 8;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, _mult, _shift, out_zp, vl);
            vint8mf2_t _res2 = requantize_m2_s(_acc2, _mult, _shift, out_zp, vl);
            vint8mf2_t _res3 = requantize_m2_s(_acc3, _mult, _shift, out_zp, vl);
            vint8mf2_t _res4 = requantize_m2_s(_acc4, _mult, _shift, out_zp, vl);
            vint8mf2_t _res5 = requantize_m2_s(_acc5, _mult, _shift, out_zp, vl);
            vint8mf2_t _res6 = requantize_m2_s(_acc6, _mult, _shift, out_zp, vl);
            vint8mf2_t _res7 = requantize_m2_s(_acc7, _mult, _shift, out_zp, vl);

            vse8_v_i8mf2(output0, _res0, vl);
            vse8_v_i8mf2(output0 + packn * 1, _res1, vl);
            vse8_v_i8mf2(output0 + packn * 2, _res2, vl);
            vse8_v_i8mf2(output0 + packn * 3, _res3, vl);
            vse8_v_i8mf2(output0 + packn * 4, _res4, vl);
            vse8_v_i8mf2(output0 + packn * 5, _res5, vl);
            vse8_v_i8mf2(output0 + packn * 6, _res6, vl);
            vse8_v_i8mf2(output0 + packn * 7, _res7, vl);

            output0 += packn * 8;
        }
        for (; t + 3 < n; t += 4) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc2 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc3 = vmv_v_v_i32m2(_acc0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, img0[1], _kernel0, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, img0[2], _kernel0, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, img0[3], _kernel0, vl);

                img0 += 4;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, _mult, _shift, out_zp, vl);
            vint8mf2_t _res2 = requantize_m2_s(_acc2, _mult, _shift, out_zp, vl);
            vint8mf2_t _res3 = requantize_m2_s(_acc3, _mult, _shift, out_zp, vl);

            vse8_v_i8mf2(output0, _res0, vl);
            vse8_v_i8mf2(output0 + packn * 1, _res1, vl);
            vse8_v_i8mf2(output0 + packn * 2, _res2, vl);
            vse8_v_i8mf2(output0 + packn * 3, _res3, vl);

            output0 += packn * 4;
        }
        for (; t + 1 < n; t += 2) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, img0[1], _kernel0, vl);
                img0 += 2;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, _mult, _shift, out_zp, vl);

            vse8_v_i8mf2(output0, _res0, vl);
            vse8_v_i8mf2(output0 + packn * 1, _res1, vl);
            output0 += packn * 2;
        }
        for (; t < n; t++) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                img0 += 1;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vse8_v_i8mf2(output0, _res0, vl);
            output0 += packn * 1;
        }
    }

    /* tail output_channel */
    if (oc < m) {
        vl = vsetvl_e32m2(m - oc);
        vint32m2_t _mult = vle32_v_i32m2(mult + oc, vl);
        vint32m2_t _shift = vle32_v_i32m2(shift + oc, vl);
        _shift = vrsub_vx_i32m2(_shift, -1, vl);

        int8_t *output0 = output_data + oc * n;
        const int32_t *img0 = (const int32_t *)input_data;
        const int32_t *b0 = bias_data + oc;

        int t = 0;
        for (; t + 11 < n; t += 12) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc2 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc3 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc4 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc5 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc6 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc7 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc8 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc9 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acca = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _accb = vmv_v_v_i32m2(_acc0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, img0[1], _kernel0, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, img0[2], _kernel0, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, img0[3], _kernel0, vl);
                _acc4 = vmaqa_vx_i32m2(_acc4, img0[4], _kernel0, vl);
                _acc5 = vmaqa_vx_i32m2(_acc5, img0[5], _kernel0, vl);
                _acc6 = vmaqa_vx_i32m2(_acc6, img0[6], _kernel0, vl);
                _acc7 = vmaqa_vx_i32m2(_acc7, img0[7], _kernel0, vl);
                _acc8 = vmaqa_vx_i32m2(_acc8, img0[8], _kernel0, vl);
                _acc9 = vmaqa_vx_i32m2(_acc9, img0[9], _kernel0, vl);
                _acca = vmaqa_vx_i32m2(_acca, img0[10], _kernel0, vl);
                _accb = vmaqa_vx_i32m2(_accb, img0[11], _kernel0, vl);

                img0 += 12;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, _mult, _shift, out_zp, vl);
            vint8mf2_t _res2 = requantize_m2_s(_acc2, _mult, _shift, out_zp, vl);
            vint8mf2_t _res3 = requantize_m2_s(_acc3, _mult, _shift, out_zp, vl);
            vint8mf2_t _res4 = requantize_m2_s(_acc4, _mult, _shift, out_zp, vl);
            vint8mf2_t _res5 = requantize_m2_s(_acc5, _mult, _shift, out_zp, vl);
            vint8mf2_t _res6 = requantize_m2_s(_acc6, _mult, _shift, out_zp, vl);
            vint8mf2_t _res7 = requantize_m2_s(_acc7, _mult, _shift, out_zp, vl);
            vint8mf2_t _res8 = requantize_m2_s(_acc8, _mult, _shift, out_zp, vl);
            vint8mf2_t _res9 = requantize_m2_s(_acc9, _mult, _shift, out_zp, vl);
            vint8mf2_t _resa = requantize_m2_s(_acca, _mult, _shift, out_zp, vl);
            vint8mf2_t _resb = requantize_m2_s(_accb, _mult, _shift, out_zp, vl);

            vse8_v_i8mf2(output0, _res0, vl);
            vse8_v_i8mf2(output0 + vl * 1, _res1, vl);
            vse8_v_i8mf2(output0 + vl * 2, _res2, vl);
            vse8_v_i8mf2(output0 + vl * 3, _res3, vl);
            vse8_v_i8mf2(output0 + vl * 4, _res4, vl);
            vse8_v_i8mf2(output0 + vl * 5, _res5, vl);
            vse8_v_i8mf2(output0 + vl * 6, _res6, vl);
            vse8_v_i8mf2(output0 + vl * 7, _res7, vl);
            vse8_v_i8mf2(output0 + vl * 8, _res8, vl);
            vse8_v_i8mf2(output0 + vl * 9, _res9, vl);
            vse8_v_i8mf2(output0 + vl * 10, _resa, vl);
            vse8_v_i8mf2(output0 + vl * 11, _resb, vl);

            output0 += vl * 12;
        }
        for (; t + 7 < n; t += 8) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc2 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc3 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc4 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc5 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc6 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc7 = vmv_v_v_i32m2(_acc0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, img0[1], _kernel0, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, img0[2], _kernel0, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, img0[3], _kernel0, vl);
                _acc4 = vmaqa_vx_i32m2(_acc4, img0[4], _kernel0, vl);
                _acc5 = vmaqa_vx_i32m2(_acc5, img0[5], _kernel0, vl);
                _acc6 = vmaqa_vx_i32m2(_acc6, img0[6], _kernel0, vl);
                _acc7 = vmaqa_vx_i32m2(_acc7, img0[7], _kernel0, vl);

                img0 += 8;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, _mult, _shift, out_zp, vl);
            vint8mf2_t _res2 = requantize_m2_s(_acc2, _mult, _shift, out_zp, vl);
            vint8mf2_t _res3 = requantize_m2_s(_acc3, _mult, _shift, out_zp, vl);
            vint8mf2_t _res4 = requantize_m2_s(_acc4, _mult, _shift, out_zp, vl);
            vint8mf2_t _res5 = requantize_m2_s(_acc5, _mult, _shift, out_zp, vl);
            vint8mf2_t _res6 = requantize_m2_s(_acc6, _mult, _shift, out_zp, vl);
            vint8mf2_t _res7 = requantize_m2_s(_acc7, _mult, _shift, out_zp, vl);

            vse8_v_i8mf2(output0, _res0, vl);
            vse8_v_i8mf2(output0 + vl * 1, _res1, vl);
            vse8_v_i8mf2(output0 + vl * 2, _res2, vl);
            vse8_v_i8mf2(output0 + vl * 3, _res3, vl);
            vse8_v_i8mf2(output0 + vl * 4, _res4, vl);
            vse8_v_i8mf2(output0 + vl * 5, _res5, vl);
            vse8_v_i8mf2(output0 + vl * 6, _res6, vl);
            vse8_v_i8mf2(output0 + vl * 7, _res7, vl);

            output0 += vl * 8;
        }
        for (; t + 3 < n; t += 4) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc2 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc3 = vmv_v_v_i32m2(_acc0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, img0[1], _kernel0, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, img0[2], _kernel0, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, img0[3], _kernel0, vl);

                img0 += 4;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, _mult, _shift, out_zp, vl);
            vint8mf2_t _res2 = requantize_m2_s(_acc2, _mult, _shift, out_zp, vl);
            vint8mf2_t _res3 = requantize_m2_s(_acc3, _mult, _shift, out_zp, vl);

            vse8_v_i8mf2(output0, _res0, vl);
            vse8_v_i8mf2(output0 + vl * 1, _res1, vl);
            vse8_v_i8mf2(output0 + vl * 2, _res2, vl);
            vse8_v_i8mf2(output0 + vl * 3, _res3, vl);

            output0 += vl * 4;
        }
        for (; t + 1 < n; t += 2) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, img0[1], _kernel0, vl);
                img0 += 2;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, _mult, _shift, out_zp, vl);

            vse8_v_i8mf2(output0, _res0, vl);
            vse8_v_i8mf2(output0 + vl * 1, _res1, vl);
            output0 += vl * 2;
        }
        for (; t < n; t++) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                img0 += 1;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vse8_v_i8mf2(output0, _res0, vl);
            output0 += vl * 1;
        }
    }
}

/**************************************************************
 * dst - output: [m/packn, n, packn]
 * sa - kernel:  [m/packn, k, packn]
 * sb - input:   [n/8, k, 8]
 **************************************************************/
void shl_rvv_ncxhwx_gemm_8xpackn_int8(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                      int32_t *bias, int m, int k, int n, int ldc, int32_t out_zp,
                                      int32_t *mult, int32_t *shift)
{
    int8_t *kernel_data = (int8_t *)sa;
    int8_t *input_data = (int8_t *)sb;
    int8_t *output_data = dst;
    // please use fuse_zp2bias option in hhb, thus bias_data wont be NULL
    int32_t *bias_data = bias;

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    int vl = vsetvl_e32m2(packn);

    int oc = 0;
    for (; oc + packn - 1 < m; oc += packn) {
        vint32m2_t _mult = vle32_v_i32m2(mult + oc, vl);
        vint32m2_t _shift = vle32_v_i32m2(shift + oc, vl);
        _shift = vrsub_vx_i32m2(_shift, -1, vl);

        int8_t *output0 = output_data + oc * n;
        const int32_t *img0 = (const int32_t *)input_data;
        const int32_t *b0 = bias_data + oc;

        int t = 0;
        for (; t + 7 < n; t += 8) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc2 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc3 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc4 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc5 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc6 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc7 = vmv_v_v_i32m2(_acc0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, img0[1], _kernel0, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, img0[2], _kernel0, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, img0[3], _kernel0, vl);
                _acc4 = vmaqa_vx_i32m2(_acc4, img0[4], _kernel0, vl);
                _acc5 = vmaqa_vx_i32m2(_acc5, img0[5], _kernel0, vl);
                _acc6 = vmaqa_vx_i32m2(_acc6, img0[6], _kernel0, vl);
                _acc7 = vmaqa_vx_i32m2(_acc7, img0[7], _kernel0, vl);

                img0 += 8;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, _mult, _shift, out_zp, vl);
            vint8mf2_t _res2 = requantize_m2_s(_acc2, _mult, _shift, out_zp, vl);
            vint8mf2_t _res3 = requantize_m2_s(_acc3, _mult, _shift, out_zp, vl);
            vint8mf2_t _res4 = requantize_m2_s(_acc4, _mult, _shift, out_zp, vl);
            vint8mf2_t _res5 = requantize_m2_s(_acc5, _mult, _shift, out_zp, vl);
            vint8mf2_t _res6 = requantize_m2_s(_acc6, _mult, _shift, out_zp, vl);
            vint8mf2_t _res7 = requantize_m2_s(_acc7, _mult, _shift, out_zp, vl);

            vse8_v_i8mf2(output0, _res0, vl);
            vse8_v_i8mf2(output0 + packn * 1, _res1, vl);
            vse8_v_i8mf2(output0 + packn * 2, _res2, vl);
            vse8_v_i8mf2(output0 + packn * 3, _res3, vl);
            vse8_v_i8mf2(output0 + packn * 4, _res4, vl);
            vse8_v_i8mf2(output0 + packn * 5, _res5, vl);
            vse8_v_i8mf2(output0 + packn * 6, _res6, vl);
            vse8_v_i8mf2(output0 + packn * 7, _res7, vl);

            output0 += packn * 8;
        }
        for (; t + 3 < n; t += 4) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc2 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc3 = vmv_v_v_i32m2(_acc0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, img0[1], _kernel0, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, img0[2], _kernel0, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, img0[3], _kernel0, vl);

                img0 += 4;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, _mult, _shift, out_zp, vl);
            vint8mf2_t _res2 = requantize_m2_s(_acc2, _mult, _shift, out_zp, vl);
            vint8mf2_t _res3 = requantize_m2_s(_acc3, _mult, _shift, out_zp, vl);

            vse8_v_i8mf2(output0, _res0, vl);
            vse8_v_i8mf2(output0 + packn * 1, _res1, vl);
            vse8_v_i8mf2(output0 + packn * 2, _res2, vl);
            vse8_v_i8mf2(output0 + packn * 3, _res3, vl);

            output0 += packn * 4;
        }
        for (; t + 1 < n; t += 2) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, img0[1], _kernel0, vl);
                img0 += 2;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, _mult, _shift, out_zp, vl);

            vse8_v_i8mf2(output0, _res0, vl);
            vse8_v_i8mf2(output0 + packn * 1, _res1, vl);
            output0 += packn * 2;
        }
        for (; t < n; t++) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                img0 += 1;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vse8_v_i8mf2(output0, _res0, vl);
            output0 += packn * 1;
        }
    }

    /* tail output_channel */
    if (oc < m) {
        vl = vsetvl_e32m2(m - oc);
        vint32m2_t _mult = vle32_v_i32m2(mult + oc, vl);
        vint32m2_t _shift = vle32_v_i32m2(shift + oc, vl);
        _shift = vrsub_vx_i32m2(_shift, -1, vl);

        int8_t *output0 = output_data + oc * n;
        const int32_t *img0 = (const int32_t *)input_data;
        const int32_t *b0 = bias_data + oc;

        int t = 0;
        for (; t + 7 < n; t += 8) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc2 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc3 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc4 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc5 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc6 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc7 = vmv_v_v_i32m2(_acc0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, img0[1], _kernel0, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, img0[2], _kernel0, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, img0[3], _kernel0, vl);
                _acc4 = vmaqa_vx_i32m2(_acc4, img0[4], _kernel0, vl);
                _acc5 = vmaqa_vx_i32m2(_acc5, img0[5], _kernel0, vl);
                _acc6 = vmaqa_vx_i32m2(_acc6, img0[6], _kernel0, vl);
                _acc7 = vmaqa_vx_i32m2(_acc7, img0[7], _kernel0, vl);

                img0 += 8;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, _mult, _shift, out_zp, vl);
            vint8mf2_t _res2 = requantize_m2_s(_acc2, _mult, _shift, out_zp, vl);
            vint8mf2_t _res3 = requantize_m2_s(_acc3, _mult, _shift, out_zp, vl);
            vint8mf2_t _res4 = requantize_m2_s(_acc4, _mult, _shift, out_zp, vl);
            vint8mf2_t _res5 = requantize_m2_s(_acc5, _mult, _shift, out_zp, vl);
            vint8mf2_t _res6 = requantize_m2_s(_acc6, _mult, _shift, out_zp, vl);
            vint8mf2_t _res7 = requantize_m2_s(_acc7, _mult, _shift, out_zp, vl);

            vse8_v_i8mf2(output0, _res0, vl);
            vse8_v_i8mf2(output0 + vl * 1, _res1, vl);
            vse8_v_i8mf2(output0 + vl * 2, _res2, vl);
            vse8_v_i8mf2(output0 + vl * 3, _res3, vl);
            vse8_v_i8mf2(output0 + vl * 4, _res4, vl);
            vse8_v_i8mf2(output0 + vl * 5, _res5, vl);
            vse8_v_i8mf2(output0 + vl * 6, _res6, vl);
            vse8_v_i8mf2(output0 + vl * 7, _res7, vl);

            output0 += vl * 8;
        }
        for (; t + 3 < n; t += 4) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc2 = vmv_v_v_i32m2(_acc0, vl);
            vint32m2_t _acc3 = vmv_v_v_i32m2(_acc0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, img0[1], _kernel0, vl);
                _acc2 = vmaqa_vx_i32m2(_acc2, img0[2], _kernel0, vl);
                _acc3 = vmaqa_vx_i32m2(_acc3, img0[3], _kernel0, vl);

                img0 += 4;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, _mult, _shift, out_zp, vl);
            vint8mf2_t _res2 = requantize_m2_s(_acc2, _mult, _shift, out_zp, vl);
            vint8mf2_t _res3 = requantize_m2_s(_acc3, _mult, _shift, out_zp, vl);

            vse8_v_i8mf2(output0, _res0, vl);
            vse8_v_i8mf2(output0 + vl * 1, _res1, vl);
            vse8_v_i8mf2(output0 + vl * 2, _res2, vl);
            vse8_v_i8mf2(output0 + vl * 3, _res3, vl);

            output0 += vl * 4;
        }
        for (; t + 1 < n; t += 2) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);
            vint32m2_t _acc1 = vmv_v_v_i32m2(_acc0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                _acc1 = vmaqa_vx_i32m2(_acc1, img0[1], _kernel0, vl);
                img0 += 2;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, _mult, _shift, out_zp, vl);

            vse8_v_i8mf2(output0, _res0, vl);
            vse8_v_i8mf2(output0 + vl * 1, _res1, vl);
            output0 += vl * 2;
        }
        for (; t < n; t++) {
            const int8_t *k0 = kernel_data + oc * k;
            vint32m2_t _acc0 = vle32_v_i32m2(b0, vl);

            for (int c = 0; c + 3 < k; c += 4) {
                vint8m2_t _kernel0 = vle8_v_i8m2(k0, vl * 4);
                k0 += vl * 4;
                _acc0 = vmaqa_vx_i32m2(_acc0, img0[0], _kernel0, vl);
                img0 += 1;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, _mult, _shift, out_zp, vl);
            vse8_v_i8mf2(output0, _res0, vl);
            output0 += vl * 1;
        }
    }
}
#endif
