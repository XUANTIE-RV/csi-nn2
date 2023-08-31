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
#ifdef SHL_USE_DOT_INT4
static vint8mf4_t requantize_m2(vint32m2_t _src, int32_t multiplier, int32_t shift, int32_t out_zp,
                                int vl)
{
    vint32m2_t _mulh = vmulh_vx_i32m2(_src, multiplier, vl);
    _mulh = vssra_vx_i32m2(_mulh, -shift - 1, vl);
    _mulh = vadd_vx_i32m2(_mulh, out_zp, vl);
    vint16m1_t _tmp1 = vnclip_wx_i16m1(_mulh, 0, vl);
    vint8mf2_t _tmp2 = vnclip_wx_i8mf2(_tmp1, 0, vl);
    vint8mf4_t _res = vpnclip_wx_i8mf4(vreinterpret_v_i8mf2_i16mf2(_tmp2), 0, vl / 2);
    return _res;
}

static vint8mf8_t requantize_m1(vint32m1_t _src, int32_t multiplier, int32_t shift, int32_t out_zp,
                                int vl)
{
    vint32m1_t _mulh = vmulh_vx_i32m1(_src, multiplier, vl);
    _mulh = vssra_vx_i32m1(_mulh, -shift - 1, vl);
    _mulh = vadd_vx_i32m1(_mulh, out_zp, vl);
    vint16mf2_t _tmp1 = vnclip_wx_i16mf2(_mulh, 0, vl);
    vint8mf4_t _tmp2 = vnclip_wx_i8mf4(_tmp1, 0, vl);
    vint8mf8_t _res = vpnclip_wx_i8mf8(vreinterpret_v_i8mf4_i16mf4(_tmp2), 0, vl / 2);
    return _res;
}

/*************************************************************
 * note: VLEN = 128
 * layerout: input/output-[n, h, w , c]  kernel-[o, h, w, i]
 *************************************************************/
void shl_rvv_reorder_input_n8_int4_dot(int8_t *a, int8_t *sa, int m, int k, int ldx)
{
    if (k % 4 == 0) {
        int i = 0;
        // m8
        for (; i + 7 < m; i += 8) {
            int j = 0;
            // k16
            int32_t *in_ptr0 = (int32_t *)a;
            int32_t *out_ptr0 = (int32_t *)sa;
            for (; j + 15 < k; j += 16) {
                vint32m2_t _nf0, _nf1, _nf2, _nf3;
                vlsseg4e32_v_i32m2(&_nf0, &_nf1, &_nf2, &_nf3, in_ptr0, k * sizeof(int8_t), 8);
                in_ptr0 += 4;
                vse32_v_i32m2(out_ptr0, _nf0, 8);
                out_ptr0 += 8;
                vse32_v_i32m2(out_ptr0, _nf1, 8);
                out_ptr0 += 8;
                vse32_v_i32m2(out_ptr0, _nf2, 8);
                out_ptr0 += 8;
                vse32_v_i32m2(out_ptr0, _nf3, 8);
                out_ptr0 += 8;
            }
            for (; j + 3 < k; j += 4) {
                vint32m2_t _input = vlse32_v_i32m2(in_ptr0, k * sizeof(int8_t), 8);
                in_ptr0++;
                vse32_v_i32m2(out_ptr0, _input, 8);
                out_ptr0 += 8;
            }
            if (j < k) {
                int8_t *in_ptr1 = (int8_t *)in_ptr0;
                int8_t *out_ptr1 = (int8_t *)out_ptr0;
                for (int c = 0; c < 8; c++) {
                    vint8m1_t _input1 = vle8_v_i8m1(in_ptr1, k & 3);
                    in_ptr1 += k;
                    vse8_v_i8m1(out_ptr1, _input1, 4);
                    out_ptr1 += 4;
                }
            }
            a += 8 * k;
            sa += 8 * k;
        }
        // m4
        for (; i + 3 < m; i += 4) {
            int j = 0;
            int32_t *in_ptr0 = (int32_t *)a;
            int32_t *out_ptr0 = (int32_t *)sa;
            for (; j + 15 < k; j += 16) {
                vint32m1_t _nf0, _nf1, _nf2, _nf3;
                vlsseg4e32_v_i32m1(&_nf0, &_nf1, &_nf2, &_nf3, in_ptr0, k * sizeof(int8_t), 4);
                in_ptr0 += 4;
                vse32_v_i32m1(out_ptr0, _nf0, 4);
                out_ptr0 += 4;
                vse32_v_i32m1(out_ptr0, _nf1, 4);
                out_ptr0 += 4;
                vse32_v_i32m1(out_ptr0, _nf2, 4);
                out_ptr0 += 4;
                vse32_v_i32m1(out_ptr0, _nf3, 4);
                out_ptr0 += 4;
            }
            for (; j + 3 < k; j += 4) {
                vint32m1_t _input = vlse32_v_i32m1(in_ptr0, k * sizeof(int8_t), 4);
                in_ptr0++;
                vse32_v_i32m1(out_ptr0, _input, 4);
                out_ptr0 += 4;
            }
            if (j < k) {
                int8_t *in_ptr1 = (int8_t *)in_ptr0;
                int8_t *out_ptr1 = (int8_t *)out_ptr0;
                for (int c = 0; c < 4; c++) {
                    vint8m1_t _input1 = vle8_v_i8m1(in_ptr1, k & 3);
                    in_ptr1 += k;
                    vse8_v_i8m1(out_ptr1, _input1, 4);
                    out_ptr1 += 4;
                }
            }
            a += 4 * k;
            sa += 4 * k;
        }
        // m2
        for (; i + 1 < m; i += 2) {
            int j = 0;
            for (; j + 3 < k; j += 4) {
                int8_t *in_ptr = a + j;
                for (int c = 0; c < 2; c++) {
                    vint8m1_t _input = vle8_v_i8m1(in_ptr, 4);
                    in_ptr += k;
                    vse8_v_i8m1(sa, _input, 4);
                    sa += 4;
                }
            }
            if (j < k) {
                int8_t *in_ptr = a + j;
                for (int c = 0; c < 2; c++) {
                    vint8m1_t _input = vle8_v_i8m1(in_ptr, k & 3);
                    in_ptr += k;
                    vse8_v_i8m1(sa, _input, k & 3);
                    sa += 4;
                }
            }
            a += 2 * k;
        }
        // m1
        for (; i < m; i++) {
            memcpy(sa, a, k * sizeof(int8_t));
        }
    } else {
        shl_rvv_reorder_kernel_n8_int8_dot(a, sa, m, k, ldx);
    }
}

// 和 shl_rvv_reorder_kernel_n8_int8 实现相同， 可以直接调用 shl_rvv_reorder_kernel_n8_int8
void shl_rvv_reorder_kernel_n8_int4(int8_t *b, int8_t *sb, int n, int k, int ldx)
{
    // TODO:
}

void shl_rvv_gemm_8x8_int4_dot(int8_t *dst, const int8_t *sa, const int8_t *sb, int m, int k, int n,
                               int ldc, int32_t *bias, int32_t out_zp, int32_t *mult,
                               int32_t *shift)
{
    int8_t *input_data = (int8_t *)sa;
    int8_t *kernel_data = (int8_t *)sb;
    int8_t *output_data = dst;
    // please use fuse_zp2bias option in hhb, thus bias_data wont be NULL
    int32_t *bias_data = bias;
    int vl = 0;
    int i = 0;
    // m8 loop
    vl = vsetvl_e32m2(8);
    for (; i + 7 < m; i += 8) {
        int8_t *kernel_ptr = kernel_data;

        int8_t *out_ptr0 = output_data;
        int8_t *out_ptr1 = out_ptr0 + ldc;
        int8_t *out_ptr2 = out_ptr1 + ldc;
        int8_t *out_ptr3 = out_ptr2 + ldc;
        int8_t *out_ptr4 = out_ptr3 + ldc;
        int8_t *out_ptr5 = out_ptr4 + ldc;
        int8_t *out_ptr6 = out_ptr5 + ldc;
        int8_t *out_ptr7 = out_ptr6 + ldc;  // ldc = m =  h * w * inc
        int j = 0;
        // n8m8 loop
        for (; j + 7 < n; j += 8) {
            int32_t *in_ptr = (int32_t *)input_data;
            vint32m2_t _acc0 = vle32_v_i32m2(bias_data + j, 8);
            vint32m2_t _acc1 = vle32_v_i32m2(bias_data + j, 8);
            vint32m2_t _acc2 = vle32_v_i32m2(bias_data + j, 8);
            vint32m2_t _acc3 = vle32_v_i32m2(bias_data + j, 8);
            vint32m2_t _acc4 = vle32_v_i32m2(bias_data + j, 8);
            vint32m2_t _acc5 = vle32_v_i32m2(bias_data + j, 8);
            vint32m2_t _acc6 = vle32_v_i32m2(bias_data + j, 8);
            vint32m2_t _acc7 = vle32_v_i32m2(bias_data + j, 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _kernel = vle8_v_i8m2(kernel_ptr, 32);
                _acc0 = vpmaqa_vx_i32m2(_acc0, in_ptr[0], _kernel, 8);
                _acc1 = vpmaqa_vx_i32m2(_acc1, in_ptr[1], _kernel, 8);
                _acc2 = vpmaqa_vx_i32m2(_acc2, in_ptr[2], _kernel, 8);
                _acc3 = vpmaqa_vx_i32m2(_acc3, in_ptr[3], _kernel, 8);
                _acc4 = vpmaqa_vx_i32m2(_acc4, in_ptr[4], _kernel, 8);
                _acc5 = vpmaqa_vx_i32m2(_acc5, in_ptr[5], _kernel, 8);
                _acc6 = vpmaqa_vx_i32m2(_acc6, in_ptr[6], _kernel, 8);
                _acc7 = vpmaqa_vx_i32m2(_acc7, in_ptr[7], _kernel, 8);

                in_ptr += 8;
                kernel_ptr += 32;
            }
            vint8mf4_t _res0 = requantize_m2(_acc0, mult[j], shift[j], out_zp, 8);
            vint8mf4_t _res1 = requantize_m2(_acc1, mult[j], shift[j], out_zp, 8);
            vint8mf4_t _res2 = requantize_m2(_acc2, mult[j], shift[j], out_zp, 8);
            vint8mf4_t _res3 = requantize_m2(_acc3, mult[j], shift[j], out_zp, 8);
            vint8mf4_t _res4 = requantize_m2(_acc4, mult[j], shift[j], out_zp, 8);
            vint8mf4_t _res5 = requantize_m2(_acc5, mult[j], shift[j], out_zp, 8);
            vint8mf4_t _res6 = requantize_m2(_acc6, mult[j], shift[j], out_zp, 8);
            vint8mf4_t _res7 = requantize_m2(_acc7, mult[j], shift[j], out_zp, 8);

            vse8_v_i8mf4(out_ptr0, _res0, 4);
            vse8_v_i8mf4(out_ptr1, _res1, 4);
            vse8_v_i8mf4(out_ptr2, _res2, 4);
            vse8_v_i8mf4(out_ptr3, _res3, 4);
            vse8_v_i8mf4(out_ptr4, _res4, 4);
            vse8_v_i8mf4(out_ptr5, _res5, 4);
            vse8_v_i8mf4(out_ptr6, _res6, 4);
            vse8_v_i8mf4(out_ptr7, _res7, 4);
            out_ptr0 += 4;
            out_ptr1 += 4;
            out_ptr2 += 4;
            out_ptr3 += 4;
            out_ptr4 += 4;
            out_ptr5 += 4;
            out_ptr6 += 4;
            out_ptr7 += 4;
        }
        // m8n4
        for (; j + 3 < n; j += 4) {
            int32_t *in_ptr = (int32_t *)input_data;
            vint32m1_t _acc0 = vle32_v_i32m1(bias_data + j, 4);
            vint32m1_t _acc1 = vle32_v_i32m1(bias_data + j, 4);
            vint32m1_t _acc2 = vle32_v_i32m1(bias_data + j, 4);
            vint32m1_t _acc3 = vle32_v_i32m1(bias_data + j, 4);
            vint32m1_t _acc4 = vle32_v_i32m1(bias_data + j, 4);
            vint32m1_t _acc5 = vle32_v_i32m1(bias_data + j, 4);
            vint32m1_t _acc6 = vle32_v_i32m1(bias_data + j, 4);
            vint32m1_t _acc7 = vle32_v_i32m1(bias_data + j, 4);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _kernel = vle8_v_i8m1(kernel_ptr, 16);
                _acc0 = vpmaqa_vx_i32m1(_acc0, in_ptr[0], _kernel, 4);
                _acc1 = vpmaqa_vx_i32m1(_acc1, in_ptr[1], _kernel, 4);
                _acc2 = vpmaqa_vx_i32m1(_acc2, in_ptr[2], _kernel, 4);
                _acc3 = vpmaqa_vx_i32m1(_acc3, in_ptr[3], _kernel, 4);
                _acc4 = vpmaqa_vx_i32m1(_acc4, in_ptr[4], _kernel, 4);
                _acc5 = vpmaqa_vx_i32m1(_acc5, in_ptr[5], _kernel, 4);
                _acc6 = vpmaqa_vx_i32m1(_acc6, in_ptr[6], _kernel, 4);
                _acc7 = vpmaqa_vx_i32m1(_acc7, in_ptr[7], _kernel, 4);

                in_ptr += 8;
                kernel_ptr += 16;
            }
            vint8mf8_t _res0 = requantize_m1(_acc0, mult[j], shift[j], out_zp, 4);
            vint8mf8_t _res1 = requantize_m1(_acc1, mult[j], shift[j], out_zp, 4);
            vint8mf8_t _res2 = requantize_m1(_acc2, mult[j], shift[j], out_zp, 4);
            vint8mf8_t _res3 = requantize_m1(_acc3, mult[j], shift[j], out_zp, 4);
            vint8mf8_t _res4 = requantize_m1(_acc4, mult[j], shift[j], out_zp, 4);
            vint8mf8_t _res5 = requantize_m1(_acc5, mult[j], shift[j], out_zp, 4);
            vint8mf8_t _res6 = requantize_m1(_acc6, mult[j], shift[j], out_zp, 4);
            vint8mf8_t _res7 = requantize_m1(_acc7, mult[j], shift[j], out_zp, 4);
            vse8_v_i8mf8(out_ptr0, _res0, 2);
            vse8_v_i8mf8(out_ptr1, _res1, 2);
            vse8_v_i8mf8(out_ptr2, _res2, 2);
            vse8_v_i8mf8(out_ptr3, _res3, 2);
            vse8_v_i8mf8(out_ptr4, _res4, 2);
            vse8_v_i8mf8(out_ptr5, _res5, 2);
            vse8_v_i8mf8(out_ptr6, _res6, 2);
            vse8_v_i8mf8(out_ptr7, _res7, 2);
            out_ptr0 += 2;
            out_ptr1 += 2;
            out_ptr2 += 2;
            out_ptr3 += 2;
            out_ptr4 += 2;
            out_ptr5 += 2;
            out_ptr6 += 2;
            out_ptr7 += 2;
        }
        // m8n2
        for (; j + 1 < n; j += 2) {
            // TODO:
        }

        input_data += 8 * k;
        output_data += 8 * ldc;
    }
    // m4
    for (; i + 3 < m; i += 4) {
        int8_t *kernel_ptr = kernel_data;

        int8_t *out_ptr0 = output_data;
        int8_t *out_ptr1 = out_ptr0 + ldc;
        int8_t *out_ptr2 = out_ptr1 + ldc;
        int8_t *out_ptr3 = out_ptr2 + ldc;
        int j = 0;
        // m4n8 loop
        for (; j + 7 < n; j += 8) {
            int32_t *in_ptr = (int32_t *)input_data;
            vint32m2_t _acc0 = vle32_v_i32m2(bias_data + j, 8);
            vint32m2_t _acc1 = vle32_v_i32m2(bias_data + j, 8);
            vint32m2_t _acc2 = vle32_v_i32m2(bias_data + j, 8);
            vint32m2_t _acc3 = vle32_v_i32m2(bias_data + j, 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _kernel = vle8_v_i8m2(kernel_ptr, 32);
                _acc0 = vpmaqa_vx_i32m2(_acc0, in_ptr[0], _kernel, 8);
                _acc1 = vpmaqa_vx_i32m2(_acc1, in_ptr[1], _kernel, 8);
                _acc2 = vpmaqa_vx_i32m2(_acc2, in_ptr[2], _kernel, 8);
                _acc3 = vpmaqa_vx_i32m2(_acc3, in_ptr[3], _kernel, 8);

                in_ptr += 4;
                kernel_ptr += 32;
            }
            vint8mf4_t _res0 = requantize_m2(_acc0, mult[j], shift[j], out_zp, 8);
            vint8mf4_t _res1 = requantize_m2(_acc1, mult[j], shift[j], out_zp, 8);
            vint8mf4_t _res2 = requantize_m2(_acc2, mult[j], shift[j], out_zp, 8);
            vint8mf4_t _res3 = requantize_m2(_acc3, mult[j], shift[j], out_zp, 8);
            vse8_v_i8mf4(out_ptr0, _res0, 4);
            vse8_v_i8mf4(out_ptr1, _res1, 4);
            vse8_v_i8mf4(out_ptr2, _res2, 4);
            vse8_v_i8mf4(out_ptr3, _res3, 4);
            out_ptr0 += 4;
            out_ptr1 += 4;
            out_ptr2 += 4;
            out_ptr3 += 4;
        }
        // m4n4
        for (; j + 3 < n; j += 4) {
            int32_t *in_ptr = (int32_t *)input_data;
            vint32m1_t _acc0 = vle32_v_i32m1(bias_data + j, 4);
            vint32m1_t _acc1 = vle32_v_i32m1(bias_data + j, 4);
            vint32m1_t _acc2 = vle32_v_i32m1(bias_data + j, 4);
            vint32m1_t _acc3 = vle32_v_i32m1(bias_data + j, 4);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _kernel = vle8_v_i8m1(kernel_ptr, 16);
                _acc0 = vpmaqa_vx_i32m1(_acc0, in_ptr[0], _kernel, 4);
                _acc1 = vpmaqa_vx_i32m1(_acc1, in_ptr[1], _kernel, 4);
                _acc2 = vpmaqa_vx_i32m1(_acc2, in_ptr[2], _kernel, 4);
                _acc3 = vpmaqa_vx_i32m1(_acc3, in_ptr[3], _kernel, 4);

                in_ptr += 4;
                kernel_ptr += 16;
            }
            vint8mf8_t _res0 = requantize_m1(_acc0, mult[j], shift[j], out_zp, 4);
            vint8mf8_t _res1 = requantize_m1(_acc1, mult[j], shift[j], out_zp, 4);
            vint8mf8_t _res2 = requantize_m1(_acc2, mult[j], shift[j], out_zp, 4);
            vint8mf8_t _res3 = requantize_m1(_acc3, mult[j], shift[j], out_zp, 4);
            vse8_v_i8mf8(out_ptr0, _res0, 2);
            vse8_v_i8mf8(out_ptr1, _res1, 2);
            vse8_v_i8mf8(out_ptr2, _res2, 2);
            vse8_v_i8mf8(out_ptr3, _res3, 2);
            out_ptr0 += 2;
            out_ptr1 += 2;
            out_ptr2 += 2;
            out_ptr3 += 2;
        }
        // m4n2
        for (; j + 1 < n; j += 2) {
            // TODO:
        }

        input_data += 4 * k;
        output_data += 4 * ldc;
    }
    // m2
    for (; i + 1 < m; i += 2) {
        int8_t *kernel_ptr = kernel_data;

        int8_t *out_ptr0 = output_data;
        int8_t *out_ptr1 = out_ptr0 + ldc;
        int j = 0;
        // m2n8 loop
        for (; j + 7 < n; j += 8) {
            int32_t *in_ptr = (int32_t *)input_data;
            vint32m2_t _acc0 = vle32_v_i32m2(bias_data + j, 8);
            vint32m2_t _acc1 = vle32_v_i32m2(bias_data + j, 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _kernel = vle8_v_i8m2(kernel_ptr, 32);
                _acc0 = vpmaqa_vx_i32m2(_acc0, in_ptr[0], _kernel, 8);
                _acc1 = vpmaqa_vx_i32m2(_acc1, in_ptr[1], _kernel, 8);
                in_ptr += 2;
                kernel_ptr += 32;
            }
            vint8mf4_t _res0 = requantize_m2(_acc0, mult[j], shift[j], out_zp, 8);
            vint8mf4_t _res1 = requantize_m2(_acc1, mult[j], shift[j], out_zp, 8);
            vse8_v_i8mf4(out_ptr0, _res0, 4);
            vse8_v_i8mf4(out_ptr1, _res1, 4);
            out_ptr0 += 4;
            out_ptr1 += 4;
        }
        // m2n4
        for (; j + 3 < n; j += 4) {
            int32_t *in_ptr = (int32_t *)input_data;
            vint32m1_t _acc0 = vle32_v_i32m1(bias_data + j, 4);
            vint32m1_t _acc1 = vle32_v_i32m1(bias_data + j, 4);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _kernel = vle8_v_i8m1(kernel_ptr, 16);
                _acc0 = vpmaqa_vx_i32m1(_acc0, in_ptr[0], _kernel, 4);
                _acc1 = vpmaqa_vx_i32m1(_acc1, in_ptr[1], _kernel, 4);
                in_ptr += 2;
                kernel_ptr += 16;
            }
            vint8mf8_t _res0 = requantize_m1(_acc0, mult[j], shift[j], out_zp, 4);
            vint8mf8_t _res1 = requantize_m1(_acc1, mult[j], shift[j], out_zp, 4);
            vse8_v_i8mf8(out_ptr0, _res0, 2);
            vse8_v_i8mf8(out_ptr1, _res1, 2);
            out_ptr0 += 2;
            out_ptr1 += 2;
        }
        // m2n2
        for (; j + 1 < n; j += 2) {
            // TODO:
        }

        input_data += 2 * k;
        output_data += 2 * ldc;
    }
    // m1
    for (; i < m; i++) {
        int8_t *kernel_ptr = kernel_data;

        int8_t *out_ptr0 = output_data;
        int j = 0;
        // m1n8 loop
        for (; j + 7 < n; j += 8) {
            int32_t *in_ptr = (int32_t *)input_data;
            vint32m2_t _acc0 = vle32_v_i32m2(bias_data + j, 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _kernel = vle8_v_i8m2(kernel_ptr, 32);
                _acc0 = vpmaqa_vx_i32m2(_acc0, in_ptr[0], _kernel, 8);
                in_ptr += 1;
                kernel_ptr += 32;
            }
            vint8mf4_t _res0 = requantize_m2(_acc0, mult[j], shift[j], out_zp, 8);
            vse8_v_i8mf4(out_ptr0, _res0, 4);
            out_ptr0 += 4;
        }
        // m1n4
        for (; j + 3 < n; j += 4) {
            int32_t *in_ptr = (int32_t *)input_data;
            vint32m1_t _acc0 = vle32_v_i32m1(bias_data + j, 4);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _kernel = vle8_v_i8m1(kernel_ptr, 16);
                _acc0 = vpmaqa_vx_i32m1(_acc0, in_ptr[0], _kernel, 4);
                in_ptr += 1;
                kernel_ptr += 16;
            }
            vint8mf8_t _res0 = requantize_m1(_acc0, mult[j], shift[j], out_zp, 4);
            vse8_v_i8mf8(out_ptr0, _res0, 2);
            out_ptr0 += 2;
        }
        // m1n2
        for (; j + 1 < n; j += 2) {
            // TODO:
        }
    }
}

#endif