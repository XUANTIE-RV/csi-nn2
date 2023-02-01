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
#ifdef SHL_USE_DOT_INT8
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

static vint8mf4_t requantize_m1(vint32m1_t _src, int32_t multiplier, int32_t shift, int32_t out_zp,
                                int vl)
{
    vint32m1_t _mulh = vmulh_vx_i32m1(_src, multiplier, vl);
    _mulh = vssra_vx_i32m1(_mulh, -shift - 1, vl);
    _mulh = vadd_vx_i32m1(_mulh, out_zp, vl);
    vint16mf2_t _tmp1 = vnclip_wx_i16mf2(_mulh, 0, vl);
    vint8mf4_t _tmp2 = vnclip_wx_i8mf4(_tmp1, 0, vl);
    return _tmp2;
}

static int8_t requantize_single(int32_t src, int32_t multiplier, int32_t shift, int32_t out_zp)
{
    int64_t src_64 = (int64_t)src;
    int64_t mult_64 = (int64_t)multiplier;
    int64_t mulw = src_64 * multiplier;
    int32_t nudge = mulw >= 0 ? (1 << 30) : (1 - (1 << 30));
    int32_t mulh = (int32_t)((mulw + nudge) / (1ll << 31));
    int32_t res = mulh >> (-shift);
    res += out_zp;
    if (res > 127) res = 127;
    if (res < -128) res = -128;
    return (int8_t)res;
}

static vint8mf2_t requantize_m2_s(vint32m2_t _src, int32_t *multiplier, int32_t *shift,
                                  int32_t out_zp, int vl)
{
    vint32m2_t _mult = vle32_v_i32m2(multiplier, vl);
    vint32m2_t _shift = vle32_v_i32m2(shift, vl);
    vint32m2_t _mulh = vmulh_vv_i32m2(_src, _mult, vl);
    _shift = vrsub_vx_i32m2(_shift, -1, vl);
    _mulh = vssra_vv_i32m2(_mulh, vreinterpret_v_i32m2_u32m2(_shift), vl);
    _mulh = vadd_vx_i32m2(_mulh, out_zp, vl);
    vint16m1_t _tmp1 = vnclip_wx_i16m1(_mulh, 0, vl);
    vint8mf2_t _tmp2 = vnclip_wx_i8mf2(_tmp1, 0, vl);
    return _tmp2;
}

static vint8mf4_t requantize_m1_s(vint32m1_t _src, int32_t *multiplier, int32_t *shift,
                                  int32_t out_zp, int vl)
{
    vint32m1_t _mult = vle32_v_i32m1(multiplier, vl);
    vint32m1_t _shift = vle32_v_i32m1(shift, vl);
    vint32m1_t _mulh = vmulh_vv_i32m1(_src, _mult, vl);
    _shift = vrsub_vx_i32m1(_shift, -1, vl);
    _mulh = vssra_vv_i32m1(_mulh, vreinterpret_v_i32m1_u32m1(_shift), vl);
    _mulh = vadd_vx_i32m1(_mulh, out_zp, vl);
    vint16mf2_t _tmp1 = vnclip_wx_i16mf2(_mulh, 0, vl);
    vint8mf4_t _tmp2 = vnclip_wx_i8mf4(_tmp1, 0, vl);
    return _tmp2;
}

// vlen=128
void shl_rvv_gemm_8x8_int32(int32_t *dst, const int8_t *sa, const int8_t *sb, int32_t *bias, int m,
                            int k, int n, int ldc)
{
    int8_t *kernel_data = (int8_t *)sa;
    int8_t *input_data = (int8_t *)sb;
    int32_t *output_data = dst;
    // please use fuse_zp2bias option in hhb, thus bias_data wont be NULL
    int32_t *bias_data = bias;

    int vl = 0;
    int i = 0;
    // m8 loop
    vl = vsetvl_e32m2(8);
    for (; i + 7 < m; i += 8) {
        int8_t *in_ptr = input_data;

        int32_t *out_ptr0 = output_data;
        int32_t *out_ptr1 = out_ptr0 + ldc;
        int32_t *out_ptr2 = out_ptr1 + ldc;
        int32_t *out_ptr3 = out_ptr2 + ldc;
        int32_t *out_ptr4 = out_ptr3 + ldc;
        int32_t *out_ptr5 = out_ptr4 + ldc;
        int32_t *out_ptr6 = out_ptr5 + ldc;
        int32_t *out_ptr7 = out_ptr6 + ldc;
        int j = 0;
        // m8n8 loop
        for (; j + 7 < n; j += 8) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m2_t _acc0 = vmv_v_x_i32m2(bias_data[0], 8);
            vint32m2_t _acc1 = vmv_v_x_i32m2(bias_data[1], 8);
            vint32m2_t _acc2 = vmv_v_x_i32m2(bias_data[2], 8);
            vint32m2_t _acc3 = vmv_v_x_i32m2(bias_data[3], 8);
            vint32m2_t _acc4 = vmv_v_x_i32m2(bias_data[4], 8);
            vint32m2_t _acc5 = vmv_v_x_i32m2(bias_data[5], 8);
            vint32m2_t _acc6 = vmv_v_x_i32m2(bias_data[6], 8);
            vint32m2_t _acc7 = vmv_v_x_i32m2(bias_data[7], 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _input = vle8_v_i8m2(in_ptr, 32);
                _acc0 = vmaqa_vx_i32m2(_acc0, kernel_ptr[0], _input, 8);
                _acc1 = vmaqa_vx_i32m2(_acc1, kernel_ptr[1], _input, 8);
                _acc2 = vmaqa_vx_i32m2(_acc2, kernel_ptr[2], _input, 8);
                _acc3 = vmaqa_vx_i32m2(_acc3, kernel_ptr[3], _input, 8);
                _acc4 = vmaqa_vx_i32m2(_acc4, kernel_ptr[4], _input, 8);
                _acc5 = vmaqa_vx_i32m2(_acc5, kernel_ptr[5], _input, 8);
                _acc6 = vmaqa_vx_i32m2(_acc6, kernel_ptr[6], _input, 8);
                _acc7 = vmaqa_vx_i32m2(_acc7, kernel_ptr[7], _input, 8);

                kernel_ptr += 8;
                in_ptr += 32;
            }
            vse32_v_i32m2(out_ptr0, _acc0, 8);
            vse32_v_i32m2(out_ptr1, _acc1, 8);
            vse32_v_i32m2(out_ptr2, _acc2, 8);
            vse32_v_i32m2(out_ptr3, _acc3, 8);
            vse32_v_i32m2(out_ptr4, _acc4, 8);
            vse32_v_i32m2(out_ptr5, _acc5, 8);
            vse32_v_i32m2(out_ptr6, _acc6, 8);
            vse32_v_i32m2(out_ptr7, _acc7, 8);
            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
            out_ptr4 += 8;
            out_ptr5 += 8;
            out_ptr6 += 8;
            out_ptr7 += 8;
        }
        // m8n4
        for (; j + 3 < n; j += 4) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m1_t _acc0 = vmv_v_x_i32m1(bias_data[0], 4);
            vint32m1_t _acc1 = vmv_v_x_i32m1(bias_data[1], 4);
            vint32m1_t _acc2 = vmv_v_x_i32m1(bias_data[2], 4);
            vint32m1_t _acc3 = vmv_v_x_i32m1(bias_data[3], 4);
            vint32m1_t _acc4 = vmv_v_x_i32m1(bias_data[4], 4);
            vint32m1_t _acc5 = vmv_v_x_i32m1(bias_data[5], 4);
            vint32m1_t _acc6 = vmv_v_x_i32m1(bias_data[6], 4);
            vint32m1_t _acc7 = vmv_v_x_i32m1(bias_data[7], 4);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 16);
                _acc0 = vmaqa_vx_i32m1(_acc0, kernel_ptr[0], _input, 4);
                _acc1 = vmaqa_vx_i32m1(_acc1, kernel_ptr[1], _input, 4);
                _acc2 = vmaqa_vx_i32m1(_acc2, kernel_ptr[2], _input, 4);
                _acc3 = vmaqa_vx_i32m1(_acc3, kernel_ptr[3], _input, 4);
                _acc4 = vmaqa_vx_i32m1(_acc4, kernel_ptr[4], _input, 4);
                _acc5 = vmaqa_vx_i32m1(_acc5, kernel_ptr[5], _input, 4);
                _acc6 = vmaqa_vx_i32m1(_acc6, kernel_ptr[6], _input, 4);
                _acc7 = vmaqa_vx_i32m1(_acc7, kernel_ptr[7], _input, 4);

                kernel_ptr += 8;
                in_ptr += 16;
            }
            vse32_v_i32m1(out_ptr0, _acc0, 4);
            vse32_v_i32m1(out_ptr1, _acc1, 4);
            vse32_v_i32m1(out_ptr2, _acc2, 4);
            vse32_v_i32m1(out_ptr3, _acc3, 4);
            vse32_v_i32m1(out_ptr4, _acc4, 4);
            vse32_v_i32m1(out_ptr5, _acc5, 4);
            vse32_v_i32m1(out_ptr6, _acc6, 4);
            vse32_v_i32m1(out_ptr7, _acc7, 4);
            out_ptr0 += 4;
            out_ptr1 += 4;
            out_ptr2 += 4;
            out_ptr3 += 4;
            out_ptr4 += 4;
            out_ptr5 += 4;
            out_ptr6 += 4;
            out_ptr7 += 4;
        }
        // m8n2
        for (; j + 1 < n; j += 2) {
            int8_t *kernel_ptr = kernel_data;
            vint32m2_t _acc0 = vle32_v_i32m2(bias_data, 8);
            vint32m2_t _acc1 = vle32_v_i32m2(bias_data, 8);

            int32_t *in_ptr0 = (int32_t *)in_ptr;
            int32_t *in_ptr1 = (int32_t *)(in_ptr + k);

            out_ptr1 = out_ptr0 + 1;
            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _kernel = vle8_v_i8m2(kernel_ptr, 32);
                _acc0 = vmaqa_vx_i32m2(_acc0, in_ptr0[0], _kernel, 8);
                _acc1 = vmaqa_vx_i32m2(_acc1, in_ptr1[0], _kernel, 8);
                in_ptr0++;
                in_ptr1++;
                kernel_ptr += 32;
            }
            vsse32_v_i32m2(out_ptr0, ldc * sizeof(int32_t), _acc0, 8);
            vsse32_v_i32m2(out_ptr1, ldc * sizeof(int32_t), _acc1, 8);
            out_ptr0 += 2;
            in_ptr += 2 * k;
        }
        // m8n1
        for (; j < n; j++) {
            int8_t *kernel_ptr = kernel_data;
            vint32m2_t _acc0 = vle32_v_i32m2(bias_data, 8);
            int32_t *in_ptr0 = (int32_t *)in_ptr;
            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _kernel = vle8_v_i8m2(kernel_ptr, 32);
                _acc0 = vmaqa_vx_i32m2(_acc0, in_ptr0[0], _kernel, 8);
                in_ptr0++;
                kernel_ptr += 32;
            }
            vsse32_v_i32m2(out_ptr0, ldc * sizeof(int32_t), _acc0, 8);
            out_ptr0 += 1;
            in_ptr += 1 * k;
        }
        kernel_data += 8 * k;
        output_data += 8 * ldc;
        bias_data += 8;
    }
    // m4
    for (; i + 3 < m; i += 4) {
        int8_t *in_ptr = input_data;

        int32_t *out_ptr0 = output_data;
        int32_t *out_ptr1 = out_ptr0 + ldc;
        int32_t *out_ptr2 = out_ptr1 + ldc;
        int32_t *out_ptr3 = out_ptr2 + ldc;
        int j = 0;
        // m4n8 loop
        for (; j + 7 < n; j += 8) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m2_t _acc0 = vmv_v_x_i32m2(bias_data[0], 8);
            vint32m2_t _acc1 = vmv_v_x_i32m2(bias_data[1], 8);
            vint32m2_t _acc2 = vmv_v_x_i32m2(bias_data[2], 8);
            vint32m2_t _acc3 = vmv_v_x_i32m2(bias_data[3], 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _input = vle8_v_i8m2(in_ptr, 32);
                _acc0 = vmaqa_vx_i32m2(_acc0, kernel_ptr[0], _input, 8);
                _acc1 = vmaqa_vx_i32m2(_acc1, kernel_ptr[1], _input, 8);
                _acc2 = vmaqa_vx_i32m2(_acc2, kernel_ptr[2], _input, 8);
                _acc3 = vmaqa_vx_i32m2(_acc3, kernel_ptr[3], _input, 8);

                kernel_ptr += 4;
                in_ptr += 32;
            }
            vse32_v_i32m2(out_ptr0, _acc0, 8);
            vse32_v_i32m2(out_ptr1, _acc1, 8);
            vse32_v_i32m2(out_ptr2, _acc2, 8);
            vse32_v_i32m2(out_ptr3, _acc3, 8);
            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
        }
        // m4n4
        for (; j + 3 < n; j += 4) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m1_t _acc0 = vmv_v_x_i32m1(bias_data[0], 4);
            vint32m1_t _acc1 = vmv_v_x_i32m1(bias_data[1], 4);
            vint32m1_t _acc2 = vmv_v_x_i32m1(bias_data[2], 4);
            vint32m1_t _acc3 = vmv_v_x_i32m1(bias_data[3], 4);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 16);
                _acc0 = vmaqa_vx_i32m1(_acc0, kernel_ptr[0], _input, 4);
                _acc1 = vmaqa_vx_i32m1(_acc1, kernel_ptr[1], _input, 4);
                _acc2 = vmaqa_vx_i32m1(_acc2, kernel_ptr[2], _input, 4);
                _acc3 = vmaqa_vx_i32m1(_acc3, kernel_ptr[3], _input, 4);

                kernel_ptr += 4;
                in_ptr += 16;
            }
            vse32_v_i32m1(out_ptr0, _acc0, 4);
            vse32_v_i32m1(out_ptr1, _acc1, 4);
            vse32_v_i32m1(out_ptr2, _acc2, 4);
            vse32_v_i32m1(out_ptr3, _acc3, 4);
            out_ptr0 += 4;
            out_ptr1 += 4;
            out_ptr2 += 4;
            out_ptr3 += 4;
        }
        // m4n2
        for (; j + 1 < n; j += 2) {
            int8_t *kernel_ptr = kernel_data;
            vint32m1_t _acc0 = vle32_v_i32m1(bias_data, 4);
            vint32m1_t _acc1 = vle32_v_i32m1(bias_data, 4);

            int32_t *in_ptr0 = (int32_t *)in_ptr;
            int32_t *in_ptr1 = (int32_t *)(in_ptr + k);

            out_ptr1 = out_ptr0 + 1;
            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _kernel = vle8_v_i8m1(kernel_ptr, 16);
                _acc0 = vmaqa_vx_i32m1(_acc0, in_ptr0[0], _kernel, 4);
                _acc1 = vmaqa_vx_i32m1(_acc1, in_ptr1[0], _kernel, 4);
                in_ptr0++;
                in_ptr1++;
                kernel_ptr += 16;
            }
            vsse32_v_i32m1(out_ptr0, ldc * sizeof(int32_t), _acc0, 4);
            vsse32_v_i32m1(out_ptr1, ldc * sizeof(int32_t), _acc1, 4);
            out_ptr0 += 2;
            in_ptr += 2 * k;
        }
        // m4n1
        for (; j < n; j++) {
            int8_t *kernel_ptr = kernel_data;
            vint32m1_t _acc0 = vle32_v_i32m1(bias_data, 4);
            int32_t *in_ptr0 = (int32_t *)in_ptr;
            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _kernel = vle8_v_i8m1(kernel_ptr, 16);
                _acc0 = vmaqa_vx_i32m1(_acc0, in_ptr0[0], _kernel, 4);
                in_ptr0++;
                kernel_ptr += 16;
            }
            vsse32_v_i32m1(out_ptr0, ldc * sizeof(int32_t), _acc0, 4);
            out_ptr0 += 1;
            in_ptr += 1 * k;
        }
        kernel_data += 4 * k;
        output_data += 4 * ldc;
        bias_data += 4;
    }
    // m2
    for (; i + 1 < m; i += 2) {
        int8_t *in_ptr = input_data;

        int32_t *out_ptr0 = output_data;
        int32_t *out_ptr1 = out_ptr0 + ldc;
        int j = 0;
        // m2n8 loop
        for (; j + 7 < n; j += 8) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m2_t _acc0 = vmv_v_x_i32m2(bias_data[0], 8);
            vint32m2_t _acc1 = vmv_v_x_i32m2(bias_data[1], 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _input = vle8_v_i8m2(in_ptr, 32);
                _acc0 = vmaqa_vx_i32m2(_acc0, kernel_ptr[0], _input, 8);
                _acc1 = vmaqa_vx_i32m2(_acc1, kernel_ptr[1], _input, 8);

                kernel_ptr += 2;
                in_ptr += 32;
            }
            vse32_v_i32m2(out_ptr0, _acc0, 8);
            vse32_v_i32m2(out_ptr1, _acc1, 8);
            out_ptr0 += 8;
            out_ptr1 += 8;
        }
        // m2n4
        for (; j + 3 < n; j += 4) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m1_t _acc0 = vmv_v_x_i32m1(bias_data[0], 4);
            vint32m1_t _acc1 = vmv_v_x_i32m1(bias_data[1], 4);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 16);
                _acc0 = vmaqa_vx_i32m1(_acc0, kernel_ptr[0], _input, 4);
                _acc1 = vmaqa_vx_i32m1(_acc1, kernel_ptr[1], _input, 4);

                kernel_ptr += 2;
                in_ptr += 16;
            }
            vse32_v_i32m1(out_ptr0, _acc0, 4);
            vse32_v_i32m1(out_ptr1, _acc1, 4);
            out_ptr0 += 4;
            out_ptr1 += 4;
        }
        // m2n_tail
        for (; j < n; j++) {
            int32_t acc0 = bias_data[0];
            int32_t acc1 = bias_data[1];
            int8_t *k0 = kernel_data;
            int c = 0;
            for (; c + 3 < k; c += 4) {
                acc0 += k0[0] * in_ptr[c + 0];
                acc0 += k0[1] * in_ptr[c + 1];
                acc0 += k0[2] * in_ptr[c + 2];
                acc0 += k0[3] * in_ptr[c + 3];
                acc1 += k0[4] * in_ptr[c + 0];
                acc1 += k0[5] * in_ptr[c + 1];
                acc1 += k0[6] * in_ptr[c + 2];
                acc1 += k0[7] * in_ptr[c + 3];
                k0 += 8;
            }
            *out_ptr0++ = acc0;
            *out_ptr1++ = acc1;
            in_ptr += k;
        }
        kernel_data += 2 * k;
        output_data += 2 * ldc;
        bias_data += 2;
    }
    // m1
    for (; i < m; i++) {
        int8_t *in_ptr = input_data;
        int32_t *out_ptr0 = output_data;
        int j = 0;
        // m1n8 loop
        for (; j + 7 < n; j += 8) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m2_t _acc0 = vmv_v_x_i32m2(bias_data[0], 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _input = vle8_v_i8m2(in_ptr, 32);
                _acc0 = vmaqa_vx_i32m2(_acc0, kernel_ptr[0], _input, 8);
                kernel_ptr += 1;
                in_ptr += 32;
            }
            vse32_v_i32m2(out_ptr0, _acc0, 8);
            out_ptr0 += 8;
        }
        // m1n4
        for (; j + 3 < n; j += 4) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m1_t _acc0 = vmv_v_x_i32m1(bias_data[0], 4);
            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 16);
                _acc0 = vmaqa_vx_i32m1(_acc0, kernel_ptr[0], _input, 4);
                kernel_ptr += 1;
                in_ptr += 16;
            }
            vse32_v_i32m1(out_ptr0, _acc0, 4);
            out_ptr0 += 4;
        }
        // m1n_tail
        for (; j < n; j++) {
            int32_t acc0 = bias_data[0];
            for (int c = 0; c < k; c++) {
                acc0 += kernel_data[c] * in_ptr[c];
            }
            *out_ptr0++ = acc0;
            in_ptr += k;
        }
    }
}

void shl_rvv_gemm_8x8_int8_dot(int8_t *dst, const int8_t *sa, const int8_t *sb, int32_t *bias,
                               int m, int k, int n, int ldc, int32_t out_zp, int32_t *mult,
                               int32_t *shift)
{
    int8_t *kernel_data = (int8_t *)sa;
    int8_t *input_data = (int8_t *)sb;
    int8_t *output_data = dst;
    // please use fuse_zp2bias option in hhb, thus bias_data wont be NULL
    int32_t *bias_data = bias;

    int vl = 0;
    int i = 0;
    // m8 loop
    vl = vsetvl_e32m2(8);
    for (; i + 7 < m; i += 8) {
        int8_t *in_ptr = input_data;

        int8_t *out_ptr0 = output_data;
        int8_t *out_ptr1 = out_ptr0 + ldc;
        int8_t *out_ptr2 = out_ptr1 + ldc;
        int8_t *out_ptr3 = out_ptr2 + ldc;
        int8_t *out_ptr4 = out_ptr3 + ldc;
        int8_t *out_ptr5 = out_ptr4 + ldc;
        int8_t *out_ptr6 = out_ptr5 + ldc;
        int8_t *out_ptr7 = out_ptr6 + ldc;
        int j = 0;
        // m8n8 loop
        for (; j + 7 < n; j += 8) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m2_t _acc0 = vmv_v_x_i32m2(bias_data[0], 8);
            vint32m2_t _acc1 = vmv_v_x_i32m2(bias_data[1], 8);
            vint32m2_t _acc2 = vmv_v_x_i32m2(bias_data[2], 8);
            vint32m2_t _acc3 = vmv_v_x_i32m2(bias_data[3], 8);
            vint32m2_t _acc4 = vmv_v_x_i32m2(bias_data[4], 8);
            vint32m2_t _acc5 = vmv_v_x_i32m2(bias_data[5], 8);
            vint32m2_t _acc6 = vmv_v_x_i32m2(bias_data[6], 8);
            vint32m2_t _acc7 = vmv_v_x_i32m2(bias_data[7], 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _input = vle8_v_i8m2(in_ptr, 32);
                _acc0 = vmaqa_vx_i32m2(_acc0, kernel_ptr[0], _input, 8);
                _acc1 = vmaqa_vx_i32m2(_acc1, kernel_ptr[1], _input, 8);
                _acc2 = vmaqa_vx_i32m2(_acc2, kernel_ptr[2], _input, 8);
                _acc3 = vmaqa_vx_i32m2(_acc3, kernel_ptr[3], _input, 8);
                _acc4 = vmaqa_vx_i32m2(_acc4, kernel_ptr[4], _input, 8);
                _acc5 = vmaqa_vx_i32m2(_acc5, kernel_ptr[5], _input, 8);
                _acc6 = vmaqa_vx_i32m2(_acc6, kernel_ptr[6], _input, 8);
                _acc7 = vmaqa_vx_i32m2(_acc7, kernel_ptr[7], _input, 8);

                kernel_ptr += 8;
                in_ptr += 32;
            }
            vint8mf2_t _res0 = requantize_m2(_acc0, mult[0], shift[0], out_zp, 8);
            vint8mf2_t _res1 = requantize_m2(_acc1, mult[1], shift[1], out_zp, 8);
            vint8mf2_t _res2 = requantize_m2(_acc2, mult[2], shift[2], out_zp, 8);
            vint8mf2_t _res3 = requantize_m2(_acc3, mult[3], shift[3], out_zp, 8);
            vint8mf2_t _res4 = requantize_m2(_acc4, mult[4], shift[4], out_zp, 8);
            vint8mf2_t _res5 = requantize_m2(_acc5, mult[5], shift[5], out_zp, 8);
            vint8mf2_t _res6 = requantize_m2(_acc6, mult[6], shift[6], out_zp, 8);
            vint8mf2_t _res7 = requantize_m2(_acc7, mult[7], shift[7], out_zp, 8);

            vse8_v_i8mf2(out_ptr0, _res0, 8);
            vse8_v_i8mf2(out_ptr1, _res1, 8);
            vse8_v_i8mf2(out_ptr2, _res2, 8);
            vse8_v_i8mf2(out_ptr3, _res3, 8);
            vse8_v_i8mf2(out_ptr4, _res4, 8);
            vse8_v_i8mf2(out_ptr5, _res5, 8);
            vse8_v_i8mf2(out_ptr6, _res6, 8);
            vse8_v_i8mf2(out_ptr7, _res7, 8);
            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
            out_ptr4 += 8;
            out_ptr5 += 8;
            out_ptr6 += 8;
            out_ptr7 += 8;
        }
        // m8n4
        for (; j + 3 < n; j += 4) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m1_t _acc0 = vmv_v_x_i32m1(bias_data[0], 4);
            vint32m1_t _acc1 = vmv_v_x_i32m1(bias_data[1], 4);
            vint32m1_t _acc2 = vmv_v_x_i32m1(bias_data[2], 4);
            vint32m1_t _acc3 = vmv_v_x_i32m1(bias_data[3], 4);
            vint32m1_t _acc4 = vmv_v_x_i32m1(bias_data[4], 4);
            vint32m1_t _acc5 = vmv_v_x_i32m1(bias_data[5], 4);
            vint32m1_t _acc6 = vmv_v_x_i32m1(bias_data[6], 4);
            vint32m1_t _acc7 = vmv_v_x_i32m1(bias_data[7], 4);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 16);
                _acc0 = vmaqa_vx_i32m1(_acc0, kernel_ptr[0], _input, 4);
                _acc1 = vmaqa_vx_i32m1(_acc1, kernel_ptr[1], _input, 4);
                _acc2 = vmaqa_vx_i32m1(_acc2, kernel_ptr[2], _input, 4);
                _acc3 = vmaqa_vx_i32m1(_acc3, kernel_ptr[3], _input, 4);
                _acc4 = vmaqa_vx_i32m1(_acc4, kernel_ptr[4], _input, 4);
                _acc5 = vmaqa_vx_i32m1(_acc5, kernel_ptr[5], _input, 4);
                _acc6 = vmaqa_vx_i32m1(_acc6, kernel_ptr[6], _input, 4);
                _acc7 = vmaqa_vx_i32m1(_acc7, kernel_ptr[7], _input, 4);

                kernel_ptr += 8;
                in_ptr += 16;
            }
            vint8mf4_t _res0 = requantize_m1(_acc0, mult[0], shift[0], out_zp, 4);
            vint8mf4_t _res1 = requantize_m1(_acc1, mult[1], shift[1], out_zp, 4);
            vint8mf4_t _res2 = requantize_m1(_acc2, mult[2], shift[2], out_zp, 4);
            vint8mf4_t _res3 = requantize_m1(_acc3, mult[3], shift[3], out_zp, 4);
            vint8mf4_t _res4 = requantize_m1(_acc4, mult[4], shift[4], out_zp, 4);
            vint8mf4_t _res5 = requantize_m1(_acc5, mult[5], shift[5], out_zp, 4);
            vint8mf4_t _res6 = requantize_m1(_acc6, mult[6], shift[6], out_zp, 4);
            vint8mf4_t _res7 = requantize_m1(_acc7, mult[7], shift[7], out_zp, 4);
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
        // m8n2
        for (; j + 1 < n; j += 2) {
            int8_t *kernel_ptr = kernel_data;
            vint32m2_t _acc0 = vle32_v_i32m2(bias_data, 8);
            vint32m2_t _acc1 = vle32_v_i32m2(bias_data, 8);

            int32_t *in_ptr0 = (int32_t *)in_ptr;
            int32_t *in_ptr1 = (int32_t *)(in_ptr + k);

            out_ptr1 = out_ptr0 + 1;
            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _kernel = vle8_v_i8m2(kernel_ptr, 32);
                _acc0 = vmaqa_vx_i32m2(_acc0, in_ptr0[0], _kernel, 8);
                _acc1 = vmaqa_vx_i32m2(_acc1, in_ptr1[0], _kernel, 8);
                in_ptr0++;
                in_ptr1++;
                kernel_ptr += 32;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, mult, shift, out_zp, 8);
            vint8mf2_t _res1 = requantize_m2_s(_acc1, mult, shift, out_zp, 8);
            vsse8_v_i8mf2(out_ptr0, ldc * sizeof(int8_t), _res0, 8);
            vsse8_v_i8mf2(out_ptr1, ldc * sizeof(int8_t), _res1, 8);
            out_ptr0 += 2;
            in_ptr += 2 * k;
        }
        // m8n1
        for (; j < n; j++) {
            int8_t *kernel_ptr = kernel_data;
            vint32m2_t _acc0 = vle32_v_i32m2(bias_data, 8);
            int32_t *in_ptr0 = (int32_t *)in_ptr;
            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _kernel = vle8_v_i8m2(kernel_ptr, 32);
                _acc0 = vmaqa_vx_i32m2(_acc0, in_ptr0[0], _kernel, 8);
                in_ptr0++;
                kernel_ptr += 32;
            }
            vint8mf2_t _res0 = requantize_m2_s(_acc0, mult, shift, out_zp, 8);
            vsse8_v_i8mf2(out_ptr0, ldc * sizeof(int8_t), _res0, 8);
            out_ptr0 += 1;
            in_ptr += 1 * k;
        }
        kernel_data += 8 * k;
        output_data += 8 * ldc;
        bias_data += 8;
        mult += 8;
        shift += 8;
    }
    // m4
    for (; i + 3 < m; i += 4) {
        int8_t *in_ptr = input_data;

        int8_t *out_ptr0 = output_data;
        int8_t *out_ptr1 = out_ptr0 + ldc;
        int8_t *out_ptr2 = out_ptr1 + ldc;
        int8_t *out_ptr3 = out_ptr2 + ldc;
        int j = 0;
        // m4n8 loop
        for (; j + 7 < n; j += 8) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m2_t _acc0 = vmv_v_x_i32m2(bias_data[0], 8);
            vint32m2_t _acc1 = vmv_v_x_i32m2(bias_data[1], 8);
            vint32m2_t _acc2 = vmv_v_x_i32m2(bias_data[2], 8);
            vint32m2_t _acc3 = vmv_v_x_i32m2(bias_data[3], 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _input = vle8_v_i8m2(in_ptr, 32);
                _acc0 = vmaqa_vx_i32m2(_acc0, kernel_ptr[0], _input, 8);
                _acc1 = vmaqa_vx_i32m2(_acc1, kernel_ptr[1], _input, 8);
                _acc2 = vmaqa_vx_i32m2(_acc2, kernel_ptr[2], _input, 8);
                _acc3 = vmaqa_vx_i32m2(_acc3, kernel_ptr[3], _input, 8);

                kernel_ptr += 4;
                in_ptr += 32;
            }
            vint8mf2_t _res0 = requantize_m2(_acc0, mult[0], shift[0], out_zp, 8);
            vint8mf2_t _res1 = requantize_m2(_acc1, mult[1], shift[1], out_zp, 8);
            vint8mf2_t _res2 = requantize_m2(_acc2, mult[2], shift[2], out_zp, 8);
            vint8mf2_t _res3 = requantize_m2(_acc3, mult[3], shift[3], out_zp, 8);
            vse8_v_i8mf2(out_ptr0, _res0, 8);
            vse8_v_i8mf2(out_ptr1, _res1, 8);
            vse8_v_i8mf2(out_ptr2, _res2, 8);
            vse8_v_i8mf2(out_ptr3, _res3, 8);
            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
        }
        // m4n4
        for (; j + 3 < n; j += 4) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m1_t _acc0 = vmv_v_x_i32m1(bias_data[0], 4);
            vint32m1_t _acc1 = vmv_v_x_i32m1(bias_data[1], 4);
            vint32m1_t _acc2 = vmv_v_x_i32m1(bias_data[2], 4);
            vint32m1_t _acc3 = vmv_v_x_i32m1(bias_data[3], 4);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 16);
                _acc0 = vmaqa_vx_i32m1(_acc0, kernel_ptr[0], _input, 4);
                _acc1 = vmaqa_vx_i32m1(_acc1, kernel_ptr[1], _input, 4);
                _acc2 = vmaqa_vx_i32m1(_acc2, kernel_ptr[2], _input, 4);
                _acc3 = vmaqa_vx_i32m1(_acc3, kernel_ptr[3], _input, 4);

                kernel_ptr += 4;
                in_ptr += 16;
            }
            vint8mf4_t _res0 = requantize_m1(_acc0, mult[0], shift[0], out_zp, 4);
            vint8mf4_t _res1 = requantize_m1(_acc1, mult[1], shift[1], out_zp, 4);
            vint8mf4_t _res2 = requantize_m1(_acc2, mult[2], shift[2], out_zp, 4);
            vint8mf4_t _res3 = requantize_m1(_acc3, mult[3], shift[3], out_zp, 4);
            vse8_v_i8mf4(out_ptr0, _res0, 4);
            vse8_v_i8mf4(out_ptr1, _res1, 4);
            vse8_v_i8mf4(out_ptr2, _res2, 4);
            vse8_v_i8mf4(out_ptr3, _res3, 4);
            out_ptr0 += 4;
            out_ptr1 += 4;
            out_ptr2 += 4;
            out_ptr3 += 4;
        }
        // m4n2
        for (; j + 1 < n; j += 2) {
            int8_t *kernel_ptr = kernel_data;
            vint32m1_t _acc0 = vle32_v_i32m1(bias_data, 4);
            vint32m1_t _acc1 = vle32_v_i32m1(bias_data, 4);

            int32_t *in_ptr0 = (int32_t *)in_ptr;
            int32_t *in_ptr1 = (int32_t *)(in_ptr + k);

            out_ptr1 = out_ptr0 + 1;
            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _kernel = vle8_v_i8m1(kernel_ptr, 16);
                _acc0 = vmaqa_vx_i32m1(_acc0, in_ptr0[0], _kernel, 4);
                _acc1 = vmaqa_vx_i32m1(_acc1, in_ptr1[0], _kernel, 4);
                in_ptr0++;
                in_ptr1++;
                kernel_ptr += 16;
            }
            vint8mf4_t _res0 = requantize_m1_s(_acc0, mult, shift, out_zp, 4);
            vint8mf4_t _res1 = requantize_m1_s(_acc1, mult, shift, out_zp, 4);
            vsse8_v_i8mf4(out_ptr0, ldc * sizeof(int8_t), _res0, 4);
            vsse8_v_i8mf4(out_ptr1, ldc * sizeof(int8_t), _res1, 4);
            out_ptr0 += 2;
            in_ptr += 2 * k;
        }
        // m4n1
        for (; j < n; j++) {
            int8_t *kernel_ptr = kernel_data;
            vint32m1_t _acc0 = vle32_v_i32m1(bias_data, 4);
            int32_t *in_ptr0 = (int32_t *)in_ptr;
            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _kernel = vle8_v_i8m1(kernel_ptr, 16);
                _acc0 = vmaqa_vx_i32m1(_acc0, in_ptr0[0], _kernel, 4);
                in_ptr0++;
                kernel_ptr += 16;
            }
            vint8mf4_t _res0 = requantize_m1_s(_acc0, mult, shift, out_zp, 4);
            vsse8_v_i8mf4(out_ptr0, ldc * sizeof(int8_t), _res0, 4);
            out_ptr0 += 1;
            in_ptr += 1 * k;
        }
        kernel_data += 4 * k;
        output_data += 4 * ldc;
        bias_data += 4;
        mult += 4;
        shift += 4;
    }
    // m2
    for (; i + 1 < m; i += 2) {
        int8_t *in_ptr = input_data;

        int8_t *out_ptr0 = output_data;
        int8_t *out_ptr1 = out_ptr0 + ldc;
        int j = 0;
        // m2n8 loop
        for (; j + 7 < n; j += 8) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m2_t _acc0 = vmv_v_x_i32m2(bias_data[0], 8);
            vint32m2_t _acc1 = vmv_v_x_i32m2(bias_data[1], 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _input = vle8_v_i8m2(in_ptr, 32);
                _acc0 = vmaqa_vx_i32m2(_acc0, kernel_ptr[0], _input, 8);
                _acc1 = vmaqa_vx_i32m2(_acc1, kernel_ptr[1], _input, 8);

                kernel_ptr += 2;
                in_ptr += 32;
            }
            vint8mf2_t _res0 = requantize_m2(_acc0, mult[0], shift[0], out_zp, 8);
            vint8mf2_t _res1 = requantize_m2(_acc1, mult[1], shift[1], out_zp, 8);
            vse8_v_i8mf2(out_ptr0, _res0, 8);
            vse8_v_i8mf2(out_ptr1, _res1, 8);
            out_ptr0 += 8;
            out_ptr1 += 8;
        }
        // m2n4
        for (; j + 3 < n; j += 4) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m1_t _acc0 = vmv_v_x_i32m1(bias_data[0], 4);
            vint32m1_t _acc1 = vmv_v_x_i32m1(bias_data[1], 4);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 16);
                _acc0 = vmaqa_vx_i32m1(_acc0, kernel_ptr[0], _input, 4);
                _acc1 = vmaqa_vx_i32m1(_acc1, kernel_ptr[1], _input, 4);

                kernel_ptr += 2;
                in_ptr += 16;
            }
            vint8mf4_t _res0 = requantize_m1(_acc0, mult[0], shift[0], out_zp, 4);
            vint8mf4_t _res1 = requantize_m1(_acc1, mult[1], shift[1], out_zp, 4);
            vse8_v_i8mf4(out_ptr0, _res0, 4);
            vse8_v_i8mf4(out_ptr1, _res1, 4);
            out_ptr0 += 4;
            out_ptr1 += 4;
        }
        // m2n_tail
        for (; j < n; j++) {
            int32_t acc0 = bias_data[0];
            int32_t acc1 = bias_data[1];
            int8_t *k0 = kernel_data;
            int c = 0;
            for (; c + 3 < k; c += 4) {
                acc0 += k0[0] * in_ptr[c + 0];
                acc0 += k0[1] * in_ptr[c + 1];
                acc0 += k0[2] * in_ptr[c + 2];
                acc0 += k0[3] * in_ptr[c + 3];
                acc1 += k0[4] * in_ptr[c + 0];
                acc1 += k0[5] * in_ptr[c + 1];
                acc1 += k0[6] * in_ptr[c + 2];
                acc1 += k0[7] * in_ptr[c + 3];
                k0 += 8;
            }
            *out_ptr0++ = requantize_single(acc0, mult[0], shift[0], out_zp);
            *out_ptr1++ = requantize_single(acc1, mult[1], shift[1], out_zp);
            in_ptr += k;
        }
        kernel_data += 2 * k;
        output_data += 2 * ldc;
        bias_data += 2;
        mult += 2;
        shift += 2;
    }
    // m1
    for (; i < m; i++) {
        int8_t *in_ptr = input_data;
        int8_t *out_ptr0 = output_data;
        int j = 0;
        // m1n8 loop
        for (; j + 7 < n; j += 8) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m2_t _acc0 = vmv_v_x_i32m2(bias_data[0], 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _input = vle8_v_i8m2(in_ptr, 32);
                _acc0 = vmaqa_vx_i32m2(_acc0, kernel_ptr[0], _input, 8);
                kernel_ptr += 1;
                in_ptr += 32;
            }
            vint8mf2_t _res0 = requantize_m2(_acc0, mult[0], shift[0], out_zp, 8);
            vse8_v_i8mf2(out_ptr0, _res0, 8);
            out_ptr0 += 8;
        }
        // m1n4
        for (; j + 3 < n; j += 4) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m1_t _acc0 = vmv_v_x_i32m1(bias_data[0], 4);
            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 16);
                _acc0 = vmaqa_vx_i32m1(_acc0, kernel_ptr[0], _input, 4);
                kernel_ptr += 1;
                in_ptr += 16;
            }
            vint8mf4_t _res0 = requantize_m1(_acc0, mult[0], shift[0], out_zp, 4);
            vse8_v_i8mf4(out_ptr0, _res0, 4);
            out_ptr0 += 4;
        }
        // m1n_tail
        for (; j < n; j++) {
            int32_t acc0 = bias_data[0];
            for (int c = 0; c < k; c++) {
                acc0 += kernel_data[c] * in_ptr[c];
            }
            *out_ptr0++ = requantize_single(acc0, mult[0], shift[0], out_zp);
            in_ptr += k;
        }
    }
}

/*************************************************************
    note: VLEN = 256
*************************************************************/
void shl_rvv256_gemm_8x16_int32(int32_t *dst, const int8_t *sa, const int8_t *sb, int32_t *bias,
                                int m, int k, int n, int ldc)
{
    int8_t *kernel_data = (int8_t *)sa;
    int8_t *input_data = (int8_t *)sb;
    int32_t *output_data = dst;
    // please use fuse_zp2bias option in hhb, thus bias_data wont be NULL
    int32_t *bias_data = bias;

    int vl = 0;
    int i = 0;
    // m8 loop
    vl = vsetvl_e32m2(16);
    for (; i + 7 < m; i += 8) {
        int8_t *in_ptr = input_data;

        int32_t *out_ptr0 = output_data;
        int32_t *out_ptr1 = out_ptr0 + ldc;
        int32_t *out_ptr2 = out_ptr1 + ldc;
        int32_t *out_ptr3 = out_ptr2 + ldc;
        int32_t *out_ptr4 = out_ptr3 + ldc;
        int32_t *out_ptr5 = out_ptr4 + ldc;
        int32_t *out_ptr6 = out_ptr5 + ldc;
        int32_t *out_ptr7 = out_ptr6 + ldc;
        int j = 0;
        // m8n16 loop
        for (; j + 15 < n; j += 16) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m2_t _acc0 = vmv_v_x_i32m2(bias_data[0], 16);
            vint32m2_t _acc1 = vmv_v_x_i32m2(bias_data[1], 16);
            vint32m2_t _acc2 = vmv_v_x_i32m2(bias_data[2], 16);
            vint32m2_t _acc3 = vmv_v_x_i32m2(bias_data[3], 16);
            vint32m2_t _acc4 = vmv_v_x_i32m2(bias_data[4], 16);
            vint32m2_t _acc5 = vmv_v_x_i32m2(bias_data[5], 16);
            vint32m2_t _acc6 = vmv_v_x_i32m2(bias_data[6], 16);
            vint32m2_t _acc7 = vmv_v_x_i32m2(bias_data[7], 16);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _input = vle8_v_i8m2(in_ptr, 64);
                _acc0 = vmaqa_vx_i32m2(_acc0, kernel_ptr[0], _input, 16);
                _acc1 = vmaqa_vx_i32m2(_acc1, kernel_ptr[1], _input, 16);
                _acc2 = vmaqa_vx_i32m2(_acc2, kernel_ptr[2], _input, 16);
                _acc3 = vmaqa_vx_i32m2(_acc3, kernel_ptr[3], _input, 16);
                _acc4 = vmaqa_vx_i32m2(_acc4, kernel_ptr[4], _input, 16);
                _acc5 = vmaqa_vx_i32m2(_acc5, kernel_ptr[5], _input, 16);
                _acc6 = vmaqa_vx_i32m2(_acc6, kernel_ptr[6], _input, 16);
                _acc7 = vmaqa_vx_i32m2(_acc7, kernel_ptr[7], _input, 16);

                kernel_ptr += 8;
                in_ptr += 64;
            }
            vse32_v_i32m2(out_ptr0, _acc0, 16);
            vse32_v_i32m2(out_ptr1, _acc1, 16);
            vse32_v_i32m2(out_ptr2, _acc2, 16);
            vse32_v_i32m2(out_ptr3, _acc3, 16);
            vse32_v_i32m2(out_ptr4, _acc4, 16);
            vse32_v_i32m2(out_ptr5, _acc5, 16);
            vse32_v_i32m2(out_ptr6, _acc6, 16);
            vse32_v_i32m2(out_ptr7, _acc7, 16);
            out_ptr0 += 16;
            out_ptr1 += 16;
            out_ptr2 += 16;
            out_ptr3 += 16;
            out_ptr4 += 16;
            out_ptr5 += 16;
            out_ptr6 += 16;
            out_ptr7 += 16;
        }
        // m8n8
        for (; j + 7 < n; j += 8) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m1_t _acc0 = vmv_v_x_i32m1(bias_data[0], 8);
            vint32m1_t _acc1 = vmv_v_x_i32m1(bias_data[1], 8);
            vint32m1_t _acc2 = vmv_v_x_i32m1(bias_data[2], 8);
            vint32m1_t _acc3 = vmv_v_x_i32m1(bias_data[3], 8);
            vint32m1_t _acc4 = vmv_v_x_i32m1(bias_data[4], 8);
            vint32m1_t _acc5 = vmv_v_x_i32m1(bias_data[5], 8);
            vint32m1_t _acc6 = vmv_v_x_i32m1(bias_data[6], 8);
            vint32m1_t _acc7 = vmv_v_x_i32m1(bias_data[7], 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 32);
                _acc0 = vmaqa_vx_i32m1(_acc0, kernel_ptr[0], _input, 8);
                _acc1 = vmaqa_vx_i32m1(_acc1, kernel_ptr[1], _input, 8);
                _acc2 = vmaqa_vx_i32m1(_acc2, kernel_ptr[2], _input, 8);
                _acc3 = vmaqa_vx_i32m1(_acc3, kernel_ptr[3], _input, 8);
                _acc4 = vmaqa_vx_i32m1(_acc4, kernel_ptr[4], _input, 8);
                _acc5 = vmaqa_vx_i32m1(_acc5, kernel_ptr[5], _input, 8);
                _acc6 = vmaqa_vx_i32m1(_acc6, kernel_ptr[6], _input, 8);
                _acc7 = vmaqa_vx_i32m1(_acc7, kernel_ptr[7], _input, 8);

                kernel_ptr += 8;
                in_ptr += 32;
            }
            vse32_v_i32m1(out_ptr0, _acc0, 8);
            vse32_v_i32m1(out_ptr1, _acc1, 8);
            vse32_v_i32m1(out_ptr2, _acc2, 8);
            vse32_v_i32m1(out_ptr3, _acc3, 8);
            vse32_v_i32m1(out_ptr4, _acc4, 8);
            vse32_v_i32m1(out_ptr5, _acc5, 8);
            vse32_v_i32m1(out_ptr6, _acc6, 8);
            vse32_v_i32m1(out_ptr7, _acc7, 8);
            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
            out_ptr4 += 8;
            out_ptr5 += 8;
            out_ptr6 += 8;
            out_ptr7 += 8;
        }
        // m8n4
        for (; j + 3 < n; j += 4) {
            int8_t *kernel_ptr = kernel_data;
            vint32m1_t _acc0 = vle32_v_i32m1(bias_data, 8);
            vint32m1_t _acc1 = vle32_v_i32m1(bias_data, 8);
            vint32m1_t _acc2 = vle32_v_i32m1(bias_data, 8);
            vint32m1_t _acc3 = vle32_v_i32m1(bias_data, 8);

            int32_t *in_ptr0 = (int32_t *)in_ptr;
            int32_t *in_ptr1 = in_ptr0 + k;
            int32_t *in_ptr2 = in_ptr1 + k;
            int32_t *in_ptr3 = in_ptr2 + k;

            out_ptr1 = out_ptr0 + 1;
            out_ptr2 = out_ptr0 + 2;
            out_ptr3 = out_ptr0 + 3;
            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _kernel = vle8_v_i8m1(kernel_ptr, 32);
                _acc0 = vmaqa_vx_i32m1(_acc0, in_ptr0[0], _kernel, 8);
                _acc1 = vmaqa_vx_i32m1(_acc1, in_ptr1[0], _kernel, 8);
                _acc2 = vmaqa_vx_i32m1(_acc2, in_ptr2[0], _kernel, 8);
                _acc3 = vmaqa_vx_i32m1(_acc3, in_ptr3[0], _kernel, 8);
                in_ptr0++;
                in_ptr1++;
                in_ptr2++;
                in_ptr3++;
                kernel_ptr += 32;
            }
            vsse32_v_i32m1(out_ptr0, ldc * sizeof(int32_t), _acc0, 8);
            vsse32_v_i32m1(out_ptr1, ldc * sizeof(int32_t), _acc1, 8);
            vsse32_v_i32m1(out_ptr2, ldc * sizeof(int32_t), _acc2, 8);
            vsse32_v_i32m1(out_ptr3, ldc * sizeof(int32_t), _acc3, 8);
            out_ptr0 += 4;
            in_ptr += 4 * k;
        }
        // m8n2
        for (; j + 1 < n; j += 2) {
            int8_t *kernel_ptr = kernel_data;
            vint32m1_t _acc0 = vle32_v_i32m1(bias_data, 8);
            vint32m1_t _acc1 = vle32_v_i32m1(bias_data, 8);

            int32_t *in_ptr0 = (int32_t *)in_ptr;
            int32_t *in_ptr1 = in_ptr0 + k;

            out_ptr1 = out_ptr0 + 1;
            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _kernel = vle8_v_i8m1(kernel_ptr, 32);
                _acc0 = vmaqa_vx_i32m1(_acc0, in_ptr0[0], _kernel, 8);
                _acc1 = vmaqa_vx_i32m1(_acc1, in_ptr1[0], _kernel, 8);
                in_ptr0++;
                in_ptr1++;
                kernel_ptr += 32;
            }
            vsse32_v_i32m1(out_ptr0, ldc * sizeof(int32_t), _acc0, 8);
            vsse32_v_i32m1(out_ptr1, ldc * sizeof(int32_t), _acc1, 8);
            out_ptr0 += 2;
            in_ptr += 2 * k;
        }
        // m8n1
        for (; j < n; j++) {
            int8_t *kernel_ptr = kernel_data;
            vint32m1_t _acc0 = vle32_v_i32m1(bias_data, 8);
            int32_t *in_ptr0 = (int32_t *)in_ptr;

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _kernel = vle8_v_i8m1(kernel_ptr, 32);
                _acc0 = vmaqa_vx_i32m1(_acc0, in_ptr0[0], _kernel, 8);
                in_ptr0++;
                kernel_ptr += 32;
            }
            vsse32_v_i32m1(out_ptr0, ldc * sizeof(int32_t), _acc0, 8);
            out_ptr0 += 1;
            in_ptr += 1 * k;
        }
        kernel_data += 8 * k;
        output_data += 8 * ldc;
        bias_data += 8;
    }
    // m4
    for (; i + 3 < m; i += 4) {
        int8_t *in_ptr = input_data;

        int32_t *out_ptr0 = output_data;
        int32_t *out_ptr1 = out_ptr0 + ldc;
        int32_t *out_ptr2 = out_ptr1 + ldc;
        int32_t *out_ptr3 = out_ptr2 + ldc;
        int j = 0;
        // m4n16 loop
        for (; j + 15 < n; j += 16) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m2_t _acc0 = vmv_v_x_i32m2(bias_data[0], 16);
            vint32m2_t _acc1 = vmv_v_x_i32m2(bias_data[1], 16);
            vint32m2_t _acc2 = vmv_v_x_i32m2(bias_data[2], 16);
            vint32m2_t _acc3 = vmv_v_x_i32m2(bias_data[3], 16);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _input = vle8_v_i8m2(in_ptr, 64);
                _acc0 = vmaqa_vx_i32m2(_acc0, kernel_ptr[0], _input, 16);
                _acc1 = vmaqa_vx_i32m2(_acc1, kernel_ptr[1], _input, 16);
                _acc2 = vmaqa_vx_i32m2(_acc2, kernel_ptr[2], _input, 16);
                _acc3 = vmaqa_vx_i32m2(_acc3, kernel_ptr[3], _input, 16);

                kernel_ptr += 4;
                in_ptr += 64;
            }
            vse32_v_i32m2(out_ptr0, _acc0, 16);
            vse32_v_i32m2(out_ptr1, _acc1, 16);
            vse32_v_i32m2(out_ptr2, _acc2, 16);
            vse32_v_i32m2(out_ptr3, _acc3, 16);
            out_ptr0 += 16;
            out_ptr1 += 16;
            out_ptr2 += 16;
            out_ptr3 += 16;
        }
        // m4n8
        for (; j + 7 < n; j += 8) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m1_t _acc0 = vmv_v_x_i32m1(bias_data[0], 8);
            vint32m1_t _acc1 = vmv_v_x_i32m1(bias_data[1], 8);
            vint32m1_t _acc2 = vmv_v_x_i32m1(bias_data[2], 8);
            vint32m1_t _acc3 = vmv_v_x_i32m1(bias_data[3], 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 16);
                _acc0 = vmaqa_vx_i32m1(_acc0, kernel_ptr[0], _input, 8);
                _acc1 = vmaqa_vx_i32m1(_acc1, kernel_ptr[1], _input, 8);
                _acc2 = vmaqa_vx_i32m1(_acc2, kernel_ptr[2], _input, 8);
                _acc3 = vmaqa_vx_i32m1(_acc3, kernel_ptr[3], _input, 8);

                kernel_ptr += 4;
                in_ptr += 32;
            }
            vse32_v_i32m1(out_ptr0, _acc0, 8);
            vse32_v_i32m1(out_ptr1, _acc1, 8);
            vse32_v_i32m1(out_ptr2, _acc2, 8);
            vse32_v_i32m1(out_ptr3, _acc3, 8);
            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
        }
        for (; j < n; j++) {
            int8_t *kernel_ptr = kernel_data;
            int32_t acc0 = bias_data[0];
            int32_t acc1 = bias_data[1];
            int32_t acc2 = bias_data[2];
            int32_t acc3 = bias_data[3];
            int c = 0;
            for (; c + 3 < k; c += 4) {
                acc0 += kernel_ptr[0] * in_ptr[0] + kernel_ptr[1] * in_ptr[1] +
                        kernel_ptr[2] * in_ptr[2] + kernel_ptr[3] * in_ptr[3];
                acc1 += kernel_ptr[4] * in_ptr[0] + kernel_ptr[5] * in_ptr[1] +
                        kernel_ptr[6] * in_ptr[2] + kernel_ptr[7] * in_ptr[3];
                acc2 += kernel_ptr[8] * in_ptr[0] + kernel_ptr[9] * in_ptr[1] +
                        kernel_ptr[10] * in_ptr[2] + kernel_ptr[11] * in_ptr[3];
                acc3 += kernel_ptr[12] * in_ptr[0] + kernel_ptr[13] * in_ptr[1] +
                        kernel_ptr[14] * in_ptr[2] + kernel_ptr[15] * in_ptr[3];
            }
            *out_ptr0++ = acc0;
            *out_ptr1++ = acc1;
            *out_ptr2++ = acc2;
            *out_ptr3++ = acc3;
            in_ptr += k;
        }
        kernel_data += 4 * k;
        output_data += 4 * ldc;
        bias_data += 4;
    }

    // m2
    for (; i + 1 < m; i += 2) {
        int8_t *in_ptr = input_data;

        int32_t *out_ptr0 = output_data;
        int32_t *out_ptr1 = out_ptr0 + ldc;
        int j = 0;
        // m2n16 loop
        for (; j + 15 < n; j += 16) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m2_t _acc0 = vmv_v_x_i32m2(bias_data[0], 16);
            vint32m2_t _acc1 = vmv_v_x_i32m2(bias_data[1], 16);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _input = vle8_v_i8m2(in_ptr, 64);
                _acc0 = vmaqa_vx_i32m2(_acc0, kernel_ptr[0], _input, 16);
                _acc1 = vmaqa_vx_i32m2(_acc1, kernel_ptr[1], _input, 16);

                kernel_ptr += 2;
                in_ptr += 64;
            }
            vse32_v_i32m2(out_ptr0, _acc0, 16);
            vse32_v_i32m2(out_ptr1, _acc1, 16);
            out_ptr0 += 16;
            out_ptr1 += 16;
        }
        // m2n8
        for (; j + 7 < n; j += 8) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m1_t _acc0 = vmv_v_x_i32m1(bias_data[0], 8);
            vint32m1_t _acc1 = vmv_v_x_i32m1(bias_data[1], 8);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 32);
                _acc0 = vmaqa_vx_i32m1(_acc0, kernel_ptr[0], _input, 8);
                _acc1 = vmaqa_vx_i32m1(_acc1, kernel_ptr[1], _input, 8);

                kernel_ptr += 2;
                in_ptr += 32;
            }
            vse32_v_i32m1(out_ptr0, _acc0, 8);
            vse32_v_i32m1(out_ptr1, _acc1, 8);
            out_ptr0 += 8;
            out_ptr1 += 8;
        }
        for (; j < n; j++) {
            int8_t *kernel_ptr = kernel_data;
            int32_t acc0 = bias_data[0];
            int32_t acc1 = bias_data[1];
            int c = 0;
            for (; c + 3 < k; c += 4) {
                acc0 += kernel_ptr[0] * in_ptr[0] + kernel_ptr[1] * in_ptr[1] +
                        kernel_ptr[2] * in_ptr[2] + kernel_ptr[3] * in_ptr[3];
                acc1 += kernel_ptr[4] * in_ptr[0] + kernel_ptr[5] * in_ptr[1] +
                        kernel_ptr[6] * in_ptr[2] + kernel_ptr[7] * in_ptr[3];
            }
            *out_ptr0++ = acc0;
            *out_ptr1++ = acc1;
            in_ptr += k;
        }
        kernel_data += 2 * k;
        output_data += 2 * ldc;
        bias_data += 2;
    }

    // m1
    for (; i < m; i++) {
        int8_t *in_ptr = input_data;
        int32_t *out_ptr0 = output_data;
        int j = 0;
        // m1n16 loop
        for (; j + 15 < n; j += 16) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m2_t _acc0 = vmv_v_x_i32m2(bias_data[0], 16);

            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m2_t _input = vle8_v_i8m2(in_ptr, 64);
                _acc0 = vmaqa_vx_i32m2(_acc0, kernel_ptr[0], _input, 16);
                kernel_ptr += 1;
                in_ptr += 64;
            }
            vse32_v_i32m2(out_ptr0, _acc0, 16);
            out_ptr0 += 16;
        }
        // m1n8
        for (; j + 7 < n; j += 8) {
            int32_t *kernel_ptr = (int32_t *)kernel_data;
            vint32m1_t _acc0 = vmv_v_x_i32m1(bias_data[0], 8);
            int c = 0;
            for (; c + 3 < k; c += 4) {
                vint8m1_t _input = vle8_v_i8m1(in_ptr, 32);
                _acc0 = vmaqa_vx_i32m1(_acc0, kernel_ptr[0], _input, 8);
                kernel_ptr += 1;
                in_ptr += 32;
            }
            vse32_v_i32m1(out_ptr0, _acc0, 8);
            out_ptr0 += 8;
        }
        // m1n_tail
        for (; j < n; j++) {
            int32_t acc0 = bias_data[0];
            for (int c = 0; c < k; c++) {
                acc0 += kernel_data[c] * in_ptr[c];
            }
            *out_ptr0++ = acc0;
            in_ptr += k;
        }
    }
}

#endif