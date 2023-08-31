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
static vint8m1_t requantize_4xn(vint32m4_t _src, int32_t *mult, int32_t *shift, int32_t out_zp,
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
 * note: support flexible vlen
 * dst - output: [batch, out_nodes]
 * sa - input:   [batch, in_nodes]
 * sb - weight:  [out_nodes, in_nodes]
 *************************************************************/
void shl_rvv_gemm_4xn_int8(int8_t *dst, const int8_t *sa, const int8_t *sb, const int32_t *bias,
                           int batch, int in_nodes, int out_nodes, int ldc, int32_t out_zp,
                           int32_t *mult, int32_t *shift)
{
    int8_t *input_data = (int8_t *)sa;
    int8_t *kernel_data = (int8_t *)sb;
    int8_t *output_data = dst;
    int32_t *bias_data = (int32_t *)bias;
    const int vlenb = csrr_vlenb();

    int i = 0;
    for (; i + 3 < batch; i += 4) {
        const int8_t *k_ptr = kernel_data;
        int8_t *out_ptr0 = output_data;
        int8_t *out_ptr1 = out_ptr0 + ldc;
        int8_t *out_ptr2 = out_ptr1 + ldc;
        int8_t *out_ptr3 = out_ptr2 + ldc;

        int j = 0;
        while (j < out_nodes) {
            int vl = vsetvl_e8m1(out_nodes - j);
            int8_t *in_ptr = input_data;
            vint32m4_t _acc0 = vle32_v_i32m4(bias + j, vl);
            vint32m4_t _acc1 = _acc0;
            vint32m4_t _acc2 = _acc0;
            vint32m4_t _acc3 = _acc0;

            for (int k = 0; k < in_nodes; k++) {
                vint8m1_t _k = vle8_v_i8m1(k_ptr, vl);
                vint16m2_t _mul0 = vwmul_vx_i16m2(_k, in_ptr[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_k, in_ptr[1], vl);
                vint16m2_t _mul2 = vwmul_vx_i16m2(_k, in_ptr[2], vl);
                vint16m2_t _mul3 = vwmul_vx_i16m2(_k, in_ptr[3], vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                _acc2 = vwmacc_vx_i32m4(_acc2, 1, _mul2, vl);
                _acc3 = vwmacc_vx_i32m4(_acc3, 1, _mul3, vl);
                in_ptr += 4;
                k_ptr += vl;
            }

            vint8m1_t _res0 = requantize_4xn(_acc0, mult + j, shift + j, out_zp, vl);
            vint8m1_t _res1 = requantize_4xn(_acc1, mult + j, shift + j, out_zp, vl);
            vint8m1_t _res2 = requantize_4xn(_acc2, mult + j, shift + j, out_zp, vl);
            vint8m1_t _res3 = requantize_4xn(_acc3, mult + j, shift + j, out_zp, vl);
            vse8_v_i8m1(out_ptr0, _res0, vl);
            vse8_v_i8m1(out_ptr1, _res1, vl);
            vse8_v_i8m1(out_ptr2, _res2, vl);
            vse8_v_i8m1(out_ptr3, _res3, vl);
            out_ptr0 += vl;
            out_ptr1 += vl;
            out_ptr2 += vl;
            out_ptr3 += vl;
            j += vl;
        }
        input_data += 4 * in_nodes;
        output_data += 4 * out_nodes;
    }
    for (; i + 1 < batch; i += 2) {
        const int8_t *k_ptr = kernel_data;
        int8_t *out_ptr0 = output_data;
        int8_t *out_ptr1 = out_ptr0 + ldc;

        int j = 0;
        while (j < out_nodes) {
            int vl = vsetvl_e8m1(out_nodes - j);
            int8_t *in_ptr = input_data;
            vint32m4_t _acc0 = vle32_v_i32m4(bias + j, vl);
            vint32m4_t _acc1 = _acc0;

            for (int k = 0; k < in_nodes; k++) {
                vint8m1_t _k = vle8_v_i8m1(k_ptr, vl);
                vint16m2_t _mul0 = vwmul_vx_i16m2(_k, in_ptr[0], vl);
                vint16m2_t _mul1 = vwmul_vx_i16m2(_k, in_ptr[1], vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, 1, _mul1, vl);
                in_ptr += 2;
                k_ptr += vl;
            }

            vint8m1_t _res0 = requantize_4xn(_acc0, mult + j, shift + j, out_zp, vl);
            vint8m1_t _res1 = requantize_4xn(_acc1, mult + j, shift + j, out_zp, vl);
            vse8_v_i8m1(out_ptr0, _res0, vl);
            vse8_v_i8m1(out_ptr1, _res1, vl);
            out_ptr0 += vl;
            out_ptr1 += vl;
            j += vl;
        }
        input_data += 2 * in_nodes;
        output_data += 2 * out_nodes;
    }
    for (; i < batch; i += 1) {
        const int8_t *k_ptr = kernel_data;
        int8_t *out_ptr0 = output_data;

        int j = 0;
        while (j < out_nodes) {
            int vl = vsetvl_e8m1(out_nodes - j);
            int8_t *in_ptr = input_data;
            vint32m4_t _acc0 = vle32_v_i32m4(bias + j, vl);

            for (int k = 0; k < in_nodes; k++) {
                vint8m1_t _k = vle8_v_i8m1(k_ptr, vl);
                vint16m2_t _mul0 = vwmul_vx_i16m2(_k, in_ptr[0], vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, 1, _mul0, vl);
                in_ptr += 1;
                k_ptr += vl;
            }

            vint8m1_t _res0 = requantize_4xn(_acc0, mult + j, shift + j, out_zp, vl);
            vse8_v_i8m1(out_ptr0, _res0, vl);
            out_ptr0 += vl;
            j += vl;
        }
        input_data += in_nodes;
        output_data += out_nodes;
    }
}
