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

#ifdef RVV_1_0_0
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
#elif defined RVV_0_7_1
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
#endif

int shl_rvv_dwconv3x3s1_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params)
{
    if (input->layout == CSINN_LAYOUT_NCHW) {
        shl_rvv_tensor_ndarray_to_nc1xc0_replace_int8(input);
    }
    if (output->layout == CSINN_LAYOUT_NCHW) {
        output->dim[1] /= input->dim[4];
        output->dim[4] = input->dim[4];
        output->dim_count = 5;
        output->layout = CSINN_LAYOUT_NC1HWC0;
    }
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)kernel->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1] * input->dim[4];  // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];
    int32_t out_c = in_c;
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int32_t *multiplier = (int32_t *)shl_mem_alloc(out_c * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(out_c * sizeof(int32_t));

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);

    int8_t *input_padd_buf = (int8_t *)shl_mem_alloc((in_h + params->pad_top + params->pad_down) *
                                                     (in_w + params->pad_left + params->pad_right) *
                                                     in_c * sizeof(int8_t));

    shl_rvv_pad_input_packn_int8(input_data, input_padd_buf, in_c, in_h, in_w,
                                 in_h + params->pad_top + params->pad_down,
                                 in_w + params->pad_left + params->pad_right, params->pad_top,
                                 params->pad_left, input->qinfo->zero_point);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

    if (kernel->quant_channel > 1) {
        for (int c = 0; c < out_c; c++) {
            multiplier[c] = kernel->qinfo[c].multiplier;
            shift[c] = kernel->qinfo[c].shift;
        }
    } else if (kernel->quant_channel == 1) {
        for (int c = 0; c < out_c; c++) {
            multiplier[c] = kernel->qinfo[0].multiplier;
            shift[c] = kernel->qinfo[0].shift;
        }
    }

#ifdef RVV_1_0_0
#pragma omp parallel for num_threads(1)
    for (int c = 0; c + packn - 1 < in_c; c += packn) {
        int8_t *out0 = output_data + c * out_h * out_w;
        int8_t *out1 = out0 + out_w * packn;

        const int8_t *r0 = input_padd_buf + c * in_h * in_w;
        const int8_t *r1 = r0 + in_w * packn;
        const int8_t *r2 = r1 + in_w * packn;
        const int8_t *r3 = r2 + in_w * packn;

        const int8_t *kernel0 = kernel_data + c * 9;

        vint16m1_t _k00 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0, vl), 0, vl);
        vint16m1_t _k01 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 1 * packn, vl), 0, vl);
        vint16m1_t _k02 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 2 * packn, vl), 0, vl);
        vint16m1_t _k10 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 3 * packn, vl), 0, vl);
        vint16m1_t _k11 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 4 * packn, vl), 0, vl);
        vint16m1_t _k12 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 5 * packn, vl), 0, vl);
        vint16m1_t _k20 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 6 * packn, vl), 0, vl);
        vint16m1_t _k21 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 7 * packn, vl), 0, vl);
        vint16m1_t _k22 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 8 * packn, vl), 0, vl);

        // please use fuse_zp2bias option in hhb, thus bias_data wont be NULL
        vint32m2_t _bias0 = vle32_v_i32m2(bias_data + c, vl);

        vint32m2_t _mult = vle32_v_i32m2(multiplier + c, vl);
        vint32m2_t _shift = vle32_v_i32m2(shift + c, vl);
        _shift = vrsub_vx_i32m2(_shift, -1, vl);
        int32_t out_zp = output->qinfo->zero_point;

        int h = 0;
        // h2 loop
        for (; h + 1 < out_h; h += 2) {
            int w = 0;
            // h2w4 loop
            for (; w + 3 < out_w; w += 4) {
                vint32m2_t _acc00 = _bias0;
                vint32m2_t _acc01 = _bias0;
                vint32m2_t _acc02 = _bias0;
                vint32m2_t _acc03 = _bias0;
                vint32m2_t _acc10 = _bias0;
                vint32m2_t _acc11 = _bias0;
                vint32m2_t _acc12 = _bias0;
                vint32m2_t _acc13 = _bias0;

                vint16m1_t _r00 = vwadd_vx_i16m1(vle8_v_i8mf2(r0, vl), 0, vl);
                vint16m1_t _r01 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 1 * packn, vl), 0, vl);
                vint16m1_t _r02 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 2 * packn, vl), 0, vl);
                vint16m1_t _r03 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 3 * packn, vl), 0, vl);
                vint16m1_t _r04 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 4 * packn, vl), 0, vl);
                vint16m1_t _r05 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 5 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k02, _r02, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k00, _r01, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k01, _r02, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k02, _r03, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k00, _r02, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k01, _r03, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k02, _r04, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k00, _r03, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k01, _r04, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k02, _r05, vl);

                vint16m1_t _r10 = vwadd_vx_i16m1(vle8_v_i8mf2(r1, vl), 0, vl);
                vint16m1_t _r11 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 1 * packn, vl), 0, vl);
                vint16m1_t _r12 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 2 * packn, vl), 0, vl);
                vint16m1_t _r13 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 3 * packn, vl), 0, vl);
                vint16m1_t _r14 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 4 * packn, vl), 0, vl);
                vint16m1_t _r15 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 5 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k12, _r12, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k10, _r11, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k11, _r12, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k12, _r13, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k10, _r12, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k11, _r13, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k12, _r14, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k10, _r13, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k11, _r14, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k12, _r15, vl);  //
                _acc10 = vwmacc_vv_i32m2(_acc10, _k00, _r10, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k01, _r11, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k02, _r12, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k00, _r11, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k01, _r12, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k02, _r13, vl);
                _acc12 = vwmacc_vv_i32m2(_acc12, _k00, _r12, vl);
                _acc12 = vwmacc_vv_i32m2(_acc12, _k01, _r13, vl);
                _acc12 = vwmacc_vv_i32m2(_acc12, _k02, _r14, vl);
                _acc13 = vwmacc_vv_i32m2(_acc13, _k00, _r13, vl);
                _acc13 = vwmacc_vv_i32m2(_acc13, _k01, _r14, vl);
                _acc13 = vwmacc_vv_i32m2(_acc13, _k02, _r15, vl);

                vint16m1_t _r20 = vwadd_vx_i16m1(vle8_v_i8mf2(r2, vl), 0, vl);
                vint16m1_t _r21 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 1 * packn, vl), 0, vl);
                vint16m1_t _r22 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 2 * packn, vl), 0, vl);
                vint16m1_t _r23 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 3 * packn, vl), 0, vl);
                vint16m1_t _r24 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 4 * packn, vl), 0, vl);
                vint16m1_t _r25 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 5 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k22, _r22, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k20, _r21, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k21, _r22, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k22, _r23, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k20, _r22, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k21, _r23, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k22, _r24, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k20, _r23, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k21, _r24, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k22, _r25, vl);  //
                _acc10 = vwmacc_vv_i32m2(_acc10, _k10, _r20, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k11, _r21, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k12, _r22, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k10, _r21, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k11, _r22, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k12, _r23, vl);
                _acc12 = vwmacc_vv_i32m2(_acc12, _k10, _r22, vl);
                _acc12 = vwmacc_vv_i32m2(_acc12, _k11, _r23, vl);
                _acc12 = vwmacc_vv_i32m2(_acc12, _k12, _r24, vl);
                _acc13 = vwmacc_vv_i32m2(_acc13, _k10, _r23, vl);
                _acc13 = vwmacc_vv_i32m2(_acc13, _k11, _r24, vl);
                _acc13 = vwmacc_vv_i32m2(_acc13, _k12, _r25, vl);

                vint16m1_t _r30 = vwadd_vx_i16m1(vle8_v_i8mf2(r3, vl), 0, vl);
                vint16m1_t _r31 = vwadd_vx_i16m1(vle8_v_i8mf2(r3 + 1 * packn, vl), 0, vl);
                vint16m1_t _r32 = vwadd_vx_i16m1(vle8_v_i8mf2(r3 + 2 * packn, vl), 0, vl);
                vint16m1_t _r33 = vwadd_vx_i16m1(vle8_v_i8mf2(r3 + 3 * packn, vl), 0, vl);
                vint16m1_t _r34 = vwadd_vx_i16m1(vle8_v_i8mf2(r3 + 4 * packn, vl), 0, vl);
                vint16m1_t _r35 = vwadd_vx_i16m1(vle8_v_i8mf2(r3 + 5 * packn, vl), 0, vl);

                _acc10 = vwmacc_vv_i32m2(_acc10, _k20, _r30, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k21, _r31, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k22, _r32, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k20, _r31, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k21, _r32, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k22, _r33, vl);
                _acc12 = vwmacc_vv_i32m2(_acc12, _k20, _r32, vl);
                _acc12 = vwmacc_vv_i32m2(_acc12, _k21, _r33, vl);
                _acc12 = vwmacc_vv_i32m2(_acc12, _k22, _r34, vl);
                _acc13 = vwmacc_vv_i32m2(_acc13, _k20, _r33, vl);
                _acc13 = vwmacc_vv_i32m2(_acc13, _k21, _r34, vl);
                _acc13 = vwmacc_vv_i32m2(_acc13, _k22, _r35, vl);

                vint8mf2_t _res00 = requantize_m2_s(_acc00, _mult, _shift, out_zp, vl);
                vint8mf2_t _res01 = requantize_m2_s(_acc01, _mult, _shift, out_zp, vl);
                vint8mf2_t _res02 = requantize_m2_s(_acc02, _mult, _shift, out_zp, vl);
                vint8mf2_t _res03 = requantize_m2_s(_acc03, _mult, _shift, out_zp, vl);
                vint8mf2_t _res10 = requantize_m2_s(_acc10, _mult, _shift, out_zp, vl);
                vint8mf2_t _res11 = requantize_m2_s(_acc11, _mult, _shift, out_zp, vl);
                vint8mf2_t _res12 = requantize_m2_s(_acc12, _mult, _shift, out_zp, vl);
                vint8mf2_t _res13 = requantize_m2_s(_acc13, _mult, _shift, out_zp, vl);

                vse8_v_i8mf2(out0, _res00, vl);
                vse8_v_i8mf2(out0 + packn * 1, _res01, vl);
                vse8_v_i8mf2(out0 + packn * 2, _res02, vl);
                vse8_v_i8mf2(out0 + packn * 3, _res03, vl);
                vse8_v_i8mf2(out1, _res10, vl);
                vse8_v_i8mf2(out1 + packn * 1, _res11, vl);
                vse8_v_i8mf2(out1 + packn * 2, _res12, vl);
                vse8_v_i8mf2(out1 + packn * 3, _res13, vl);

                out0 += packn * 4;
                out1 += packn * 4;

                r0 += packn * 4;
                r1 += packn * 4;
                r2 += packn * 4;
                r3 += packn * 4;
            }
            for (; w + 1 < out_w; w += 2) {
                vint32m2_t _acc00 = _bias0;
                vint32m2_t _acc01 = _bias0;
                vint32m2_t _acc10 = _bias0;
                vint32m2_t _acc11 = _bias0;

                vint16m1_t _r00 = vwadd_vx_i16m1(vle8_v_i8mf2(r0, vl), 0, vl);
                vint16m1_t _r01 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 1 * packn, vl), 0, vl);
                vint16m1_t _r02 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 2 * packn, vl), 0, vl);
                vint16m1_t _r03 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 3 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k02, _r02, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k00, _r01, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k01, _r02, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k02, _r03, vl);

                vint16m1_t _r10 = vwadd_vx_i16m1(vle8_v_i8mf2(r1, vl), 0, vl);
                vint16m1_t _r11 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 1 * packn, vl), 0, vl);
                vint16m1_t _r12 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 2 * packn, vl), 0, vl);
                vint16m1_t _r13 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 3 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k12, _r12, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k10, _r11, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k11, _r12, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k12, _r13, vl);  //
                _acc10 = vwmacc_vv_i32m2(_acc10, _k00, _r10, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k01, _r11, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k02, _r12, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k00, _r11, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k01, _r12, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k02, _r13, vl);

                vint16m1_t _r20 = vwadd_vx_i16m1(vle8_v_i8mf2(r2, vl), 0, vl);
                vint16m1_t _r21 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 1 * packn, vl), 0, vl);
                vint16m1_t _r22 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 2 * packn, vl), 0, vl);
                vint16m1_t _r23 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 3 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k22, _r22, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k20, _r21, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k21, _r22, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k22, _r23, vl);  //
                _acc10 = vwmacc_vv_i32m2(_acc10, _k10, _r20, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k11, _r21, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k12, _r22, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k10, _r21, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k11, _r22, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k12, _r23, vl);

                vint16m1_t _r30 = vwadd_vx_i16m1(vle8_v_i8mf2(r3, vl), 0, vl);
                vint16m1_t _r31 = vwadd_vx_i16m1(vle8_v_i8mf2(r3 + 1 * packn, vl), 0, vl);
                vint16m1_t _r32 = vwadd_vx_i16m1(vle8_v_i8mf2(r3 + 2 * packn, vl), 0, vl);
                vint16m1_t _r33 = vwadd_vx_i16m1(vle8_v_i8mf2(r3 + 3 * packn, vl), 0, vl);

                _acc10 = vwmacc_vv_i32m2(_acc10, _k20, _r30, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k21, _r31, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k22, _r32, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k20, _r31, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k21, _r32, vl);
                _acc11 = vwmacc_vv_i32m2(_acc11, _k22, _r33, vl);

                vint8mf2_t _res00 = requantize_m2_s(_acc00, _mult, _shift, out_zp, vl);
                vint8mf2_t _res01 = requantize_m2_s(_acc01, _mult, _shift, out_zp, vl);
                vint8mf2_t _res10 = requantize_m2_s(_acc10, _mult, _shift, out_zp, vl);
                vint8mf2_t _res11 = requantize_m2_s(_acc11, _mult, _shift, out_zp, vl);

                vse8_v_i8mf2(out0, _res00, vl);
                vse8_v_i8mf2(out0 + packn * 1, _res01, vl);
                vse8_v_i8mf2(out1, _res10, vl);
                vse8_v_i8mf2(out1 + packn * 1, _res11, vl);

                out0 += packn * 2;
                out1 += packn * 2;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
                r3 += packn * 2;
            }
            for (; w < out_w; w++) {
                vint32m2_t _acc00 = _bias0;
                vint32m2_t _acc10 = _bias0;

                vint16m1_t _r00 = vwadd_vx_i16m1(vle8_v_i8mf2(r0, vl), 0, vl);
                vint16m1_t _r01 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 1 * packn, vl), 0, vl);
                vint16m1_t _r02 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k02, _r02, vl);

                vint16m1_t _r10 = vwadd_vx_i16m1(vle8_v_i8mf2(r1, vl), 0, vl);
                vint16m1_t _r11 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 1 * packn, vl), 0, vl);
                vint16m1_t _r12 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k12, _r12, vl);  //
                _acc10 = vwmacc_vv_i32m2(_acc10, _k00, _r10, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k01, _r11, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k02, _r12, vl);

                vint16m1_t _r20 = vwadd_vx_i16m1(vle8_v_i8mf2(r2, vl), 0, vl);
                vint16m1_t _r21 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 1 * packn, vl), 0, vl);
                vint16m1_t _r22 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k22, _r22, vl);  //
                _acc10 = vwmacc_vv_i32m2(_acc10, _k10, _r20, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k11, _r21, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k12, _r22, vl);

                vint16m1_t _r30 = vwadd_vx_i16m1(vle8_v_i8mf2(r3, vl), 0, vl);
                vint16m1_t _r31 = vwadd_vx_i16m1(vle8_v_i8mf2(r3 + 1 * packn, vl), 0, vl);
                vint16m1_t _r32 = vwadd_vx_i16m1(vle8_v_i8mf2(r3 + 2 * packn, vl), 0, vl);

                _acc10 = vwmacc_vv_i32m2(_acc10, _k20, _r30, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k21, _r31, vl);
                _acc10 = vwmacc_vv_i32m2(_acc10, _k22, _r32, vl);

                vint8mf2_t _res00 = requantize_m2_s(_acc00, _mult, _shift, out_zp, vl);
                vint8mf2_t _res10 = requantize_m2_s(_acc10, _mult, _shift, out_zp, vl);

                vse8_v_i8mf2(out0, _res00, vl);
                vse8_v_i8mf2(out1, _res10, vl);

                out0 += packn * 1;
                out1 += packn * 1;

                r0 += packn * 1;
                r1 += packn * 1;
                r2 += packn * 1;
                r3 += packn * 1;
            }
            r0 += (2 + in_w) * packn;
            r1 += (2 + in_w) * packn;
            r2 += (2 + in_w) * packn;
            r3 += (2 + in_w) * packn;

            out0 += out_w * packn;
            out1 += out_w * packn;
        }
        for (; h < out_h; h++) {
            int w = 0;
            // h1w4 loop
            for (; w + 3 < out_w; w += 4) {
                vint32m2_t _acc00 = _bias0;
                vint32m2_t _acc01 = _bias0;
                vint32m2_t _acc02 = _bias0;
                vint32m2_t _acc03 = _bias0;

                vint16m1_t _r00 = vwadd_vx_i16m1(vle8_v_i8mf2(r0, vl), 0, vl);
                vint16m1_t _r01 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 1 * packn, vl), 0, vl);
                vint16m1_t _r02 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 2 * packn, vl), 0, vl);
                vint16m1_t _r03 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 3 * packn, vl), 0, vl);
                vint16m1_t _r04 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 4 * packn, vl), 0, vl);
                vint16m1_t _r05 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 5 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k02, _r02, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k00, _r01, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k01, _r02, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k02, _r03, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k00, _r02, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k01, _r03, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k02, _r04, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k00, _r03, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k01, _r04, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k02, _r05, vl);

                vint16m1_t _r10 = vwadd_vx_i16m1(vle8_v_i8mf2(r1, vl), 0, vl);
                vint16m1_t _r11 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 1 * packn, vl), 0, vl);
                vint16m1_t _r12 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 2 * packn, vl), 0, vl);
                vint16m1_t _r13 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 3 * packn, vl), 0, vl);
                vint16m1_t _r14 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 4 * packn, vl), 0, vl);
                vint16m1_t _r15 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 5 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k12, _r12, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k10, _r11, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k11, _r12, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k12, _r13, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k10, _r12, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k11, _r13, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k12, _r14, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k10, _r13, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k11, _r14, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k12, _r15, vl);

                vint16m1_t _r20 = vwadd_vx_i16m1(vle8_v_i8mf2(r2, vl), 0, vl);
                vint16m1_t _r21 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 1 * packn, vl), 0, vl);
                vint16m1_t _r22 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 2 * packn, vl), 0, vl);
                vint16m1_t _r23 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 3 * packn, vl), 0, vl);
                vint16m1_t _r24 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 4 * packn, vl), 0, vl);
                vint16m1_t _r25 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 5 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k22, _r22, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k20, _r21, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k21, _r22, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k22, _r23, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k20, _r22, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k21, _r23, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k22, _r24, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k20, _r23, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k21, _r24, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k22, _r25, vl);

                vint8mf2_t _res00 = requantize_m2_s(_acc00, _mult, _shift, out_zp, vl);
                vint8mf2_t _res01 = requantize_m2_s(_acc01, _mult, _shift, out_zp, vl);
                vint8mf2_t _res02 = requantize_m2_s(_acc02, _mult, _shift, out_zp, vl);
                vint8mf2_t _res03 = requantize_m2_s(_acc03, _mult, _shift, out_zp, vl);

                vse8_v_i8mf2(out0, _res00, vl);
                vse8_v_i8mf2(out0 + packn * 1, _res01, vl);
                vse8_v_i8mf2(out0 + packn * 2, _res02, vl);
                vse8_v_i8mf2(out0 + packn * 3, _res03, vl);

                out0 += packn * 4;

                r0 += packn * 4;
                r1 += packn * 4;
                r2 += packn * 4;
            }
            for (; w + 1 < out_w; w += 2) {
                vint32m2_t _acc00 = _bias0;
                vint32m2_t _acc01 = _bias0;

                vint16m1_t _r00 = vwadd_vx_i16m1(vle8_v_i8mf2(r0, vl), 0, vl);
                vint16m1_t _r01 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 1 * packn, vl), 0, vl);
                vint16m1_t _r02 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 2 * packn, vl), 0, vl);
                vint16m1_t _r03 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 3 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k02, _r02, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k00, _r01, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k01, _r02, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k02, _r03, vl);

                vint16m1_t _r10 = vwadd_vx_i16m1(vle8_v_i8mf2(r1, vl), 0, vl);
                vint16m1_t _r11 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 1 * packn, vl), 0, vl);
                vint16m1_t _r12 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 2 * packn, vl), 0, vl);
                vint16m1_t _r13 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 3 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k12, _r12, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k10, _r11, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k11, _r12, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k12, _r13, vl);

                vint16m1_t _r20 = vwadd_vx_i16m1(vle8_v_i8mf2(r2, vl), 0, vl);
                vint16m1_t _r21 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 1 * packn, vl), 0, vl);
                vint16m1_t _r22 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 2 * packn, vl), 0, vl);
                vint16m1_t _r23 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 3 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k22, _r22, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k20, _r21, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k21, _r22, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k22, _r23, vl);

                vint8mf2_t _res00 = requantize_m2_s(_acc00, _mult, _shift, out_zp, vl);
                vint8mf2_t _res01 = requantize_m2_s(_acc01, _mult, _shift, out_zp, vl);

                vse8_v_i8mf2(out0, _res00, vl);
                vse8_v_i8mf2(out0 + packn * 1, _res01, vl);

                out0 += packn * 2;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
            }
            for (; w < out_w; w++) {
                vint32m2_t _acc00 = _bias0;

                vint16m1_t _r00 = vwadd_vx_i16m1(vle8_v_i8mf2(r0, vl), 0, vl);
                vint16m1_t _r01 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 1 * packn, vl), 0, vl);
                vint16m1_t _r02 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k02, _r02, vl);

                vint16m1_t _r10 = vwadd_vx_i16m1(vle8_v_i8mf2(r1, vl), 0, vl);
                vint16m1_t _r11 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 1 * packn, vl), 0, vl);
                vint16m1_t _r12 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k12, _r12, vl);

                vint16m1_t _r20 = vwadd_vx_i16m1(vle8_v_i8mf2(r2, vl), 0, vl);
                vint16m1_t _r21 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 1 * packn, vl), 0, vl);
                vint16m1_t _r22 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k22, _r22, vl);

                vint8mf2_t _res00 = requantize_m2_s(_acc00, _mult, _shift, out_zp, vl);

                vse8_v_i8mf2(out0, _res00, vl);

                out0 += packn * 1;

                r0 += packn * 1;
                r1 += packn * 1;
                r2 += packn * 1;
            }
        }
    }
#elif defined RVV_0_7_1
#pragma omp parallel for num_threads(1)
    for (int c = 0; c + packn - 1 < in_c; c += packn) {
        int8_t *out0 = output_data + c * out_h * out_w;
        int8_t *out1 = out0 + out_w * packn;

        const int8_t *r0 = input_padd_buf + c * in_h * in_w;
        const int8_t *r1 = r0 + in_w * packn;
        const int8_t *r2 = r1 + in_w * packn;
        const int8_t *r3 = r2 + in_w * packn;

        const int8_t *kernel0 = kernel_data + c * 9;

        vint16m2_t _k00 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0, vl), 0, vl);
        vint16m2_t _k01 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 1 * packn, vl), 0, vl);
        vint16m2_t _k02 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 2 * packn, vl), 0, vl);
        vint16m2_t _k10 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 3 * packn, vl), 0, vl);
        vint16m2_t _k11 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 4 * packn, vl), 0, vl);
        vint16m2_t _k12 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 5 * packn, vl), 0, vl);
        vint16m2_t _k20 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 6 * packn, vl), 0, vl);
        vint16m2_t _k21 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 7 * packn, vl), 0, vl);
        vint16m2_t _k22 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 8 * packn, vl), 0, vl);

        // please use fuse_zp2bias option in hhb, thus bias_data wont be NULL
        vint32m4_t _bias0 = vle32_v_i32m4(bias_data + c, vl);

        vint32m4_t _mult = vle32_v_i32m4(multiplier + c, vl);
        vint32m4_t _shift = vle32_v_i32m4(shift + c, vl);
        _shift = vrsub_vx_i32m4(_shift, -1, vl);
        int32_t out_zp = output->qinfo->zero_point;

        int h = 0;
        // h2 loop
        for (; h + 1 < out_h; h += 2) {
            int w = 0;
            // h2w2 loop
            for (; w + 1 < out_w; w += 2) {
                vint32m4_t _acc00 = _bias0;
                vint32m4_t _acc01 = _bias0;
                vint32m4_t _acc10 = _bias0;
                vint32m4_t _acc11 = _bias0;

                vint16m2_t _r00 = vwadd_vx_i16m2(vle8_v_i8m1(r0, vl), 0, vl);
                vint16m2_t _r01 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 1 * packn, vl), 0, vl);
                vint16m2_t _r02 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 2 * packn, vl), 0, vl);
                vint16m2_t _r03 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 3 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k02, _r02, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k00, _r01, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k01, _r02, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k02, _r03, vl);

                vint16m2_t _r10 = vwadd_vx_i16m2(vle8_v_i8m1(r1, vl), 0, vl);
                vint16m2_t _r11 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 1 * packn, vl), 0, vl);
                vint16m2_t _r12 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 2 * packn, vl), 0, vl);
                vint16m2_t _r13 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 3 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k12, _r12, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k10, _r11, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k11, _r12, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k12, _r13, vl);  //
                _acc10 = vwmacc_vv_i32m4(_acc10, _k00, _r10, vl);
                _acc10 = vwmacc_vv_i32m4(_acc10, _k01, _r11, vl);
                _acc10 = vwmacc_vv_i32m4(_acc10, _k02, _r12, vl);
                _acc11 = vwmacc_vv_i32m4(_acc11, _k00, _r11, vl);
                _acc11 = vwmacc_vv_i32m4(_acc11, _k01, _r12, vl);
                _acc11 = vwmacc_vv_i32m4(_acc11, _k02, _r13, vl);

                vint16m2_t _r20 = vwadd_vx_i16m2(vle8_v_i8m1(r2, vl), 0, vl);
                vint16m2_t _r21 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 1 * packn, vl), 0, vl);
                vint16m2_t _r22 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 2 * packn, vl), 0, vl);
                vint16m2_t _r23 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 3 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k22, _r22, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k20, _r21, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k21, _r22, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k22, _r23, vl);  //
                _acc10 = vwmacc_vv_i32m4(_acc10, _k10, _r20, vl);
                _acc10 = vwmacc_vv_i32m4(_acc10, _k11, _r21, vl);
                _acc10 = vwmacc_vv_i32m4(_acc10, _k12, _r22, vl);
                _acc11 = vwmacc_vv_i32m4(_acc11, _k10, _r21, vl);
                _acc11 = vwmacc_vv_i32m4(_acc11, _k11, _r22, vl);
                _acc11 = vwmacc_vv_i32m4(_acc11, _k12, _r23, vl);

                vint16m2_t _r30 = vwadd_vx_i16m2(vle8_v_i8m1(r3, vl), 0, vl);
                vint16m2_t _r31 = vwadd_vx_i16m2(vle8_v_i8m1(r3 + 1 * packn, vl), 0, vl);
                vint16m2_t _r32 = vwadd_vx_i16m2(vle8_v_i8m1(r3 + 2 * packn, vl), 0, vl);
                vint16m2_t _r33 = vwadd_vx_i16m2(vle8_v_i8m1(r3 + 3 * packn, vl), 0, vl);

                _acc10 = vwmacc_vv_i32m4(_acc10, _k20, _r30, vl);
                _acc10 = vwmacc_vv_i32m4(_acc10, _k21, _r31, vl);
                _acc10 = vwmacc_vv_i32m4(_acc10, _k22, _r32, vl);
                _acc11 = vwmacc_vv_i32m4(_acc11, _k20, _r31, vl);
                _acc11 = vwmacc_vv_i32m4(_acc11, _k21, _r32, vl);
                _acc11 = vwmacc_vv_i32m4(_acc11, _k22, _r33, vl);

                vint8m1_t _res00 = requantize_m4_s(_acc00, _mult, _shift, out_zp, vl);
                vint8m1_t _res01 = requantize_m4_s(_acc01, _mult, _shift, out_zp, vl);
                vint8m1_t _res10 = requantize_m4_s(_acc10, _mult, _shift, out_zp, vl);
                vint8m1_t _res11 = requantize_m4_s(_acc11, _mult, _shift, out_zp, vl);

                vse8_v_i8m1(out0, _res00, vl);
                vse8_v_i8m1(out0 + packn * 1, _res01, vl);
                vse8_v_i8m1(out1, _res10, vl);
                vse8_v_i8m1(out1 + packn * 1, _res11, vl);

                out0 += packn * 2;
                out1 += packn * 2;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
                r3 += packn * 2;
            }
            for (; w < out_w; w++) {
                vint32m4_t _acc00 = _bias0;
                vint32m4_t _acc10 = _bias0;

                vint16m2_t _r00 = vwadd_vx_i16m2(vle8_v_i8m1(r0, vl), 0, vl);
                vint16m2_t _r01 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 1 * packn, vl), 0, vl);
                vint16m2_t _r02 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k02, _r02, vl);

                vint16m2_t _r10 = vwadd_vx_i16m2(vle8_v_i8m1(r1, vl), 0, vl);
                vint16m2_t _r11 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 1 * packn, vl), 0, vl);
                vint16m2_t _r12 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k12, _r12, vl);  //
                _acc10 = vwmacc_vv_i32m4(_acc10, _k00, _r10, vl);
                _acc10 = vwmacc_vv_i32m4(_acc10, _k01, _r11, vl);
                _acc10 = vwmacc_vv_i32m4(_acc10, _k02, _r12, vl);

                vint16m2_t _r20 = vwadd_vx_i16m2(vle8_v_i8m1(r2, vl), 0, vl);
                vint16m2_t _r21 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 1 * packn, vl), 0, vl);
                vint16m2_t _r22 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k22, _r22, vl);  //
                _acc10 = vwmacc_vv_i32m4(_acc10, _k10, _r20, vl);
                _acc10 = vwmacc_vv_i32m4(_acc10, _k11, _r21, vl);
                _acc10 = vwmacc_vv_i32m4(_acc10, _k12, _r22, vl);

                vint16m2_t _r30 = vwadd_vx_i16m2(vle8_v_i8m1(r3, vl), 0, vl);
                vint16m2_t _r31 = vwadd_vx_i16m2(vle8_v_i8m1(r3 + 1 * packn, vl), 0, vl);
                vint16m2_t _r32 = vwadd_vx_i16m2(vle8_v_i8m1(r3 + 2 * packn, vl), 0, vl);

                _acc10 = vwmacc_vv_i32m4(_acc10, _k20, _r30, vl);
                _acc10 = vwmacc_vv_i32m4(_acc10, _k21, _r31, vl);
                _acc10 = vwmacc_vv_i32m4(_acc10, _k22, _r32, vl);

                vint8m1_t _res00 = requantize_m4_s(_acc00, _mult, _shift, out_zp, vl);
                vint8m1_t _res10 = requantize_m4_s(_acc10, _mult, _shift, out_zp, vl);

                vse8_v_i8m1(out0, _res00, vl);
                vse8_v_i8m1(out1, _res10, vl);

                out0 += packn * 1;
                out1 += packn * 1;

                r0 += packn * 1;
                r1 += packn * 1;
                r2 += packn * 1;
                r3 += packn * 1;
            }
            r0 += (2 + in_w) * packn;
            r1 += (2 + in_w) * packn;
            r2 += (2 + in_w) * packn;
            r3 += (2 + in_w) * packn;

            out0 += out_w * packn;
            out1 += out_w * packn;
        }
        for (; h < out_h; h++) {
            int w = 0;
            // h1w4 loop
            for (; w + 3 < out_w; w += 4) {
                vint32m4_t _acc00 = _bias0;
                vint32m4_t _acc01 = _bias0;
                vint32m4_t _acc02 = _bias0;
                vint32m4_t _acc03 = _bias0;

                vint16m2_t _r00 = vwadd_vx_i16m2(vle8_v_i8m1(r0, vl), 0, vl);
                vint16m2_t _r01 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 1 * packn, vl), 0, vl);
                vint16m2_t _r02 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 2 * packn, vl), 0, vl);
                vint16m2_t _r03 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 3 * packn, vl), 0, vl);
                vint16m2_t _r04 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 4 * packn, vl), 0, vl);
                vint16m2_t _r05 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 5 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k02, _r02, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k00, _r01, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k01, _r02, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k02, _r03, vl);
                _acc02 = vwmacc_vv_i32m4(_acc02, _k00, _r02, vl);
                _acc02 = vwmacc_vv_i32m4(_acc02, _k01, _r03, vl);
                _acc02 = vwmacc_vv_i32m4(_acc02, _k02, _r04, vl);
                _acc03 = vwmacc_vv_i32m4(_acc03, _k00, _r03, vl);
                _acc03 = vwmacc_vv_i32m4(_acc03, _k01, _r04, vl);
                _acc03 = vwmacc_vv_i32m4(_acc03, _k02, _r05, vl);

                vint16m2_t _r10 = vwadd_vx_i16m2(vle8_v_i8m1(r1, vl), 0, vl);
                vint16m2_t _r11 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 1 * packn, vl), 0, vl);
                vint16m2_t _r12 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 2 * packn, vl), 0, vl);
                vint16m2_t _r13 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 3 * packn, vl), 0, vl);
                vint16m2_t _r14 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 4 * packn, vl), 0, vl);
                vint16m2_t _r15 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 5 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k12, _r12, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k10, _r11, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k11, _r12, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k12, _r13, vl);
                _acc02 = vwmacc_vv_i32m4(_acc02, _k10, _r12, vl);
                _acc02 = vwmacc_vv_i32m4(_acc02, _k11, _r13, vl);
                _acc02 = vwmacc_vv_i32m4(_acc02, _k12, _r14, vl);
                _acc03 = vwmacc_vv_i32m4(_acc03, _k10, _r13, vl);
                _acc03 = vwmacc_vv_i32m4(_acc03, _k11, _r14, vl);
                _acc03 = vwmacc_vv_i32m4(_acc03, _k12, _r15, vl);

                vint16m2_t _r20 = vwadd_vx_i16m2(vle8_v_i8m1(r2, vl), 0, vl);
                vint16m2_t _r21 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 1 * packn, vl), 0, vl);
                vint16m2_t _r22 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 2 * packn, vl), 0, vl);
                vint16m2_t _r23 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 3 * packn, vl), 0, vl);
                vint16m2_t _r24 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 4 * packn, vl), 0, vl);
                vint16m2_t _r25 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 5 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k22, _r22, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k20, _r21, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k21, _r22, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k22, _r23, vl);
                _acc02 = vwmacc_vv_i32m4(_acc02, _k20, _r22, vl);
                _acc02 = vwmacc_vv_i32m4(_acc02, _k21, _r23, vl);
                _acc02 = vwmacc_vv_i32m4(_acc02, _k22, _r24, vl);
                _acc03 = vwmacc_vv_i32m4(_acc03, _k20, _r23, vl);
                _acc03 = vwmacc_vv_i32m4(_acc03, _k21, _r24, vl);
                _acc03 = vwmacc_vv_i32m4(_acc03, _k22, _r25, vl);

                vint8m1_t _res00 = requantize_m4_s(_acc00, _mult, _shift, out_zp, vl);
                vint8m1_t _res01 = requantize_m4_s(_acc01, _mult, _shift, out_zp, vl);
                vint8m1_t _res02 = requantize_m4_s(_acc02, _mult, _shift, out_zp, vl);
                vint8m1_t _res03 = requantize_m4_s(_acc03, _mult, _shift, out_zp, vl);

                vse8_v_i8m1(out0, _res00, vl);
                vse8_v_i8m1(out0 + packn * 1, _res01, vl);
                vse8_v_i8m1(out0 + packn * 2, _res02, vl);
                vse8_v_i8m1(out0 + packn * 3, _res03, vl);

                out0 += packn * 4;

                r0 += packn * 4;
                r1 += packn * 4;
                r2 += packn * 4;
            }
            for (; w + 1 < out_w; w += 2) {
                vint32m4_t _acc00 = _bias0;
                vint32m4_t _acc01 = _bias0;

                vint16m2_t _r00 = vwadd_vx_i16m2(vle8_v_i8m1(r0, vl), 0, vl);
                vint16m2_t _r01 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 1 * packn, vl), 0, vl);
                vint16m2_t _r02 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 2 * packn, vl), 0, vl);
                vint16m2_t _r03 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 3 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k02, _r02, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k00, _r01, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k01, _r02, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k02, _r03, vl);

                vint16m2_t _r10 = vwadd_vx_i16m2(vle8_v_i8m1(r1, vl), 0, vl);
                vint16m2_t _r11 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 1 * packn, vl), 0, vl);
                vint16m2_t _r12 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 2 * packn, vl), 0, vl);
                vint16m2_t _r13 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 3 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k12, _r12, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k10, _r11, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k11, _r12, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k12, _r13, vl);

                vint16m2_t _r20 = vwadd_vx_i16m2(vle8_v_i8m1(r2, vl), 0, vl);
                vint16m2_t _r21 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 1 * packn, vl), 0, vl);
                vint16m2_t _r22 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 2 * packn, vl), 0, vl);
                vint16m2_t _r23 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 3 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k22, _r22, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k20, _r21, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k21, _r22, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k22, _r23, vl);

                vint8m1_t _res00 = requantize_m4_s(_acc00, _mult, _shift, out_zp, vl);
                vint8m1_t _res01 = requantize_m4_s(_acc01, _mult, _shift, out_zp, vl);

                vse8_v_i8m1(out0, _res00, vl);
                vse8_v_i8m1(out0 + packn * 1, _res01, vl);

                out0 += packn * 2;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
            }
            for (; w < out_w; w++) {
                vint32m4_t _acc00 = _bias0;

                vint16m2_t _r00 = vwadd_vx_i16m2(vle8_v_i8m1(r0, vl), 0, vl);
                vint16m2_t _r01 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 1 * packn, vl), 0, vl);
                vint16m2_t _r02 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k02, _r02, vl);

                vint16m2_t _r10 = vwadd_vx_i16m2(vle8_v_i8m1(r1, vl), 0, vl);
                vint16m2_t _r11 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 1 * packn, vl), 0, vl);
                vint16m2_t _r12 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k12, _r12, vl);

                vint16m2_t _r20 = vwadd_vx_i16m2(vle8_v_i8m1(r2, vl), 0, vl);
                vint16m2_t _r21 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 1 * packn, vl), 0, vl);
                vint16m2_t _r22 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k22, _r22, vl);

                vint8m1_t _res00 = requantize_m4_s(_acc00, _mult, _shift, out_zp, vl);

                vse8_v_i8m1(out0, _res00, vl);

                out0 += packn * 1;

                r0 += packn * 1;
                r1 += packn * 1;
                r2 += packn * 1;
            }
        }
    }
#endif
    shl_mem_free(input_padd_buf);
    shl_mem_free(multiplier);
    shl_mem_free(shift);
    return CSINN_TRUE;
}

int shl_rvv_dwconv3x3s2_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params)
{
    if (input->layout == CSINN_LAYOUT_NCHW) {
        shl_rvv_tensor_ndarray_to_nc1xc0_replace_int8(input);
    }
    if (output->layout == CSINN_LAYOUT_NCHW) {
        output->dim[1] /= input->dim[4];
        output->dim[4] = input->dim[4];
        output->dim_count = 5;
        output->layout = CSINN_LAYOUT_NC1HWC0;
    }
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)kernel->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1] * input->dim[4];  // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];
    int32_t out_c = in_c;
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int32_t *multiplier = (int32_t *)shl_mem_alloc(out_c * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(out_c * sizeof(int32_t));

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);

    int8_t *input_padd_buf = (int8_t *)shl_mem_alloc((in_h + params->pad_top + params->pad_down) *
                                                     (in_w + params->pad_left + params->pad_right) *
                                                     in_c * sizeof(int8_t));

    shl_rvv_pad_input_packn_int8(input_data, input_padd_buf, in_c, in_h, in_w,
                                 in_h + params->pad_top + params->pad_down,
                                 in_w + params->pad_left + params->pad_right, params->pad_top,
                                 params->pad_left, input->qinfo->zero_point);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

    int tailstep = (in_w - 2 * out_w + in_w) * packn;

    if (kernel->quant_channel > 1) {
        for (int c = 0; c < out_c; c++) {
            multiplier[c] = kernel->qinfo[c].multiplier;
            shift[c] = kernel->qinfo[c].shift;
        }
    } else if (kernel->quant_channel == 1) {
        for (int c = 0; c < out_c; c++) {
            multiplier[c] = kernel->qinfo[0].multiplier;
            shift[c] = kernel->qinfo[0].shift;
        }
    }

#ifdef RVV_1_0_0
#pragma omp parallel for num_threads(1)
    for (int c = 0; c + packn - 1 < in_c; c += packn) {
        int8_t *out0 = output_data + c * out_h * out_w;

        int8_t *r0 = input_padd_buf + c * in_h * in_w;
        int8_t *r1 = r0 + in_w * packn;
        int8_t *r2 = r1 + in_w * packn;

        const int8_t *kernel0 = kernel_data + c * 9;

        vint16m1_t _k00 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0, vl), 0, vl);
        vint16m1_t _k01 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 1 * packn, vl), 0, vl);
        vint16m1_t _k02 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 2 * packn, vl), 0, vl);
        vint16m1_t _k10 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 3 * packn, vl), 0, vl);
        vint16m1_t _k11 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 4 * packn, vl), 0, vl);
        vint16m1_t _k12 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 5 * packn, vl), 0, vl);
        vint16m1_t _k20 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 6 * packn, vl), 0, vl);
        vint16m1_t _k21 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 7 * packn, vl), 0, vl);
        vint16m1_t _k22 = vwadd_vx_i16m1(vle8_v_i8mf2(kernel0 + 8 * packn, vl), 0, vl);

        // please use fuse_zp2bias option in hhb, thus bias_data wont be NULL
        vint32m2_t _bias0 = vle32_v_i32m2(bias_data + c, vl);

        vint32m2_t _mult = vle32_v_i32m2(multiplier + c, vl);
        vint32m2_t _shift = vle32_v_i32m2(shift + c, vl);
        _shift = vrsub_vx_i32m2(_shift, -1, vl);
        int32_t out_zp = output->qinfo->zero_point;

        for (int h = 0; h < out_h; h++) {
            int w = 0;
            for (; w + 3 < out_w; w += 4) {
                vint32m2_t _acc00 = _bias0;
                vint32m2_t _acc01 = _bias0;
                vint32m2_t _acc02 = _bias0;
                vint32m2_t _acc03 = _bias0;

                vint16m1_t _r00 = vwadd_vx_i16m1(vle8_v_i8mf2(r0, vl), 0, vl);
                vint16m1_t _r01 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 1 * packn, vl), 0, vl);
                vint16m1_t _r02 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 2 * packn, vl), 0, vl);
                vint16m1_t _r03 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 3 * packn, vl), 0, vl);
                vint16m1_t _r04 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 4 * packn, vl), 0, vl);
                vint16m1_t _r05 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 5 * packn, vl), 0, vl);
                vint16m1_t _r06 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 6 * packn, vl), 0, vl);
                vint16m1_t _r07 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 7 * packn, vl), 0, vl);
                vint16m1_t _r08 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 8 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k02, _r02, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k00, _r02, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k01, _r03, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k02, _r04, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k00, _r04, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k01, _r05, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k02, _r06, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k00, _r06, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k01, _r07, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k02, _r08, vl);

                vint16m1_t _r10 = vwadd_vx_i16m1(vle8_v_i8mf2(r1, vl), 0, vl);
                vint16m1_t _r11 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 1 * packn, vl), 0, vl);
                vint16m1_t _r12 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 2 * packn, vl), 0, vl);
                vint16m1_t _r13 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 3 * packn, vl), 0, vl);
                vint16m1_t _r14 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 4 * packn, vl), 0, vl);
                vint16m1_t _r15 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 5 * packn, vl), 0, vl);
                vint16m1_t _r16 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 6 * packn, vl), 0, vl);
                vint16m1_t _r17 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 7 * packn, vl), 0, vl);
                vint16m1_t _r18 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 8 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k12, _r12, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k10, _r12, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k11, _r13, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k12, _r14, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k10, _r14, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k11, _r15, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k12, _r16, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k10, _r16, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k11, _r17, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k12, _r18, vl);

                vint16m1_t _r20 = vwadd_vx_i16m1(vle8_v_i8mf2(r2, vl), 0, vl);
                vint16m1_t _r21 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 1 * packn, vl), 0, vl);
                vint16m1_t _r22 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 2 * packn, vl), 0, vl);
                vint16m1_t _r23 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 3 * packn, vl), 0, vl);
                vint16m1_t _r24 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 4 * packn, vl), 0, vl);
                vint16m1_t _r25 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 5 * packn, vl), 0, vl);
                vint16m1_t _r26 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 6 * packn, vl), 0, vl);
                vint16m1_t _r27 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 7 * packn, vl), 0, vl);
                vint16m1_t _r28 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 8 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k22, _r22, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k20, _r22, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k21, _r23, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k22, _r24, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k20, _r24, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k21, _r25, vl);
                _acc02 = vwmacc_vv_i32m2(_acc02, _k22, _r26, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k20, _r26, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k21, _r27, vl);
                _acc03 = vwmacc_vv_i32m2(_acc03, _k22, _r28, vl);

                vint8mf2_t _res00 = requantize_m2_s(_acc00, _mult, _shift, out_zp, vl);
                vint8mf2_t _res01 = requantize_m2_s(_acc01, _mult, _shift, out_zp, vl);
                vint8mf2_t _res02 = requantize_m2_s(_acc02, _mult, _shift, out_zp, vl);
                vint8mf2_t _res03 = requantize_m2_s(_acc03, _mult, _shift, out_zp, vl);

                vse8_v_i8mf2(out0, _res00, vl);
                vse8_v_i8mf2(out0 + packn * 1, _res01, vl);
                vse8_v_i8mf2(out0 + packn * 2, _res02, vl);
                vse8_v_i8mf2(out0 + packn * 3, _res03, vl);

                out0 += packn * 4;

                r0 += packn * 8;
                r1 += packn * 8;
                r2 += packn * 8;
            }
            for (; w + 1 < out_w; w += 2) {
                vint32m2_t _acc00 = _bias0;
                vint32m2_t _acc01 = _bias0;

                vint16m1_t _r00 = vwadd_vx_i16m1(vle8_v_i8mf2(r0, vl), 0, vl);
                vint16m1_t _r01 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 1 * packn, vl), 0, vl);
                vint16m1_t _r02 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 2 * packn, vl), 0, vl);
                vint16m1_t _r03 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 3 * packn, vl), 0, vl);
                vint16m1_t _r04 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 4 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k02, _r02, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k00, _r02, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k01, _r03, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k02, _r04, vl);

                vint16m1_t _r10 = vwadd_vx_i16m1(vle8_v_i8mf2(r1, vl), 0, vl);
                vint16m1_t _r11 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 1 * packn, vl), 0, vl);
                vint16m1_t _r12 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 2 * packn, vl), 0, vl);
                vint16m1_t _r13 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 3 * packn, vl), 0, vl);
                vint16m1_t _r14 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 4 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k12, _r12, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k10, _r12, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k11, _r13, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k12, _r14, vl);

                vint16m1_t _r20 = vwadd_vx_i16m1(vle8_v_i8mf2(r2, vl), 0, vl);
                vint16m1_t _r21 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 1 * packn, vl), 0, vl);
                vint16m1_t _r22 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 2 * packn, vl), 0, vl);
                vint16m1_t _r23 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 3 * packn, vl), 0, vl);
                vint16m1_t _r24 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 4 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k22, _r22, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k20, _r22, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k21, _r23, vl);
                _acc01 = vwmacc_vv_i32m2(_acc01, _k22, _r24, vl);

                vint8mf2_t _res00 = requantize_m2_s(_acc00, _mult, _shift, out_zp, vl);
                vint8mf2_t _res01 = requantize_m2_s(_acc01, _mult, _shift, out_zp, vl);

                vse8_v_i8mf2(out0, _res00, vl);
                vse8_v_i8mf2(out0 + packn * 1, _res01, vl);

                out0 += packn * 2;

                r0 += packn * 4;
                r1 += packn * 4;
                r2 += packn * 4;
            }
            for (; w < out_w; w++) {
                vint32m2_t _acc00 = _bias0;

                vint16m1_t _r00 = vwadd_vx_i16m1(vle8_v_i8mf2(r0, vl), 0, vl);
                vint16m1_t _r01 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 1 * packn, vl), 0, vl);
                vint16m1_t _r02 = vwadd_vx_i16m1(vle8_v_i8mf2(r0 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k02, _r02, vl);

                vint16m1_t _r10 = vwadd_vx_i16m1(vle8_v_i8mf2(r1, vl), 0, vl);
                vint16m1_t _r11 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 1 * packn, vl), 0, vl);
                vint16m1_t _r12 = vwadd_vx_i16m1(vle8_v_i8mf2(r1 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k12, _r12, vl);

                vint16m1_t _r20 = vwadd_vx_i16m1(vle8_v_i8mf2(r2, vl), 0, vl);
                vint16m1_t _r21 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 1 * packn, vl), 0, vl);
                vint16m1_t _r22 = vwadd_vx_i16m1(vle8_v_i8mf2(r2 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m2(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m2(_acc00, _k22, _r22, vl);

                vint8mf2_t _res00 = requantize_m2_s(_acc00, _mult, _shift, out_zp, vl);

                vse8_v_i8mf2(out0, _res00, vl);

                out0 += packn * 1;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
            }
            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
#elif defined RVV_0_7_1
#pragma omp parallel for num_threads(1)
    for (int c = 0; c + packn - 1 < in_c; c += packn) {
        int8_t *out0 = output_data + c * out_h * out_w;

        int8_t *r0 = input_padd_buf + c * in_h * in_w;
        int8_t *r1 = r0 + in_w * packn;
        int8_t *r2 = r1 + in_w * packn;

        const int8_t *kernel0 = kernel_data + c * 9;

        vint16m2_t _k00 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0, vl), 0, vl);
        vint16m2_t _k01 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 1 * packn, vl), 0, vl);
        vint16m2_t _k02 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 2 * packn, vl), 0, vl);
        vint16m2_t _k10 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 3 * packn, vl), 0, vl);
        vint16m2_t _k11 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 4 * packn, vl), 0, vl);
        vint16m2_t _k12 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 5 * packn, vl), 0, vl);
        vint16m2_t _k20 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 6 * packn, vl), 0, vl);
        vint16m2_t _k21 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 7 * packn, vl), 0, vl);
        vint16m2_t _k22 = vwadd_vx_i16m2(vle8_v_i8m1(kernel0 + 8 * packn, vl), 0, vl);

        // please use fuse_zp2bias option in hhb, thus bias_data wont be NULL
        vint32m4_t _bias0 = vle32_v_i32m4(bias_data + c, vl);

        vint32m4_t _mult = vle32_v_i32m4(multiplier + c, vl);
        vint32m4_t _shift = vle32_v_i32m4(shift + c, vl);
        _shift = vrsub_vx_i32m4(_shift, -1, vl);
        int32_t out_zp = output->qinfo->zero_point;

        for (int h = 0; h < out_h; h++) {
            int w = 0;
            for (; w + 1 < out_w; w += 2) {
                vint32m4_t _acc00 = _bias0;
                vint32m4_t _acc01 = _bias0;

                vint16m2_t _r00 = vwadd_vx_i16m2(vle8_v_i8m1(r0, vl), 0, vl);
                vint16m2_t _r01 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 1 * packn, vl), 0, vl);
                vint16m2_t _r02 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 2 * packn, vl), 0, vl);
                vint16m2_t _r03 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 3 * packn, vl), 0, vl);
                vint16m2_t _r04 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 4 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k02, _r02, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k00, _r02, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k01, _r03, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k02, _r04, vl);

                vint16m2_t _r10 = vwadd_vx_i16m2(vle8_v_i8m1(r1, vl), 0, vl);
                vint16m2_t _r11 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 1 * packn, vl), 0, vl);
                vint16m2_t _r12 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 2 * packn, vl), 0, vl);
                vint16m2_t _r13 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 3 * packn, vl), 0, vl);
                vint16m2_t _r14 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 4 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k12, _r12, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k10, _r12, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k11, _r13, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k12, _r14, vl);

                vint16m2_t _r20 = vwadd_vx_i16m2(vle8_v_i8m1(r2, vl), 0, vl);
                vint16m2_t _r21 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 1 * packn, vl), 0, vl);
                vint16m2_t _r22 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 2 * packn, vl), 0, vl);
                vint16m2_t _r23 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 3 * packn, vl), 0, vl);
                vint16m2_t _r24 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 4 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k22, _r22, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k20, _r22, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k21, _r23, vl);
                _acc01 = vwmacc_vv_i32m4(_acc01, _k22, _r24, vl);

                vint8m1_t _res00 = requantize_m4_s(_acc00, _mult, _shift, out_zp, vl);
                vint8m1_t _res01 = requantize_m4_s(_acc01, _mult, _shift, out_zp, vl);

                vse8_v_i8m1(out0, _res00, vl);
                vse8_v_i8m1(out0 + packn * 1, _res01, vl);

                out0 += packn * 2;

                r0 += packn * 4;
                r1 += packn * 4;
                r2 += packn * 4;
            }
            for (; w < out_w; w++) {
                vint32m4_t _acc00 = _bias0;

                vint16m2_t _r00 = vwadd_vx_i16m2(vle8_v_i8m1(r0, vl), 0, vl);
                vint16m2_t _r01 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 1 * packn, vl), 0, vl);
                vint16m2_t _r02 = vwadd_vx_i16m2(vle8_v_i8m1(r0 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k00, _r00, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k01, _r01, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k02, _r02, vl);

                vint16m2_t _r10 = vwadd_vx_i16m2(vle8_v_i8m1(r1, vl), 0, vl);
                vint16m2_t _r11 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 1 * packn, vl), 0, vl);
                vint16m2_t _r12 = vwadd_vx_i16m2(vle8_v_i8m1(r1 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k10, _r10, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k11, _r11, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k12, _r12, vl);

                vint16m2_t _r20 = vwadd_vx_i16m2(vle8_v_i8m1(r2, vl), 0, vl);
                vint16m2_t _r21 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 1 * packn, vl), 0, vl);
                vint16m2_t _r22 = vwadd_vx_i16m2(vle8_v_i8m1(r2 + 2 * packn, vl), 0, vl);

                _acc00 = vwmacc_vv_i32m4(_acc00, _k20, _r20, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k21, _r21, vl);
                _acc00 = vwmacc_vv_i32m4(_acc00, _k22, _r22, vl);

                vint8m1_t _res00 = requantize_m4_s(_acc00, _mult, _shift, out_zp, vl);

                vse8_v_i8m1(out0, _res00, vl);

                out0 += packn * 1;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
            }
            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
#endif
    shl_mem_free(input_padd_buf);
    shl_mem_free(multiplier);
    shl_mem_free(shift);
    return CSINN_TRUE;
}

/****************************************************************************
 * packn = vlenb / sizeof(int8_t) / 2
 * maxk = ksize_h * ksize_w
 * constrain: out_c % packn = 0 and in_ch = 1
 * layout: [out_c, 1, ksize_h, ksize_w] ==> [out_c/packn, 1, maxk, packn]
 ***************************************************************************/
void shl_rvv_dwconv_reorder_kernel_packn_int8(struct csinn_tensor *kernel,
                                              struct csinn_conv2d_params *params)
{
    int8_t *kernel_data = (int8_t *)kernel->data;
    const int out_ch = kernel->dim[0];
    const int maxk = kernel->dim[2] * kernel->dim[3];
    int8_t *kernel_trans = (int8_t *)shl_mem_alloc(out_ch * maxk * sizeof(int8_t));

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);

    for (int oc = 0; oc + packn - 1 < out_ch; oc += packn) {
        int8_t *ksrc = kernel_data + oc * maxk;
        int8_t *kdst = kernel_trans + oc * maxk;
        for (int ic = 0; ic < maxk; ic++) {
            vint8m1_t _tmp = vlse8_v_i8m1(ksrc + ic, maxk * sizeof(int8_t), vl);
            vse8_v_i8m1(kdst, _tmp, vl);
            kdst += vl;
        }
    }
    memcpy(kernel_data, kernel_trans, out_ch * maxk * sizeof(int8_t));
    shl_mem_free(kernel_trans);
}
