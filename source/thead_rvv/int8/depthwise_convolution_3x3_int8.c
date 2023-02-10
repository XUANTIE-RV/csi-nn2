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
#include "shl_thead_rvv.h"

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

int shl_rvv_dwconv3x3s1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params)
{
    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(input);
    }
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)kernel->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];  // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int8_t *input_padd_buf = (int8_t *)shl_mem_alloc((in_h + params->pad_top + params->pad_down) *
                                                     (in_w + params->pad_left + params->pad_right) *
                                                     in_c * sizeof(int8_t));

    shl_rvv_pad_input_int8(input_data, input_padd_buf, in_c, in_h, in_w,
                           in_h + params->pad_top + params->pad_down,
                           in_w + params->pad_left + params->pad_right, params->pad_top,
                           params->pad_left, input->qinfo->zero_point);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

#pragma omp parallel for num_threads(1)
    for (int c = 0; c < in_c; c++) {
        int8_t *outptr0 = output_data;
        int8_t *outptr1 = outptr0 + out_w;

        // please use fuse_zp2bias option in hhb, thus bias_data wont be NULL
        int32_t bias0 = bias_data[c];

        int8_t *img0 = input_padd_buf + c * in_h * in_w;
        int8_t *r0 = img0;
        int8_t *r1 = r0 + in_w;
        int8_t *r2 = r1 + in_w;
        int8_t *r3 = r2 + in_w;

        const int8_t *kernel0 = kernel_data + c * 9;

        int8_t k00 = kernel0[0];
        int8_t k01 = kernel0[1];
        int8_t k02 = kernel0[2];
        int8_t k10 = kernel0[3];
        int8_t k11 = kernel0[4];
        int8_t k12 = kernel0[5];
        int8_t k20 = kernel0[6];
        int8_t k21 = kernel0[7];
        int8_t k22 = kernel0[8];
        int vl;
        int h = 0;
        // h2 loop
        for (; h + 1 < out_h; h += 2) {
            int w = out_w;
            // h2w8 loop
            while (w > 0) {
                vl = vsetvl_e32m4(w);
                vint32m4_t _acc0 = vmv_v_x_i32m4(bias0, vl);
                vint32m4_t _acc1 = vmv_v_x_i32m4(bias0, vl);

                vint8m1_t _r0_0_7 = vle8_v_i8m1(r0, vl);
                vint8m1_t _r0_1_8 = vle8_v_i8m1(r0 + 1, vl);
                vint8m1_t _r0_2_9 = vle8_v_i8m1(r0 + 2, vl);

                vint8m1_t _r1_0_7 = vle8_v_i8m1(r1, vl);
                vint8m1_t _r1_1_8 = vle8_v_i8m1(r1 + 1, vl);
                vint8m1_t _r1_2_9 = vle8_v_i8m1(r1 + 2, vl);

                vint8m1_t _r2_0_7 = vle8_v_i8m1(r2, vl);
                vint8m1_t _r2_1_8 = vle8_v_i8m1(r2 + 1, vl);
                vint8m1_t _r2_2_9 = vle8_v_i8m1(r2 + 2, vl);

                vint8m1_t _r3_0_7 = vle8_v_i8m1(r3, vl);
                vint8m1_t _r3_1_8 = vle8_v_i8m1(r3 + 1, vl);
                vint8m1_t _r3_2_9 = vle8_v_i8m1(r3 + 2, vl);

                vint16m2_t _r0_0_7_w = vwadd_vx_i16m2(_r0_0_7, 0, vl);  // widden 8->16
                vint16m2_t _r0_1_8_w = vwadd_vx_i16m2(_r0_1_8, 0, vl);
                vint16m2_t _r0_2_9_w = vwadd_vx_i16m2(_r0_2_9, 0, vl);

                vint16m2_t _r1_0_7_w = vwadd_vx_i16m2(_r1_0_7, 0, vl);
                vint16m2_t _r1_1_8_w = vwadd_vx_i16m2(_r1_1_8, 0, vl);
                vint16m2_t _r1_2_9_w = vwadd_vx_i16m2(_r1_2_9, 0, vl);

                vint16m2_t _r2_0_7_w = vwadd_vx_i16m2(_r2_0_7, 0, vl);
                vint16m2_t _r2_1_8_w = vwadd_vx_i16m2(_r2_1_8, 0, vl);
                vint16m2_t _r2_2_9_w = vwadd_vx_i16m2(_r2_2_9, 0, vl);

                vint16m2_t _r3_0_7_w = vwadd_vx_i16m2(_r3_0_7, 0, vl);
                vint16m2_t _r3_1_8_w = vwadd_vx_i16m2(_r3_1_8, 0, vl);
                vint16m2_t _r3_2_9_w = vwadd_vx_i16m2(_r3_2_9, 0, vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, k00, _r0_0_7_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k01, _r0_1_8_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k02, _r0_2_9_w, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, k00, _r1_0_7_w, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, k01, _r1_1_8_w, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, k02, _r1_2_9_w, vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, k10, _r1_0_7_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k11, _r1_1_8_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k12, _r1_2_9_w, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, k10, _r2_0_7_w, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, k11, _r2_1_8_w, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, k12, _r2_2_9_w, vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, k20, _r2_0_7_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k21, _r2_1_8_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k22, _r2_2_9_w, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, k20, _r3_0_7_w, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, k21, _r3_1_8_w, vl);
                _acc1 = vwmacc_vx_i32m4(_acc1, k22, _r3_2_9_w, vl);

                // int32_t q1z2 = (k00 + k01 + k02 + k10 + k11 + k12 +
                //                 k20 + k21 + k22) * input->qinfo->zero_point;
                // _acc0 = vsub_vx_i32m2(_acc0, q1z2, vl);
                // _acc1 = vsub_vx_i32m2(_acc1, q1z2, vl);

                // vint16m1_t _mul0_0 = vwmul_vx_i16m1(_r0_0_7, k00, vl);
                // vint16m1_t _mul0_1 = vwmul_vx_i16m1(_r0_1_8, k01, vl);
                // vint16m1_t _mul0_2 = vwmul_vx_i16m1(_r0_2_9, k02, vl);
                // vint16m1_t _mul1_0 = vwmul_vx_i16m1(_r1_0_7, k00, vl);
                // vint16m1_t _mul1_1 = vwmul_vx_i16m1(_r1_1_8, k01, vl);
                // vint16m1_t _mul1_2 = vwmul_vx_i16m1(_r1_2_9, k02, vl);

                // vint16m1_t _mul0_3 = vwmul_vx_i16m1(_r1_0_7, k10, vl);
                // vint16m1_t _mul0_4 = vwmul_vx_i16m1(_r1_1_8, k11, vl);
                // vint16m1_t _mul0_5 = vwmul_vx_i16m1(_r1_2_9, k12, vl);
                // vint16m1_t _mul1_3 = vwmul_vx_i16m1(_r2_0_7, k10, vl);
                // vint16m1_t _mul1_4 = vwmul_vx_i16m1(_r2_1_8, k11, vl);
                // vint16m1_t _mul1_5 = vwmul_vx_i16m1(_r2_2_9, k12, vl);

                // vint16m1_t _mul0_6 = vwmul_vx_i16m1(_r2_0_7, k20, vl);
                // vint16m1_t _mul0_7 = vwmul_vx_i16m1(_r2_1_8, k21, vl);
                // vint16m1_t _mul0_8 = vwmul_vx_i16m1(_r2_2_9, k22, vl);
                // vint16m1_t _mul1_6 = vwmul_vx_i16m1(_r3_0_7, k20, vl);
                // vint16m1_t _mul1_7 = vwmul_vx_i16m1(_r3_1_8, k21, vl);
                // vint16m1_t _mul1_8 = vwmul_vx_i16m1(_r3_2_9, k22, vl);

                // _acc0 = vwadd_wv_i32m2(_acc0, _mul0_0, vl);
                // _acc0 = vwadd_wv_i32m2(_acc0, _mul0_1, vl);
                // _acc0 = vwadd_wv_i32m2(_acc0, _mul0_2, vl);
                // _acc1 = vwadd_wv_i32m2(_acc1, _mul1_0, vl);
                // _acc1 = vwadd_wv_i32m2(_acc1, _mul1_1, vl);
                // _acc1 = vwadd_wv_i32m2(_acc1, _mul1_2, vl);

                // _acc0 = vwadd_wv_i32m2(_acc0, _mul0_3, vl);
                // _acc0 = vwadd_wv_i32m2(_acc0, _mul0_4, vl);
                // _acc0 = vwadd_wv_i32m2(_acc0, _mul0_5, vl);
                // _acc1 = vwadd_wv_i32m2(_acc1, _mul1_3, vl);
                // _acc1 = vwadd_wv_i32m2(_acc1, _mul1_4, vl);
                // _acc1 = vwadd_wv_i32m2(_acc1, _mul1_5, vl);

                // _acc0 = vwadd_wv_i32m2(_acc0, _mul0_6, vl);
                // _acc0 = vwadd_wv_i32m2(_acc0, _mul0_7, vl);
                // _acc0 = vwadd_wv_i32m2(_acc0, _mul0_8, vl);
                // _acc1 = vwadd_wv_i32m2(_acc1, _mul1_6, vl);
                // _acc1 = vwadd_wv_i32m2(_acc1, _mul1_7, vl);
                // _acc1 = vwadd_wv_i32m2(_acc1, _mul1_8, vl);
                vint8m1_t _res0, _res1;
                if (kernel->quant_channel > 1) {
                    _res0 = requantize_m4(_acc0, kernel->qinfo[c].multiplier,
                                          kernel->qinfo[c].shift, output->qinfo->zero_point, vl);
                    _res1 = requantize_m4(_acc1, kernel->qinfo[c].multiplier,
                                          kernel->qinfo[c].shift, output->qinfo->zero_point, vl);
                } else if (kernel->quant_channel == 1) {
                    _res0 = requantize_m4(_acc0, kernel->qinfo[0].multiplier,
                                          kernel->qinfo[0].shift, output->qinfo->zero_point, vl);
                    _res1 = requantize_m4(_acc1, kernel->qinfo[0].multiplier,
                                          kernel->qinfo[0].shift, output->qinfo->zero_point, vl);
                }
                vse8_v_i8m1(outptr0, _res0, vl);
                vse8_v_i8m1(outptr1, _res1, vl);

                r0 += vl;
                r1 += vl;
                r2 += vl;
                r3 += vl;
                outptr0 += vl;
                outptr1 += vl;
                w -= vl;
            }
            r0 += 2 + in_w;
            r1 += 2 + in_w;
            r2 += 2 + in_w;
            r3 += 2 + in_w;
            outptr0 += out_w;
            outptr1 += out_w;
        }
        for (; h < out_h; h++) {
            int w = out_w;
            // h2w8 loop
            while (w > 0) {
                vl = vsetvl_e32m4(w);
                vint32m4_t _acc0 = vmv_v_x_i32m4(bias0, vl);

                vint8m1_t _r0_0_7 = vle8_v_i8m1(r0, vl);
                vint8m1_t _r0_1_8 = vle8_v_i8m1(r0 + 1, vl);
                vint8m1_t _r0_2_9 = vle8_v_i8m1(r0 + 2, vl);

                vint8m1_t _r1_0_7 = vle8_v_i8m1(r1, vl);
                vint8m1_t _r1_1_8 = vle8_v_i8m1(r1 + 1, vl);
                vint8m1_t _r1_2_9 = vle8_v_i8m1(r1 + 2, vl);

                vint8m1_t _r2_0_7 = vle8_v_i8m1(r2, vl);
                vint8m1_t _r2_1_8 = vle8_v_i8m1(r2 + 1, vl);
                vint8m1_t _r2_2_9 = vle8_v_i8m1(r2 + 2, vl);

                vint16m2_t _r0_0_7_w = vwadd_vx_i16m2(_r0_0_7, 0, vl);  // widden 8->16
                vint16m2_t _r0_1_8_w = vwadd_vx_i16m2(_r0_1_8, 0, vl);
                vint16m2_t _r0_2_9_w = vwadd_vx_i16m2(_r0_2_9, 0, vl);

                vint16m2_t _r1_0_7_w = vwadd_vx_i16m2(_r1_0_7, 0, vl);
                vint16m2_t _r1_1_8_w = vwadd_vx_i16m2(_r1_1_8, 0, vl);
                vint16m2_t _r1_2_9_w = vwadd_vx_i16m2(_r1_2_9, 0, vl);

                vint16m2_t _r2_0_7_w = vwadd_vx_i16m2(_r2_0_7, 0, vl);
                vint16m2_t _r2_1_8_w = vwadd_vx_i16m2(_r2_1_8, 0, vl);
                vint16m2_t _r2_2_9_w = vwadd_vx_i16m2(_r2_2_9, 0, vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, k00, _r0_0_7_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k01, _r0_1_8_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k02, _r0_2_9_w, vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, k10, _r1_0_7_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k11, _r1_1_8_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k12, _r1_2_9_w, vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, k20, _r2_0_7_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k21, _r2_1_8_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k22, _r2_2_9_w, vl);

                vint8m1_t _res0;
                if (kernel->quant_channel > 1) {
                    _res0 = requantize_m4(_acc0, kernel->qinfo[c].multiplier,
                                          kernel->qinfo[c].shift, output->qinfo->zero_point, vl);
                } else if (kernel->quant_channel == 1) {
                    _res0 = requantize_m4(_acc0, kernel->qinfo[0].multiplier,
                                          kernel->qinfo[0].shift, output->qinfo->zero_point, vl);
                }
                vse8_v_i8m1(outptr0, _res0, vl);

                r0 += vl;
                r1 += vl;
                r2 += vl;
                outptr0 += vl;
                w -= vl;
            }
        }
        output_data += out_h * out_w;
    }
    shl_mem_free(input_padd_buf);
    return CSINN_TRUE;
}

int shl_rvv_dwconv3x3s2_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params)
{
    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(input);
    }
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)kernel->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];  // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int8_t *input_padd_buf = (int8_t *)shl_mem_alloc((in_h + params->pad_top + params->pad_down) *
                                                     (in_w + params->pad_left + params->pad_right) *
                                                     in_c * sizeof(int8_t));

    shl_rvv_pad_input_int8(input_data, input_padd_buf, in_c, in_h, in_w,
                           in_h + params->pad_top + params->pad_down,
                           in_w + params->pad_left + params->pad_right, params->pad_top,
                           params->pad_left, input->qinfo->zero_point);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

    int tailstep = in_w - 2 * out_w + in_w;

#pragma omp parallel for num_threads(1)
    for (int c = 0; c < in_c; c++) {
        int8_t *outptr0 = output_data;

        int32_t bias0 = bias_data[c];

        int8_t *img0 = input_padd_buf + c * in_h * in_w;
        int8_t *r0 = img0;
        int8_t *r1 = r0 + in_w;
        int8_t *r2 = r1 + in_w;

        const int8_t *kernel0 = kernel_data + c * 9;

        int8_t k00 = kernel0[0];
        int8_t k01 = kernel0[1];
        int8_t k02 = kernel0[2];
        int8_t k10 = kernel0[3];
        int8_t k11 = kernel0[4];
        int8_t k12 = kernel0[5];
        int8_t k20 = kernel0[6];
        int8_t k21 = kernel0[7];
        int8_t k22 = kernel0[8];
        int vl;

        for (int h = 0; h < out_h; h++) {
            int w = out_w;
            while (w > 0) {
                vl = vsetvl_e32m4(w);
                vint32m4_t _acc0 = vmv_v_x_i32m4(bias0, vl);

                // vint8mf2_t _r0_0_7, _r0_1_8;
                // vint8mf2_t _r1_0_7, _r1_1_8;
                // vint8mf2_t _r2_0_7, _r2_1_8;
                // vlseg2e8_v_i8mf2(&_r0_0_7, &_r0_1_8, r0, vl);
                // r0 += 2;
                // vint8mf2_t _r0_2_9 = vlse8_v_i8mf2(r0, 2 * sizeof(int8_t), vl);
                // r0 += (vl - 1) * 2;

                // vlseg2e8_v_i8mf2(&_r1_0_7, &_r1_1_8, r1, vl);
                // r1 += 2;
                // vint8mf2_t _r1_2_9 = vlse8_v_i8mf2(r1, 2 * sizeof(int8_t), vl);
                // r1 += (vl - 1) * 2;

                // vlseg2e8_v_i8mf2(&_r2_0_7, &_r2_1_8, r2, vl);
                // r2 += 2;
                // vint8mf2_t _r2_2_9 = vlse8_v_i8mf2(r2, 2 * sizeof(int8_t), vl);
                // r2 += (vl - 1) * 2;

                vint8m1_t _r0_0_7 = vlse8_v_i8m1(r0, 2 * sizeof(int8_t), vl);
                r0 += 1;
                vint8m1_t _r0_1_8 = vlse8_v_i8m1(r0, 2 * sizeof(int8_t), vl);
                r0 += 1;
                vint8m1_t _r0_2_9 = vlse8_v_i8m1(r0, 2 * sizeof(int8_t), vl);
                r0 += (vl - 1) * 2;

                vint8m1_t _r1_0_7 = vlse8_v_i8m1(r1, 2 * sizeof(int8_t), vl);
                r1 += 1;
                vint8m1_t _r1_1_8 = vlse8_v_i8m1(r1, 2 * sizeof(int8_t), vl);
                r1 += 1;
                vint8m1_t _r1_2_9 = vlse8_v_i8m1(r1, 2 * sizeof(int8_t), vl);
                r1 += (vl - 1) * 2;

                vint8m1_t _r2_0_7 = vlse8_v_i8m1(r2, 2 * sizeof(int8_t), vl);
                r2 += 1;
                vint8m1_t _r2_1_8 = vlse8_v_i8m1(r2, 2 * sizeof(int8_t), vl);
                r2 += 1;
                vint8m1_t _r2_2_9 = vlse8_v_i8m1(r2, 2 * sizeof(int8_t), vl);
                r2 += (vl - 1) * 2;

                vint16m2_t _r0_0_7_w = vwadd_vx_i16m2(_r0_0_7, 0, vl);  // widden 8->16
                vint16m2_t _r0_1_8_w = vwadd_vx_i16m2(_r0_1_8, 0, vl);
                vint16m2_t _r0_2_9_w = vwadd_vx_i16m2(_r0_2_9, 0, vl);

                vint16m2_t _r1_0_7_w = vwadd_vx_i16m2(_r1_0_7, 0, vl);
                vint16m2_t _r1_1_8_w = vwadd_vx_i16m2(_r1_1_8, 0, vl);
                vint16m2_t _r1_2_9_w = vwadd_vx_i16m2(_r1_2_9, 0, vl);

                vint16m2_t _r2_0_7_w = vwadd_vx_i16m2(_r2_0_7, 0, vl);
                vint16m2_t _r2_1_8_w = vwadd_vx_i16m2(_r2_1_8, 0, vl);
                vint16m2_t _r2_2_9_w = vwadd_vx_i16m2(_r2_2_9, 0, vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, k00, _r0_0_7_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k01, _r0_1_8_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k02, _r0_2_9_w, vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, k10, _r1_0_7_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k11, _r1_1_8_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k12, _r1_2_9_w, vl);

                _acc0 = vwmacc_vx_i32m4(_acc0, k20, _r2_0_7_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k21, _r2_1_8_w, vl);
                _acc0 = vwmacc_vx_i32m4(_acc0, k22, _r2_2_9_w, vl);

                vint8m1_t _res0;
                if (kernel->quant_channel > 1) {
                    _res0 = requantize_m4(_acc0, kernel->qinfo[c].multiplier,
                                          kernel->qinfo[c].shift, output->qinfo->zero_point, vl);
                } else if (kernel->quant_channel == 1) {
                    _res0 = requantize_m4(_acc0, kernel->qinfo[0].multiplier,
                                          kernel->qinfo[0].shift, output->qinfo->zero_point, vl);
                }
                vse8_v_i8m1(outptr0, _res0, vl);
                outptr0 += vl;
                w -= vl;
            }
            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
        output_data += out_h * out_w;
    }
    shl_mem_free(input_padd_buf);
    return CSINN_TRUE;
}
