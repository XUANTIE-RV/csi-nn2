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
 * note: VLEN = 128/256 ... flexible vlen
 *************************************************************/
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

int shl_rvv_dwconv_nhwc_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)kernel->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int batch = input->dim[0];
    int in_h = input->dim[1];
    int in_w = input->dim[2];
    int in_c = input->dim[3];  // group = in_channel

    int out_h = output->dim[1];
    int out_w = output->dim[2];
    int out_c = output->dim[3];

    int kernel_h = kernel->dim[1];
    int kernel_w = kernel->dim[2];
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;

    int dilation_w = params->dilation_width;
    int dilation_h = params->dilation_height;

    int pad_top = params->pad_top;
    int pad_down = params->pad_down;
    int pad_left = params->pad_left;
    int pad_right = params->pad_right;

    int padded_in_h = in_h + pad_top + pad_down;
    int padded_in_w = in_w + pad_left + pad_right;

    int32_t *multiplier = (int32_t *)shl_mem_alloc(out_c * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(out_c * sizeof(int32_t));
    int32_t out_zp = output->qinfo->zero_point;

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

    int8_t *input_padd_buf =
        (int8_t *)shl_mem_alloc(padded_in_h * padded_in_w * in_c * sizeof(int8_t));

    for (int b = 0; b < batch; b++) {
        shl_rvv_pad_input_nhwc_int8(input_data, input_padd_buf, in_h, in_w, in_c, padded_in_h,
                                    padded_in_w, pad_top, pad_left, input->qinfo->zero_point);
        const int8_t *src = (int8_t *)input_padd_buf;
        int8_t *dst = output_data + b * out_h * out_w * out_c;
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int vl;
                int c = 0;
                while (c < in_c) {
                    vl = vsetvl_e8m1(in_c - c);
                    vint32m4_t _acc = vle32_v_i32m4(bias_data + c, vl);
                    vint32m4_t _mult = vle32_v_i32m4(multiplier + c, vl);
                    vint32m4_t _shift = vle32_v_i32m4(shift + c, vl);
                    _shift = vrsub_vx_i32m4(_shift, -1, vl);
                    int in_x_origin = ow * stride_w;
                    int in_y_origin = oh * stride_h;
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int w = in_x_origin + dilation_w * kw;
                            int h = in_y_origin + dilation_h * kh;
                            const int8_t *in_ptr = src + (h * padded_in_w + w) * in_c + c;
                            const int8_t *k_ptr = kernel_data + (kh * kernel_w + kw) * in_c + c;
                            vint16m2_t _rxx = vwadd_vx_i16m2(vle8_v_i8m1(in_ptr, vl), 0, vl);
                            vint16m2_t _kxx = vwadd_vx_i16m2(vle8_v_i8m1(k_ptr, vl), 0, vl);
                            _acc = vwmacc_vv_i32m4(_acc, _rxx, _kxx, vl);
                        }
                    }
                    vint8m1_t _res = requantize_m4_s(_acc, _mult, _shift, out_zp, vl);
                    int8_t *out_ptr = dst + (oh * out_w + ow) * out_c + c;
                    vse8_v_i8m1(out_ptr, _res, vl);
                    c += vl;
                }
            }
        }
        input_data += in_h * in_w * in_c;
    }
    shl_mem_free(input_padd_buf);
    shl_mem_free(multiplier);
    shl_mem_free(shift);
    return CSINN_TRUE;
}