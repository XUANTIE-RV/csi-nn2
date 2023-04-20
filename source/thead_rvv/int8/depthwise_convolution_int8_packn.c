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

#include "shl_thead_rvv.h"

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

int shl_rvv_dwconv_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)kernel->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int batch = input->dim[0];
    int in_c = input->dim[1];  // group = in_channel
    int in_h = input->dim[2];
    int in_w = input->dim[3];

    int out_c = output->dim[1];
    int out_h = output->dim[2];
    int out_w = output->dim[3];

    int kernel_h = kernel->dim[2];
    int kernel_w = kernel->dim[3];
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;

    int padded_in_h = in_h + params->pad_top + params->pad_down;
    int padded_in_w = in_w + params->pad_left + params->pad_right;

    int32_t *multiplier = (int32_t *)shl_mem_alloc(out_c * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(out_c * sizeof(int32_t));

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

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    const int vl = vsetvl_e8m1(packn);

    int8_t *input_padd_buf =
        (int8_t *)shl_mem_alloc(in_c * padded_in_h * padded_in_w * sizeof(int8_t));

#pragma omp parallel for num_threads(1)
    for (int b = 0; b < batch; b++) {
        shl_rvv_pad_input_packn_int8(input_data, input_padd_buf, in_c, in_h, in_w, padded_in_h,
                                     padded_in_w, params->pad_top, params->pad_left,
                                     input->qinfo->zero_point);

        for (int c = 0; c + packn <= in_c; c += packn) {
            const int8_t *src = input_padd_buf + c * padded_in_h * padded_in_w;
            int8_t *dst = output_data + c * out_h * out_w;
            const int8_t *kernel0 = kernel_data + c * kernel_h * kernel_w;

            vint32m4_t _bias0 = vle32_v_i32m4(bias_data + c, vl);  // bias_data != NULL
            vint32m4_t _mult = vle32_v_i32m4(multiplier + c, vl);
            vint32m4_t _shift = vle32_v_i32m4(shift + c, vl);
            _shift = vrsub_vx_i32m4(_shift, -1, vl);
            int32_t out_zp = output->qinfo->zero_point;

            for (int oh = 0; oh < out_h; oh++) {
                int i_h_start = oh * stride_h;
                int i_h_end = i_h_start + kernel_h;
                for (int ow = 0; ow < out_w; ow++) {
                    int i_w_start = ow * stride_w;
                    int i_w_end = i_w_start + kernel_w;

                    vint32m4_t _acc = _bias0;
                    for (int ih = i_h_start, i_kh = 0; ih < i_h_end; ih++, i_kh++) {
                        for (int iw = i_w_start, i_kw = 0; iw < i_w_end; iw++, i_kw++) {
                            const int8_t *in_ptr = src + (ih * padded_in_w + iw) * packn;
                            const int8_t *k_ptr = kernel0 + (i_kh * kernel_w + i_kw) * packn;
                            vint16m2_t _rxx = vwadd_vx_i16m2(vle8_v_i8m1(in_ptr, vl), 0, vl);
                            vint16m2_t _kxx = vwadd_vx_i16m2(vle8_v_i8m1(k_ptr, vl), 0, vl);
                            _acc = vwmacc_vv_i32m4(_acc, _rxx, _kxx, vl);
                        }
                    }
                    vint8m1_t _res = requantize_m4_s(_acc, _mult, _shift, out_zp, vl);
                    int8_t *out_ptr = dst + (oh * out_w + ow) * packn;
                    vse8_v_i8m1(out_ptr, _res, vl);
                }
            }
        }
        input_data += in_c * in_h * in_w;
        output_data += out_c * out_h * out_w;
    }
    return CSINN_TRUE;
}
