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

int shl_rvv_dwconv1d_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv1d_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)kernel->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int batch = input->dim[0];
    int in_c = input->dim[1];  // group = in_channel
    int in_w = input->dim[2];

    int out_c = output->dim[1];
    int out_w = output->dim[2];

    int kernel_w = kernel->dim[2];
    int stride_w = params->stride_width;
    int dilation_w = params->dilation_width;

    int pad_left = params->pad_left;
    int pad_right = params->pad_right;
    int padded_in_w = in_w + pad_left + pad_right;

    const int32_t depth_multiplier = out_c / in_c;
    assert(in_c * depth_multiplier ==
           out_c);  // The input and output channels are equal for dw convolution

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

    for (int b = 0; b < batch; b++) {
        const int8_t *src = input_data + b * in_c * in_w;
        int8_t *dst = output_data + b * out_c * out_w;
        for (int ow = 0; ow < out_w; ow++) {
            for (int m = 0; m < depth_multiplier; m++) {
                int c = 0;
                while (c < in_c) {
                    int oc = c * depth_multiplier + m;
                    int vl = vsetvl_e8m1(in_c - c);
                    vint32m4_t _acc =
                        vlse32_v_i32m4(bias_data + oc, depth_multiplier * sizeof(int32_t), vl);
                    vint32m4_t _mult =
                        vlse32_v_i32m4(multiplier + oc, depth_multiplier * sizeof(int32_t), vl);
                    vint32m4_t _shift =
                        vlse32_v_i32m4(shift + oc, depth_multiplier * sizeof(int32_t), vl);
                    _shift = vrsub_vx_i32m4(_shift, -1, vl);
                    int in_x_origin = ow * stride_w;
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int w = in_x_origin + dilation_w * kw;
                        if ((w >= 0) && (w < in_w)) {
                            const int8_t *in_ptr = src + c * in_w + w;
                            int8_t *k_ptr = kernel_data + oc * kernel_w + kw;
                            vint16m2_t _rxx = vwadd_vx_i16m2(vlse8_v_i8m1(in_ptr, in_w, vl), 0, vl);
                            vint16m2_t _kxx = vwadd_vx_i16m2(
                                vlse8_v_i8m1(k_ptr, kernel_w * depth_multiplier, vl), 0, vl);
                            _acc = vwmacc_vv_i32m4(_acc, _rxx, _kxx, vl);
                        }
                    }
                    vint32m4_t _mulh = vmulh_vv_i32m4(_acc, _mult, vl);
                    _mulh = vssra_vv_i32m4(_mulh, vreinterpret_v_i32m4_u32m4(_shift), vl);
                    _mulh = vadd_vx_i32m4(_mulh, out_zp, vl);
                    vint16m2_t _res0 = vnclip_wx_i16m2(_mulh, 0, vl);
                    vint8m1_t _res1 = vnclip_wx_i8m1(_res0, 0, vl);
                    int8_t *out_ptr = dst + oc * out_w + ow;
                    vsse8_v_i8m1(out_ptr, out_w * depth_multiplier, _res1, vl);
                    c += vl;
                }
            }
        }
    }

    return CSINN_TRUE;
}
