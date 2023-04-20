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
int shl_rvv_dwconv_nhwc_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

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

    __fp16 *input_padd_buf =
        (__fp16 *)shl_mem_alloc(padded_in_h * padded_in_w * in_c * sizeof(__fp16));

    for (int b = 0; b < batch; b++) {
        shl_rvv_pad_input_nhwc_fp16(input_data, input_padd_buf, in_h, in_w, in_c, padded_in_h,
                                    padded_in_w, pad_top, pad_left);
        const __fp16 *src = (__fp16 *)input_padd_buf;
        __fp16 *dst = output_data + b * out_h * out_w * out_c;
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int vl;
                int c = 0;
                while (c < in_c) {
                    vl = vsetvl_e16m1(in_c - c);
                    vfloat16m1_t _acc = vle16_v_f16m1(bias_data + c, vl);
                    int in_x_origin = ow * stride_w;
                    int in_y_origin = oh * stride_h;
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int w = in_x_origin + dilation_w * kw;
                            int h = in_y_origin + dilation_h * kh;
                            const __fp16 *in_ptr = src + (h * padded_in_w + w) * in_c + c;
                            const __fp16 *k_ptr = kernel_data + (kh * kernel_w + kw) * in_c + c;
                            vfloat16m1_t _rxx = vle16_v_f16m1(in_ptr, vl);
                            vfloat16m1_t _kxx = vle16_v_f16m1(k_ptr, vl);
                            _acc = vfmacc_vv_f16m1(_acc, _rxx, _kxx, vl);
                        }
                    }
                    __fp16 *out_ptr = dst + (oh * out_w + ow) * out_c + c;
                    vse16_v_f16m1(out_ptr, _acc, vl);
                    c += vl;
                }
            }
        }
        input_data += in_h * in_w * in_c;
    }
    shl_mem_free(input_padd_buf);
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}