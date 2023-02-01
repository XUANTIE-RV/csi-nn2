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

/*************************************************************
 * note: VLEN = 128/256 ... flexible vlen
 *************************************************************/
int shl_rvv_dwconv_nhwc_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

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

    float *input_padd_buf =
        (float *)shl_mem_alloc(padded_in_h * padded_in_w * in_c * sizeof(float));

    for (int b = 0; b < batch; b++) {
        shl_rvv_pad_input_nhwc_fp32(input_data, input_padd_buf, in_h, in_w, in_c, padded_in_h,
                                    padded_in_w, pad_top, pad_left);
        const float *src = (float *)input_padd_buf;
        float *dst = output_data + b * out_h * out_w * out_c;
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int vl;
                int c = 0;
                while (c < in_c) {
                    vl = vsetvl_e32m1(in_c - c);
                    vfloat32m1_t _acc = vle32_v_f32m1(bias_data + c, vl);
                    int in_x_origin = (ow * stride_w) - pad_left;
                    int in_y_origin = (oh * stride_h) - pad_top;
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int w = in_x_origin + dilation_w * kw;
                            int h = in_y_origin + dilation_h * kh;
                            const float *in_ptr = src + (h * padded_in_w + w) * in_c + c;
                            const float *k_ptr = kernel_data + (kh * kernel_w + kw) * in_c + c;
                            vfloat32m1_t _rxx = vle32_v_f32m1(in_ptr, vl);
                            vfloat32m1_t _kxx = vle32_v_f32m1(k_ptr, vl);
                            _acc = vfmacc_vv_f32m1(_acc, _rxx, _kxx, vl);
                        }
                    }
                    float *out_ptr = dst + (oh * out_w + ow) * out_c + c;
                    vse32_v_f32m1(out_ptr, _acc, vl);
                    c += vl;
                }
            }
        }
        input_data += in_h * in_w * in_c;
    }
    shl_mem_free(input_padd_buf);
    return CSINN_TRUE;
}