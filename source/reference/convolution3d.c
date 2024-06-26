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

#include "reference/ref.h"

int shl_ref_conv3d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                       struct csinn_conv3d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    const int32_t batch = input->dim[0];
    const int32_t in_channel = input->dim[1];
    const int32_t in_depth = input->dim[2];
    const int32_t in_height = input->dim[3];
    const int32_t in_width = input->dim[4];

    // const int filter_outchannel = kernel->dim[0];
    // const int filter_inchannel = kernel->dim[1];
    const int32_t filter_depth = kernel->dim[2];
    const int32_t filter_height = kernel->dim[3];
    const int32_t filter_width = kernel->dim[4];

    // int output_batch = output->dim[0];
    const int32_t output_channel = output->dim[1];
    const int32_t output_depth = output->dim[2];
    const int32_t output_height = output->dim[3];
    const int32_t output_width = output->dim[4];

    const int32_t dilation_depth = params->dilation_depth;
    const int32_t dilation_height = params->dilation_height;
    const int32_t dilation_width = params->dilation_width;

    for (int32_t out_b = 0; out_b < batch; ++out_b) {
        for (int32_t out_ch = 0; out_ch < output_channel; ++out_ch) {
            for (int32_t out_d = 0; out_d < output_depth; ++out_d) {
                for (int32_t out_h = 0; out_h < output_height; ++out_h) {
                    for (int32_t out_w = 0; out_w < output_width; ++out_w) {
                        const int32_t in_d_origin =
                            (out_d * params->stride_depth) - params->pad_front;
                        const int32_t in_h_origin =
                            (out_h * params->stride_height) - params->pad_top;
                        const int32_t in_w_origin =
                            (out_w * params->stride_width) - params->pad_left;

                        float acc = 0.0f;
                        for (int32_t in_ch = 0; in_ch < in_channel; ++in_ch) {
                            for (int32_t filter_d = 0; filter_d < filter_depth; ++filter_d) {
                                for (int32_t filter_h = 0; filter_h < filter_height; ++filter_h) {
                                    for (int32_t filter_w = 0; filter_w < filter_width;
                                         ++filter_w) {
                                        int32_t in_d = in_d_origin + dilation_depth * filter_d;
                                        int32_t in_h = in_h_origin + dilation_height * filter_h;
                                        int32_t in_w = in_w_origin + dilation_width * filter_w;
                                        // If the location is outside the bounds of the input image,
                                        // use zero as a default value.
                                        if ((in_d >= 0) && (in_d < in_depth) && (in_h >= 0) &&
                                            (in_h < in_height) && (in_w >= 0) &&
                                            (in_w < in_width)) {
                                            int32_t input_idx = shl_ref_get_index_5(
                                                input->dim, out_b, in_ch, in_d, in_h, in_w);
                                            float input_val = input_data[input_idx];
                                            int32_t filter_idx =
                                                shl_ref_get_index_5(kernel->dim, out_ch, in_ch,
                                                                    filter_d, filter_h, filter_w);
                                            float filter_val = kernel_data[filter_idx];
                                            acc += input_val * filter_val;
                                        }
                                    }
                                }
                            }
                        }
                        float bias_val = 0.0f;
                        if (bias_data != NULL && bias->dim_count != 0) {
                            bias_val = bias_data[out_ch];
                        }
                        int32_t output_idx =
                            shl_ref_get_index_5(output->dim, out_b, out_ch, out_d, out_h, out_w);
                        output_data[output_idx] = acc + bias_val;
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

int shl_ref_conv3d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv3d_params *params)
{
    return shl_ref_conv_callback_base(input, output, kernel, bias, params, shl_ref_conv3d_f32);
}
