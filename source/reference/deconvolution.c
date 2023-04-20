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

#include "shl_ref.h"

static int shl_ref_deconv2d_nhwc_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                     struct csinn_conv2d_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    float *filter_data = kernel->data;
    float *bias_data = bias->data;
    const int batches = input->dim[0];
    const int input_depth = input->dim[3];
    const int output_depth = output->dim[3];
    const int input_height = input->dim[1];
    const int input_width = input->dim[2];
    const int filter_height = kernel->dim[1];
    const int filter_width = kernel->dim[2];
    const int output_height = output->dim[1];
    const int output_width = output->dim[2];
    const int output_batch = output->dim[0];

    int num_elements = csinn_tensor_size(output);
    memset(output_data, 0, num_elements * sizeof(float));

    // Loop through input elements one at a time.
    for (int batch = 0; batch < batches; ++batch) {
        for (int in_y = 0; in_y < input_height; ++in_y) {
            for (int in_x = 0; in_x < input_width; ++in_x) {
                for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                    // Loop through the output elements it will influence.
                    const int out_x_origin = (in_x * params->stride_width) - params->pad_left;
                    const int out_y_origin = (in_y * params->stride_height) - params->pad_top;
                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                                // Compute output element location.
                                const int out_x = out_x_origin + filter_x;
                                const int out_y = out_y_origin + filter_y;
                                // We cannot accumulate out of bounds.
                                if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) &&
                                    (out_y < output_height)) {
                                    float input_value = input_data[shl_ref_get_index(
                                        input->dim, batch, in_y, in_x, in_channel)];
                                    float filter_value = filter_data[shl_ref_get_index(
                                        kernel->dim, out_channel, filter_y, filter_x, in_channel)];
                                    output_data[shl_ref_get_index(output->dim, batch, out_y, out_x,
                                                                  out_channel)] +=
                                        input_value * filter_value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (bias->dim_count != 0) {
        for (int batch = 0; batch < output_batch; batch++) {
            for (int o_y = 0; o_y < output_height; o_y++) {
                for (int o_x = 0; o_x < output_width; o_x++) {
                    for (int o_channel = 0; o_channel < output_depth; ++o_channel) {
                        output_data[shl_ref_get_index(output->dim, batch, o_y, o_x, o_channel)] +=
                            bias_data[o_channel];
                    }
                }
            }
        }
    }

    return CSINN_TRUE;
}

static int shl_ref_deconv2d_nchw_f32(struct csinn_tensor *o_input, struct csinn_tensor *o_output,
                                     struct csinn_tensor *o_kernel, struct csinn_tensor *o_bias,
                                     struct csinn_conv2d_params *params)
{
    struct csinn_tensor *input = shl_ref_nchw_to_nhwc_f32(o_input);
    struct csinn_tensor *output = shl_ref_nchw_to_nhwc_f32(o_output);
    int32_t permute[4] = {1, 2, 3, 0};
    struct csinn_tensor *kernel = shl_ref_deconv_kernel_nchw_to_nhwc_f32(o_kernel, permute);
    struct csinn_tensor *bias = o_bias;

    shl_ref_deconv2d_nhwc_f32(input, output, kernel, bias, params);

    shl_ref_nhwc_to_nchw_f32(o_output, output);
    shl_ref_free_float_tensor(input);
    return CSINN_TRUE;
}

int shl_ref_depthwise_deconv2d_nhwc_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    float *filter_data = kernel->data;
    float *bias_data = bias->data;
    const int batches = input->dim[0];
    const int input_depth = input->dim[3];
    const int output_depth = output->dim[3];
    const int input_height = input->dim[1];
    const int input_width = input->dim[2];
    const int filter_height = kernel->dim[1];
    const int filter_width = kernel->dim[2];
    const int output_height = output->dim[1];
    const int output_width = output->dim[2];
    const int output_batch = output->dim[0];

    int num_elements = csinn_tensor_size(output);
    memset(output_data, 0, num_elements * sizeof(float));

    // Loop through input elements one at a time.
    for (int batch = 0; batch < batches; ++batch) {
        for (int in_y = 0; in_y < input_height; ++in_y) {
            for (int in_x = 0; in_x < input_width; ++in_x) {
                for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                    // Loop through the output elements it will influence.
                    const int out_x_origin = (in_x * params->stride_width) - params->pad_left;
                    const int out_y_origin = (in_y * params->stride_height) - params->pad_top;
                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            // Compute output element location.
                            const int out_x = out_x_origin + filter_x;
                            const int out_y = out_y_origin + filter_y;
                            // We cannot accumulate out of bounds.
                            if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) &&
                                (out_y < output_height)) {
                                float input_value = input_data[shl_ref_get_index(
                                    input->dim, batch, in_y, in_x, in_channel)];
                                float filter_value = filter_data[shl_ref_get_index(
                                    kernel->dim, 0, filter_y, filter_x, in_channel)];
                                output_data[shl_ref_get_index(output->dim, batch, out_y, out_x,
                                                              in_channel)] +=
                                    input_value * filter_value;
                            }
                        }
                    }
                }
            }
        }
    }
    if (bias->dim_count != 0) {
        for (int batch = 0; batch < output_batch; batch++) {
            for (int o_y = 0; o_y < output_height; o_y++) {
                for (int o_x = 0; o_x < output_width; o_x++) {
                    for (int o_channel = 0; o_channel < output_depth; ++o_channel) {
                        output_data[shl_ref_get_index(output->dim, batch, o_y, o_x, o_channel)] +=
                            bias_data[o_channel];
                    }
                }
            }
        }
    }

    return CSINN_TRUE;
}

int shl_ref_depthwise_deconv2d_nchw_f32(struct csinn_tensor *o_input, struct csinn_tensor *o_output,
                                        struct csinn_tensor *o_kernel, struct csinn_tensor *o_bias,
                                        struct csinn_conv2d_params *params)
{
    struct csinn_tensor *input = shl_ref_nchw_to_nhwc_f32(o_input);
    struct csinn_tensor *output = shl_ref_nchw_to_nhwc_f32(o_output);
    int32_t permute[4] = {1, 2, 3, 0};
    struct csinn_tensor *kernel = shl_ref_deconv_kernel_nchw_to_nhwc_f32(o_kernel, permute);
    struct csinn_tensor *bias = o_bias;
    shl_ref_depthwise_deconv2d_nhwc_f32(input, output, kernel, bias, params);

    shl_ref_nhwc_to_nchw_f32(o_output, output);
    shl_ref_free_float_tensor(input);
    return CSINN_TRUE;
}

int shl_ref_depthwise_deconv2d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params)
{
    if (params->base.layout == CSINN_LAYOUT_NCHW) {
        shl_ref_depthwise_deconv2d_nchw_f32(input, output, kernel, bias, params);
    } else if (params->base.layout == CSINN_LAYOUT_NHWC) {
        shl_ref_depthwise_deconv2d_nhwc_f32(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    return CSINN_TRUE;
}

int shl_ref_depthwise_deconv2d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                     struct csinn_conv2d_params *params)
{
    return shl_ref_conv_callback_base(input, output, kernel, bias, params,
                                      shl_ref_depthwise_deconv2d_f32);
}

int shl_ref_deconv2d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv2d_params *params)
{
    if (params->base.layout == CSINN_LAYOUT_NCHW) {
        shl_ref_deconv2d_nchw_f32(input, output, kernel, bias, params);
    } else if (params->base.layout == CSINN_LAYOUT_NHWC) {
        shl_ref_deconv2d_nhwc_f32(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    return CSINN_TRUE;
}

int shl_ref_deconv2d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                           struct csinn_conv2d_params *params)
{
    return shl_ref_conv_callback_base(input, output, kernel, bias, params, shl_ref_deconv2d_f32);
}
