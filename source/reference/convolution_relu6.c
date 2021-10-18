/*
 * Copyright (C) 2016-2020 C-SKY Limited. All rights reserved.
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

#include "csi_nn.h"
#include "csi_utils.h"

int csi_conv2d_relu6_u8(struct csi_tensor *o_input,
                        struct csi_tensor *o_output,
                        struct csi_tensor *o_kernel,
                        struct csi_tensor *o_bias,
                        struct conv2d_params *params)
{
#ifdef CSI_AVX_OPT
    float *float_input_data;
    float *float_kernel_data;
    float *float_bias_data;
    float *float_output_data;
    struct csi_tensor float_input;
    struct csi_tensor float_kernel;
    struct csi_tensor float_bias;
    struct csi_tensor float_output;
    uint8_t *input_data = o_input->data;
    uint8_t *kernel_data = o_kernel->data;
    int32_t *bias_data = o_bias->data;
    uint8_t *output_data = o_output->data;
    int input_size = 1;
    int kernel_size = 1;
    int output_size = 1;

    for (int i = 0; i < o_input->dim_count; i++) {
        input_size *= o_input->dim[i];
    }
    for (int i = 0; i < o_kernel->dim_count; i++) {
        kernel_size *= o_kernel->dim[i];
    }
    for (int i = 0; i < o_output->dim_count; i++) {
        output_size *= o_output->dim[i];
    }
    int bias_size = o_output->dim[1];

    memcpy(&float_input, o_input, sizeof(struct csi_tensor));
    memcpy(&float_kernel, o_kernel, sizeof(struct csi_tensor));
    memcpy(&float_bias, o_bias, sizeof(struct csi_tensor));
    memcpy(&float_output, o_output, sizeof(struct csi_tensor));
    float_input_data = malloc(input_size * sizeof(float));
    float_output_data = malloc(output_size * sizeof(float));
    float_kernel_data = malloc(kernel_size * sizeof(float));
    float_bias_data = malloc(bias_size * sizeof(float));
    float_input.dtype = CSINN_DTYPE_FLOAT32;
    float_input.data = float_input_data;
    float_kernel.data = float_kernel_data;
    float_bias.data = float_bias_data;
    float_output.data = float_output_data;

    for (int i = 0; i < input_size; i++) {
        float_input_data[i] = uint8_to_float(input_data[i], o_input);
    }
    for (int i = 0; i < kernel_size; i++) {
        float_kernel_data[i] = uint8_to_float(kernel_data[i], o_kernel);
    }
    for (int i = 0; i < bias_size; i++) {
        float_bias_data[i] = bias_data[i] * o_kernel->scale * o_input->scale;
    }

    csi_conv2d_init(&float_input, &float_output, &float_kernel, &float_bias, params);
    csi_conv2d(&float_input, &float_output, &float_kernel, &float_bias, params);

    for (int i = 0; i < output_size; i++) {
        if (float_output_data[i] < 0) {
            float_output_data[i] = 0;
        }else if (float_output_data[i] > 0) {
            float_output_data[i] = 6;
        }
        output_data[i] = float_to_uint8(float_output_data[i], o_output);
    }
    free(float_input_data);
    free(float_kernel_data);
    free(float_bias_data);
    free(float_output_data);
#else
    struct csi_tensor* input;
    struct csi_tensor* output;
    struct csi_tensor* kernel;
    struct csi_tensor* bias = o_bias;
    input =  csi_nchw_to_nhwc_8(o_input);
    kernel = csi_nchw_to_nhwc_8(o_kernel);
    output = csi_nchw_to_nhwc_8(o_output);

    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    uint8_t *kernel_data = kernel->data;
    int32_t *bias_data = bias->data;
    const int32_t dilation_width_factor = params->dilation_width;
    const int32_t dilation_height_factor = params->dilation_height;
    const int32_t input_offset = input->zero_point;
    const int32_t filter_offset = kernel->zero_point;
    const int32_t output_offset = output->zero_point;
    const int32_t output_multiplier = output->multiplier;
    const int32_t output_shift = output->shift;

    const int32_t batches = input->dim[0];
    const int32_t input_depth = input->dim[3];
    const int32_t output_depth = output->dim[3];
    const int32_t input_height = input->dim[1];
    const int32_t input_width = input->dim[2];
    const int32_t filter_height = kernel->dim[1];
    const int32_t filter_width = kernel->dim[2];
    const int32_t output_height = output->dim[1];
    const int32_t output_width = output->dim[2];

    for (int32_t batch = 0; batch < batches; ++batch) {
        #pragma omp parallel for num_threads(8)
        for (int32_t out_y = 0; out_y < output_height; ++out_y) {
            for (int32_t out_x = 0; out_x < output_width; ++out_x) {
                for (int32_t out_channel = 0; out_channel < output_depth; ++out_channel) {
                    const int32_t in_x_origin = (out_x * params->stride_width) - params->pad_left;
                    const int32_t in_y_origin = (out_y * params->stride_height) - params->pad_top;
                    int64_t acc = 0;
                    for (int32_t filter_y = 0; filter_y < filter_height; ++filter_y) {
                        for (int32_t filter_x = 0; filter_x < filter_width; ++filter_x) {
                            for (int32_t in_channel = 0; in_channel < input_depth; ++in_channel) {
                                const int32_t in_x = in_x_origin + dilation_width_factor * filter_x;
                                const int32_t in_y = in_y_origin + dilation_height_factor * filter_y;
                                // If the location is outside the bounds of the input image,
                                // use zero as a default value.
                                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                    (in_y < input_height)) {
                                    int32_t input_index =
                                        csi_get_index(input->dim, batch, in_y, in_x, in_channel);
                                    int32_t input_val = input_data[input_index];
                                    int32_t filter_index = csi_get_index(
                                        kernel->dim, out_channel, filter_y, filter_x, in_channel);
                                    int32_t filter_val = kernel_data[filter_index];
                                    acc +=
                                        (filter_val - filter_offset) * (input_val - input_offset);
                                }
                            }
                        }
                    }
                    if (bias->dim_count != 0) {
                        acc += bias_data[out_channel];
                    }
                    acc = conv_relu6_out_u8(acc, input, output, kernel);
                    output_data[csi_get_index(output->dim, batch, out_y, out_x, out_channel)] = acc;
                }
            }
        }
    }
    csi_nhwc_to_nchw_8(o_output, output);
    free(input->data);
    free(input);
    free(kernel->data);
    free(kernel);
#endif
    return CSINN_TRUE;
}


int csi_conv2d_relu6_i8(struct csi_tensor *o_input,
                        struct csi_tensor *o_output,
                        struct csi_tensor *o_kernel,
                        struct csi_tensor *o_bias,
                        struct conv2d_params *params)
{
#ifdef CSI_AVX_OPT
    float *float_input_data;
    float *float_kernel_data;
    float *float_bias_data;
    float *float_output_data;
    struct csi_tensor float_input;
    struct csi_tensor float_kernel;
    struct csi_tensor float_bias;
    struct csi_tensor float_output;
    uint8_t *input_data = o_input->data;
    uint8_t *kernel_data = o_kernel->data;
    int32_t *bias_data = o_bias->data;
    uint8_t *output_data = o_output->data;
    int input_size = 1;
    int kernel_size = 1;
    int output_size = 1;

    for (int i = 0; i < o_input->dim_count; i++) {
        input_size *= o_input->dim[i];
    }
    for (int i = 0; i < o_kernel->dim_count; i++) {
        kernel_size *= o_kernel->dim[i];
    }
    for (int i = 0; i < o_output->dim_count; i++) {
        output_size *= o_output->dim[i];
    }
    int bias_size = o_output->dim[1];

    memcpy(&float_input, o_input, sizeof(struct csi_tensor));
    memcpy(&float_kernel, o_kernel, sizeof(struct csi_tensor));
    memcpy(&float_bias, o_bias, sizeof(struct csi_tensor));
    memcpy(&float_output, o_output, sizeof(struct csi_tensor));
    float_input_data = malloc(input_size * sizeof(float));
    float_output_data = malloc(output_size * sizeof(float));
    float_kernel_data = malloc(kernel_size * sizeof(float));
    float_bias_data = malloc(bias_size * sizeof(float));
    float_input.dtype = CSINN_DTYPE_FLOAT32;
    float_input.data = float_input_data;
    float_kernel.data = float_kernel_data;
    float_bias.data = float_bias_data;
    float_output.data = float_output_data;

    for (int i = 0; i < input_size; i++) {
        float_input_data[i] = int8_to_float(input_data[i], o_input);
    }
    for (int i = 0; i < kernel_size; i++) {
        float_kernel_data[i] = int8_to_float(kernel_data[i], o_kernel);
    }
    for (int i = 0; i < bias_size; i++) {
        float_bias_data[i] = bias_data[i] * o_kernel->scale * o_input->scale;
    }

    csi_conv2d_init(&float_input, &float_output, &float_kernel, &float_bias, params);
    csi_conv2d(&float_input, &float_output, &float_kernel, &float_bias, params);

    for (int i = 0; i < output_size; i++) {
        if (float_output_data[i] < 0) {
            float_output_data[i] = 0;
        }else if (float_output_data[i] > 0) {
            float_output_data[i] = 6;
        }
        output_data[i] = float_to_int8(float_output_data[i], o_output);
    }
    free(float_input_data);
    free(float_kernel_data);
    free(float_bias_data);
    free(float_output_data);
#else
    struct csi_tensor* input;
    struct csi_tensor* output;
    struct csi_tensor* kernel;
    struct csi_tensor* bias = o_bias;
    input =  csi_nchw_to_nhwc_8(o_input);
    kernel = csi_nchw_to_nhwc_8(o_kernel);
    output = csi_nchw_to_nhwc_8(o_output);

    int8_t *input_data = input->data;
    int8_t *output_data = output->data;
    int8_t *kernel_data = kernel->data;
    int32_t *bias_data = bias->data;
    const int32_t dilation_width_factor = params->dilation_width;
    const int32_t dilation_height_factor = params->dilation_height;
    const int32_t input_offset = input->zero_point;
    const int32_t filter_offset = kernel->zero_point;
    const int32_t output_offset = output->zero_point;
    const int32_t output_multiplier = output->multiplier;
    const int32_t output_shift = output->shift;

    const int32_t batches = input->dim[0];
    const int32_t input_depth = input->dim[3];
    const int32_t output_depth = output->dim[3];
    const int32_t input_height = input->dim[1];
    const int32_t input_width = input->dim[2];
    const int32_t filter_height = kernel->dim[1];
    const int32_t filter_width = kernel->dim[2];
    const int32_t output_height = output->dim[1];
    const int32_t output_width = output->dim[2];

    for (int32_t batch = 0; batch < batches; ++batch) {
        #pragma omp parallel for num_threads(8)
        for (int32_t out_y = 0; out_y < output_height; ++out_y) {
            for (int32_t out_x = 0; out_x < output_width; ++out_x) {
                for (int32_t out_channel = 0; out_channel < output_depth; ++out_channel) {
                    const int32_t in_x_origin = (out_x * params->stride_width) - params->pad_left;
                    const int32_t in_y_origin = (out_y * params->stride_height) - params->pad_top;
                    int64_t acc = 0;
                    for (int32_t filter_y = 0; filter_y < filter_height; ++filter_y) {
                        for (int32_t filter_x = 0; filter_x < filter_width; ++filter_x) {
                            for (int32_t in_channel = 0; in_channel < input_depth; ++in_channel) {
                                const int32_t in_x = in_x_origin + dilation_width_factor * filter_x;
                                const int32_t in_y = in_y_origin + dilation_height_factor * filter_y;
                                // If the location is outside the bounds of the input image,
                                // use zero as a default value.
                                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                    (in_y < input_height)) {
                                    int32_t input_index =
                                        csi_get_index(input->dim, batch, in_y, in_x, in_channel);
                                    int32_t input_val = input_data[input_index];
                                    int32_t filter_index = csi_get_index(
                                        kernel->dim, out_channel, filter_y, filter_x, in_channel);
                                    int32_t filter_val = kernel_data[filter_index];
                                    acc +=
                                        (filter_val - filter_offset) * (input_val - input_offset);
                                }
                            }
                        }
                    }
                    if (bias->dim_count != 0) {
                        acc += bias_data[out_channel];
                    }
                    acc = conv_relu6_out_i8(acc, input, output, kernel);
                    output_data[csi_get_index(output->dim, batch, out_y, out_x, out_channel)] = acc;
                }
            }
        }
    }
    csi_nhwc_to_nchw_8(o_output, output);
    free(input->data);
    free(input);
    free(kernel->data);
    free(kernel);
#endif
    return CSINN_TRUE;
}


int csi_depthwise_conv2d_relu6_u8(struct csi_tensor *o_input,
                                  struct csi_tensor *o_output,
                                  struct csi_tensor *o_kernel,
                                  struct csi_tensor *o_bias,
                                  struct conv2d_params *params)
{
    struct csi_tensor* input;
    struct csi_tensor* output;
    struct csi_tensor* kernel;
    struct csi_tensor* bias = o_bias;
    input =  csi_nchw_to_nhwc_8(o_input);
    kernel = csi_nchw_to_nhwc_8(o_kernel);
    output = csi_nchw_to_nhwc_8(o_output);

    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    uint8_t *kernel_data = kernel->data;
    int32_t *bias_data = bias->data;
    const int32_t dilation_width_factor = params->dilation_width;
    const int32_t dilation_height_factor = params->dilation_height;
    const int32_t batches = input->dim[0];
    const int32_t input_depth = input->dim[3];
    const int32_t output_depth = output->dim[3];
    const int32_t input_height = input->dim[1];
    const int32_t input_width = input->dim[2];
    const int32_t filter_height = kernel->dim[1];
    const int32_t filter_width = kernel->dim[2];
    const int32_t output_height = output->dim[1];
    const int32_t output_width = output->dim[2];
    const int32_t depth_multiplier = output_depth / input_depth;
    const int32_t input_offset = input->zero_point;
    const int32_t filter_offset = kernel->zero_point;
    const int32_t output_offset = output->zero_point;
    const int32_t output_multiplier = output->multiplier;
    const int32_t output_shift = output->shift;

    for (int32_t b = 0; b < batches; ++b) {
        #pragma omp parallel for num_threads(8)
        for (int32_t out_y = 0; out_y < output_height; ++out_y) {
            for (int32_t out_x = 0; out_x < output_width; ++out_x) {
                for (int32_t ic = 0; ic < input_depth; ++ic) {
                    for (int32_t m = 0; m < depth_multiplier; m++) {
                        const int32_t oc = m + ic * depth_multiplier;
                        const int32_t in_x_origin = (out_x * params->stride_width) - params->pad_left;
                        const int32_t in_y_origin = (out_y * params->stride_height) - params->pad_top;
                        int64_t acc = 0;
                        for (int32_t filter_y = 0; filter_y < filter_height; ++filter_y) {
                            for (int32_t filter_x = 0; filter_x < filter_width; ++filter_x) {
                                const int32_t in_x = in_x_origin + dilation_width_factor * filter_x;
                                const int32_t in_y =
                                    in_y_origin + dilation_height_factor * filter_y;
                                // If the location is outside the bounds of the input image,
                                // use zero as a default value.
                                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                    (in_y < input_height)) {
                                    int32_t input_val =
                                        input_data[csi_get_index(input->dim, b, in_y, in_x, ic)];
                                    int32_t filter_val = kernel_data[csi_get_index(
                                        kernel->dim, ic, filter_y, filter_x, m)];
                                    acc +=
                                        (filter_val - filter_offset) * (input_val - input_offset);
                                }
                            }
                        }
                        if (bias->dim_count != 0) {
                            acc += bias_data[oc];
                        }
                        acc = conv_relu6_out_u8(acc, input, output, kernel);
                        output_data[csi_get_index(output->dim, b, out_y, out_x, oc)] =
                            acc;
                    }
                }
            }
        }
    }
    csi_nhwc_to_nchw_8(o_output, output);
    free(input->data);
    free(input);
    free(kernel->data);
    free(kernel);
    return CSINN_TRUE;
}

int csi_depthwise_conv2d_relu6_i8(struct csi_tensor *o_input,
                                  struct csi_tensor *o_output,
                                  struct csi_tensor *o_kernel,
                                  struct csi_tensor *o_bias,
                                  struct conv2d_params *params)
{
    struct csi_tensor* input;
    struct csi_tensor* output;
    struct csi_tensor* kernel;
    struct csi_tensor* bias = o_bias;
    input =  csi_nchw_to_nhwc_8(o_input);
    kernel = csi_nchw_to_nhwc_8(o_kernel);
    output = csi_nchw_to_nhwc_8(o_output);

    int8_t *input_data = input->data;
    int8_t *output_data = output->data;
    int8_t *kernel_data = kernel->data;
    int32_t *bias_data = bias->data;
    const int32_t dilation_width_factor = params->dilation_width;
    const int32_t dilation_height_factor = params->dilation_height;
    const int32_t batches = input->dim[0];
    const int32_t input_depth = input->dim[3];
    const int32_t output_depth = output->dim[3];
    const int32_t input_height = input->dim[1];
    const int32_t input_width = input->dim[2];
    const int32_t filter_height = kernel->dim[1];
    const int32_t filter_width = kernel->dim[2];
    const int32_t output_height = output->dim[1];
    const int32_t output_width = output->dim[2];
    const int32_t depth_multiplier = output_depth / input_depth;
    const int32_t input_offset = input->zero_point;
    const int32_t filter_offset = kernel->zero_point;
    const int32_t output_offset = output->zero_point;
    const int32_t output_multiplier = output->multiplier;
    const int32_t output_shift = output->shift;

    for (int32_t b = 0; b < batches; ++b) {
        #pragma omp parallel for num_threads(8)
        for (int32_t out_y = 0; out_y < output_height; ++out_y) {
            for (int32_t out_x = 0; out_x < output_width; ++out_x) {
                for (int32_t ic = 0; ic < input_depth; ++ic) {
                    for (int32_t m = 0; m < depth_multiplier; m++) {
                        const int32_t oc = m + ic * depth_multiplier;
                        const int32_t in_x_origin = (out_x * params->stride_width) - params->pad_left;
                        const int32_t in_y_origin = (out_y * params->stride_height) - params->pad_top;
                        int64_t acc = 0;
                        for (int32_t filter_y = 0; filter_y < filter_height; ++filter_y) {
                            for (int32_t filter_x = 0; filter_x < filter_width; ++filter_x) {
                                const int32_t in_x = in_x_origin + dilation_width_factor * filter_x;
                                const int32_t in_y =
                                    in_y_origin + dilation_height_factor * filter_y;
                                // If the location is outside the bounds of the input image,
                                // use zero as a default value.
                                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                    (in_y < input_height)) {
                                    int32_t input_val =
                                        input_data[csi_get_index(input->dim, b, in_y, in_x, ic)];
                                    int32_t filter_val = kernel_data[csi_get_index(
                                        kernel->dim, ic, filter_y, filter_x, m)];
                                    acc +=
                                        (filter_val - filter_offset) * (input_val - input_offset);
                                }
                            }
                        }
                        if (bias->dim_count != 0) {
                            acc += bias_data[oc];
                        }
                        acc = conv_relu6_out_i8(acc, input, output, kernel);
                        output_data[csi_get_index(output->dim, b, out_y, out_x, oc)] =
                            acc;
                    }
                }
            }
        }
    }
    csi_nhwc_to_nchw_8(o_output, output);
    free(input->data);
    free(input);
    free(kernel->data);
    free(kernel);
    return CSINN_TRUE;
}


int csi_conv2d_relu6_init(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct csi_tensor *kernel,
                          struct csi_tensor *bias,
                          struct conv2d_params *params)
{
    if(params->wscales != NULL && params->wzps != NULL){
        if (params->layout == CSINN_NCHW) {
            if (params->group == 1) {
                params->bc = csi_bc_map(params->api, CSINN_OP_CONV2D_CHANNEL_RELU6, input->dtype);
            } else if (params->group == output->dim[1]) {
                params->bc = csi_bc_map(params->api, CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6, input->dtype);
            } else {
                return CSINN_FALSE;
            }
            if (params->bc == NULL) {
                return CSINN_UNSUPPORT_DTYPE;
            }
        } else {
            return CSINN_UNSUPPORT_LAYOUT;
        }
        return CSINN_TRUE;
    }
    if (params->layout == CSINN_NCHW) {
        if (params->group == 1) {
            params->bc = csi_bc_map(params->api, CSINN_OP_CONV2D_RELU6, input->dtype);
        } else if (params->group == output->dim[1]) {
            params->bc = csi_bc_map(params->api, CSINN_OP_DEPTHWISE_CONV2D_RELU6, input->dtype);
        } else {
            return CSINN_FALSE;
        }
        if (params->bc == NULL) {
                return CSINN_UNSUPPORT_DTYPE;
        }
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    return CSINN_TRUE;
}

int csi_conv2d_relu6(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct csi_tensor *kernel,
                     struct csi_tensor *bias,
                     struct conv2d_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, kernel, bias, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
