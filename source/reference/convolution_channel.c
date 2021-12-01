/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

#include "csi_ref.h"

static float csi_ref_uint8_to_float_channel(uint8_t i, float scale, int32_t zero_point)
{
    return ((float)i - zero_point) * scale;
}

static float csi_ref_int8_to_float_channel(int8_t i, float scale, int32_t zero_point)
{
    return ((float)i - zero_point) * scale;
}

static int channel_kernel_to_common(struct csi_tensor *float_kernel, struct csi_tensor *o_kernel,
                                     struct conv2d_params *params)
{
    float *float_kernel_data = float_kernel->data;
    int kernel_size = csi_tensor_size(o_kernel);
    for (int i = 0; i < o_kernel->dim[0]; i++) {
        int per_cahnnel = kernel_size / o_kernel->dim[0];
        for (int j = 0; j < per_cahnnel; j++) {
            int index = i * per_cahnnel + j;
            if (o_kernel->dtype == CSINN_DTYPE_UINT8) {
                uint8_t *kernel_data = o_kernel->data;
                float_kernel_data[index] = csi_ref_uint8_to_float_channel(kernel_data[index],
                    o_kernel->qinfo[i].scale, o_kernel->qinfo[i].zero_point);
            } else if (o_kernel->dtype == CSINN_DTYPE_INT8) {
                int8_t *kernel_data = o_kernel->data;
                float_kernel_data[index] = csi_ref_int8_to_float_channel(kernel_data[index],
                    o_kernel->qinfo[i].scale, o_kernel->qinfo[i].zero_point);
            } else {
                return CSINN_FALSE;
            }
        }
    }
}

static void channel_bias_to_common(struct csi_tensor *float_bias, struct csi_tensor *bias,
                                   struct csi_tensor *input, struct csi_tensor *kernel)
{
    int32_t *bias_data = bias->data;
    float *float_bias_data = float_bias->data;
    int bias_size = csi_tensor_size(bias);
    for (int i = 0; i < bias_size; i++) {
        float_bias_data[i] = bias_data[i] * kernel->qinfo[i].scale * input->qinfo->scale;
    }
}

static int csi_ref_conv2d_channel_nchw_quant(struct csi_tensor *o_input,
                                             struct csi_tensor *o_output,
                                             struct csi_tensor *o_kernel,
                                             struct csi_tensor *o_bias,
                                             struct conv2d_params *params)
{
    struct csi_tensor *float_input = csi_ref_convert_float_tensor(o_input);
    struct csi_tensor *float_kernel = csi_ref_alloc_float_tensor(o_kernel);
    struct csi_tensor *float_bias = csi_ref_alloc_float_tensor(o_bias);
    struct csi_tensor *float_output = csi_ref_alloc_float_tensor(o_output);
    channel_kernel_to_common(float_kernel, o_kernel, params);
    channel_bias_to_common(float_bias, o_bias, o_input, o_kernel);
    csi_ref_conv2d_f32(float_input, float_output, float_kernel, float_bias, params);
    csi_tensor_data_convert(o_output, float_output);
    csi_ref_conv_free_float_tensor(float_input, float_output, float_kernel, float_bias);

    return CSINN_TRUE;
}

static int csi_ref_depthwise_conv2d_channel_nchw_u8(struct csi_tensor *o_input,
                                                    struct csi_tensor *o_output,
                                                    struct csi_tensor *o_kernel,
                                                    struct csi_tensor *o_bias,
                                                    struct conv2d_params *params)
{
    struct csi_tensor* input;
    struct csi_tensor* output;
    struct csi_tensor* kernel;
    struct csi_tensor* bias = o_bias;
    input =  csi_ref_nchw_to_nhwc_8(o_input);
    kernel = csi_ref_nchw_to_nhwc_8(o_kernel);
    output = csi_ref_nchw_to_nhwc_8(o_output);

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
    const int32_t input_offset = input->qinfo->zero_point;
    const int32_t output_offset = output->qinfo->zero_point;
    const int32_t output_multiplier = output->qinfo->multiplier;
    const int32_t output_shift = output->qinfo->shift;

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
                                        input_data[csi_ref_get_index(input->dim, b, in_y, in_x, ic)];
                                    int32_t filter_val = kernel_data[csi_ref_get_index(
                                        kernel->dim, ic, filter_y, filter_x, m)];
                                    acc +=
                                        (filter_val - o_kernel->qinfo[oc].zero_point) * (input_val - input_offset);
                                }
                            }
                        }
                        if (bias->dim_count != 0) {
                            acc += bias_data[oc];
                        }

                        uint8_t out = csi_ref_quantize_channel_u8(acc, input, output, o_kernel->qinfo[oc].scale);
                        output_data[csi_ref_get_index(output->dim, b, out_y, out_x, oc)] = out;
                    }
                }
            }
        }
    }
    csi_ref_nhwc_to_nchw_8(o_output, output);
    free(input->data);
    free(input);
    free(kernel->data);
    free(kernel);
    return CSINN_TRUE;
}

static int csi_ref_depthwise_conv2d_channel_nchw_i8(struct csi_tensor *o_input,
                                                    struct csi_tensor *o_output,
                                                    struct csi_tensor *o_kernel,
                                                    struct csi_tensor *o_bias,
                                                    struct conv2d_params *params)
{
    struct csi_tensor* input;
    struct csi_tensor* output;
    struct csi_tensor* kernel;
    struct csi_tensor* bias = o_bias;
    input =  csi_ref_nchw_to_nhwc_8(o_input);
    kernel = csi_ref_nchw_to_nhwc_8(o_kernel);
    output = csi_ref_nchw_to_nhwc_8(o_output);

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
    const int32_t input_offset = input->qinfo->zero_point;
    const int32_t output_offset = output->qinfo->zero_point;
    const int32_t output_multiplier = output->qinfo->multiplier;
    const int32_t output_shift = output->qinfo->shift;

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
                                        input_data[csi_ref_get_index(input->dim, b, in_y, in_x, ic)];
                                    int32_t filter_val = kernel_data[csi_ref_get_index(
                                        kernel->dim, ic, filter_y, filter_x, m)];
                                    acc +=
                                        (filter_val - o_kernel->qinfo[oc].zero_point) * (input_val - input_offset);
                                }
                            }
                        }
                        if (bias->dim_count != 0) {
                            acc += bias_data[oc];
                        }

                        int8_t out = csi_ref_quantize_channel_i8(acc, input, output, o_kernel->qinfo[oc].scale);
                        output_data[csi_ref_get_index(output->dim, b, out_y, out_x, oc)] = out;
                    }
                }
            }
        }
    }
    csi_ref_nhwc_to_nchw_8(o_output, output);
    free(input->data);
    free(input);
    free(kernel->data);
    free(kernel);
    return CSINN_TRUE;
}

static int csi_ref_group_conv2d_channel_nchw_quant(struct csi_tensor *o_input,
                                                   struct csi_tensor *o_output,
                                                   struct csi_tensor *o_kernel,
                                                   struct csi_tensor *o_bias,
                                                   struct conv2d_params *params)
{
    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *kernel = csi_alloc_tensor(NULL);
    struct csi_tensor *bias = csi_alloc_tensor(NULL);
    struct conv2d_params pparams;

    csi_tensor_copy(input, o_input);
    csi_tensor_copy(output, o_output);
    csi_tensor_copy(kernel, o_kernel);
    csi_tensor_copy(bias, o_bias);
    memcpy(&pparams, params, sizeof(struct conv2d_params));

    input->dim[1] /= params->group;
    output->dim[1] /= params->group;
    kernel->dim[0] /= params->group;
    bias->dim[0] /= params->group;

    pparams.group = 1;
    int input_size = csi_tensor_size(input);
    int output_size = csi_tensor_size(output);
    int kernel_size = csi_tensor_size(kernel);

    int8_t *input_data = o_input->data;
    int8_t *output_data = o_output->data;
    int8_t *kernel_data = o_kernel->data;
    int32_t *bias_data = o_bias->data;
    for (int i = 0; i < params->group; i++) {
        input->data = input_data + i * input_size;
        output->data = output_data + i * output_size;
        kernel->data = kernel_data + i * kernel_size;
        if (bias->data && bias->dim_count != 0) {
            bias->data = bias_data + i * o_output->dim[1] / params->group;
        }
        kernel->qinfo = o_kernel->qinfo + i * o_output->dim[1] / params->group;

        csi_ref_conv2d_channel_nchw_quant(input, output, kernel, bias, &pparams);
    }
    return CSINN_TRUE;
}

int csi_ref_conv2d_channel_quant(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct csi_tensor *kernel,
                                 struct csi_tensor *bias,
                                 struct conv2d_params *params)
{
    if (params->base.layout == CSINN_NCHW) {
        csi_ref_conv2d_channel_nchw_quant(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_ref_conv2d_channel_relu_quant(struct csi_tensor *input,
                                      struct csi_tensor *output,
                                      struct csi_tensor *kernel,
                                      struct csi_tensor *bias,
                                      struct conv2d_params *params)
{
    csi_ref_conv2d_channel_quant(input, output, kernel, bias, params);
    struct relu_params *rp = calloc(1, sizeof(struct relu_params));
    memcpy(&(rp->base), &(params->base), sizeof(struct csi_params_base));
    csi_relu_init(output, output, rp);
    csi_relu(output, output, rp);
    return CSINN_TRUE;
}

int csi_ref_conv2d_channel_relu6_quant(struct csi_tensor *input,
                                       struct csi_tensor *output,
                                       struct csi_tensor *kernel,
                                       struct csi_tensor *bias,
                                       struct conv2d_params *params)
{
    csi_ref_conv2d_channel_quant(input, output, kernel, bias, params);
    struct relu_params *rp = calloc(1, sizeof(struct relu_params));
    memcpy(&(rp->base), &(params->base), sizeof(struct csi_params_base));
    csi_relu6_init(output, output, rp);
    csi_relu6(output, output, rp);
    return CSINN_TRUE;
}


int csi_ref_depthwise_conv2d_channel_quant(struct csi_tensor *input,
                                           struct csi_tensor *output,
                                           struct csi_tensor *kernel,
                                           struct csi_tensor *bias,
                                           struct conv2d_params *params)
{
    if (params->base.layout == CSINN_NCHW) {
        if (input->dtype == CSINN_DTYPE_UINT8) {
            csi_ref_depthwise_conv2d_channel_nchw_u8(input, output, kernel, bias, params);
        } else if (input->dtype == CSINN_DTYPE_INT8) {
            csi_ref_depthwise_conv2d_channel_nchw_i8(input, output, kernel, bias, params);
        } else {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_ref_depthwise_conv2d_channel_relu_quant(struct csi_tensor *input,
                                                struct csi_tensor *output,
                                                struct csi_tensor *kernel,
                                                struct csi_tensor *bias,
                                                struct conv2d_params *params)
{
    csi_ref_depthwise_conv2d_channel_quant(input, output, kernel, bias, params);
    struct relu_params *rp = calloc(1, sizeof(struct relu_params));
    memcpy(&(rp->base), &(params->base), sizeof(struct csi_params_base));
    csi_relu_init(output, output, rp);
    csi_relu(output, output, rp);
}

int csi_ref_depthwise_conv2d_channel_relu6_quant(struct csi_tensor *input,
                                                 struct csi_tensor *output,
                                                 struct csi_tensor *kernel,
                                                 struct csi_tensor *bias,
                                                 struct conv2d_params *params)
{
    csi_ref_depthwise_conv2d_channel_quant(input, output, kernel, bias, params);
    struct relu_params *rp = calloc(1, sizeof(struct relu_params));
    memcpy(&(rp->base), &(params->base), sizeof(struct csi_params_base));
    csi_relu6_init(output, output, rp);
    csi_relu6(output, output, rp);
}

int csi_ref_group_conv2d_channel_quant(struct csi_tensor *input,
                                       struct csi_tensor *output,
                                       struct csi_tensor *kernel,
                                       struct csi_tensor *bias,
                                       struct conv2d_params *params)
{
    if (params->base.layout == CSINN_NCHW) {
        csi_ref_group_conv2d_channel_nchw_quant(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_ref_group_conv2d_channel_relu_quant(struct csi_tensor *input,
                                            struct csi_tensor *output,
                                            struct csi_tensor *kernel,
                                            struct csi_tensor *bias,
                                            struct conv2d_params *params)
{
    csi_ref_group_conv2d_channel_quant(input, output, kernel, bias, params);
    struct relu_params *rp = calloc(1, sizeof(struct relu_params));
    memcpy(&(rp->base), &(params->base), sizeof(struct csi_params_base));
    csi_relu_init(output, output, rp);
    csi_relu(output, output, rp);
}
