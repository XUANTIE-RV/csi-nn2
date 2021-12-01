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

/* CSI-NN2 version 1.8.x */

#include "csi_ref.h"
#ifdef CSI_AVX_OPT
#include "conv_avx.c"
#endif

/* reference https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/conv.h */

static int csi_ref_conv2d_nhwc_f32(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct csi_tensor *kernel,
                                   struct csi_tensor *bias,
                                   struct conv2d_params *params)
{
    float *input_data  = input->data;
    float *output_data = output->data;
    float *kernel_data = kernel->data;
    float *bias_data   = bias->data;
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

    for (int32_t batch = 0; batch < batches; ++batch) {
        for (int32_t out_y = 0; out_y < output_height; ++out_y) {
            for (int32_t out_x = 0; out_x < output_width; ++out_x) {
                for (int32_t out_channel = 0; out_channel < output_depth; ++out_channel) {
                    const int32_t in_x_origin = (out_x * params->stride_width) - params->pad_left;
                    const int32_t in_y_origin = (out_y * params->stride_height) - params->pad_top;
                    float acc = 0;
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
                                        csi_ref_get_index(input->dim, batch, in_y, in_x, in_channel);
                                    float input_val = input_data[input_index];
                                    int32_t filter_index = csi_ref_get_index(
                                        kernel->dim, out_channel, filter_y, filter_x, in_channel);
                                    float filter_val = kernel_data[filter_index];
                                    acc += (input_val * filter_val);
                                }
                            }
                        }
                    }
                    float bias_value = 0.0f;
                    if (bias_data && bias->dim_count != 0) {
                        bias_value = bias_data[out_channel];
                    }
                    output_data[csi_ref_get_index(output->dim, batch, out_y, out_x, out_channel)] = acc + bias_value;
                }
            }
        }
    }

    return CSINN_TRUE;
}

static int csi_ref_conv2d_nchw_f32(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct csi_tensor *kernel,
                                   struct csi_tensor *bias,
                                   struct conv2d_params *params)
{
#ifdef CSI_AVX_OPT
    struct csi_tensor *t_input = csi_alloc_tensor(NULL);
    csi_tensor_copy(t_input, input);
    int32_t pad_b[4] = {0, params->pad_top, params->pad_left, 0};
    int32_t pad_a[4] = {0, params->pad_down, params->pad_right, 0};
    t_input->dim[2] = input->dim[2] + params->pad_top + params->pad_down;
    t_input->dim[3] = input->dim[3] + params->pad_left + params->pad_right;
    t_input->data = malloc(t_input->dim[0] * t_input->dim[1] *
                           t_input->dim[2] * t_input->dim[3] * 4);
    struct pad_params pparams;
    pparams.base.layout = CSINN_LAYOUT_NCHW;
    pparams.base.api = CSINN_REF;
    pparams.pad_before = pad_b;
    pparams.pad_after = pad_a;
    pparams.pad_num = 4;
    pparams.pad_mode = 0;
    pparams.pad_value = 0;
    pparams.base.name = "tmp_pad";
    csi_pad_init(input, t_input, &pparams);
    csi_pad(input, t_input, &pparams);

    struct csi_tensor *t_kernel = csi_alloc_tensor(NULL);
    conv_trans_kernel_avx(kernel, t_kernel);
    conv_im2col_sgemm_avx(t_input, output, t_kernel, bias,
                          kernel->dim[3], kernel->dim[2],
                          params->stride_width, params->stride_height);

    free(t_input->data);
    free(t_kernel->data);
#else
    struct csi_tensor* t_input;
    struct csi_tensor* t_output;
    struct csi_tensor* t_kernel;
    struct csi_tensor* t_bias = bias;
    t_input =  csi_ref_nchw_to_nhwc_f32(input);
    t_kernel = csi_ref_nchw_to_nhwc_f32(kernel);
    t_output = csi_ref_nchw_to_nhwc_f32(output);
    csi_ref_conv2d_nhwc_f32(t_input, t_output, t_kernel, t_bias, params);
    csi_ref_nhwc_to_nchw_f32(output, t_output);
    free(t_input->data);
    free(t_input);
    free(t_kernel->data);
    free(t_kernel);

#endif
    return CSINN_TRUE;
}

static int csi_ref_depthwise_conv2d_nhwc_f32(struct csi_tensor *input,
                                             struct csi_tensor *output,
                                             struct csi_tensor *kernel,
                                             struct csi_tensor *bias,
                                             struct conv2d_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    float *kernel_data = kernel->data;
    float *bias_data = bias->data;
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

    assert(input_depth * depth_multiplier == output_depth);    // The input and output channels are equal for dw convolution

    for (int32_t b = 0; b < batches; ++b) {
        for (int32_t out_y = 0; out_y < output_height; ++out_y) {
            for (int32_t out_x = 0; out_x < output_width; ++out_x) {
                for (int32_t ic = 0; ic < input_depth; ++ic) {
                    for (int32_t m = 0; m < depth_multiplier; m++) {
                        const int32_t oc = m + ic * depth_multiplier;
                        const int32_t in_x_origin = (out_x * params->stride_width) - params->pad_left;
                        const int32_t in_y_origin = (out_y * params->stride_height) - params->pad_top;
                        float acc = 0;
                        for (int32_t filter_y = 0; filter_y < filter_height; ++filter_y) {
                            for (int32_t filter_x = 0; filter_x < filter_width; ++filter_x) {
                                const int32_t in_x = in_x_origin + dilation_width_factor * filter_x;
                                const int32_t in_y = in_y_origin + dilation_height_factor * filter_y;
                                // If the location is outside the bounds of the input image,
                                // use zero as a default value.
                                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                    (in_y < input_height)) {
                                    float input_val =
                                        input_data[csi_ref_get_index(input->dim, b, in_y, in_x, ic)];
                                    float filter_val = kernel_data[csi_ref_get_index(
                                        kernel->dim, 0, filter_y, filter_x, oc)];
                                    acc += (filter_val) * (input_val);
                                }
                            }
                        }
                        if (bias_data && bias->dim_count != 0) {
                            acc += bias_data[oc];
                        }
                        output_data[csi_ref_get_index(output->dim, b, out_y, out_x, oc)] = acc;
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_ref_depthwise_conv2d_nchw_f32(struct csi_tensor *input,
                                             struct csi_tensor *output,
                                             struct csi_tensor *kernel,
                                             struct csi_tensor *bias,
                                             struct conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    const int32_t dilation_width_factor = params->dilation_width;
    const int32_t dilation_height_factor = params->dilation_height;
    const int32_t batches = input->dim[0];
    const int32_t input_depth = input->dim[1];
    const int32_t output_depth = output->dim[1];
    const int32_t input_height = input->dim[2];
    const int32_t input_width = input->dim[3];
    const int32_t filter_height = kernel->dim[2];
    const int32_t filter_width = kernel->dim[3];
    const int32_t output_height = output->dim[2];
    const int32_t output_width = output->dim[3];
    const int32_t depth_multiplier = output_depth / input_depth;
    assert(input_depth * depth_multiplier == output_depth);    // The input and output channels are equal for dw convolution

    for (int32_t b = 0; b < batches; ++b) {
        for (int32_t ic = 0; ic < input_depth; ++ic) {
            for (int32_t out_y = 0; out_y < output_height; ++out_y) {
                for (int32_t out_x = 0; out_x < output_width; ++out_x) {
                    for (int32_t m = 0; m < depth_multiplier; m++) {
                        const int32_t oc = m + ic * depth_multiplier;
                        const int32_t in_x_origin = (out_x * params->stride_width) - params->pad_left;
                        const int32_t in_y_origin = (out_y * params->stride_height) - params->pad_top;
                        float acc = 0;
                        for (int32_t filter_y = 0; filter_y < filter_height; ++filter_y) {
                            for (int32_t filter_x = 0; filter_x < filter_width; ++filter_x) {
                                const int32_t in_x = in_x_origin + dilation_width_factor * filter_x;
                                const int32_t in_y = in_y_origin + dilation_height_factor * filter_y;
                                // If the location is outside the bounds of the input image,
                                // use zero as a default value.
                                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                                    (in_y < input_height)) {
                                    float input_val =
                                        input_data[csi_ref_get_index(input->dim, b, ic, in_y, in_x)];
                                    float filter_val = kernel_data[csi_ref_get_index(
                                        kernel->dim, oc, 0, filter_y, filter_x)];
                                    acc += (filter_val) * (input_val);
                                }
                            }
                        }
                        if (bias_data && bias->dim_count != 0) {
                            acc += bias_data[oc];
                        }
                        output_data[csi_ref_get_index(output->dim, b, oc, out_y, out_x)] = acc;
                    }
                }
            }
        }
    }

}

static int csi_ref_group_conv2d_nhwc_f32(struct csi_tensor *o_input,
                                         struct csi_tensor *o_output,
                                         struct csi_tensor *o_kernel,
                                         struct csi_tensor *o_bias,
                                         struct conv2d_params *params)
{
    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *kernel = csi_alloc_tensor(NULL);
    struct csi_tensor *bias = csi_alloc_tensor(NULL);

    csi_tensor_copy(input, o_input);
    csi_tensor_copy(output, o_output);
    csi_tensor_copy(kernel, o_kernel);
    csi_tensor_copy(bias, o_bias);

    input->dim[3] /= params->group;
    output->dim[3] /= params->group;
    kernel->dim[0] /= params->group;

    int input_size = csi_tensor_size(input);
    int output_size = csi_tensor_size(output);
    int kernel_size = csi_tensor_size(kernel);

    float *input_data = o_input->data;
    float *output_data = o_output->data;
    float *kernel_data = o_kernel->data;
    float *bias_data = o_bias->data;
    for (int i = 0; i < params->group; i++) {
        input->data = input_data + i * input_size;
        output->data = output_data + i * output_size;
        kernel->data = kernel_data + i * kernel_size;
        if (bias->data && bias->dim_count != 0) {
            bias->data = bias_data + i * o_output->dim[3] / params->group;
        }
        csi_ref_conv2d_nhwc_f32(input, output, kernel, bias, params);
    }
    return CSINN_TRUE;
}

static int csi_ref_group_conv2d_nchw_f32(struct csi_tensor *o_input,
                                         struct csi_tensor *o_output,
                                         struct csi_tensor *o_kernel,
                                         struct csi_tensor *o_bias,
                                         struct conv2d_params *params)
{
    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *kernel = csi_alloc_tensor(NULL);
    struct csi_tensor *bias = csi_alloc_tensor(NULL);

    csi_tensor_copy(input, o_input);
    csi_tensor_copy(output, o_output);
    csi_tensor_copy(kernel, o_kernel);
    csi_tensor_copy(bias, o_bias);

    input->dim[1] /= params->group;
    output->dim[1] /= params->group;
    kernel->dim[0] /= params->group;

    int input_size = csi_tensor_size(input);
    int output_size = csi_tensor_size(output);
    int kernel_size = csi_tensor_size(kernel);

    float *input_data = o_input->data;
    float *output_data = o_output->data;
    float *kernel_data = o_kernel->data;
    float *bias_data = o_bias->data;
    for (int i = 0; i < params->group; i++) {
        input->data = input_data + i * input_size;
        output->data = output_data + i * output_size;
        kernel->data = kernel_data + i * kernel_size;
        if (bias->data && bias->dim_count != 0) {
            bias->data = bias_data + i * o_output->dim[1] / params->group;
        }
        csi_ref_conv2d_nchw_f32(input, output, kernel, bias, params);
    }
    return CSINN_TRUE;
}

int csi_ref_conv2d_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct csi_tensor *kernel,
                       struct csi_tensor *bias,
                       struct conv2d_params *params)
{
    if (params->base.layout == CSINN_LAYOUT_NHWC) {
        csi_ref_conv2d_nhwc_f32(input, output, kernel, bias, params);
    } else if (params->base.layout == CSINN_LAYOUT_NCHW) {
        csi_ref_conv2d_nchw_f32(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_ref_conv2d_quant(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct csi_tensor *kernel,
                         struct csi_tensor *bias,
                         struct conv2d_params *params)
{
    return csi_ref_conv_callback_base(input, output, kernel, bias, params, csi_ref_conv2d_f32);
}

int csi_ref_depthwise_conv2d_f32(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct csi_tensor *kernel,
                                 struct csi_tensor *bias,
                                 struct conv2d_params *params)
{
    if (params->base.layout == CSINN_LAYOUT_NHWC) {
        csi_ref_depthwise_conv2d_nhwc_f32(input, output, kernel, bias, params);
    } else if (params->base.layout == CSINN_LAYOUT_NCHW) {
        csi_ref_depthwise_conv2d_nchw_f32(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_ref_depthwise_conv2d_quant(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct csi_tensor *kernel,
                                   struct csi_tensor *bias,
                                   struct conv2d_params *params)
{
    return csi_ref_conv_callback_base(input, output, kernel, bias, params, csi_ref_depthwise_conv2d_f32);
}

int csi_ref_group_conv2d_f32(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct csi_tensor *kernel,
                             struct csi_tensor *bias,
                             struct conv2d_params *params)
{
    if (params->base.layout == CSINN_LAYOUT_NHWC) {
        csi_ref_group_conv2d_nhwc_f32(input, output, kernel, bias, params);
    } else if (params->base.layout == CSINN_LAYOUT_NCHW) {
        csi_ref_group_conv2d_nchw_f32(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_ref_group_conv2d_quant(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct csi_tensor *kernel,
                               struct csi_tensor *bias,
                               struct conv2d_params *params)
{
    return csi_ref_conv_callback_base(input, output, kernel, bias, params, csi_ref_group_conv2d_f32);
}

