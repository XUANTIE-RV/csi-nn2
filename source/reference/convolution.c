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
#ifdef CSI_AVX_OPT
#include "conv_avx.c"
#endif

/* reference https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/reference/conv.h */

static int csi_conv2d_nhwc_f32(struct csi_tensor *input,
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
                                        csi_get_index(input->dim, batch, in_y, in_x, in_channel);
                                    float input_val = input_data[input_index];
                                    int32_t filter_index = csi_get_index(
                                        kernel->dim, out_channel, filter_y, filter_x, in_channel);
                                    float filter_val = kernel_data[filter_index];
                                    acc += (input_val * filter_val);
                                }
                            }
                        }
                    }
                    float bias_value = 0.0f;
                    if (bias_data) {
                        bias_value = bias_data[out_channel];
                    }
                    output_data[csi_get_index(output->dim, batch, out_y, out_x, out_channel)] = acc + bias_value;
                }
            }
        }
    }

    return CSINN_TRUE;
}

static int csi_conv2d_nchw_f32(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct csi_tensor *kernel,
                               struct csi_tensor *bias,
                               struct conv2d_params *params)
{
#ifdef CSI_AVX_OPT
    struct csi_tensor t_input;
    memcpy(&t_input, input, sizeof(struct csi_tensor));
    int32_t pad_b[4] = {0, params->pad_top, params->pad_left, 0};
    int32_t pad_a[4] = {0, params->pad_down, params->pad_right, 0};
    t_input.dim[2] = input->dim[2] + params->pad_top + params->pad_down;
    t_input.dim[3] = input->dim[3] + params->pad_left + params->pad_right;
    t_input.data = malloc(t_input.dim[0] * t_input.dim[1] *
                           t_input.dim[2] * t_input.dim[3] * 4);
    struct pad_params pparams;
    pparams.layout = CSINN_NCHW;
    pparams.api = CSINN_REF;
    pparams.pad_before = pad_b;
    pparams.pad_after = pad_a;
    pparams.pad_mode = 0;
    pparams.pad_value = 0;
    csi_pad_init(input, &t_input, &pparams);
    csi_pad(input, &t_input, &pparams);

    struct csi_tensor t_kernel;
    conv_trans_kernel_avx(kernel, &t_kernel);
    conv_im2col_sgemm_avx(&t_input, output, &t_kernel, bias,
                          kernel->dim[3], kernel->dim[2],
                          params->stride_width, params->stride_height);

    free(t_input.data);
    free(t_kernel.data);
#else
    struct csi_tensor* t_input;
    struct csi_tensor* t_output;
    struct csi_tensor* t_kernel;
    struct csi_tensor* t_bias = bias;
    t_input =  csi_nchw_to_nhwc_f32(input);
    t_kernel = csi_nchw_to_nhwc_f32(kernel);
    t_output = csi_nchw_to_nhwc_f32(output);
    int out = csi_conv2d_nhwc_f32(t_input, t_output, t_kernel, t_bias, params);
    csi_nhwc_to_nchw_f32(output, t_output);
    free(t_input->data);
    free(t_input);
    free(t_kernel->data);
    free(t_kernel);

#endif
    return CSINN_TRUE;
}

static int csi_conv2d_nhwc_u8(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct csi_tensor *kernel,
                              struct csi_tensor *bias,
                              struct conv2d_params *params)
{
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
                    acc = csi_quantize_u8(acc, output_offset, output_multiplier, output_shift);
                    output_data[csi_get_index(output->dim, batch, out_y, out_x, out_channel)] = acc;
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_conv2d_nhwc_i8(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct csi_tensor *kernel,
                              struct csi_tensor *bias,
                              struct conv2d_params *params)
{
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
                    acc = csi_quantize_i8(acc, output_offset, output_multiplier, output_shift);
                    output_data[csi_get_index(output->dim, batch, out_y, out_x, out_channel)] = acc;
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_depthwise_conv2d_nhwc_f32(struct csi_tensor *input,
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
    assert(input_depth == output_depth);    // The input and output channels are equal for dw convolution

    for (int32_t b = 0; b < batches; ++b) {
        for (int32_t out_y = 0; out_y < output_height; ++out_y) {
            for (int32_t out_x = 0; out_x < output_width; ++out_x) {
                for (int32_t ic = 0; ic < input_depth; ++ic) {
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
                                    input_data[csi_get_index(input->dim, b, in_y, in_x, ic)];
                                float filter_val = kernel_data[csi_get_index(
                                    kernel->dim, 0, filter_y, filter_x, ic)];
                                acc += (filter_val) * (input_val);
                            }
                        }
                    }
                    if (bias_data) {
                        acc += bias_data[ic];
                    }
                    output_data[csi_get_index(output->dim, b, out_y, out_x, ic)] = acc;
                }
            }
        }
    }
    return CSINN_TRUE;
}


static int csi_depthwise_conv2d_nchw_f32(struct csi_tensor *input,
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
    assert(input_depth == output_depth);    // The input and output channels are equal for dw convolution

    for (int32_t b = 0; b < batches; ++b) {
        for (int32_t ic = 0; ic < input_depth; ++ic) {
            for (int32_t out_y = 0; out_y < output_height; ++out_y) {
                for (int32_t out_x = 0; out_x < output_width; ++out_x) {

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
                                    input_data[csi_get_index(input->dim, b, ic, in_y, in_x)];
                                float filter_val = kernel_data[csi_get_index(
                                    kernel->dim, ic, 0, filter_y, filter_x)];
                                acc += (filter_val) * (input_val);
                            }
                        }
                    }
                    if (bias_data) {
                        acc += bias_data[ic];
                    }
                    output_data[csi_get_index(output->dim, b, ic, out_y, out_x)] = acc;

                }
            }
        }
    }

}

static int csi_depthwise_conv2d_nhwc_u8(struct csi_tensor *input,
                                        struct csi_tensor *output,
                                        struct csi_tensor *kernel,
                                        struct csi_tensor *bias,
                                        struct conv2d_params *params)
{
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
                        output_data[csi_get_index(output->dim, b, out_y, out_x, oc)] =
                            csi_quantize_u8(acc, output_offset, output_multiplier, output_shift);
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_depthwise_conv2d_nhwc_i8(struct csi_tensor *input,
                                        struct csi_tensor *output,
                                        struct csi_tensor *kernel,
                                        struct csi_tensor *bias,
                                        struct conv2d_params *params)
{
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
                        output_data[csi_get_index(output->dim, b, out_y, out_x, oc)] =
                            csi_quantize_i8(acc, output_offset, output_multiplier, output_shift);
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_group_conv2d_nhwc_u8(struct csi_tensor *o_input,
                                    struct csi_tensor *o_output,
                                    struct csi_tensor *o_kernel,
                                    struct csi_tensor *o_bias,
                                    struct conv2d_params *params)
{
    struct csi_tensor input;
    struct csi_tensor output;
    struct csi_tensor kernel;
    struct csi_tensor bias;

    memcpy(&input, o_input, sizeof(struct csi_tensor));
    memcpy(&output, o_output, sizeof(struct csi_tensor));
    memcpy(&kernel, o_kernel, sizeof(struct csi_tensor));
    memcpy(&bias, o_bias, sizeof(struct csi_tensor));

    input.dim[3] /= params->group;
    output.dim[3] /= params->group;
    kernel.dim[0] /= params->group;

    int input_size = 1;
    int output_size = 1;
    int kernel_size = 1;

    for (int i = 0; i < input.dim_count; i++) {
        input_size *= input.dim[i];
        output_size *= output.dim[i];
        kernel_size *= kernel.dim[i];
    }

    uint8_t *input_data = o_input->data;
    uint8_t *output_data = o_output->data;
    uint8_t *kernel_data = o_kernel->data;
    int32_t *bias_data = o_bias->data;
    for (int i = 0; i < params->group; i++) {
        input.data = input_data + i * input_size;
        output.data = output_data + i * output_size;
        kernel.data = kernel_data + i * kernel_size;
        bias.data = bias_data + i * o_output->dim[3] / params->group;
        csi_conv2d_nhwc_u8(&input, &output, &kernel, &bias, params);
    }
    return CSINN_TRUE;
}

static int csi_group_conv2d_nhwc_i8(struct csi_tensor *o_input,
                                    struct csi_tensor *o_output,
                                    struct csi_tensor *o_kernel,
                                    struct csi_tensor *o_bias,
                                    struct conv2d_params *params)
{
    struct csi_tensor input;
    struct csi_tensor output;
    struct csi_tensor kernel;
    struct csi_tensor bias;

    memcpy(&input, o_input, sizeof(struct csi_tensor));
    memcpy(&output, o_output, sizeof(struct csi_tensor));
    memcpy(&kernel, o_kernel, sizeof(struct csi_tensor));
    memcpy(&bias, o_bias, sizeof(struct csi_tensor));

    input.dim[3] /= params->group;
    output.dim[3] /= params->group;
    kernel.dim[0] /= params->group;

    int input_size = 1;
    int output_size = 1;
    int kernel_size = 1;

    for (int i = 0; i < input.dim_count; i++) {
        input_size *= input.dim[i];
        output_size *= output.dim[i];
        kernel_size *= kernel.dim[i];
    }

    int8_t *input_data = o_input->data;
    int8_t *output_data = o_output->data;
    int8_t *kernel_data = o_kernel->data;
    int32_t *bias_data = o_bias->data;
    for (int i = 0; i < params->group; i++) {
        input.data = input_data + i * input_size;
        output.data = output_data + i * output_size;
        kernel.data = kernel_data + i * kernel_size;
        bias.data = bias_data + i * o_output->dim[3] / params->group;
        csi_conv2d_nhwc_i8(&input, &output, &kernel, &bias, params);
    }
    return CSINN_TRUE;
}

static int csi_conv2d_nchw_u8(struct csi_tensor *o_input,
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

    struct csi_tensor t_input;
    memcpy(&t_input, &float_input, sizeof(struct csi_tensor));
    int32_t pad_b[4] = {0, params->pad_top, params->pad_left, 0};
    int32_t pad_a[4] = {0, params->pad_down, params->pad_right, 0};
    t_input.dim[2] = float_input.dim[2] + params->pad_top + params->pad_down;
    t_input.dim[3] = float_input.dim[3] + params->pad_left + params->pad_right;
    t_input.data = malloc(t_input.dim[0] * t_input.dim[1] *
                           t_input.dim[2] * t_input.dim[3] * 4);
    struct pad_params pparams;
    pparams.layout = CSINN_NCHW;
    pparams.api = CSINN_REF;
    pparams.pad_before = pad_b;
    pparams.pad_after = pad_a;
    pparams.pad_mode = 0;
    pparams.pad_value = 0;
    csi_pad_init(&float_input, &t_input, &pparams);
    csi_pad(&float_input, &t_input, &pparams);

    struct csi_tensor t_kernel;
    conv_trans_kernel_avx(&float_kernel, &t_kernel);
    conv_im2col_sgemm_avx(&t_input, &float_output, &t_kernel, &float_bias,
                          o_kernel->dim[3], o_kernel->dim[2],
                          params->stride_width, params->stride_height);


    for (int i = 0; i < output_size; i++) {
        output_data[i] = float_to_uint8(float_output_data[i], o_output);
    }
    free(float_input_data);
    free(float_kernel_data);
    free(float_bias_data);
    free(float_output_data);
    free(t_input.data);
    free(t_kernel.data);
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
                    acc = csi_quantize_u8(acc, output_offset, output_multiplier, output_shift);
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

static int csi_conv2d_nchw_i8(struct csi_tensor *o_input,
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
    int8_t *input_data = o_input->data;
    int8_t *kernel_data = o_kernel->data;
    int32_t *bias_data = o_bias->data;
    int8_t *output_data = o_output->data;
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

    struct csi_tensor t_input;
    memcpy(&t_input, &float_input, sizeof(struct csi_tensor));
    int32_t pad_b[4] = {0, params->pad_top, params->pad_left, 0};
    int32_t pad_a[4] = {0, params->pad_down, params->pad_right, 0};
    t_input.dim[2] = float_input.dim[2] + params->pad_top + params->pad_down;
    t_input.dim[3] = float_input.dim[3] + params->pad_left + params->pad_right;
    t_input.data = malloc(t_input.dim[0] * t_input.dim[1] *
                           t_input.dim[2] * t_input.dim[3] * 4);
    struct pad_params pparams;
    pparams.layout = CSINN_NCHW;
    pparams.api = CSINN_REF;
    pparams.pad_before = pad_b;
    pparams.pad_after = pad_a;
    pparams.pad_mode = 0;
    pparams.pad_value = 0;
    csi_pad_init(&float_input, &t_input, &pparams);
    csi_pad(&float_input, &t_input, &pparams);

    struct csi_tensor t_kernel;
    conv_trans_kernel_avx(&float_kernel, &t_kernel);
    conv_im2col_sgemm_avx(&t_input, &float_output, &t_kernel, &float_bias,
                          o_kernel->dim[3], o_kernel->dim[2],
                          params->stride_width, params->stride_height);


    for (int i = 0; i < output_size; i++) {
        output_data[i] = float_to_int8(float_output_data[i], o_output);
    }
    free(float_input_data);
    free(float_kernel_data);
    free(float_bias_data);
    free(float_output_data);
    free(t_input.data);
    free(t_kernel.data);
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
                    acc = csi_quantize_i8(acc, output_offset, output_multiplier, output_shift);
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


static int csi_depthwise_conv2d_nchw_u8(struct csi_tensor *o_input,
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
                        output_data[csi_get_index(output->dim, b, out_y, out_x, oc)] =
                            csi_quantize_u8(acc, output_offset, output_multiplier, output_shift);
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

static int csi_depthwise_conv2d_nchw_i8(struct csi_tensor *o_input,
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
                        output_data[csi_get_index(output->dim, b, out_y, out_x, oc)] =
                            csi_quantize_i8(acc, output_offset, output_multiplier, output_shift);
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

static int csi_group_conv2d_nchw_u8(struct csi_tensor *o_input,
                                    struct csi_tensor *o_output,
                                    struct csi_tensor *o_kernel,
                                    struct csi_tensor *o_bias,
                                    struct conv2d_params *params)
{
    struct csi_tensor input;
    struct csi_tensor output;
    struct csi_tensor kernel;
    struct csi_tensor bias;

    memcpy(&input, o_input, sizeof(struct csi_tensor));
    memcpy(&output, o_output, sizeof(struct csi_tensor));
    memcpy(&kernel, o_kernel, sizeof(struct csi_tensor));
    memcpy(&bias, o_bias, sizeof(struct csi_tensor));

    input.dim[1] /= params->group;
    output.dim[1] /= params->group;
    kernel.dim[0] /= params->group;

    int input_size = 1;
    int output_size = 1;
    int kernel_size = 1;

    for (int i = 0; i < input.dim_count; i++) {
        input_size *= input.dim[i];
        output_size *= output.dim[i];
        kernel_size *= kernel.dim[i];
    }

    uint8_t *input_data = o_input->data;
    uint8_t *output_data = o_output->data;
    uint8_t *kernel_data = o_kernel->data;
    uint8_t *bias_data = o_bias->data;
    for (int i = 0; i < params->group; i++) {
        input.data = input_data + i * input_size;
        output.data = output_data + i * output_size;
        kernel.data = kernel_data + i * kernel_size;
        bias.data = bias_data + i * o_output->dim[1] / params->group;
        csi_conv2d_nchw_u8(&input, &output, &kernel, &bias, params);
    }
    return CSINN_TRUE;
}

static int csi_group_conv2d_nchw_i8(struct csi_tensor *o_input,
                                    struct csi_tensor *o_output,
                                    struct csi_tensor *o_kernel,
                                    struct csi_tensor *o_bias,
                                    struct conv2d_params *params)
{
    struct csi_tensor input;
    struct csi_tensor output;
    struct csi_tensor kernel;
    struct csi_tensor bias;

    memcpy(&input, o_input, sizeof(struct csi_tensor));
    memcpy(&output, o_output, sizeof(struct csi_tensor));
    memcpy(&kernel, o_kernel, sizeof(struct csi_tensor));
    memcpy(&bias, o_bias, sizeof(struct csi_tensor));

    input.dim[1] /= params->group;
    output.dim[1] /= params->group;
    kernel.dim[0] /= params->group;

    int input_size = 1;
    int output_size = 1;
    int kernel_size = 1;

    for (int i = 0; i < input.dim_count; i++) {
        input_size *= input.dim[i];
        output_size *= output.dim[i];
        kernel_size *= kernel.dim[i];
    }

    int8_t *input_data = o_input->data;
    int8_t *output_data = o_output->data;
    int8_t *kernel_data = o_kernel->data;
    int8_t *bias_data = o_bias->data;
    for (int i = 0; i < params->group; i++) {
        input.data = input_data + i * input_size;
        output.data = output_data + i * output_size;
        kernel.data = kernel_data + i * kernel_size;
        bias.data = bias_data + i * o_output->dim[1] / params->group;
        csi_conv2d_nchw_i8(&input, &output, &kernel, &bias, params);
    }
    return CSINN_TRUE;
}

static int csi_group_conv2d_nhwc_f32(struct csi_tensor *input,
                                     struct csi_tensor *output,
                                     struct csi_tensor *kernel,
                                     struct csi_tensor *bias,
                                     struct conv2d_params *params)
{
    int input_size = 1;
    int output_size = 1;
    int kernel_size = 1;

    input->dim[3] /= params->group;
    output->dim[3] /= params->group;
    kernel->dim[0] /= params->group;

    for (int i = 0; i < input->dim_count; i++) {
        input_size  *= input->dim[i];
        output_size *= output->dim[i];
        kernel_size *= kernel->dim[i];
    }

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

    for(int i = 0; i < params->group; i++) {
        float *input_data = (float *)input->data + i * input_size;
        float *output_data = (float *)output->data + i * output_size;
        float *kernel_data = (float *)kernel->data + i * kernel_size;
        float *bias_data = (float *)bias->data + i * output_depth;
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
                                            csi_get_index(input->dim, batch, in_y, in_x, in_channel);
                                        float input_val = input_data[input_index];
                                        int32_t filter_index = csi_get_index(
                                            kernel->dim, out_channel, filter_y, filter_x, in_channel);
                                        float filter_val = kernel_data[filter_index];
                                        acc += (filter_val) * (input_val);
                                    }
                                }
                            }
                        }
                        if (bias_data != NULL) {
                            acc += bias_data[out_channel];
                        }
                        output_data[csi_get_index(output->dim, batch, out_y, out_x, out_channel)] = acc;
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

int csi_conv2d_f32(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct csi_tensor *kernel,
                   struct csi_tensor *bias,
                   struct conv2d_params *params)
{
    if (params->layout == CSINN_NHWC) {
        csi_conv2d_nhwc_f32(input, output, kernel, bias, params);
    } else if (params->layout == CSINN_NCHW) {
        csi_conv2d_nchw_f32(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_conv2d_u8(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct csi_tensor *kernel,
                  struct csi_tensor *bias,
                  struct conv2d_params *params)
{
    if (params->layout == CSINN_NCHW) {
        csi_conv2d_nchw_u8(input, output, kernel, bias, params);
    } else if (params->layout == CSINN_NHWC) {
        csi_conv2d_nhwc_u8(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_conv2d_i8(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct csi_tensor *kernel,
                  struct csi_tensor *bias,
                  struct conv2d_params *params)
{
    if (params->layout == CSINN_NCHW) {
        csi_conv2d_nchw_i8(input, output, kernel, bias, params);
    } else if (params->layout == CSINN_NHWC) {
        csi_conv2d_nhwc_i8(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_depthwise_conv2d_f32(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct csi_tensor *kernel,
                             struct csi_tensor *bias,
                             struct conv2d_params *params)
{
    if (params->layout == CSINN_NHWC) {
        csi_depthwise_conv2d_nhwc_f32(input, output, kernel, bias, params);
    } else if (params->layout == CSINN_NCHW) {
        csi_depthwise_conv2d_nchw_f32(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_depthwise_conv2d_u8(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct csi_tensor *kernel,
                            struct csi_tensor *bias,
                            struct conv2d_params *params)
{
    if (params->layout == CSINN_NCHW) {
        csi_depthwise_conv2d_nchw_u8(input, output, kernel, bias, params);
    } else if (params->layout == CSINN_NHWC) {
        csi_depthwise_conv2d_nhwc_u8(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_depthwise_conv2d_i8(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct csi_tensor *kernel,
                            struct csi_tensor *bias,
                            struct conv2d_params *params)
{
    if (params->layout == CSINN_NCHW) {
        csi_depthwise_conv2d_nchw_i8(input, output, kernel, bias, params);
    } else if (params->layout == CSINN_NHWC) {
        csi_depthwise_conv2d_nhwc_i8(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_group_conv2d_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct csi_tensor *kernel,
                        struct csi_tensor *bias,
                        struct conv2d_params *params)
{
    if (params->layout == CSINN_NHWC) {
        csi_group_conv2d_nhwc_f32(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_group_conv2d_u8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct csi_tensor *kernel,
                        struct csi_tensor *bias,
                        struct conv2d_params *params)
{
    if (params->layout == CSINN_NCHW) {
        csi_group_conv2d_nchw_u8(input, output, kernel, bias, params);
    } else if (params->layout == CSINN_NHWC) {
        csi_group_conv2d_nhwc_u8(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_group_conv2d_i8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct csi_tensor *kernel,
                        struct csi_tensor *bias,
                        struct conv2d_params *params)
{
    if (params->layout == CSINN_NCHW) {
        csi_group_conv2d_nchw_i8(input, output, kernel, bias, params);
    } else if (params->layout == CSINN_NHWC) {
        csi_group_conv2d_nhwc_i8(input, output, kernel, bias, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_conv2d_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct csi_tensor *bias,
                    struct conv2d_params *params)
{
    if (params->wscales != NULL && params->wzps != NULL){
        if (params->layout != CSINN_NCHW){
            return CSINN_UNSUPPORT_DTYPE;
        }
        if (params->group == 1) {
            params->bc = csi_bc_map(params->api, CSINN_OP_CONV2D_CHANNEL, input->dtype);
         } else if (params->group == input->dim[1]) {
            params->bc = csi_bc_map(params->api, CSINN_OP_DEPTHWISE_CONV2D_CHANNEL, input->dtype);
        } else {
            params->bc = csi_bc_map(params->api, CSINN_OP_GROUP_CONV2D_CHANNEL, input->dtype);
        }
        if (params->bc == NULL) {
                return CSINN_UNSUPPORT_DTYPE;
            }
        return CSINN_TRUE;
    }

    if (params->layout == CSINN_NCHW || params->layout == CSINN_NHWC) {
        if (params->group == 1) {
            params->bc = csi_bc_map(params->api, CSINN_OP_CONV2D, input->dtype);
        } else if (params->group == input->dim[1] || params->group == input->dim[3]) {
            params->bc = csi_bc_map(params->api, CSINN_OP_DEPTHWISE_CONV2D, input->dtype);
        } else {
            params->bc = csi_bc_map(params->api, CSINN_OP_GROUP_CONV2D, input->dtype);
        }
        if (params->bc == NULL) {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    return CSINN_TRUE;
}

int csi_conv2d(struct csi_tensor *input,
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
