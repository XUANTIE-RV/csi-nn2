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

static int csi_lrn_nhwc_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct lrn_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    const int trailing_dim = input->dim_count - 1;
    int outer_size = 1;
    const int depth = input->dim[trailing_dim];
    int half_range = params->range / 2;

    for (int i = 0; i < trailing_dim; i++) {
        outer_size *= input->dim[i];
    }

    for (int i = 0; i < outer_size; ++i) {
        for (int c = 0; c < depth; ++c) {
            const int begin_input_c = csi_max_internal_s32(0, c - half_range);
            const int end_input_c = csi_min_internal_s32(depth, c + half_range + 1);
            float accum = 0.f;
            for (int input_c = begin_input_c; input_c < end_input_c; ++input_c) {
                const float input_val = input_data[i * depth + input_c];
                accum += input_val * input_val;
            }
            const float multiplier = pow(params->bias + params->alpha * accum / params->range, -params->beta);
            output_data[i * depth + c] = input_data[i * depth + c] * multiplier;
        }
    }
    return CSINN_TRUE;
}

static int csi_lrn_nhwc_u8(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct lrn_params *params)
{
    float *float_input_data;
    float *float_output_data;
    struct csi_tensor float_input;
    struct csi_tensor float_output;
    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    double bias_f, alpha_f, beta_f;
    int size = 1;

    for (int i = 0; i < input->dim_count; i++) {
        size *= input->dim[i];
    }

    memcpy(&float_input, input, sizeof(struct csi_tensor));
    memcpy(&float_output, output, sizeof(struct csi_tensor));
    float_input_data = malloc(size * sizeof(float));
    float_output_data = malloc(size * sizeof(float));
    float_input.data = float_input_data;
    float_output.data = float_output_data;
    float_input.dtype = CSINN_DTYPE_FLOAT32;
    float_output.dtype = CSINN_DTYPE_FLOAT32;

    for (int i = 0; i < size; i++) {
        float_input_data[i] = csi_dequantize_f32(input_data[i], input->offset,
                                                 input->multiplier, input->shift);
    }

    bias_f = csi_dequantize_f32(1, 0, params->bias_multiplier, params->bias_shift);
    alpha_f = csi_dequantize_f32(1, 0, params->alpha_multiplier, params->alpha_shift);
    beta_f = csi_dequantize_f32(1, 0, params->beta_multiplier, params->beta_shift);

    params->bias = bias_f;
    params->alpha = alpha_f;
    params->beta = beta_f;

    csi_lrn_nhwc_f32(&float_input, &float_output, params);

    for (int i = 0; i < size; i++) {
        output_data[i] = csi_quantize_f32(float_output_data[i], output->offset,
                                          output->multiplier, output->shift);
    }
    free(float_input_data);
    free(float_output_data);
    return CSINN_TRUE;
}

static int csi_lrn_nchw_f32(struct csi_tensor *o_input,
                            struct csi_tensor *o_output,
                            struct lrn_params *params)
{
    struct csi_tensor* input;
    struct csi_tensor* output;
    input =  csi_nchw_to_nhwc_f32(o_input);
    output = csi_nchw_to_nhwc_f32(o_output);

    float *input_data = input->data;
    float *output_data = output->data;
    const int trailing_dim = input->dim_count - 1;
    int outer_size = 1;
    const int depth = input->dim[trailing_dim];
    int half_range = params->range / 2;

    for (int i = 0; i < trailing_dim; i++) {
        outer_size *= input->dim[i];
    }

    for (int i = 0; i < outer_size; ++i) {
        for (int c = 0; c < depth; ++c) {
            const int begin_input_c = csi_max_internal_s32(0, c - half_range);
            const int end_input_c = csi_min_internal_s32(depth, c + half_range + 1);
            float accum = 0.f;
            for (int input_c = begin_input_c; input_c < end_input_c; ++input_c) {
                const float input_val = input_data[i * depth + input_c];
                accum += input_val * input_val;
            }
            const float multiplier = pow(params->bias + params->alpha * accum / params->range, -params->beta);
            output_data[i * depth + c] = input_data[i * depth + c] * multiplier;
        }
    }
    csi_nhwc_to_nchw_f32(o_output, output);
    return CSINN_TRUE;
}

static int csi_lrn_nchw_u8(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct lrn_params *params)
{
    float *float_input_data;
    float *float_output_data;
    struct csi_tensor float_input;
    struct csi_tensor float_output;
    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    double bias_f, alpha_f, beta_f;
    int size = 1;

    for (int i = 0; i < input->dim_count; i++) {
        size *= input->dim[i];
    }

    memcpy(&float_input, input, sizeof(struct csi_tensor));
    memcpy(&float_output, output, sizeof(struct csi_tensor));
    float_input_data = malloc(size * sizeof(float));
    float_output_data = malloc(size * sizeof(float));
    float_input.data = float_input_data;
    float_output.data = float_output_data;
    float_input.dtype = CSINN_DTYPE_FLOAT32;
    float_output.dtype = CSINN_DTYPE_FLOAT32;

    for (int i = 0; i < size; i++) {
        float_input_data[i] = csi_dequantize_f32(input_data[i], input->offset,
                                                 input->multiplier, input->shift);
    }

    bias_f = csi_dequantize_f32(1, 0, params->bias_multiplier, params->bias_shift);
    alpha_f = csi_dequantize_f32(1, 0, params->alpha_multiplier, params->alpha_shift);
    beta_f = csi_dequantize_f32(1, 0, params->beta_multiplier, params->beta_shift);

    params->bias = bias_f;
    params->alpha = alpha_f;
    params->beta = beta_f;

    csi_lrn_nchw_f32(&float_input, &float_output, params);

    for (int i = 0; i < size; i++) {
        output_data[i] = csi_quantize_f32(float_output_data[i], output->offset,
                                          output->multiplier, output->shift);
    }
    free(float_input_data);
    free(float_output_data);
    return CSINN_TRUE;
}



int csi_lrn_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct lrn_params *params)
{
    if (params->layout == CSINN_NCHW) {
        if (input->dtype == CSINN_DTYPE_UINT8) {
            params->bc = csi_lrn_nchw_u8;
        } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
            params->bc = csi_lrn_nchw_f32;
        } else {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else if (params->layout = CSINN_NHWC) {
        if (input->dtype == CSINN_DTYPE_UINT8) {
            params->bc = csi_lrn_nhwc_u8;
        } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
            params->bc = csi_lrn_nhwc_f32;
        } else {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    return CSINN_TRUE;
}

int csi_lrn(struct csi_tensor *input,
             struct csi_tensor *output,
             struct lrn_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}