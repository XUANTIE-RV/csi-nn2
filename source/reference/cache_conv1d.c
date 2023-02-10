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

#include "shl_ref.h"

int shl_ref_cache_conv1d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *weight, struct csinn_tensor *bias,
                              struct csinn_cache_conv1d_params *params)
{
    size_t data_size =
        output->dim[0] * output->dim[1] * output->dim[2] * sizeof(float);  // 512*13*2
    asr_buffer_init(&params->asr_buffer, 2 * data_size, data_size);

    struct csinn_callback *cb = params->base.cb;
    cb->exec = shl_ref_cache_conv1d_quant;
    return CSINN_TRUE;
}

int shl_ref_cache_conv1d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *weight, struct csinn_tensor *bias,
                             struct csinn_cache_conv1d_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    float *weights_data = weight->data;
    float *bias_data = bias->data;
    const int weights_dims_count = weight->dim_count;
    const int output_depth = weight->dim[weights_dims_count - 3];
    const int accum_depth = weight->dim[weights_dims_count - 2];
    const int batches = input->dim[1];
    for (int b = 0; b < batches; ++b) {
        for (int out_c = 0; out_c < output_depth; ++out_c) {
            float total = 0.f;
            for (int d = 0; d < accum_depth; ++d) {
                total += input_data[b * accum_depth + d] * weights_data[out_c * accum_depth + d];
            }
            float bias_value = 0.0f;
            if (bias->dim_count != 0) {
                bias_value = bias_data[out_c];
            }
            output_data[out_c + output_depth * b] = total + bias_value;
        }
    }
    size_t insert_lenth = output->dim[1] * input->dim[1];
    float *output_from_buffer;
    output_from_buffer =
        asr_buffer_insert_back(&params->asr_buffer, output_data, insert_lenth * sizeof(float));
    size_t output_lenth = output->dim[0] * output->dim[1] * output->dim[2];
    int32_t *shape = output->dim;
    for (int i = 0; i < shape[2]; i++) {
        int j = 0;
        for (; j < shape[1]; j++) {
            int out_pos = j * shape[2] + i;
            output_data[out_pos] = output_from_buffer[i * shape[1] + j];
        }
    }
    return CSINN_TRUE;
}

int shl_ref_cache_conv1d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *weight, struct csinn_tensor *bias,
                               struct csinn_cache_conv1d_params *params)
{
    struct csinn_tensor *float_input = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *float_output = shl_ref_tensor_transform_f32(output);
    struct csinn_tensor *float_weight = shl_ref_tensor_transform_f32(weight);
    struct csinn_tensor *float_bias = shl_ref_tensor_transform_f32(bias);

    int ret = shl_ref_cache_conv1d_f32(float_input, float_output, float_weight, float_bias, params);

    csinn_tensor_data_convert(output, float_output);

    shl_ref_tensor_transform_free_f32(float_input);
    shl_ref_tensor_transform_free_f32(float_output);
    shl_ref_tensor_transform_free_f32(float_weight);
    shl_ref_tensor_transform_free_f32(float_bias);

    return CSINN_TRUE;
}
