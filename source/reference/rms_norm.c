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

int shl_ref_rms_norm_f32(struct csinn_tensor *input, struct csinn_tensor *weight,
                         struct csinn_tensor *output, struct csinn_rms_norm_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *weight_data = (float *)weight->data;
    float eps = params->epsilon;

    int axis = params->axis >= 0 ? params->axis : (params->axis + input->dim_count);

    int32_t batches = 1;
    for (int i = 0; i < axis; i++) {
        batches *= input->dim[i];
    }
    int32_t norm_size = 1;
    for (int i = axis; i < input->dim_count; i++) {
        norm_size *= input->dim[i];
    }

    for (int b = 0; b < batches; b++) {
        float *input_ptr = input_data + b * norm_size;
        float *output_ptr = output_data + b * norm_size;

        float sum = 0.0f;
        for (int i = 0; i < norm_size; i++) {
            sum += input_ptr[i] * input_ptr[i];
        }

        float scale = 1.0 / sqrt(sum / norm_size + eps);

        for (int i = 0; i < norm_size; i++) {
            output_ptr[i] = input_ptr[i] * scale * weight_data[i];
        }
    }

    return CSINN_TRUE;
}

int shl_ref_rms_norm_quant(struct csinn_tensor *input, struct csinn_tensor *weight,
                           struct csinn_tensor *output, struct csinn_rms_norm_params *params)
{
    struct csinn_tensor *float_input = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *float_output = shl_ref_tensor_transform_f32(output);
    struct csinn_tensor *float_gamma = shl_ref_tensor_transform_f32(weight);

    int ret = shl_ref_rms_norm_f32(float_input, float_gamma, float_output, params);

    csinn_tensor_data_convert(output, float_output);

    shl_ref_tensor_transform_free_f32(float_input);
    shl_ref_tensor_transform_free_f32(float_output);
    shl_ref_tensor_transform_free_f32(float_gamma);

    return CSINN_TRUE;
}