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

int shl_ref_layer_norm_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_tensor *gamma, struct csinn_tensor *beta,
                           struct csinn_layer_norm_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *gamma_data = (float *)gamma->data;
    float *beta_data = (float *)beta->data;

    /* support negative axis */
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
        float *tmp = (float *)shl_mem_alloc(norm_size * sizeof(float));

        float mean = 0.0f;
        for (int i = 0; i < norm_size; i++) {
            mean += input_ptr[i];
        }
        mean /= norm_size;

        float sum = 0.0f;
        for (int i = 0; i < norm_size; i++) {
            tmp[i] = input_ptr[i] - mean;
            sum += tmp[i] * tmp[i];
        }
        float var = sum / norm_size;
        float std = sqrt(var + params->epsilon);

        for (int i = 0; i < norm_size; i++) {
            output_ptr[i] = tmp[i] / std * gamma_data[i] + beta_data[i];
        }
        shl_mem_free(tmp);
    }

    return CSINN_TRUE;
}

int shl_ref_layer_norm_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *gamma, struct csinn_tensor *beta,
                             struct csinn_layer_norm_params *params)
{
    struct csinn_tensor *float_input = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *float_output = shl_ref_tensor_transform_f32(output);
    struct csinn_tensor *float_gamma = shl_ref_tensor_transform_f32(gamma);
    struct csinn_tensor *float_beta = shl_ref_tensor_transform_f32(beta);

    int ret = shl_ref_layer_norm_f32(float_input, float_output, float_gamma, float_beta, params);

    csinn_tensor_data_convert(output, float_output);

    shl_ref_tensor_transform_free_f32(float_input);
    shl_ref_tensor_transform_free_f32(float_output);
    shl_ref_tensor_transform_free_f32(float_gamma);
    shl_ref_tensor_transform_free_f32(float_beta);

    return CSINN_TRUE;
}
