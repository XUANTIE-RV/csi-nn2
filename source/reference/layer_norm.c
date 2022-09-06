/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

/* CSI-NN2 version 2.0.x */

#include "shl_ref.h"

int shl_ref_layer_norm_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_tensor *gamma, struct csinn_tensor *beta,
                           struct csinn_layer_norm_params *params)
{
    int flatten_size = 0;
    flatten_size *= input->dim[0] * input->dim[1] * input->dim[2];

    float *sum = (float *)calloc(input->dim[1], sizeof(float));
    float *sum2 = (float *)calloc(input->dim[1], sizeof(float));
    float *input_data = input->data;
    float *output_data = output->data;
    float *gamma_data = gamma->data;
    float *beta_data = beta->data;

    for (int i = 0; i < input->dim[1]; i++) {
        for (int j = 0; j < input->dim[2]; j++) {
            sum[i] += input_data[j + i * input->dim[2]];
        }
        sum[i] /= input->dim[2];
    }

    for (int i = 0; i < input->dim[1]; i++) {
        for (int j = 0; j < input->dim[2]; j++) {
            input_data[j + i * input->dim[2]] -= sum[i];
            output_data[j + i * input->dim[2]] = input_data[j + i * input->dim[2]];

            input_data[j + i * input->dim[2]] =
                input_data[j + i * input->dim[2]] * input_data[j + i * input->dim[2]];
            sum2[i] += input_data[j + i * input->dim[2]];
        }
        sum2[i] /= input->dim[2];
        sum2[i] = sqrtf(sum2[i]);
    }

    for (int i = 0; i < input->dim[1]; i++) {
        for (int j = 0; j < input->dim[2]; j++) {
            output_data[j + i * input->dim[2]] =
                output_data[j + i * input->dim[2]] / sum2[i] * gamma_data[j] + beta_data[j];
        }
    }

    free(sum);
    free(sum2);

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
