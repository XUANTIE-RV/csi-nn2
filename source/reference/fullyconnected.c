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

static int csi_fullyconnected_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct csi_tensor *weights,
                            struct csi_tensor *bias,
                            struct fc_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    float *weights_data = weights->data;
    float *bias_data = bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int batches = output->dim[0];
    const int output_depth = weights->dim[weights_dims_count - 2];
    const int accum_depth = weights->dim[weights_dims_count - 1];
    for (int b = 0; b < batches; ++b) {
        for (int out_c = 0; out_c < output_depth; ++out_c) {
            float total = 0.f;
            for (int d = 0; d < accum_depth; ++d) {
                total += input_data[b * accum_depth + d] * weights_data[out_c * accum_depth + d];
            }
            float bias_value = 0.0f;
            if (bias_data != NULL) {
                bias_value = bias_data[out_c];
            }
            output_data[out_c + output_depth * b] = total + bias_value;
        }
    }
    return CSINN_TRUE;
}

static int csi_fullyconnected_u8(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct csi_tensor *weights,
                            struct csi_tensor *bias,
                            struct fc_params *params)
{
    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    uint8_t *weights_data = weights->data;
    int32_t *bias_data = bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int batches = output->dim[0];
    const int output_depth = weights->dim[weights_dims_count - 2];
    const int accum_depth = weights->dim[weights_dims_count - 1];
    for (int b = 0; b < batches; ++b) {
        #pragma omp parallel for num_threads(8)
        for (int out_c = 0; out_c < output_depth; ++out_c) {
            int32_t acc = 0;
            for (int d = 0; d < accum_depth; ++d) {
                int32_t input_val = input_data[b * accum_depth + d];
                int32_t filter_val = weights_data[out_c * accum_depth + d];
                acc += (filter_val + weights->offset) * (input_val + input->offset);
            }
            if (bias_data != NULL) {
                acc += bias_data[out_c];
            }

            output_data[out_c + output_depth * b] =
                csi_quantize_u8(acc, output->offset, output->multiplier, output->shift);
        }
    }
    return CSINN_TRUE;
}

int csi_fullyconnected_init(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct csi_tensor *weights,
                            struct csi_tensor *bias,
                            struct fc_params *params)
{
    if (input->dtype == CSINN_DTYPE_UINT8) {
        params->bc = csi_fullyconnected_u8;
    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        params->bc = csi_fullyconnected_f32;
    } else {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_fullyconnected(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct csi_tensor *weights,
                       struct csi_tensor *bias,
                       struct fc_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, weights, bias, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}