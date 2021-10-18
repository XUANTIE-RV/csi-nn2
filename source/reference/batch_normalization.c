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

/* https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/nn_impl.py#L1474-L1542 */
int csi_batch_normalization_f32(struct csi_tensor *input,
                                struct csi_tensor *mean,
                                struct csi_tensor *variance,
                                struct csi_tensor *gamma,
                                struct csi_tensor *beta,
                                struct csi_tensor *output,
                                struct bn_params *params)
{
    float *input_data = input->data;
    float *mean_data  = mean->data;
    float *var_data   = variance->data;
    float *beta_data  = beta->data;
    float *output_data = output->data;
    const int dims_count = input->dim_count;
    int batches = 1;

    /* compute the outer size */
    for(int i = 0; i < dims_count - 1; i++ ){
        batches *= input->dim[i];
    }

    int batch_offset = input->dim[dims_count - 1];

    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < input->dim[dims_count - 1]; ++c) {
            float intput_val = input_data[b * batch_offset + c];
            float mean_val   = mean_data[c];
            float var_val    = var_data[c];
            float beta_val   = beta_data[c];
            float result = 1/sqrt(var_val + params->epsilon);
            result *= (intput_val - mean_val);
            if (gamma != NULL) {
                float *gamma_data  = gamma->data;
                result *=  gamma_data[c];
            }
            result += beta_val;
            output_data[b * batch_offset + c] = result;
        }
    }

    return CSINN_TRUE;
}

int csi_batch_normalization_u8(struct csi_tensor *input,
                               struct csi_tensor *mean,
                               struct csi_tensor *variance,
                               struct csi_tensor *gamma,
                               struct csi_tensor *beta,
                               struct csi_tensor *output,
                               struct bn_params *params)
{
    uint8_t *input_data  = input->data;
    uint8_t *mean_data   = mean->data;
    uint8_t *var_data    = variance->data;
    uint8_t *beta_data   = beta->data;
    uint8_t *output_data = output->data;
    const int dims_count = input->dim_count;
    int batches = 1;

    /* compute the outer size */
    for(int i = 0; i < dims_count - 1; i++ ){
        batches *= input->dim[i];
    }

    int batch_offset = input->dim[dims_count - 1];

    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < input->dim[dims_count - 1]; ++c) {
            float intput_val = csi_dequantize_u8_to_f32(input_data[b * batch_offset + c], input->zero_point,
                                            input->multiplier, input->shift);
            float mean_val   = csi_dequantize_u8_to_f32(mean_data[c], mean->zero_point, mean->multiplier,
                                            mean->shift);
            float var_val    = csi_dequantize_u8_to_f32(var_data[c], variance->zero_point, variance->multiplier,
                                            variance->shift);
            float beta_val   = csi_dequantize_u8_to_f32(beta_data[c], beta->zero_point, beta->multiplier,
                                            beta->shift);
            float result = 1/sqrt(var_val + params->epsilon);
            result *= (intput_val - mean_val);
            if (gamma != NULL) {
                uint8_t *gamma_data  = gamma->data;
                result *=  csi_dequantize_u8_to_f32(gamma_data[c], gamma->zero_point, gamma->multiplier,
                                            gamma->shift);
            }
            result += beta_val;
            output_data[b * batch_offset + c] = csi_quantize_f32_to_u8(result, output->zero_point,
                                            output->multiplier, output->shift);
        }
    }

    return CSINN_TRUE;
}


int csi_batch_normalization_init(struct csi_tensor *input,
                                 struct csi_tensor *mean,
                                 struct csi_tensor *variance,
                                 struct csi_tensor *gamma,
                                 struct csi_tensor *beta,
                                 struct csi_tensor *output,
                                 struct bn_params *params)
{
    if (params->layout == CSINN_NCHW) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    params->bc = csi_bc_map(params->api, CSINN_OP_BN, input->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }

    return CSINN_TRUE;
}

int csi_batch_normalization(struct csi_tensor *input,
                            struct csi_tensor *mean,
                            struct csi_tensor *variance,
                            struct csi_tensor *gamma,
                            struct csi_tensor *beta,
                            struct csi_tensor *output,
                            struct bn_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, mean, variance, gamma, beta, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}