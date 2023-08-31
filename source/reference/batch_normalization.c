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

#include "reference/ref.h"

/* https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/nn_impl.py#L1474-L1542
 */
int shl_ref_batch_normalization_f32(struct csinn_tensor *input, struct csinn_tensor *mean,
                                    struct csinn_tensor *variance, struct csinn_tensor *gamma,
                                    struct csinn_tensor *beta, struct csinn_tensor *output,
                                    struct csinn_bn_params *params)
{
    float *input_data = input->data;
    float *mean_data = mean->data;
    float *var_data = variance->data;
    float *beta_data = beta->data;
    float *output_data = output->data;
    const int dims_count = input->dim_count;
    int batches = 1;

    /* compute the outer size */
    for (int i = 0; i < dims_count - 1; i++) {
        batches *= input->dim[i];
    }

    int batch_offset = input->dim[dims_count - 1];

    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < input->dim[dims_count - 1]; ++c) {
            float intput_val = input_data[b * batch_offset + c];
            float mean_val = mean_data[c];
            float var_val = var_data[c];
            float beta_val = beta_data[c];
            float result = 1 / sqrt(var_val + params->epsilon);
            result *= (intput_val - mean_val);
            if (gamma != NULL) {
                float *gamma_data = gamma->data;
                result *= gamma_data[c];
            }
            result += beta_val;
            output_data[b * batch_offset + c] = result;
        }
    }

    return CSINN_TRUE;
}

int shl_ref_batch_normalization_quant(struct csinn_tensor *input, struct csinn_tensor *mean,
                                      struct csinn_tensor *variance, struct csinn_tensor *gamma,
                                      struct csinn_tensor *beta, struct csinn_tensor *output,
                                      struct csinn_bn_params *params)
{
    int ret;
    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *fmean = shl_ref_tensor_transform_f32(mean);
    struct csinn_tensor *fvariance = shl_ref_tensor_transform_f32(variance);
    struct csinn_tensor *fgamma = shl_ref_tensor_transform_f32(gamma);
    struct csinn_tensor *fbeta = shl_ref_tensor_transform_f32(beta);
    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    ret = shl_ref_batch_normalization_f32(finput, fmean, fvariance, fgamma, fbeta, foutput, params);
    csinn_tensor_data_convert(output, foutput);
    shl_ref_tensor_transform_free_f32(finput);
    shl_ref_tensor_transform_free_f32(fmean);
    shl_ref_tensor_transform_free_f32(fvariance);
    shl_ref_tensor_transform_free_f32(fgamma);
    shl_ref_tensor_transform_free_f32(fbeta);
    shl_ref_tensor_transform_free_f32(foutput);
    return ret;
}
