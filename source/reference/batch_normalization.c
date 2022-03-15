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

/* CSI-NN2 version 1.12.x */

#include "csi_ref.h"

/* https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/nn_impl.py#L1474-L1542
 */
int csi_ref_batch_normalization_f32(struct csi_tensor *input, struct csi_tensor *mean,
                                    struct csi_tensor *variance, struct csi_tensor *gamma,
                                    struct csi_tensor *beta, struct csi_tensor *output,
                                    struct bn_params *params)
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

int csi_ref_batch_normalization_quant(struct csi_tensor *input, struct csi_tensor *mean,
                                      struct csi_tensor *variance, struct csi_tensor *gamma,
                                      struct csi_tensor *beta, struct csi_tensor *output,
                                      struct bn_params *params)
{
    int ret;
    struct csi_tensor *finput = csi_ref_tensor_transform_f32(input);
    struct csi_tensor *fmean = csi_ref_tensor_transform_f32(mean);
    struct csi_tensor *fvariance = csi_ref_tensor_transform_f32(variance);
    struct csi_tensor *fgamma = csi_ref_tensor_transform_f32(gamma);
    struct csi_tensor *fbeta = csi_ref_tensor_transform_f32(beta);
    struct csi_tensor *foutput = csi_ref_tensor_transform_f32(output);
    ret = csi_ref_batch_normalization_f32(finput, fmean, fvariance, fgamma, fbeta, foutput, params);
    csi_tensor_data_convert(output, foutput);
    csi_ref_tensor_transform_free_f32(finput);
    csi_ref_tensor_transform_free_f32(fmean);
    csi_ref_tensor_transform_free_f32(fvariance);
    csi_ref_tensor_transform_free_f32(fgamma);
    csi_ref_tensor_transform_free_f32(fbeta);
    csi_ref_tensor_transform_free_f32(foutput);
    return ret;
}
