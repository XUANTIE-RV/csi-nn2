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

int csi_ref_fullyconnected_f32(struct csi_tensor *input, struct csi_tensor *output,
                               struct csi_tensor *weights, struct csi_tensor *bias,
                               struct fc_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    float *weights_data = weights->data;
    float *bias_data = bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    int batches = 1;
    /* compute the outer size */
    for (int i = 0; i < output_dims_count - 1; i++) {
        batches *= output->dim[i];
    }
    const int output_depth = weights->dim[weights_dims_count - 2];
    const int accum_depth = weights->dim[weights_dims_count - 1];
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
    return CSINN_TRUE;
}

int csi_ref_fullyconnected_quant(struct csi_tensor *input, struct csi_tensor *output,
                                 struct csi_tensor *weights, struct csi_tensor *bias,
                                 struct fc_params *params)
{
    struct csi_tensor *float_input = csi_ref_tensor_transform_f32(input);
    struct csi_tensor *float_kernel = csi_ref_tensor_transform_f32(weights);
    struct csi_tensor *float_bias = csi_ref_tensor_transform_f32(bias);
    struct csi_tensor *float_output = csi_ref_tensor_transform_f32(output);
    if (params->fc_extra.fuse_zp2bias) {
        float *float_bias_data = float_bias->data;
        float *float_kernel_data = float_kernel->data;

        int k_len = weights->dim[0];
        int k_inner = csi_tensor_size(weights) / k_len;
        float sp = input->qinfo->scale * input->qinfo->zero_point;
        for (int i = 0; i < k_len; i++) {
            float t_k = 0;
            for (int j = 0; j < k_inner; j++) {
                int k_idx = i * k_inner + j;
                t_k += float_kernel_data[k_idx] * sp;
            }
            float_bias_data[i] += t_k;
        }
    }

    int ret =
        csi_ref_fullyconnected_f32(float_input, float_output, float_kernel, float_bias, params);
    csi_tensor_data_convert(output, float_output);
    csi_ref_tensor_transform_free_f32(float_input);
    csi_ref_tensor_transform_free_f32(float_output);
    csi_ref_tensor_transform_free_f32(float_kernel);
    csi_ref_tensor_transform_free_f32(float_bias);
    return ret;
}
