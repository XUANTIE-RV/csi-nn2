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
#include "csi_utils.h"

int csi_ref_prelu_f32(struct csi_tensor *input, struct csi_tensor *alpha, struct csi_tensor *output,
                      struct prelu_params *params)
{
    float *input_data = (float *)input->data;
    float *alpha_data = (float *)alpha->data;
    float *output_data = (float *)output->data;

    int axis = params->axis;

    int64_t outer_size = 1;
    for (int i = 0; i < axis; i++) {
        outer_size *= input->dim[i];
    }

    int64_t inner_size = (axis == 0 && input->dim_count == 1) ? csi_tensor_size(input) : 1;
    for (int i = axis + 1; i < input->dim_count; i++) {
        inner_size *= input->dim[i];
    }

    for (int i = 0; i < outer_size; i++) {
        for (int j = 0; j < input->dim[axis]; j++) {
            for (int k = 0; k < inner_size; k++) {
                int32_t index = i * inner_size * input->dim[axis] + j * inner_size + k;
                float input_value = input_data[index];
                if (input_value >= 0) {
                    output_data[index] = input_value;
                } else {
                    output_data[index] = input_value * alpha_data[j];
                }
            }
        }
    }
    return CSINN_TRUE;
}

int csi_ref_prelu_quant(struct csi_tensor *input, struct csi_tensor *alpha,
                        struct csi_tensor *output, struct prelu_params *params)
{
    return csi_ref_diso_callback_base(input, alpha, output, params, csi_ref_prelu_f32);
}
