/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

#include "csi_ref.h"

int csi_ref_gather_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct gather_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int inner_size = 1;
    for(int i = 1; i < input->dim_count; i++) {
        inner_size *= input->dim[i];
    }

    for(int i = 0; i < params->indices_count; i++) {
        if(params->indices[i] < input->dim[0]) {
            memcpy(output_data, input_data + params->indices[i] * inner_size, inner_size * sizeof(float));
        } else {
            for(int j = 0; j < inner_size; j++) {
                *(output_data + j) = 0.0f;
            }
        }
        output_data += inner_size;
    }
    return CSINN_TRUE;
}

int csi_ref_gather_quant(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct gather_params *params)
{
    return csi_ref_siso_callback_base(input, output, params, csi_ref_gather_f32);
}
