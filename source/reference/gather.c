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


static int csi_gather_f32(struct csi_tensor *input,
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

static int csi_gather_u8(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct gather_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;

    int inner_size = 1;
    for(int i = 1; i < input->dim_count; i++) {
        inner_size *= input->dim[i];
    }

    for(int i = 0; i < params->indices_count; i++) {
        if(params->indices[i] < input->dim[0]) {
            for(int j = 0; j < inner_size; j++) {
                *(output_data + j) = csi_requantize_u8(*(input_data + params->indices[i] * inner_size + j),
                input->offset, input->multiplier, input->shift, output->offset, output->multiplier, output->shift);
            }
        } else {
            uint8_t zero = csi_requantize_u8(0.0f, input->offset, input->multiplier, input->shift, 
                                                    output->offset, output->multiplier, output->shift);
            for(int j = 0; j < inner_size; j++) {
                *(output_data + j) = zero;
            }
        }
        output_data += inner_size;
    }
    return CSINN_TRUE;
}

int csi_gather_init(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct gather_params *params)
{
    if (input->dtype == CSINN_DTYPE_UINT8) {
        params->bc = csi_gather_u8;
    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        params->bc = csi_gather_f32;
    } else {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_gather(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct gather_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}

