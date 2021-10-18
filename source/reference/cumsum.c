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
#include <assert.h>


static int csi_cumsum_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct cumsum_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int axis = params->axis;

    // For all input arrays,
    // FlatSize() = outer_size * inner_size * cnt;
    int64_t outer_size = 1;
    for(int i = 0; i < axis; i++) {
        outer_size *= input->dim[i];
    }
    int64_t inner_size = 1;
    for(int i = axis + 1; i < input->dim_count; i++) {
        inner_size *= input->dim[i];
    }
    int cnt = input->dim[axis];

    for(int i = 0; i < outer_size; i++) {
        for(int k = 0; k < inner_size; k++) {
            float temp = 0.0f;
            for(int j = 0; j < cnt; j++) {
                temp += *(input_data + j * inner_size + k);
                if(!params->exclusive) {
                    *(output_data + j * inner_size + k) = temp;
                } else {
                    *(output_data + j * inner_size + k) = temp - *(input_data + j * inner_size + k);
                }
            }
        }
        input_data += inner_size * cnt;
        output_data += inner_size * cnt;
    }
    return CSINN_TRUE;
}

static int csi_cumsum_u8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct cumsum_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;

    int axis = params->axis;

    // For all input arrays,
    // FlatSize() = outer_size * inner_size * cnt;
    int64_t outer_size = 1;
    for(int i = 0; i < axis; i++) {
        outer_size *= input->dim[i];
    }
    int64_t inner_size = 1;
    for(int i = axis + 1; i < input->dim_count; i++) {
        inner_size *= input->dim[i];
    }
    int cnt = input->dim[axis];

    for(int i = 0; i < outer_size; i++) {
        for(int k = 0; k < inner_size; k++) {
            float temp = 0.0f;
            for(int j = 0; j < cnt; j++) {
                uint8_t input_val = *(input_data + j * inner_size + k);
                float input_temp = csi_dequantize_f32(input_val, input->offset, input->multiplier, input->shift);
                temp *= input_temp;
                float output_temp = temp;
                if(!params->exclusive) {
                    *(output_data + j * inner_size + k) = csi_quantize_f32(output_temp, output->offset, output->multiplier, output->shift);
                } else {
                    *(output_data + j * inner_size + k) = csi_quantize_f32(output_temp - input_temp, output->offset, output->multiplier, output->shift);
                }
            }
        }
        input_data += inner_size * cnt;
        output_data += inner_size * cnt;
    }
    return CSINN_TRUE;
}

int csi_cumsum_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct cumsum_params *params)
{
    if (input->dtype == CSINN_DTYPE_UINT8) {
        params->bc = csi_cumsum_u8;
    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        params->bc = csi_cumsum_f32;
    } else {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_cumsum(struct csi_tensor *input,
                struct csi_tensor *output,
                struct cumsum_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}