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

static int csi_concat_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct concat_params *params)
{
    int64_t outer_size = 1;
    for (int i = 0; i < params->axis; ++i) {
        outer_size *= output->dim[i];
    }
    // For all input arrays,
    // FlatSize() = outer_size * Dims(axis) * base_inner_size;
    int64_t base_inner_size = 1;
    for (int i = params->axis + 1; i < output->dim_count; ++i) {
        base_inner_size *= output->dim[i];
    }

    float *output_ptr = output->data;
    for (int k = 0; k < outer_size; k++) {
        for (int i = 0; i < params->inputs_count; ++i) {
            struct csi_tensor *input_item = input + i;
            float *input_item_data = input_item->data;
            const int copy_size = input_item->dim[params->axis] * base_inner_size;
            const float *input_ptr = input_item_data + k * copy_size;
            memcpy(output_ptr, input_ptr, copy_size * sizeof(float));
            output_ptr += copy_size;
        }
    }
    return CSINN_TRUE;
}

static int csi_concat_u8(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct concat_params *params)
{
    if (params->axis == -1){
        params->axis= input->dim_count -1;
    }
    int64_t outer_size = 1;
    for (int i = 0; i < params->axis; ++i) {
        outer_size *= output->dim[i];
    }
    // For all input arrays,
    // FlatSize() = outer_size * Dims(axis) * base_inner_size;
    int64_t base_inner_size = 1;
    for (int i = params->axis + 1; i < output->dim_count; ++i) {
        base_inner_size *= output->dim[i];
    }

    uint8_t *output_ptr = output->data;
    for (int k = 0; k < outer_size; k++) {
        for (int i = 0; i < params->inputs_count; ++i) {
            struct csi_tensor *input_item = input + i;
            const int copy_size = input_item->dim[params->axis] * base_inner_size;
            uint8_t *input_item_data = input_item->data;
            const uint8_t *input_ptr = input_item_data + k * copy_size;
            if (input_item->offset == output->offset &&
                input_item->multiplier == output->multiplier &&
                input_item->shift == output->shift) {
                memcpy(output_ptr, input_ptr, copy_size);
            } else {
                for (int j = 0; j < copy_size; ++j) {
                    output_ptr[j] = csi_requantize_u8(input_ptr[j], input_item->offset,
                        input_item->multiplier, input_item->shift,
                        output->offset, output->multiplier, output->shift);
                }
            }
            output_ptr += copy_size;
        }
    }
    return CSINN_TRUE;
}

int csi_concat_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct concat_params *params)
{
    if (output->dtype == CSINN_DTYPE_UINT8) {
        params->bc = csi_concat_u8;
    } else if (output->dtype == CSINN_DTYPE_FLOAT32) {
        params->bc = csi_concat_f32;
    } else {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_concat(struct csi_tensor *input,
               struct csi_tensor *output,
               struct concat_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
