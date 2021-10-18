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

int csi_stack_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct stack_params *params)
{
    int input_count = params->inputs_count;
    int axis = params->axis;

    // For all input arrays,
    // FlatSize() = outer_size * base_inner_size;
    int64_t outer_size = 1;
    for(int i = 0; i < axis; ++i) {
        outer_size *= output->dim[i];
    }
    int64_t inner_size = 1;
    for(int i = axis+1; i < output->dim_count; ++i) {
        inner_size *= output->dim[i];
    }

    int copy_size = inner_size;
    float *output_data = (float *)output->data;
    for(int i = 0; i < outer_size; ++i) {
        for(int j = 0; j < input_count; ++j) {
            struct csi_tensor *input_item = input + j;
            float *input_item_data = (float *)input_item->data;
            const float *input_ptr = input_item_data + i * copy_size;
            memcpy(output_data, input_ptr, copy_size * sizeof(float));
            output_data += copy_size;
        }
    }
    return CSINN_TRUE;
}

int csi_stack_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct stack_params *params)
{
    if (params->axis == -1){
        params->axis= input->dim_count -1;
    }
    int input_count = params->inputs_count;
    int axis = params->axis;

    // For all input arrays,
    // FlatSize() = outer_size * base_inner_size;
    int64_t outer_size = 1;
    for(int i = 0; i < axis; ++i) {
        outer_size *= output->dim[i];
    }
    int64_t inner_size = 1;
    for(int i = axis+1; i < output->dim_count; ++i) {
        inner_size *= output->dim[i];
    }

    int copy_size = inner_size;
    uint8_t *output_data = (uint8_t *)output->data;
    for(int i = 0; i < outer_size; ++i) {
        for(int j = 0; j < input_count; ++j) {
            struct csi_tensor *input_item = input + j;
            uint8_t *input_item_data = (uint8_t *)input_item->data;
            const uint8_t *input_ptr = input_item_data + i * copy_size;
            if(input_item->zero_point == output->zero_point &&
                input_item->multiplier == output->multiplier &&
                input_item->shift == output->shift) {
                memcpy(output_data, input_ptr, copy_size * sizeof(uint8_t));
            } else {
                for(int n = 0; n < copy_size; n++) {
                    output_data[j] = csi_requantize_u8(input_ptr[j], input_item->zero_point, input_item->multiplier, input_item->shift,
                                                                     output->zero_point, output->multiplier, output->shift);
                }
            }
            output_data += copy_size;
        }
    }
    return CSINN_TRUE;
}

int csi_stack_init(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct stack_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_STACK, input->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_stack(struct csi_tensor *input,
              struct csi_tensor *output,
              struct stack_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}