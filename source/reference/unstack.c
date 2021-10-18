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

int csi_unstack_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct unstack_params *params)
{
    int axis = params->axis;
    int output_count = input->dim[axis];

    // For all output arrays,
    // FlatSize() = outer_size * base_inner_size;
    int64_t outer_size = 1;
    for(int i = 0; i < axis; ++i) {
        outer_size *= input->dim[i];
    }
    int64_t inner_size = 1;
    for(int i = axis+1; i < input->dim_count; ++i) {
        inner_size *= input->dim[i];
    }

    int copy_size = inner_size;
    float *input_data = (float *)input->data;
    for(int i = 0; i < outer_size; i++) {
        for(int j = 0; j < output_count; j++) {
            struct csi_tensor *output_item = output + j;
            float *output_item_data = (float *)output_item->data;
            float *output_ptr = output_item_data + i * copy_size;
            memcpy(output_ptr, input_data, copy_size * sizeof(float));
            input_data += copy_size;
        }
    }
    return CSINN_TRUE;
}

int csi_unstack_u8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct unstack_params *params)
{
    int axis = params->axis;
    int output_count = input->dim[axis];

    // For all output arrays,
    // FlatSize() = outer_size * base_inner_size;
    int64_t outer_size = 1;
    for(int i = 0; i < axis; ++i) {
        outer_size *= input->dim[i];
    }
    int64_t inner_size = 1;
    for(int i = axis+1; i < input->dim_count; ++i) {
        inner_size *= input->dim[i];
    }

    int copy_size = inner_size;
    float *input_data = (float *)input->data;
    for(int i = 0; i < outer_size; i++) {
        for(int j = 0; j < output_count; j++) {
            struct csi_tensor *output_item = output + j;
            float *output_item_data = (float *)output_item->data;
            float *output_ptr = output_item_data + i * copy_size;
            if(output_item->zero_point == input->zero_point &&
                output_item->multiplier == input->multiplier &&
                output_item->shift == input->shift) {
                memcpy(output_ptr, input_data, copy_size * sizeof(float));
            } else {
                for(int n = 0; n < copy_size; n++) {
                    output_ptr[j] = csi_requantize_u8(input_data[j], input->zero_point, input->multiplier, input->shift,
                                                                     output_item->zero_point, output_item->multiplier, output_item->shift);
                }
            }
            input_data += copy_size;
        }
    }
    return CSINN_TRUE;
}

int csi_unstack_init(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct unstack_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_UNSTACK, input->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_unstack(struct csi_tensor *input,
                struct csi_tensor *output,
                struct unstack_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}