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

/* CSI-NN2 version 1.9.x */

#include "csi_c906.h"


int csi_c906_concat_f32(struct csi_tensor **input,
                        struct csi_tensor *output,
                        struct concat_params *params)
{
    int64_t outer_size = 1;
    for (int i = 0; i < params->axis; ++i) {
        outer_size *= output->dim[i];
    }

    int64_t base_inner_size = 1;
    for (int i = params->axis + 1; i < output->dim_count; ++i) {
        base_inner_size *= output->dim[i];
    }

    float *output_ptr = output->data;
    for (int k = 0; k < outer_size; k++) {
        for (int i = 0; i < params->inputs_count; ++i) {
            struct csi_tensor *input_item = input[i];
            float *input_item_data = input_item->data;
            const int copy_size = input_item->dim[params->axis] * base_inner_size;
            const float *input_ptr = input_item_data + k * copy_size;
            csi_c906_memcpy(output_ptr, input_ptr, copy_size * sizeof(float));
            output_ptr += copy_size;
        }
    }
    return CSINN_TRUE;
}


int csi_c906_concat_fp16(struct csi_tensor **input,
                         struct csi_tensor *output,
                         struct concat_params *params)
{
    int64_t outer_size = 1;
    for (int i = 0; i < params->axis; ++i) {
        outer_size *= output->dim[i];
    }

    int64_t base_inner_size = 1;
    for (int i = params->axis + 1; i < output->dim_count; ++i) {
        base_inner_size *= output->dim[i];
    }

    __fp16 *output_ptr = output->data;
    for (int k = 0; k < outer_size; k++) {
        for (int i = 0; i < params->inputs_count; ++i) {
            struct csi_tensor *input_item = input[i];
            __fp16 *input_item_data = input_item->data;
            const int copy_size = input_item->dim[params->axis] * base_inner_size;
            const __fp16 *input_ptr = input_item_data + k * copy_size;
            csi_c906_memcpy(output_ptr, input_ptr, copy_size * sizeof(__fp16));
            output_ptr += copy_size;
        }
    }
    return CSINN_TRUE;
}
