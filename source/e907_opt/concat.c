/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
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

#include "e907/e907.h"

int shl_e907_concat_int8(struct csinn_tensor **input, struct csinn_tensor *output,
                         struct csinn_concat_params *params)
{
    int32_t inputs_count = params->inputs_count;
    int32_t axis = params->axis;

    int64_t outer_size = 1;
    for (int i = 0; i < axis; ++i) {
        outer_size *= output->dim[i];
    }

    int64_t base_inner_size = 1;
    for (int i = axis + 1; i < output->dim_count; ++i) {
        base_inner_size *= output->dim[i];
    }

    int8_t *output_ptr = output->data;
    for (int k = 0; k < outer_size; k++) {
        for (int i = 0; i < inputs_count; ++i) {
            struct csinn_tensor *input_item = input[i];
            int8_t *input_item_data = input_item->data;
            const int copy_size = input_item->dim[axis] * base_inner_size;
            const int8_t *input_ptr = input_item_data + k * copy_size;
            memcpy(output_ptr, input_ptr, copy_size * sizeof(int8_t));
            output_ptr += copy_size;
        }
    }

    return CSINN_TRUE;
}
