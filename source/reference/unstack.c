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

/* SHL version 2.1.x */

#include "shl_ref.h"

int shl_ref_unstack_f32(struct csinn_tensor *input, struct csinn_tensor **output,
                        struct csinn_unstack_params *params)
{
    int axis = params->axis;
    int output_count = input->dim[axis];

    // For all output arrays,
    // FlatSize() = outer_size * base_inner_size;
    int64_t outer_size = 1;
    for (int i = 0; i < axis; ++i) {
        outer_size *= input->dim[i];
    }
    int64_t inner_size = 1;
    for (int i = axis + 1; i < input->dim_count; ++i) {
        inner_size *= input->dim[i];
    }

    int copy_size = inner_size;
    float *input_data = (float *)input->data;
    for (int i = 0; i < outer_size; i++) {
        for (int j = 0; j < output_count; j++) {
            struct csinn_tensor *output_item = output[j];
            float *output_item_data = (float *)output_item->data;
            float *output_ptr = output_item_data + i * copy_size;
            memcpy(output_ptr, input_data, copy_size * sizeof(float));
            input_data += copy_size;
        }
    }
    return CSINN_TRUE;
}

int shl_ref_unstack_qunat(struct csinn_tensor *input, struct csinn_tensor **output,
                          struct csinn_unstack_params *params)
{
    int ret;
    int axis = params->axis;
    int output_count = input->dim[axis];
    struct csinn_tensor *foutput[output_count];
    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input);
    for (int i = 0; i < output_count; i++) {
        foutput[i] = shl_ref_tensor_transform_f32(output[i]);
    }

    ret = shl_ref_unstack_f32(finput, foutput, params);

    for (int i = 0; i < output_count; i++) {
        csinn_tensor_data_convert(output[i], foutput[i]);
    }

    shl_ref_tensor_transform_free_f32(finput);
    for (int i = 0; i < output_count; i++) {
        shl_ref_tensor_transform_free_f32(foutput[i]);
    }
    return ret;
}
