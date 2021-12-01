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

/* CSI-NN2 version 1.8.x */

#include "csi_ref.h"

int csi_ref_concat_f32(struct csi_tensor **input,
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
            memcpy(output_ptr, input_ptr, copy_size * sizeof(float));
            output_ptr += copy_size;
        }
    }
    return CSINN_TRUE;
}

int csi_ref_concat_quant(struct csi_tensor **input,
                         struct csi_tensor *output,
                         struct concat_params *params)
{
    if (params->axis == -1){
        params->axis = input[0]->dim_count - 1;
    }

    int input_count = params->inputs_count;
    int ret;

    struct csi_tensor *finput[input_count];
    struct csi_tensor *foutput = csi_ref_tensor_transform_f32(output);
    for (int i = 0; i < input_count; i++) {
        finput[i] = csi_ref_tensor_transform_f32(input[i]);
    }

    ret = csi_ref_concat_f32(finput, foutput, params);

    csi_tensor_data_convert(output, foutput);

    csi_ref_tensor_transform_free_f32(foutput);
    for (int i = 0; i < input_count; i++) {
        csi_ref_tensor_transform_free_f32(finput[i]);
    }
    return ret;
}
