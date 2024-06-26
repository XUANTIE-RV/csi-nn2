/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

#include "reference/ref.h"

/* from tensorflow/lite/kernels/one_hot.cc */

int shl_ref_one_hot_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_one_hot_params *params)
{
    int prefix_dim_size = 1;
    for (int i = 0; i < params->axis; ++i) {
        prefix_dim_size *= input->dim[i];
    }
    if (prefix_dim_size == 0) {
        // If indices tensor is degenerate, return a degenerate tensor, just like
        // TensorFlow does.
        return CSINN_FALSE;
    }
    int suffix_dim_size = csinn_tensor_size(input) / prefix_dim_size;
    int depth = params->depth;

    /* TODO: support different on_value/off_value */
    float on_value = 1;
    float off_value = 0;

    // View the indices as a matrix of size:
    //     prefix_dim_size x suffix_dim_size
    // View the output as a matrix of size:
    //     prefix_dim_size x depth x suffix_dim_size
    // Then the output is:
    //     output(i, j, k) == (input(i, k) == j) ? on : off
    float *output_data = output->data;
    float *input_data = input->data;
    for (int i = 0; i < prefix_dim_size; ++i) {
        for (int j = 0; j < depth; ++j) {
            for (int k = 0; k < suffix_dim_size; ++k) {
                int input_value = ((int *)input_data)[i * suffix_dim_size + k];
                if (input_value == j) {
                    output_data[i * depth * suffix_dim_size + j * suffix_dim_size + k] = on_value;
                } else {
                    output_data[i * depth * suffix_dim_size + j * suffix_dim_size + k] = off_value;
                }
            }
        }
    }
    return CSINN_TRUE;
}

int shl_ref_one_hot_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_one_hot_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_one_hot_f32);
}
