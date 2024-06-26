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

int shl_ref_cumsum_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_cumsum_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int axis = params->axis;

    // For all input arrays,
    // FlatSize() = outer_size * inner_size * cnt;
    int64_t outer_size = 1;
    for (int i = 0; i < axis; i++) {
        outer_size *= input->dim[i];
    }
    int64_t inner_size = 1;
    for (int i = axis + 1; i < input->dim_count; i++) {
        inner_size *= input->dim[i];
    }
    int cnt = input->dim[axis];

    for (int i = 0; i < outer_size; i++) {
        for (int k = 0; k < inner_size; k++) {
            float temp = 0.0f;
            for (int j = 0; j < cnt; j++) {
                temp += *(input_data + j * inner_size + k);
                if (!params->exclusive) {
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

int shl_ref_cumsum_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_cumsum_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_cumsum_f32);
}
