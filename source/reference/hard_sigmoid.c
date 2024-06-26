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

int shl_ref_hard_sigmoid_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_sigmoid_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size *= input->dim[i];
    }
    for (int i = 0; i < size; i++) {
        if (input_data[i] < -2.5) {
            output_data[i] = 0;
        } else if (input_data[i] > 2.5) {
            output_data[i] = 1;
        } else {
            output_data[i] = 0.2 * input_data[i] + 0.5;
        }
    }
    return CSINN_TRUE;
}

int shl_ref_hard_sigmoid_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_sigmoid_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_hard_sigmoid_f32);
}
