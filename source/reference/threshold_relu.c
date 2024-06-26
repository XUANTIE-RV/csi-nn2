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

static float threshold_relu(float x, float theta) { return x > theta ? x : 0; }

int shl_ref_threshold_relu_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_relu_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    int size = 1;
    float theta = params->n;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }

    for (int i = 0; i < size; i++) {
        output_data[i] = threshold_relu(input_data[i], theta);
    }
    return CSINN_TRUE;
}

int shl_ref_threshold_relu_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_relu_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_threshold_relu_f32);
}
