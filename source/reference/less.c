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

/* SHL version 2.1.x */

#include "shl_ref.h"

int shl_ref_less_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    float *input0_data = input0->data;
    float *input1_data = input1->data;
    float *output_data = output->data;

    int in0_size = csinn_tensor_size(input0);
    int in1_size = csinn_tensor_size(input1);
    int max_size = (in0_size > in1_size) ? in0_size : in1_size;

    float lhs_value, rhs_value;
    for (int i = 0; i < max_size; i++) {
        lhs_value = (in0_size == 1) ? input0_data[0] : input0_data[i];
        rhs_value = (in1_size == 1) ? input1_data[0] : input1_data[i];
        output_data[i] = lhs_value < rhs_value;
    }
    return CSINN_TRUE;
}

int shl_ref_less_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params)
{
    return shl_ref_diso_callback_base(input0, input1, output, params, shl_ref_less_f32);
}
