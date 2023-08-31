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

#include "reference/ref.h"

int shl_ref_equal_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params)
{
    float *input0_data = input0->data;
    float *input1_data = input1->data;
    bool *output_data = output->data;
    int size = csinn_tensor_size(input0);

    for (int i = 0; i < size; i++) {
        output_data[i] = input0_data[i] == input1_data[i];
    }
    return CSINN_TRUE;
}

int shl_ref_equal_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int ret;
    struct csinn_tensor *finput0 = shl_ref_tensor_transform_f32(input0);
    struct csinn_tensor *finput1 = shl_ref_tensor_transform_f32(input1);
    ret = shl_ref_equal_f32(finput0, finput1, output, params);
    shl_ref_tensor_transform_free_f32(finput0);
    shl_ref_tensor_transform_free_f32(finput1);
    return ret;
}
