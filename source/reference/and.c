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

int shl_ref_and_u32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params)
{
    uint32_t *input0_data = input0->data;
    uint32_t *input1_data = input1->data;
    uint32_t *output_data = output->data;
    int size = csinn_tensor_size(input0);

    for (int i = 0; i < size; i++) {
        output_data[i] = input0_data[i] & input1_data[i];
    }
    return CSINN_TRUE;
}

int shl_ref_and_u8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params)
{
    uint8_t *input0_data = input0->data;
    uint8_t *input1_data = input1->data;
    uint8_t *output_data = output->data;
    int size = csinn_tensor_size(input0);

    for (int i = 0; i < size; i++) {
        output_data[i] = input0_data[i] & input1_data[i];
    }
    return CSINN_TRUE;
}

int shl_ref_and_i8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int8_t *input0_data = input0->data;
    int8_t *input1_data = input1->data;
    int8_t *output_data = output->data;
    int size = csinn_tensor_size(input0);

    for (int i = 0; i < size; i++) {
        output_data[i] = input0_data[i] & input1_data[i];
    }
    return CSINN_TRUE;
}
