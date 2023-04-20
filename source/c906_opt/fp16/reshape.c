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

#include "shl_c906.h"

int shl_c906_reshape_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_reshape_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    int size = csinn_tensor_byte_size(input);
    if (input_data != output_data) {
        shl_c906_memcpy(output_data, input_data, size);
    }
    return CSINN_TRUE;
}
