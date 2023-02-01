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

#include "shl_c906.h"

int shl_c906_gather_fp16(struct csinn_tensor *input, struct csinn_tensor *indices,
                         struct csinn_tensor *output, struct csinn_gather_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    int32_t *indices_data = (int32_t *)indices->data;

    int inner_size = 1;
    for (int i = params->axis + 1; i < input->dim_count; i++) {
        inner_size *= input->dim[i];
    }
    int outer_size = 1;
    for (int i = 0; i < params->axis; i++) {
        outer_size *= input->dim[i];
    }
    int indices_size = 1;
    for (int i = 0; i < indices->dim_count; i++) {
        indices_size *= indices->dim[i];
    }

    for (int i = 0; i < outer_size; i++) {
        for (int j = 0; j < indices_size; j++) {
            if (indices_data[j] < input->dim[params->axis]) {
                shl_c906_memcpy(output_data, input_data + indices_data[j] * inner_size,
                                inner_size * sizeof(__fp16));
            } else {
                memset(output_data, 0, inner_size * sizeof(__fp16));
            }
            output_data += inner_size;
        }
        input_data += inner_size * input->dim[params->axis];
    }
    return CSINN_TRUE;
}
