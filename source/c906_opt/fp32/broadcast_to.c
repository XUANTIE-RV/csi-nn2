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

int shl_c906_broadcast_to_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_broadcast_to_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int size0 = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size0 = size0 * input->dim[i];
    }

    int size1 = 1;
    for (int i = 0; i < params->shape_count - input->dim_count; i++) {
        size1 = size1 * params->shape[i];
    }

    asm volatile(
        "outer_loop:\n\t"
        "mv         a1, %3\n\t"
        "memcpy_loop:\n\t"
        "vsetvli    t0, a1, e32, m2\n\t"
        "vlw.v      v4, (%2)\n\t"
        "slli       t1, t0, 2\n\t"
        "add        %2, %2, t1\n\t"
        "sub        a1, a1, t0\n\t"
        "vsw.v      v4, (%0)\n\t"
        "add        %0, %0, t1\n\t"
        "bnez       a1, memcpy_loop\n\t"
        "addi       %4, %4, -1\n\t"
        "bnez       %4, outer_loop\n\t"

        : "=r"(output_data)  // %0
        : "0"(output_data),  // %1
          "r"(input_data),   // %2
          "r"(size0),        // %3
          "r"(size1)         // %4
        : "a1", "t0", "t1", "v4", "v5");

    return CSINN_TRUE;
}
