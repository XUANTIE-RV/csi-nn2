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

static int Multiplication(struct csinn_tensor *input, int s, int e)
{
    int res = 1;
    for (int i = s; i <= e; i++) {
        res = res * input->dim[i];
    }
    return res;
}

int shl_ref_reverse_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_reverse_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }
    int axis = params->axis;
    int num = Multiplication(input, 0, axis) / (input->dim[axis]);
    int step = Multiplication(input, axis, input->dim_count - 1) / (input->dim[axis]);
    int cnt = (input->dim[axis]) / 2;

    memcpy(output_data, input_data, size * sizeof(float));

    for (int i = 0; i < num; i++) {
        float *start_addr = output_data + i * step * (input->dim[axis]);
        float *end_addr = start_addr + step * (input->dim[axis]) - 1;
        for (int j = 0; j < cnt; j++) {
            float *temp = (float *)shl_mem_alloc(step * sizeof(float));
            memcpy(temp, start_addr, step * sizeof(float));
            memcpy(start_addr, end_addr - step + 1, step * sizeof(float));
            memcpy(end_addr - step + 1, temp, step * sizeof(float));
            start_addr += step;
            end_addr -= step;
            shl_mem_free(temp);
        }
    }
    return CSINN_TRUE;
}

int shl_ref_reverse_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_reverse_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_reverse_f32);
}
