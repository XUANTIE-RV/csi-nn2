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

static int Multiplication(int32_t *dim, int s, int e)
{
    int res = 1;
    for (int i = s; i <= e; i++) {
        res = res * dim[i];
    }
    return res;
}

int shl_ref_tile_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_tile_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int reps_count = params->reps_num;
    assert(reps_count == input->dim_count);

    int in_size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        in_size *= input->dim[i];
    }
    int out_size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        out_size *= params->reps[i];
    }
    out_size = out_size * in_size;

    for (int dim_idx = reps_count - 1; dim_idx >= 0; dim_idx--) {
        int reps_num = params->reps[dim_idx];
        int num = Multiplication(input->dim, 0, dim_idx) / (input->dim[dim_idx]);
        int step = Multiplication(input->dim, dim_idx, input->dim_count - 1) *
                   Multiplication(params->reps, dim_idx, reps_count - 1) / (params->reps[dim_idx]);
        float *temp = (float *)shl_mem_alloc(reps_num * num * step * sizeof(float));
        float *temp_cpy_addr = temp;
        for (int input_pre_i = 0; input_pre_i < num; input_pre_i++) {
            for (int rep_i = 0; rep_i < reps_num; rep_i++) {
                memcpy(temp_cpy_addr, input_data, step * sizeof(float));
                temp_cpy_addr += step;
            }
            input_data += step;
        }
        memcpy(output_data, temp, reps_num * num * step * sizeof(float));
        input_data = output_data;
        shl_mem_free(temp);
        temp = NULL;
    }
    memcpy(output_data, input_data, out_size * sizeof(float));
    return CSINN_TRUE;
}

int shl_ref_tile_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tile_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_tile_f32);
}
