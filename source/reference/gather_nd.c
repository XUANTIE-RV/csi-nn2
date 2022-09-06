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

/* CSI-NN2 version 2.0.x */

#include "shl_ref.h"

static int Multiplication(int32_t *input, int s, int e)
{
    int res = 1;
    for (int i = s; i <= e; i++) {
        res = res * input[i];
    }
    return res;
}

int shl_ref_gather_nd_f32(struct csinn_tensor *input, struct csinn_tensor *indices,
                          struct csinn_tensor *output, struct csinn_gather_nd_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    uint32_t *indices_data = (uint32_t *)indices->data;

    int in_size = 1, indices_size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        in_size *= input->dim[i];
    }
    for (int i = 0; i < indices->dim_count; i++) {
        indices_size *= indices->dim[i];
    }

    // indices.shape[-1] must be <= params.rank
    int indices_last_dim = indices->dim[indices->dim_count - 1];
    int axis = indices_last_dim - 1;

    int indices_outer_size = 1;
    indices_outer_size = indices_size / indices_last_dim;

    int input_outer_size = 1;
    for (int i = 0; i < axis; i++) {
        input_outer_size *= input->dim[i];
    }
    int input_inner_size = 1;
    for (int i = axis + 1; i < input->dim_count; i++) {
        input_inner_size *= input->dim[i];
    }

    float *in_copy_addr = NULL;
    int dim_over_flag = 0;
    for (int i = 0; i < indices_outer_size; i++) {
        int input_outer_idx = 0;
        for (int j = 0; j < indices_last_dim; j++) {
            int indices_val = indices_data[i * indices_last_dim + j];
            if (indices_val >= input->dim[j]) {
                dim_over_flag = 1;
                break;
            } else {
                input_outer_idx +=
                    indices_val * Multiplication(input->dim, j + 1, indices_last_dim - 1);
            }
        }
        if (dim_over_flag == 1) {
            dim_over_flag = 0;
            for (int n = 0; n < input_inner_size; n++) {
                *(output_data + n) = 0.0f;
            }
        } else {
            in_copy_addr = input_data + input_outer_idx * input_inner_size;
            memcpy(output_data, in_copy_addr, input_inner_size * sizeof(float));
        }
        output_data += input_inner_size;
    }
    return CSINN_TRUE;
}

int shl_ref_gather_nd_quant(struct csinn_tensor *input, struct csinn_tensor *indices,
                            struct csinn_tensor *output, struct csinn_gather_nd_params *params)
{
    int ret;
    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    ret = shl_ref_gather_nd_f32(finput, indices, foutput, params);
    csinn_tensor_data_convert(output, foutput);
    shl_ref_tensor_transform_free_f32(finput);
    shl_ref_tensor_transform_free_f32(foutput);
    return ret;
}
