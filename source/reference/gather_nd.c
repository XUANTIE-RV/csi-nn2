/*
 * Copyright (C) 2016-2020 C-SKY Limited. All rights reserved.
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

#include "csi_nn.h"
#include "csi_utils.h"


static int Multiplication(int *input, int s, int e)
{
    int res = 1;
    for(int i=s; i<=e; i++) {
        res = res * input[i];
    }
    return res;
}

static int csi_gather_nd_f32(struct csi_tensor *input,
                            struct csi_tensor *indices,
                            struct csi_tensor *output,
                            struct gather_nd_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    uint32_t *indices_data = (uint32_t *)indices->data;

    int in_size = 1, indices_size = 1;
    for(int i = 0; i < input->dim_count; i++) {
        in_size *= input->dim[i];
    }
    for(int i = 0; i < indices->dim_count; i++) {
        indices_size *= indices->dim[i];
    }

    // indices.shape[-1] must be <= params.rank
    int indices_last_dim = indices->dim[indices->dim_count - 1];
    int axis = indices_last_dim - 1;

    int indices_outer_size = 1;
    indices_outer_size = indices_size / indices_last_dim;

    int input_outer_size = 1;
    for(int i = 0; i < axis; i++) {
        input_outer_size *= input->dim[i];
    }
    int input_inner_size = 1;
    for(int i = axis + 1; i < input->dim_count; i++) {
        input_inner_size *= input->dim[i];
    }

    float *in_copy_addr = NULL;
    int dim_over_flag = 0;
    for(int i = 0; i < indices_outer_size; i++) {
        int input_outer_idx = 0;
        for(int j = 0; j < indices_last_dim; j++) {
            int indices_val = indices_data[i * indices_last_dim + j];
            if(indices_val >= input->dim[j]) {
                dim_over_flag = 1;
                break;
            } else {
                input_outer_idx += indices_val * Multiplication(input->dim, j + 1, indices_last_dim - 1);
            }
        }
        if(dim_over_flag == 1) {
            dim_over_flag = 0;
            for(int n = 0; n < input_inner_size; n++) {
                *(output_data + n) = 0.0f;
            }
        } else {
            in_copy_addr = input_data + input_outer_idx * input_inner_size;
            memcpy(output_data , in_copy_addr, input_inner_size * sizeof(float));
        }
        output_data += input_inner_size;
    }
    return CSINN_TRUE;
}


static int csi_gather_nd_u8(struct csi_tensor *input,
                            struct csi_tensor *indices,
                            struct csi_tensor *output,
                            struct gather_nd_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;
    uint32_t *indices_data = (uint32_t *)indices->data;

    int in_size = 1, indices_size = 1;
    for(int i = 0; i < input->dim_count; i++) {
        in_size *= input->dim[i];
    }
    for(int i = 0; i < indices->dim_count; i++) {
        indices_size *= indices->dim[i];
    }

    // indices.shape[-1] must be <= params.rank
    int indices_last_dim = indices->dim[indices->dim_count - 1];
    int axis = indices_last_dim - 1;

    int indices_outer_size = 1;
    indices_outer_size = indices_size / indices_last_dim;

    int input_outer_size = 1;
    for(int i = 0; i < axis; i++) {
        input_outer_size *= input->dim[i];
    }
    int input_inner_size = 1;
    for(int i = axis + 1; i < input->dim_count; i++) {
        input_inner_size *= input->dim[i];
    }

    uint8_t *in_copy_addr = NULL;
    int dim_over_flag = 0;
    for(int i = 0; i < indices_outer_size; i++) {
        int input_outer_idx = 0;
        for(int j = 0; j < indices_last_dim; j++) {
            int indices_val = indices_data[i * indices_last_dim + j];
            if(indices_val >= input->dim[j]) {
                dim_over_flag = 1;
                break;
            } else {
                input_outer_idx += indices_val * Multiplication(input->dim, j + 1, indices_last_dim - 1);
            }
        }
        if(dim_over_flag == 1) {
            dim_over_flag = 0;
            uint8_t zero = csi_requantize_u8(0.0f, input->offset, input->multiplier, input->shift, 
                                                    output->offset, output->multiplier, output->shift);
            for(int n = 0; n < input_inner_size; n++) {
                *(output_data + n) = zero;
            }
        } else {
            in_copy_addr = input_data + input_outer_idx * input_inner_size;
            for(int k = 0; k < input_inner_size; k++) {
                *(output_data + k) = csi_requantize_u8(*(in_copy_addr + k), input->offset, input->multiplier, input->shift, 
                                                                            output->offset, output->multiplier, output->shift);
            }
        }
        output_data += input_inner_size;
    }
    return CSINN_TRUE;
}


int csi_gather_nd_init(struct csi_tensor *input,
                       struct csi_tensor *indices,
                       struct csi_tensor *output,
                       struct gather_nd_params *params)
{
    if(input->dtype == CSINN_DTYPE_UINT8) {
        params->bc = csi_gather_nd_u8;
    } else if(input->dtype == CSINN_DTYPE_FLOAT32) {
        params->bc = csi_gather_nd_f32;
    } else {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_gather_nd(struct csi_tensor *input,
                  struct csi_tensor *indices,
                  struct csi_tensor *output,
                  struct gather_nd_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, indices, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}

