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

int csi_strided_slice_f32(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct strided_slice_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int slice_dim_count = params->slice_count;

    int out_size = 1;
    // for(int i = 0; i < output->dim_count; i++) {
    //     out_size *= output->dim[i];
    // }
    int outer_size = 1;
    int inner_size = 1;
    int inner_size_copy_num  = 1;

    for(int i = 0; i < input->dim_count; i++) {
        inner_size *= input->dim[i];
    }

    for(int slice_dim = 0; slice_dim < slice_dim_count; slice_dim++) {

        int begin = params->begin[slice_dim];
        int end = params->end[slice_dim];
        int stride = params->stride[slice_dim];

        if(begin >= end) {
            return CSINN_FALSE;
        }
        if(end > input->dim[slice_dim]) {
            end = input->dim[slice_dim];
        }

        inner_size /= input->dim[slice_dim];
        outer_size *= inner_size_copy_num;

        inner_size_copy_num = 1 + (end -1 - begin) / stride;
        out_size *= inner_size_copy_num;

        float *temp = (float *)malloc(outer_size * inner_size * inner_size_copy_num * sizeof(float));
        float *temp_copy = NULL;
        float *temp_addr = temp;
        for(int n = 0; n < outer_size; n++) {
            for(int i = begin; i < end; i = i + stride) {
                memcpy(temp_addr, input_data + i * inner_size, inner_size * sizeof(float));
                temp_addr += inner_size;
            }
            input_data += inner_size * input->dim[slice_dim];
        }
        if(temp != NULL) {
            free(temp_copy);
        }
        temp_copy = (float *)malloc(outer_size * inner_size * inner_size_copy_num * sizeof(float));
        memcpy(temp_copy, temp, outer_size * inner_size * inner_size_copy_num * sizeof(float));
        input_data = temp_copy;
        free(temp);
        temp = NULL;
    }
    out_size = out_size * inner_size;
    memcpy(output_data, input_data, out_size * sizeof(float));
    free(input_data);
    return CSINN_TRUE;
}

int csi_strided_slice_u8(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct strided_slice_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;
    int slice_dim_count = params->slice_count;

    int out_size = 1;
    // for(int i = 0; i < output->dim_count; i++) {
    //     out_size *= output->dim[i];
    // }
    int outer_size = 1;
    int inner_size = 1;
    int inner_size_copy_num  = 1;

    for(int i = 0; i < input->dim_count; i++) {
        inner_size *= input->dim[i];
    }

    for(int slice_dim = 0; slice_dim < slice_dim_count; slice_dim++) {

        int begin = params->begin[slice_dim];
        int end = params->end[slice_dim];
        int stride = params->stride[slice_dim];

        if(begin >= end) {
            return CSINN_FALSE;
        }
        if(end > input->dim[slice_dim]) {
            end = input->dim[slice_dim];
        }

        inner_size /= input->dim[slice_dim];
        outer_size *= inner_size_copy_num;

        inner_size_copy_num = 1 + (end -1 - begin) / stride;
        out_size *= inner_size_copy_num;

        uint8_t *temp = (uint8_t *)malloc(outer_size * inner_size * inner_size_copy_num * sizeof(uint8_t));
        uint8_t *temp_copy = NULL;
        uint8_t *temp_addr = temp;
        for(int n = 0; n < outer_size; n++) {
            for(int i = begin; i < end; i = i + stride) {
                memcpy(temp_addr, input_data + i * inner_size, inner_size * sizeof(uint8_t));
                temp_addr += inner_size;
            }
            input_data += inner_size * input->dim[slice_dim];
        }
        if(temp != NULL) {
            free(temp_copy);
        }
        temp_copy = (uint8_t *)malloc(outer_size * inner_size * inner_size_copy_num * sizeof(uint8_t));
        memcpy(temp_copy, temp, outer_size * inner_size * inner_size_copy_num * sizeof(uint8_t));
        input_data = temp_copy;
        free(temp);
        temp = NULL;
    }
    out_size = out_size * inner_size;
    for(int i = 0; i < out_size; i++) {
        output_data[i] = csi_requantize_u8(input_data[i],input->zero_point, input->multiplier, input->shift,
                                                        output->zero_point, output->multiplier, output->shift);
    }
    free(input_data);
    return CSINN_TRUE;
}

int csi_strided_slice_init(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct strided_slice_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_STRIDED_SLICE, input->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_strided_slice(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct strided_slice_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}