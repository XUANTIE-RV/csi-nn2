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

int csi_topk_f32(struct csi_tensor *input,
                 struct csi_tensor *output1,
                 struct csi_tensor *output2,
                 struct topk_params *params)
{
    float *input_data  = (float *)input->data;
    float *values_data = (float *)output1->data;
    int *indices_data  = (int *)output2->data;

    int k = params->k;
    int last_dim = input->dim[input->dim_count - 1];
    int inner_size = 1;
    for(int i = 0; i < input->dim_count - 1; i++)
    {
        inner_size *= input->dim[i];
    }
    float *input_sort_addr = input_data;
    for(int n = 0; n < inner_size; n++) {
        int *flag = (int *)calloc(last_dim, sizeof(int));
        for(int i = 0; i < k; i++) {
            values_data[i] = -FLT_MAX;
            for(int j = 0; j < last_dim; j++) {
                if(input_sort_addr[j] > values_data[i] && !flag[j]) {
                    values_data[i]  = input_sort_addr[j];
                    indices_data[i] = j;
                }
            }
            flag[indices_data[i]] = 1;
        }
        free(flag);
        flag = NULL;
        input_sort_addr += last_dim;
        values_data += k;
        indices_data += k;
    }
    return CSINN_TRUE;
}

int csi_topk_u8(struct csi_tensor *input,
                struct csi_tensor *output1,
                struct csi_tensor *output2,
                struct topk_params *params)
{
    uint8_t *input_data  = (uint8_t *)input->data;
    uint8_t *values_data = (uint8_t *)output1->data;
    int *indices_data  = (int *)output2->data;

    int k = params->k;
    int last_dim = input->dim[input->dim_count - 1];
    int inner_size = 1;
    for(int i = 0; i < input->dim_count - 1; i++)
    {
        inner_size *= input->dim[i];
    }
    uint8_t *input_sort_addr = input_data;
    for(int n = 0; n < inner_size; n++) {
        int *flag = (int *)calloc(last_dim, sizeof(int));
        for(int i = 0; i < k; i++) {
            values_data[i] = 0;
            for(int j = 0; j < last_dim; j++) {
                // >= :for k = last_dim
                if(input_sort_addr[j] >= values_data[i] && !flag[j]) {
                    values_data[i]  = input_sort_addr[j];
                    indices_data[i] = j;
                }
            }
            values_data[i] = csi_requantize_u8(values_data[i], input->zero_point, input->multiplier, input->shift,
                                                                output1->zero_point, output1->multiplier, output1->shift);
            flag[indices_data[i]] = 1;
        }
        free(flag);
        flag = NULL;
        input_sort_addr += last_dim;
        values_data += k;
        indices_data += k;
    }
    return CSINN_TRUE;
}

int csi_topk_init(struct csi_tensor *input,
                  struct csi_tensor *output1,
                  struct csi_tensor *output2,
                  struct topk_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_TOPK, input->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_topk(struct csi_tensor *input,
             struct csi_tensor *output1,
             struct csi_tensor *output2,
             struct topk_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output1, output2, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
