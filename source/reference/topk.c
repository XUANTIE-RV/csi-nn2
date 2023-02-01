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

#include "shl_ref.h"

int shl_ref_topk_f32(struct csinn_tensor *input, struct csinn_tensor *output1,
                     struct csinn_tensor *output2, struct csinn_topk_params *params)
{
    float *input_data = (float *)input->data;
    float *values_data = (float *)output1->data;
    int *indices_data = (int *)output2->data;

    int k = params->k;
    int last_dim = input->dim[input->dim_count - 1];
    int inner_size = 1;
    for (int i = 0; i < input->dim_count - 1; i++) {
        inner_size *= input->dim[i];
    }
    float *input_sort_addr = input_data;
    for (int n = 0; n < inner_size; n++) {
        int *flag = (int *)shl_mem_alloc(last_dim * sizeof(int));
        for (int i = 0; i < k; i++) {
            values_data[i] = -FLT_MAX;
            for (int j = 0; j < last_dim; j++) {
                if (input_sort_addr[j] > values_data[i] && !flag[j]) {
                    values_data[i] = input_sort_addr[j];
                    indices_data[i] = j;
                }
            }
            flag[indices_data[i]] = 1;
        }
        shl_mem_free(flag);
        flag = NULL;
        input_sort_addr += last_dim;
        values_data += k;
        indices_data += k;
    }
    return CSINN_TRUE;
}

int shl_ref_topk_quant(struct csinn_tensor *input, struct csinn_tensor *output0,
                       struct csinn_tensor *output1, struct csinn_topk_params *params)
{
    int ret;
    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *foutput0 = shl_ref_tensor_transform_f32(output0);
    ret = shl_ref_topk_f32(finput, foutput0, output1, params);
    csinn_tensor_data_convert(output0, foutput0);
    shl_ref_tensor_transform_free_f32(finput);
    shl_ref_tensor_transform_free_f32(foutput0);
    return ret;
}
