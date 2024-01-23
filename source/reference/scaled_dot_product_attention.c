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

// query: batch,np,sq,dim_head
// key: batch,np,sk,dim_head
// value: batch,np,sk,dim_head
// output: batch,np,sq,dim_head
int shl_ref_scaled_dot_product_attention_f32(struct csinn_tensor *query, struct csinn_tensor *key,
                                             struct csinn_tensor *value,
                                             struct csinn_tensor *output_tensor,
                                             struct csinn_scale_dot_attention_params *params)
{
    float *query_data = query->data;
    float *key_data = key->data;
    float *value_data = value->data;
    float *output_data = output_tensor->data;
    // np: number of heads
    // sk: sequence number of kv
    // sq: sequence number of q
    int32_t batch = query->dim[0];  // batch = 1 only
    int32_t np = query->dim[1];
    int32_t sk = key->dim[2];
    int32_t sq = query->dim[2];
    int32_t head_dim = query->dim[3];
    float norm_factor = 1.0f / params->norm_factor;

    // deal with transpose_v first
    // batch,np,sq,sk * batch,np,sk,dim_head
    if (!params->transpose_v)  // if value is [batch,np,sk,dim_head],do transpose(-2,-1)
    {
        float *value_transpose_tmp = shl_mem_alloc(batch * np * sk * head_dim * sizeof(float));
        memcpy(value_transpose_tmp, value_data, batch * np * sk * head_dim * sizeof(float));
        for (int i = 0; i < np; i++) {
            for (int j = 0; j < head_dim; j++) {
                for (int k = 0; k < sk; k++) {
                    int dim1 = i * head_dim * sk + j * sk + k,
                        dim2 = i * head_dim * sk + head_dim * k + j;
                    value_data[dim1] = value_transpose_tmp[dim2];
                }
            }
        }

        shl_mem_free(value_transpose_tmp);
    }

    // matmul_result = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    // matmul_result = matmul_result / norm_factor
    size_t matmul_res_size = batch * np * sq * sk * sizeof(float);
    float *matmul_res_data = shl_mem_alloc(matmul_res_size);
    memset(matmul_res_data, 0, matmul_res_size);
    for (int i = 0; i < np; i++)  // split into multiple threads from here.
    {
        float *mat_input1 = query_data + i * sq * head_dim;
        float *mat_input2 = key_data + i * sk * head_dim;

        for (int j = 0; j < sq; j++) {
            float max = -FLT_MAX;
            float acc_exp = 0;
            int casual_cnt = sk;
            if (params->casual) {
                casual_cnt = j + 1 + (sk - sq);
            }
            for (int k = 0; k < casual_cnt; k++) {
                float sum = 0;
                for (int l = 0; l < head_dim; l++) {
                    sum += (mat_input1[j * head_dim + l] * mat_input2[k * head_dim + l]);
                }
                sum *= norm_factor;
                // cal exp_sum
                float tmp = max;
                max = fmax(max, sum);
                acc_exp *= exp(tmp - max);
                acc_exp += exp(sum - max);
                matmul_res_data[i * sq * sk + j * sk + k] = sum;
            }
            // do softmax
            for (int k = 0; k < casual_cnt; k++) {
                *(matmul_res_data + i * sq * sk + j * sk + k) =
                    exp(*(matmul_res_data + i * sq * sk + j * sk + k) - max) / acc_exp;
            }
        }

        // context_layer = torch.matmul(attention_probs, value_layer)

        mat_input1 = matmul_res_data + i * sq * sk;
        mat_input2 = value_data + i * head_dim * sk;

        for (int j = 0; j < sq; j++) {
            for (int k = 0; k < head_dim; k++) {
                float sum = 0;
                for (int l = 0; l < sk; l++) {
                    sum += (mat_input1[j * sk + l] * mat_input2[k * sk + l]);
                }
                output_data[i * sq * head_dim + j * head_dim + k] = sum;
            }
        }
    }
    shl_mem_free(matmul_res_data);

    return CSINN_TRUE;
}

int shl_ref_scaled_dot_product_attention_quant(struct csinn_tensor *query, struct csinn_tensor *key,
                                               struct csinn_tensor *value,
                                               struct csinn_tensor *output,
                                               struct csinn_scale_dot_attention_params *params)
{
    struct csinn_tensor *float_query = shl_ref_tensor_transform_f32(query);
    struct csinn_tensor *float_key = shl_ref_tensor_transform_f32(key);
    struct csinn_tensor *float_value = shl_ref_tensor_transform_f32(value);
    struct csinn_tensor *float_output = shl_ref_tensor_transform_f32(output);
    int ret = shl_ref_scaled_dot_product_attention_f32(float_query, float_key, float_value,
                                                       float_output, params);
    csinn_tensor_data_convert(output, float_output);
    shl_ref_tensor_transform_free_f32(float_query);
    shl_ref_tensor_transform_free_f32(float_value);
    shl_ref_tensor_transform_free_f32(float_key);
    shl_ref_tensor_transform_free_f32(float_output);
    return ret;
}
