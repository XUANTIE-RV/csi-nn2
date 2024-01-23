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

#include "rvv/rvv.h"

int shl_rvv_embedding_fp32_fp32(struct csinn_tensor *input, struct csinn_tensor *weight,
                                struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int input_len = input->dim[0];
    int embd_size = weight->dim[1];
    int32_t *input_data = input->data;
    float *output_data = output->data;
    float *weight_data = weight->data;
    for (int i = 0; i < input_len; i++) {
        int token = input_data[i];
        memcpy(output_data + i * embd_size, weight_data + token * embd_size,
               embd_size * sizeof(float));
    }

    return CSINN_TRUE;
}

int shl_rvv_embedding_fp32_q8(struct csinn_tensor *input, struct csinn_tensor *weight,
                              struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int input_len = input->dim[0];
    int embd_size = weight->dim[1];
    int32_t *input_data = input->data;
    float *output_data = output->data;

    int q8_block_size = 32;

    int8_t *weight_data = weight->data;
    __fp16 *scale_data = weight->data + csinn_tensor_size(weight);
    for (int i = 0; i < input_len; i++) {
        int token = input_data[i];

        for (int j = 0; j < embd_size; ++j) {
            int input_index = token * embd_size + j;
            int output_index = i * embd_size + j;

            int8_t value = weight_data[input_index];
            __fp16 scale = scale_data[input_index / q8_block_size];
            output_data[output_index] = (float)(value * scale);
        }
    }

    return CSINN_TRUE;
}

int shl_rvv_embedding_fp32_q4(struct csinn_tensor *input, struct csinn_tensor *weight,
                              struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int input_len = input->dim[0];
    int embd_size = weight->dim[1];
    int32_t *input_data = input->data;
    float *output_data = output->data;

    int block_size = 32;
    int embd_block_num = embd_size / block_size;

    int8_t *weight_data = weight->data;
    __fp16 *scale_data = weight->data + csinn_tensor_size(weight) / 2;
    for (int i = 0; i < input_len; i++) {
        int token = input_data[i];

        for (int j = 0; j < embd_block_num; j++) {
            for (int k = 0; k < block_size / 2; k++) {
                int input_index = token * embd_size / 2 + j * block_size / 2 + k;
                int output_index = i * embd_size + j * block_size + k;

                int8_t value = weight_data[input_index];
                __fp16 scale = scale_data[input_index * 2 / block_size];
                output_data[output_index] = ((float)(value & 0xf) - 8) * (float)scale;
                output_data[output_index + block_size / 2] =
                    ((float)((value & 0xf0) >> 4) - 8) * (float)scale;
            }
        }
    }

    return CSINN_TRUE;
}

int shl_rvv_embedding_fp16_fp16(struct csinn_tensor *input, struct csinn_tensor *weight,
                                struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int input_len = input->dim[0];
    int embd_size = weight->dim[1];
    int32_t *input_data = input->data;
    __fp16 *output_data = output->data;
    __fp16 *weight_data = weight->data;
    for (int i = 0; i < input_len; i++) {
        int token = input_data[i];
        memcpy(output_data + i * embd_size, weight_data + token * embd_size,
               embd_size * sizeof(__fp16));
    }

    return CSINN_TRUE;
}

int shl_rvv_embedding_fp16_q8(struct csinn_tensor *input, struct csinn_tensor *weight,
                              struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int input_len = input->dim[0];
    int embd_size = weight->dim[1];
    int32_t *input_data = input->data;
    __fp16 *output_data = output->data;

    int q8_block_size = 32;

    int8_t *weight_data = weight->data;
    __fp16 *scale_data = weight->data + csinn_tensor_size(weight);
    for (int i = 0; i < input_len; i++) {
        int token = input_data[i];

        for (int j = 0; j < embd_size; ++j) {
            int input_index = token * embd_size + j;
            int output_index = i * embd_size + j;

            int8_t value = weight_data[input_index];
            __fp16 scale = scale_data[input_index / q8_block_size];
            output_data[output_index] = (__fp16)(value * scale);
        }
    }

    return CSINN_TRUE;
}

int shl_rvv_embedding_fp16_q4(struct csinn_tensor *input, struct csinn_tensor *weight,
                              struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int input_len = input->dim[0];
    int embd_size = weight->dim[1];
    int32_t *input_data = input->data;
    __fp16 *output_data = output->data;

    int block_size = 32;
    int embd_block_num = embd_size / block_size;

    int8_t *weight_data = weight->data;
    __fp16 *scale_data = weight->data + csinn_tensor_size(weight) / 2;
    for (int i = 0; i < input_len; i++) {
        int token = input_data[i];

        for (int j = 0; j < embd_block_num; j++) {
            for (int k = 0; k < block_size / 2; k++) {
                int input_index = token * embd_size / 2 + j * block_size / 2 + k;
                int output_index = i * embd_size + j * block_size + k;

                int8_t value = weight_data[input_index];
                __fp16 scale = scale_data[input_index * 2 / block_size];
                output_data[output_index] = ((__fp16)(value & 0xf) - 8) * (__fp16)scale;
                output_data[output_index + block_size / 2] =
                    ((__fp16)((value & 0xf0) >> 4) - 8) * (__fp16)scale;
            }
        }
    }

    return CSINN_TRUE;
}

int shl_rvv_embedding_int32(struct csinn_tensor *input, struct csinn_tensor *weight,
                            struct csinn_tensor *output, struct csinn_diso_params *params)
{
    if (output->dtype == CSINN_DTYPE_FLOAT32) {
        if (weight->dtype == CSINN_DTYPE_INT8 && weight->mtype == CSINN_MEM_TYPE_BLOCK_Q8_0) {
            return shl_rvv_embedding_fp32_q8(input, weight, output, params);
        } else if (weight->dtype == CSINN_DTYPE_INT4 &&
                   weight->mtype == CSINN_MEM_TYPE_BLOCK_Q4_0) {
            return shl_rvv_embedding_fp32_q4(input, weight, output, params);
        } else if (weight->dtype == CSINN_DTYPE_FLOAT32) {
            return shl_rvv_embedding_fp32_fp32(input, weight, output, params);
        }

    } else if (output->dtype == CSINN_DTYPE_FLOAT16) {
        if (weight->dtype == CSINN_DTYPE_INT8 && weight->mtype == CSINN_MEM_TYPE_BLOCK_Q8_0) {
            return shl_rvv_embedding_fp16_q8(input, weight, output, params);
        } else if (weight->dtype == CSINN_DTYPE_INT4 &&
                   weight->mtype == CSINN_MEM_TYPE_BLOCK_Q4_0) {
            return shl_rvv_embedding_fp16_q4(input, weight, output, params);
        } else if (weight->dtype == CSINN_DTYPE_FLOAT16) {
            return shl_rvv_embedding_fp16_fp16(input, weight, output, params);
        }
    }

    return shl_ref_embedding_quant(input, weight, output, params);
}
