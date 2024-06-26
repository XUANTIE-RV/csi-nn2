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

int shl_ref_embedding_f32(struct csinn_tensor *input, struct csinn_tensor *weight,
                          struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int input_len = input->dim[0];
    int embd_size = weight->dim[1];
    float *input_data = input->data;
    float *output_data = output->data;
    float *weight_data = weight->data;
    for (int i = 0; i < input_len; i++) {
        int token = input_data[i];
        memcpy(output_data + i * embd_size, weight_data + token * embd_size,
               embd_size * sizeof(float));
    }

    return CSINN_TRUE;
}

int shl_ref_embedding_fp16(struct csinn_tensor *input, struct csinn_tensor *weight,
                           struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int input_len = input->dim[0];
    int embd_size = weight->dim[1];
    int32_t *input_data = input->data;
    float *output_data = output->data;
    // float16
    int16_t *weight_data = weight->data;
    for (int i = 0; i < input_len; i++) {
        int token = input_data[i];

        for (int j = 0; j < embd_size; ++j) {
            int16_t value = weight_data[token * embd_size + j];
            output_data[i * embd_size + j] = shl_ref_float16_to_float32(value);
        }
    }

    return CSINN_TRUE;
}

int shl_ref_embedding_q8(struct csinn_tensor *input, struct csinn_tensor *weight,
                         struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int input_len = input->dim[0];
    int embd_size = weight->dim[1];
    int32_t *input_data = input->data;
    float *output_data = output->data;

    int q8_block_size = 32;

    int8_t *weight_data = weight->data;
    int16_t *scale_data = weight->data + csinn_tensor_size(weight);
    for (int i = 0; i < input_len; i++) {
        int token = input_data[i];

        for (int j = 0; j < embd_size; ++j) {
            int input_index = token * embd_size + j;
            int output_index = i * embd_size + j;

            int8_t value = weight_data[input_index];
            float scale = shl_ref_float16_to_float32(scale_data[input_index / q8_block_size]);
            output_data[output_index] = (float)value * scale;
        }
    }

    return CSINN_TRUE;
}

int shl_ref_embedding_q4(struct csinn_tensor *input, struct csinn_tensor *weight,
                         struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int input_len = input->dim[0];
    int embd_size = weight->dim[1];
    int32_t *input_data = input->data;
    float *output_data = output->data;

    int block_size = 32;
    int embd_block_num = embd_size / block_size;

    int8_t *weight_data = weight->data;
    int16_t *scale_data = weight->data + csinn_tensor_size(weight) / 2;
    for (int i = 0; i < input_len; i++) {
        int token = input_data[i];

        for (int j = 0; j < embd_block_num; j++) {
            for (int k = 0; k < block_size / 2; k++) {
                int input_index = token * embd_size / 2 + j * block_size / 2 + k;
                int output_index = i * embd_size + j * block_size + k;

                int8_t value = weight_data[input_index];
                float scale = shl_ref_float16_to_float32(scale_data[input_index * 2 / block_size]);
                output_data[output_index] = ((float)(value & 0xf) - 8) * scale;
                output_data[output_index + block_size / 2] =
                    ((float)((value & 0xf0) >> 4) - 8) * scale;
            }
        }
    }

    return CSINN_TRUE;
}

int shl_ref_embedding_quant(struct csinn_tensor *input, struct csinn_tensor *weight,
                            struct csinn_tensor *output, struct csinn_diso_params *params)
{
    if (weight->dtype == CSINN_DTYPE_FLOAT16) {
        return shl_ref_embedding_fp16(input, weight, output, params);
    } else if (weight->dtype == CSINN_DTYPE_INT8 && weight->mtype == CSINN_MEM_TYPE_BLOCK_Q8_0) {
        return shl_ref_embedding_q8(input, weight, output, params);
    } else if (weight->dtype == CSINN_DTYPE_INT4 && weight->mtype == CSINN_MEM_TYPE_BLOCK_Q4_0) {
        return shl_ref_embedding_q4(input, weight, output, params);
    }
    return shl_ref_diso_callback_base(input, weight, output, params, shl_ref_embedding_f32);
}
