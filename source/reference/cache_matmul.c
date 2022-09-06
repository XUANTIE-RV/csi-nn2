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

// asr data buffer
void asr_buffer_init(struct csinn_asr_buffer_t *buffer, size_t buffer_size, size_t data_lenth)
{
    buffer->buffer = shl_mem_alloc(buffer_size);
    buffer->buffer_lenth = buffer_size;
    buffer->data_lenth = data_lenth;
    buffer->writer_index = buffer_size - data_lenth;
    buffer->flag = 0;  //用来记录有没有经过位置0.有的话置为1.
}

// insert front
void *asr_buffer_insert_front(struct csinn_asr_buffer_t *buffer, void *input, size_t len)
{
    int start_position = buffer->writer_index - len;
    uint8_t *p = NULL;
    if (buffer->flag == 0) {
        if (start_position < 0) {
            buffer->flag = 1;
        }
    }
    if (start_position >= 0) {
        p = &buffer->buffer[start_position];
        memcpy(p, input, len);
        buffer->writer_index = start_position;
        if (buffer->flag == 0) {
            return (void *)&buffer->buffer[0];
        } else {
            return (void *)p;
        }
    } else {
        start_position = buffer->buffer_lenth - buffer->data_lenth;
        p = &buffer->buffer[start_position];
        memcpy(p, input, len);
        memcpy(p + len, &buffer->buffer[buffer->writer_index], buffer->data_lenth - len);
        buffer->writer_index = start_position;
        return (void *)p;
    }
}

void *asr_buffer_insert_back(struct csinn_asr_buffer_t *buffer, void *input, size_t len)
{
    int end_position = buffer->writer_index + len;
    uint8_t *p = NULL;
    if (end_position <= buffer->buffer_lenth) {
        p = &buffer->buffer[buffer->writer_index];
        memcpy(p, input, len);
        buffer->writer_index += len;
        p -= (buffer->data_lenth - len);
    } else {
        p = &buffer->buffer[buffer->writer_index + len - buffer->data_lenth];
        memcpy(&buffer->buffer[0], p, buffer->data_lenth - len);
        buffer->writer_index = buffer->data_lenth;
        memcpy(&buffer->buffer[buffer->data_lenth - len], input, len);
        p = &buffer->buffer[0];
    }
    return (void *)p;
}

// get buffer
void *asr_buffer_get_buffer(struct csinn_asr_buffer_t *buffer)
{
    return asr_buffer_insert_back(buffer, NULL, 0);
}

// reset buffer
void asr_buffer_reset(struct csinn_asr_buffer_t *buffer)
{
    free(buffer->buffer);
    buffer->writer_index = 0;
    buffer->buffer = NULL;
    buffer->buffer_lenth = 0;
    buffer->data_lenth = 0;
    buffer->flag = 0;
}

int shl_ref_cache_matmul_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *weight, struct csinn_tensor *bias,
                              struct csinn_cache_matmul_params *params)
{
    size_t data_size =
        params->shape[0] * params->shape[1] * params->shape[2] * params->shape[3] * sizeof(float);
    asr_buffer_init(&params->asr_buffer, 2 * data_size, data_size);

    int accum_depth = weight->dim[0];
    int output_depth = weight->dim[1];

    struct csinn_callback *cb = params->base.cb;
    if (input->dtype == CSINN_DTYPE_FLOAT32) {
        cb->exec = shl_ref_cache_matmul_f32;
    } else {
        cb->exec = shl_ref_cache_matmul_quant;
    }

    return CSINN_TRUE;
}

int shl_ref_cache_matmul_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *weight, struct csinn_tensor *bias,
                             struct csinn_cache_matmul_params *params)
{
    int accum_depth = weight->dim[0];
    int output_depth = weight->dim[1];
    int batches = input->dim[1];
    float *input_data = input->data;
    float *output_data = output->data;
    float *weight_data = weight->data;
    float *bias_data = bias->data;

    for (int b = 0; b < batches; ++b) {
        for (int out_c = 0; out_c < output_depth; ++out_c) {
            float total = 0.f;
            for (int d = 0; d < accum_depth; ++d) {
                total += input_data[b * accum_depth + d] * weight_data[out_c * accum_depth + d];
            }
            float bias_value = 0.0f;

            bias_value = bias_data[out_c];

            int out_pos = out_c + b * output_depth;  //如果无transpose
            output_data[out_pos] = total + bias_value;
        }
    }

    float judge =
        bias_data[0] + bias_data[1] + bias_data[2] + bias_data[3] + bias_data[4] + bias_data[5];
    size_t insert_lenth = output_depth * batches;
    float *output_from_buffer;
    if (fabs(judge) < 0.01) {
        output_from_buffer =
            asr_buffer_insert_front(&params->asr_buffer, output_data, insert_lenth * sizeof(float));
    } else {
        output_from_buffer =
            asr_buffer_insert_back(&params->asr_buffer, output_data, insert_lenth * sizeof(float));
    }
    // deal with reshape & transpose
    int32_t *shape = output->dim;

    // transpose can only be 0,2,3,1 or 0,2,1,3
    if (params->axes[2] == 3)  // 0,2,3,1
    {
        int batch = shape[3];
        int shape3 = shape[2];
        int flatten_shape = shape[1] * shape[2];
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < flatten_shape; j++) {
                int out_pos = j * batch + i;
                output_data[out_pos] = output_from_buffer[i * flatten_shape + j];
            }
        }
    } else  // 0,2,1,3
    {
        int batch = shape[2];
        int shape3 = shape[3];
        int flatten_shape = shape[1] * shape[3];
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < flatten_shape; j++) {
                int out_pos = i * shape3 + j % shape3 + batch * shape3 * (j / shape3);
                output_data[out_pos] = output_from_buffer[i * flatten_shape + j];
            }
        }
    }

    return CSINN_TRUE;
}

int shl_ref_cache_matmul_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *weight, struct csinn_tensor *bias,
                               struct csinn_cache_matmul_params *params)
{
    struct csinn_tensor *float_input = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *float_output = shl_ref_tensor_transform_f32(output);
    struct csinn_tensor *float_weight = shl_ref_tensor_transform_f32(weight);
    struct csinn_tensor *float_bias = shl_ref_tensor_transform_f32(bias);

    int ret = shl_ref_cache_matmul_f32(float_input, float_output, float_weight, float_bias, params);

    csinn_tensor_data_convert(output, float_output);

    shl_ref_tensor_transform_free_f32(float_input);
    shl_ref_tensor_transform_free_f32(float_output);
    shl_ref_tensor_transform_free_f32(float_weight);
    shl_ref_tensor_transform_free_f32(float_bias);

    return CSINN_TRUE;
}