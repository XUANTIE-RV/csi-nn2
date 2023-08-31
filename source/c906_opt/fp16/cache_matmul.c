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

#include "c906/c906.h"
#include "shl_memory.h"

// asr data buffer
void asr_buffer_init_c906(struct csinn_asr_buffer_t *buffer, size_t buffer_size, size_t data_lenth)
{
    buffer->buffer = shl_mem_alloc(buffer_size);
    buffer->buffer_lenth = buffer_size;
    buffer->data_lenth = data_lenth;
    buffer->writer_index = buffer_size - data_lenth;
    buffer->flag = 0;  //用来记录有没有经过位置0.有的话置为1.
}

// insert front
void *asr_buffer_insert_c906_front(struct csinn_asr_buffer_t *buffer, void *input, size_t len)
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

void *asr_buffer_insert_c906_back(struct csinn_asr_buffer_t *buffer, void *input, size_t len)
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
void *asr_buffer_get_buffer_c906(struct csinn_asr_buffer_t *buffer)
{
    return asr_buffer_insert_c906_back(buffer, NULL, 0);
}

// reset buffer
void asr_buffer_reset_c906(struct csinn_asr_buffer_t *buffer)
{
    shl_mem_free(buffer->buffer);
    buffer->writer_index = 0;
    buffer->buffer = NULL;
    buffer->buffer_lenth = 0;
    buffer->data_lenth = 0;
    buffer->flag = 0;
}

int shl_c906_cache_matmul_init(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *weight, struct csinn_tensor *bias,
                               struct csinn_cache_matmul_params *params)
{
    size_t data_size =
        params->shape[0] * params->shape[1] * params->shape[2] * params->shape[3] * sizeof(__fp16);
    asr_buffer_init_c906(&params->asr_buffer, 2 * data_size, data_size);

    int accum_depth = weight->dim[0];
    int output_depth = weight->dim[1];

    struct csinn_callback *cb = params->base.cb;
    if (input->dtype == CSINN_DTYPE_FLOAT16) {
        __fp16 *weight_data = (__fp16 *)weight->data;

        int n = weight->dim[0];  // out_nodes
        int k = weight->dim[1];  // in_nodes
        if (k % 16 != 0) {
            shl_debug_error("out_nodes num should be multiple of 16\n");
        }
        __fp16 *pa_reorder = (__fp16 *)shl_mem_alloc(n * k * sizeof(__fp16));
        shl_c906_reorder_weight_n16_fp16(weight_data, pa_reorder, n, k, k);

        shl_c906_memcpy(weight_data, pa_reorder, n * k * sizeof(__fp16));
        params->data = weight_data;
        shl_mem_free(pa_reorder);
        cb->exec = shl_c906_cache_matmul_fp16;
    }
    return CSINN_TRUE;
}

int shl_c906_cache_matmul_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *weight, struct csinn_tensor *bias,
                               struct csinn_cache_matmul_params *params)
{
    int accum_depth = weight->dim[0];
    int output_depth = weight->dim[1];
    int batches = input->dim[1];

    __fp16 *input_data = input->data;
    __fp16 *output_data = output->data;
    __fp16 *weights_data = params->data;
    __fp16 *bias_data = bias->data;

    int packn = 16;
    int vl = 16;
    int b = 0;
    for (; b + 3 < batches; b += 4) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_output2 = init_output + output_depth;
        __fp16 *init_output3 = init_output2 + output_depth;
        __fp16 *init_output4 = init_output3 + output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_input2 = init_input + accum_depth;
        __fp16 *init_input3 = init_input2 + accum_depth;
        __fp16 *init_input4 = init_input3 + accum_depth;

        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;
        int n = output_depth;
        while (n > 0) {
            __fp16 *in_ptr = init_input;
            __fp16 *in_ptr2 = init_input2;
            __fp16 *in_ptr3 = init_input3;
            __fp16 *in_ptr4 = init_input4;

            vfloat16m2_t _acc = vle16_v_f16m2(init_bias, vl);
            vfloat16m2_t _acc2 = vmv_v_v_f16m2(_acc, vl);
            vfloat16m2_t _acc3 = vmv_v_v_f16m2(_acc, vl);
            vfloat16m2_t _acc4 = vmv_v_v_f16m2(_acc, vl);

            init_bias += vl;
            int k = accum_depth;
            while (k > 0) {
                vfloat16m2_t _weight = vle16_v_f16m2(init_weight, vl);
                _acc = vfmacc_vf_f16m2(_acc, *in_ptr, _weight, vl);
                _acc2 = vfmacc_vf_f16m2(_acc2, *in_ptr2, _weight, vl);
                _acc3 = vfmacc_vf_f16m2(_acc3, *in_ptr3, _weight, vl);
                _acc4 = vfmacc_vf_f16m2(_acc4, *in_ptr4, _weight, vl);
                init_weight += vl;
                in_ptr++;
                in_ptr2++;
                in_ptr3++;
                in_ptr4++;
                k--;
            }
            vse16_v_f16m2(init_output, _acc, vl);
            vse16_v_f16m2(init_output2, _acc2, vl);
            vse16_v_f16m2(init_output3, _acc3, vl);
            vse16_v_f16m2(init_output4, _acc4, vl);
            init_output += vl;
            init_output2 += vl;
            init_output3 += vl;
            init_output4 += vl;
            n -= vl;
        }
    }
    for (; b + 1 < batches; b += 2) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_output2 = init_output + output_depth;
        __fp16 *init_input = input_data + b * accum_depth;
        __fp16 *init_input2 = init_input + accum_depth;

        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;
        int n = output_depth;
        while (n > 0) {
            __fp16 *in_ptr = init_input;
            __fp16 *in_ptr2 = init_input2;
            vfloat16m2_t _acc = vle16_v_f16m2(init_bias, vl);
            vfloat16m2_t _acc2 = vmv_v_v_f16m2(_acc, vl);
            init_bias += vl;
            int k = accum_depth;
            while (k > 0) {
                vfloat16m2_t _weight = vle16_v_f16m2(init_weight, vl);
                _acc = vfmacc_vf_f16m2(_acc, *in_ptr, _weight, vl);
                _acc2 = vfmacc_vf_f16m2(_acc2, *in_ptr2, _weight, vl);
                init_weight += vl;
                in_ptr++;
                in_ptr2++;
                k--;
            }
            vse16_v_f16m2(init_output, _acc, vl);
            vse16_v_f16m2(init_output2, _acc2, vl);
            init_output += vl;
            init_output2 += vl;
            n -= vl;
        }
    }
    for (; b < batches; b++) {
        __fp16 *init_output = output_data + b * output_depth;
        __fp16 *init_input = input_data + b * accum_depth;

        __fp16 *init_weight = weights_data;
        __fp16 *init_bias = bias_data;
        int n = output_depth;
        while (n > 0) {
            __fp16 *in_ptr = init_input;
            vfloat16m2_t _acc = vle16_v_f16m2(init_bias, vl);
            init_bias += vl;
            int k = accum_depth;
            while (k > 0) {
                vfloat16m2_t _weight = vle16_v_f16m2(init_weight, vl);
                _acc = vfmacc_vf_f16m2(_acc, *in_ptr, _weight, vl);
                init_weight += vl;
                in_ptr++;
                k--;
            }
            vse16_v_f16m2(init_output, _acc, vl);
            init_output += vl;
            n -= vl;
        }
    }
    __fp16 judge =
        bias_data[0] + bias_data[1] + bias_data[2] + bias_data[3] + bias_data[4] + bias_data[5];

    size_t insert_lenth = output_depth * batches;
    __fp16 *output_from_buffer;
    if (fabs(judge) < 0.01) {
        output_from_buffer = asr_buffer_insert_c906_front(&params->asr_buffer, output_data,
                                                          insert_lenth * sizeof(__fp16));
    } else {
        output_from_buffer = asr_buffer_insert_c906_back(&params->asr_buffer, output_data,
                                                         insert_lenth * sizeof(__fp16));
    }

    // deal with reshape & transpose
    int *shape = output->dim;

    // transpose can only be 0,2,3,1 or 0,2,1,3
    if (params->axes[2] == 3)  // 0,2,3,1
    {
        int batch = shape[3];
        int shape3 = shape[2];
        int flatten_shape = shape[1] * shape[2];
        __fp16 *ptr = output_from_buffer;
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < flatten_shape; j += 16) {
                int out_pos = j * batch + i;
                vfloat16m2_t _output_from_buffer;
                _output_from_buffer = vle16_v_f16m2(ptr, 16);
                vsse16_v_f16m2(output_data + out_pos, 2 * batch, _output_from_buffer, 16);
                ptr += 16;
            }
        }

    } else  // 0,2,1,3
    {
        int batch = shape[2];
        int shape3 = shape[3];
        int flatten_shape = shape[1] * shape[3];
        __fp16 *ptr = output_from_buffer;
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < flatten_shape; j += 16) {
                int out_pos = i * shape3 + j % shape3 + batch * shape3 * (j / shape3);
                vfloat16m2_t v_output_from_buffer;
                v_output_from_buffer = vle16_v_f16m2(ptr, 16);
                vse16_v_f16m2(output_data + out_pos, v_output_from_buffer, 16);
                ptr += 16;
            }
        }
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, weight);
    return CSINN_TRUE;
}
