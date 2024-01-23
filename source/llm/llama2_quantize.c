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

/*
 * Block quantization from llama.cpp
 */

#include "llm/shl_llm.h"

static void shl_block_quantize_data_q4_0(const int16_t *src, int8_t *dst, int16_t *scale,
                                         int element_size, int block_size)
{
    const int block_num = element_size / block_size;

    for (int i = 0; i < block_num; i++) {
        float max_value = 0.0f;
        float abs_max_value = 0.0f;

        for (int j = 0; j < block_size; j++) {
            int16_t fp16_value = src[i * block_size + j];
            float fp32_value = shl_ref_float16_to_float32(fp16_value);
            if (abs_max_value < fabsf(fp32_value)) {
                abs_max_value = fabsf(fp32_value);
                max_value = fp32_value;
            }
        }

        float fp32_scale = max_value / -8.0f;
        float id = fp32_scale ? 1.0f / fp32_scale : 0.0f;

        scale[i] = shl_ref_float32_to_float16(fp32_scale);

        for (int j = 0; j < block_size / 2; ++j) {
            int16_t fp16_value0 = src[i * block_size + j];
            float fp32_value0 = shl_ref_float16_to_float32(fp16_value0);
            uint8_t q4_value0 = fminf((int8_t)(fp32_value0 * id + 8.5f), 15);

            int16_t fp16_value1 = src[i * block_size + block_size / 2 + j];
            float fp32_value1 = shl_ref_float16_to_float32(fp16_value1);
            uint8_t q4_value1 = fminf((int8_t)(fp32_value1 * id + 8.5f), 15);

            dst[i * block_size / 2 + j] = q4_value0 | (q4_value1 << 4);
        }
    }
}

static void shl_block_quantize_data_q8_0(const int16_t *src, int8_t *dst, int16_t *scale,
                                         int element_size, int block_size)
{
    const int block_num = element_size / block_size;

    for (int i = 0; i < block_num; i++) {
        float max_value = 0.0f;

        for (int j = 0; j < block_size; j++) {
            int16_t fp16_value = src[i * block_size + j];
            float fp32_value = shl_ref_float16_to_float32(fp16_value);
            max_value = fmaxf(max_value, fabsf(fp32_value));
        }

        float fp32_scale = max_value / ((1 << 7) - 1);
        float id = fp32_scale ? 1.0f / fp32_scale : 0.0f;

        scale[i] = shl_ref_float32_to_float16(fp32_scale);

        for (int j = 0; j < block_size; ++j) {
            int16_t fp16_value = src[i * block_size + j];
            float fp32_value = shl_ref_float16_to_float32(fp16_value);
            const float q8_value = fp32_value * id;

            dst[i * block_size + j] = roundf(q8_value);
        }
    }
}

int shl_block_quantize(struct csinn_tensor *src, struct csinn_tensor *dst)
{
    if (dst->mtype == CSINN_MEM_TYPE_BLOCK_Q8_0) {
        int q8_block_size = 32;
        /* fp16 scale */
        int scale_size = csinn_tensor_size(src) / q8_block_size * sizeof(int16_t);
        dst->data = shl_mem_alloc(csinn_tensor_size(src) + scale_size);
        void *scale_addr = dst->data + csinn_tensor_size(src);
        shl_block_quantize_data_q8_0(src->data, dst->data, scale_addr, csinn_tensor_size(src),
                                     q8_block_size);
    } else if (dst->mtype == CSINN_MEM_TYPE_BLOCK_Q4_0) {
        int q4_block_size = 32;
        /* fp16 scale */
        int scale_size = csinn_tensor_size(src) / q4_block_size * sizeof(int16_t);
        dst->data = shl_mem_alloc(csinn_tensor_size(src) / 2 + scale_size);
        void *scale_addr = dst->data + csinn_tensor_size(src) / 2;
        shl_block_quantize_data_q4_0(src->data, dst->data, scale_addr, csinn_tensor_size(src),
                                     q4_block_size);
    } else {
        return CSINN_FALSE;
    }

    return CSINN_TRUE;
}

struct csinn_tensor *quantize_tensor(struct csinn_tensor *src, enum csinn_mem_type_enum mtype)
{
    struct csinn_tensor *ret = csinn_alloc_tensor(NULL);
    ret->mtype = mtype;
    ret->dim_count = src->dim_count;
    for (int i = 0; i < ret->dim_count; i++) {
        ret->dim[i] = src->dim[i];
    }
    if (mtype == CSINN_MEM_TYPE_BLOCK_Q8_0) {
        ret->dtype = CSINN_DTYPE_INT8;
    } else if (mtype == CSINN_MEM_TYPE_BLOCK_Q4_0) {
        ret->dtype = CSINN_DTYPE_INT4;
    } else {
        shl_debug_error("Unsupport quantize type\n");
    }
    shl_block_quantize(src, ret);
    ret->name = shl_mem_alloc(strlen(src->name) + 1);
    strcpy(ret->name, src->name);
    return ret;
}
