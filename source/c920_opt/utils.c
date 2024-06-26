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

#include "c920/c920.h"

// Only dtype conversion is supported. Layout conversion is not supported.
void *shl_c920_f32_to_input_dtype(uint32_t index, float *data, struct csinn_session *sess)
{
    struct csinn_tensor *t = sess->input[index];
    float scale = t->qinfo->scale;
    int32_t zero_point = t->qinfo->zero_point;
    uint32_t size = csinn_tensor_size(t);
    void *ret_data;
    if (t->dtype == CSINN_DTYPE_UINT8) {
        ret_data = shl_mem_alloc(size * sizeof(uint8_t));
        shl_c920_f32_to_u8(data, ret_data, zero_point, &scale, size);
    } else if (t->dtype == CSINN_DTYPE_INT8) {
        ret_data = shl_mem_alloc(size * sizeof(int8_t));
        shl_c920_f32_to_i8(data, ret_data, zero_point, &scale, size);
    } else if (t->dtype == CSINN_DTYPE_INT16) {
        ret_data = shl_mem_alloc(size * sizeof(int16_t));
        shl_rvv_f32_to_i16(data, (int16_t *)ret_data, zero_point, &scale, size);
    } else if (t->dtype == CSINN_DTYPE_INT32) {
        ret_data = shl_mem_alloc(size * sizeof(int32_t));
        shl_rvv_f32_to_i32(data, (int32_t *)ret_data, zero_point, &scale, size);
    } else if (t->dtype == CSINN_DTYPE_INT64) {
        ret_data = shl_mem_alloc(size * sizeof(int64_t));
        shl_rvv_f32_to_i64(data, (int64_t *)ret_data, size);
    } else if (t->dtype == CSINN_DTYPE_FLOAT16) {
        ret_data = shl_mem_alloc(size * sizeof(__fp16));
        shl_rvv_f32_to_f16(data, (__fp16 *)ret_data, &scale, size);
    } else if (t->dtype == CSINN_DTYPE_FLOAT32) {
        ret_data = shl_mem_alloc(size * sizeof(float));
        memcpy(ret_data, data, csinn_tensor_byte_size(t));
    } else {
        ret_data = shl_ref_f32_to_input_dtype(index, data, sess);
    }
    return ret_data;
}

// Only dtype conversion is supported. Layout conversion is not supported.
float *shl_c920_output_to_f32_dtype(uint32_t index, void *data, struct csinn_session *sess)
{
    struct csinn_tensor *t = sess->output[index];
    float scale = t->qinfo->scale;
    int32_t zero_point = t->qinfo->zero_point;
    uint32_t size = csinn_tensor_size(t);
    float *ret_data = (float *)shl_mem_alloc(size * sizeof(float));

    if (t->dtype == CSINN_DTYPE_UINT8) {
        shl_c920_u8_to_f32(data, ret_data, zero_point, &scale, size);
    } else if (t->dtype == CSINN_DTYPE_INT8) {
        shl_c920_i8_to_f32(data, ret_data, zero_point, &scale, size);
    } else if (t->dtype == CSINN_DTYPE_INT16) {
        shl_rvv_i16_to_f32((int16_t *)data, ret_data, zero_point, &scale, size);
    } else if (t->dtype == CSINN_DTYPE_INT32) {
        shl_rvv_i32_to_f32((int32_t *)data, ret_data, zero_point, &scale, size);
    } else if (t->dtype == CSINN_DTYPE_INT64) {
        shl_rvv_i64_to_f32((int64_t *)data, ret_data, size);
    } else if (t->dtype == CSINN_DTYPE_FLOAT16) {
        shl_rvv_f16_to_f32((__fp16 *)data, ret_data, &scale, size);
    } else if (t->dtype == CSINN_DTYPE_FLOAT32) {
        memcpy(ret_data, data, csinn_tensor_byte_size(t));
    } else {
        shl_debug_error("output_to_f32 unsupported dtype: %d\n", t->dtype);
        return NULL;
    }
    return ret_data;
}

bool shl_c920_get_binary_model_op_init(struct csinn_session *sess)
{
    struct shl_c920_option *option = shl_c920_get_graph_option(sess);
    if (option && option->base.binary_model_op_init) {
        return true;
    } else {
        return false;
    }
}

void shl_c920_set_binary_model_op_init(struct csinn_session *sess, bool value)
{
    struct shl_c920_option *option = shl_c920_get_graph_option(sess);
    option->base.binary_model_op_init = value;
}
