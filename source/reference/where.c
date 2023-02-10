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

#include "shl_ref.h"

int shl_ref_where_f32(struct csinn_tensor *condition, struct csinn_tensor *x,
                      struct csinn_tensor *y, struct csinn_tensor *output,
                      struct csinn_where_params *params)
{
    float *output_data = output->data;

    int out_size = csinn_tensor_size(output);
    uint8_t *c_data_b = shl_mem_alloc(out_size);
    float *x_data_b = shl_mem_alloc(out_size * sizeof(float));
    float *y_data_b = shl_mem_alloc(out_size * sizeof(float));
    struct csinn_tensor *b_c = csinn_alloc_tensor(NULL);
    struct csinn_tensor *b_x = csinn_alloc_tensor(NULL);
    struct csinn_tensor *b_y = csinn_alloc_tensor(NULL);
    csinn_tensor_copy(b_c, condition);
    b_c->dim_count = output->dim_count;
    memcpy(b_c->dim, output->dim, MAX_DIM * 4);
    csinn_tensor_copy(b_x, x);
    b_x->dim_count = output->dim_count;
    memcpy(b_x->dim, output->dim, MAX_DIM * 4);
    csinn_tensor_copy(b_y, y);
    b_y->dim_count = output->dim_count;
    memcpy(b_y->dim, output->dim, MAX_DIM * 4);

    b_c->data = c_data_b;
    b_x->data = x_data_b;
    b_y->data = y_data_b;

    if (shl_ref_broadcast_to_shape(condition, b_c, output->dim, output->dim_count) == CSINN_FALSE) {
        SHL_DEBUG_CALL(shl_debug_info("%s: broadcast condition failed.\n", __func__));
        return CSINN_FALSE;
    };

    if (shl_ref_broadcast_to_shape(x, b_x, output->dim, output->dim_count) == CSINN_FALSE) {
        SHL_DEBUG_CALL(shl_debug_info("%s: broadcast x failed.\n", __func__));
        return CSINN_FALSE;
    };

    if (shl_ref_broadcast_to_shape(y, b_y, output->dim, output->dim_count) == CSINN_FALSE) {
        SHL_DEBUG_CALL(shl_debug_info("%s: broadcast y failed.\n", __func__));
        return CSINN_FALSE;
    };

    int size0 = csinn_tensor_size(b_c);
    int size1 = csinn_tensor_size(b_x);
    int size2 = csinn_tensor_size(b_y);

    if (size0 != size1 || size1 != size2) {
        return CSINN_FALSE;
    }

    for (int i = 0; i < csinn_tensor_size(b_c); i++) {
        if (c_data_b[i] == 1) {
            output_data[i] = x_data_b[i];
        } else {
            output_data[i] = y_data_b[i];
        }
    }

    shl_ref_tensor_transform_free_f32(b_c);
    shl_ref_tensor_transform_free_f32(b_x);
    shl_ref_tensor_transform_free_f32(b_y);

    return CSINN_TRUE;
}

int shl_ref_where_quant(struct csinn_tensor *condition, struct csinn_tensor *x,
                        struct csinn_tensor *y, struct csinn_tensor *output,
                        struct csinn_where_params *params)
{
    struct csinn_tensor *float_x = shl_ref_tensor_transform_f32(x);
    struct csinn_tensor *float_y = shl_ref_tensor_transform_f32(y);
    struct csinn_tensor *float_output = shl_ref_tensor_transform_f32(output);
    int ret = shl_ref_where_f32(condition, float_x, float_y, float_output, params);
    csinn_tensor_data_convert(output, float_output);
    shl_ref_tensor_transform_free_f32(float_output);
    shl_ref_tensor_transform_free_f32(float_x);
    shl_ref_tensor_transform_free_f32(float_y);
    return ret;
}
