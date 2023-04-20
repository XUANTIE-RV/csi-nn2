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

/* SHL version 2.1.x */

#include <math.h>

#include "shl_ref.h"

int shl_ref_where_softmax_f32(struct csinn_tensor *condition, struct csinn_tensor *y,
                              struct csinn_tensor *output,
                              struct csinn_where_softmax_params *params)
{
    bool *condition_data = condition->data;
    struct csinn_tensor *where_input[2] = {condition, y};

    int where_out_size = 0;
    int32_t broadcast_dim[MAX_DIM];
    int32_t broadcast_dim_count;
    for (int i = 0; i < 2; i++) {
        int t_size = csinn_tensor_size(where_input[i]);
        if (t_size > where_out_size) {
            where_out_size = t_size;
            memcpy(broadcast_dim, where_input[i]->dim, MAX_DIM * 4);
            broadcast_dim_count = where_input[i]->dim_count;
        }
    }

    uint8_t *c_data_b = shl_mem_alloc(where_out_size);
    float *y_data_b = shl_mem_alloc(where_out_size * 4);
    float *where_output_data = shl_mem_alloc(where_out_size * 4);
    struct csinn_tensor *b_c = csinn_alloc_tensor(NULL);
    struct csinn_tensor *b_y = csinn_alloc_tensor(NULL);
    struct csinn_tensor *where_out = csinn_alloc_tensor(NULL);

    csinn_tensor_copy(where_out, output);
    where_out->dim_count = broadcast_dim_count;
    memcpy(where_out->dim, broadcast_dim, MAX_DIM * 4);

    csinn_tensor_copy(b_c, condition);
    b_c->dim_count = broadcast_dim_count;
    memcpy(b_c->dim, broadcast_dim, MAX_DIM * 4);

    csinn_tensor_copy(b_y, y);
    b_y->dim_count = broadcast_dim_count;
    memcpy(b_y->dim, broadcast_dim, MAX_DIM * 4);

    b_c->data = c_data_b;
    b_y->data = y_data_b;
    where_out->data = where_output_data;

    if (shl_ref_broadcast_to_shape(condition, b_c, output->dim, output->dim_count) == CSINN_FALSE) {
        SHL_DEBUG_CALL(shl_debug_info("%s: broadcast condition failed.\n", __func__));
        return CSINN_FALSE;
    };

    if (shl_ref_broadcast_to_shape(y, b_y, output->dim, output->dim_count) == CSINN_FALSE) {
        SHL_DEBUG_CALL(shl_debug_info("%s: broadcast y failed.\n", __func__));
        return CSINN_FALSE;
    };

    int size0 = csinn_tensor_size(b_c);
    int size2 = csinn_tensor_size(b_y);

    if (size0 != size2) {
        return CSINN_FALSE;
    }

    for (int i = 0; i < csinn_tensor_size(b_c); i++) {
        if (c_data_b[i] == 1) {
            where_output_data[i] = -(float)HUGE_VALF;
        } else {
            where_output_data[i] = y_data_b[i];
        }
    }
    struct csinn_softmax_params *soft_params =
        csinn_alloc_params(sizeof(struct csinn_softmax_params), NULL);
    soft_params->axis = params->axis;
    int ret = shl_ref_softmax_f32(where_out, output, soft_params);

    shl_ref_tensor_transform_free_f32(b_c);
    shl_ref_tensor_transform_free_f32(b_y);

    shl_mem_free(where_output_data);
    csinn_free_params(soft_params);
    csinn_free_tensor(where_out);

    return ret;
}

int shl_ref_where_softmax_quant(struct csinn_tensor *condition, struct csinn_tensor *y,
                                struct csinn_tensor *output,
                                struct csinn_where_softmax_params *params)
{
    struct csinn_tensor *float_y = shl_ref_tensor_transform_f32(y);
    struct csinn_tensor *float_output = shl_ref_tensor_transform_f32(output);
    int ret = shl_ref_where_softmax_f32(condition, float_y, float_output, params);
    csinn_tensor_data_convert(output, float_output);
    shl_ref_tensor_transform_free_f32(float_output);
    shl_ref_tensor_transform_free_f32(float_y);
    return ret;
}
