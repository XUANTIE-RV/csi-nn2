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

int shl_ref_where_f32(struct csinn_tensor *condition, struct csinn_tensor *x,
                      struct csinn_tensor *y, struct csinn_tensor *output,
                      struct csinn_where_params *params)
{
    float *condition_data = condition->data;
    float *x_data = x->data;
    float *y_data = y->data;
    float *output_data = output->data;
    for (int i = 0; i < csinn_tensor_size(condition); i++) {
        if (condition_data[i] != 0) {
            output_data[i] = x_data[i];
        } else {
            output_data[i] = y_data[i];
        }
    }
    return CSINN_TRUE;
}

int shl_ref_where_quant(struct csinn_tensor *condition, struct csinn_tensor *x,
                        struct csinn_tensor *y, struct csinn_tensor *output,
                        struct csinn_where_params *params)
{
    struct csinn_tensor *float_input = shl_ref_tensor_transform_f32(condition);
    struct csinn_tensor *float_x = shl_ref_tensor_transform_f32(x);
    struct csinn_tensor *float_y = shl_ref_tensor_transform_f32(y);
    struct csinn_tensor *float_output = shl_ref_tensor_transform_f32(output);
    int ret = shl_ref_where_f32(float_input, float_x, float_y, float_output, params);
    csinn_tensor_data_convert(output, float_output);
    shl_ref_tensor_transform_free_f32(float_input);
    shl_ref_tensor_transform_free_f32(float_output);
    shl_ref_tensor_transform_free_f32(float_x);
    shl_ref_tensor_transform_free_f32(float_y);
    return ret;
}
