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

#include "shl_ref.h"

int shl_ref_cast_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_cast_params *params)
{
    float *input_data = (float *)input->data;
    int size = csinn_tensor_size(input);
    if (params->dtype == CSINN_DTYPE_BOOL) {
        bool *output_data = (bool *)output->data;
        for (int i = 0; i < size; i++) {
            output_data[i] = (bool)(input_data[i]);
        }
    } else if (params->dtype == CSINN_DTYPE_INT8) {
        int8_t *output_data = (int8_t *)output->data;
        for (int i = 0; i < size; i++) {
            output_data[i] = (int8_t)(input_data[i]);
        }
    } else {
        shl_debug_error("Unsupport destination type of float input\n");
    }
    return CSINN_TRUE;
}

int shl_ref_cast_bool(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_cast_params *params)
{
    bool *input_data = (bool *)input->data;
    int size = csinn_tensor_size(input);
    if (params->dtype == CSINN_DTYPE_INT8) {
        int8_t *output_data = (int8_t *)output->data;
        for (int i = 0; i < size; i++) {
            output_data[i] = (int8_t)(input_data[i]);
        }
    } else if (params->dtype == CSINN_DTYPE_INT64) {
        int64_t *output_data = (int64_t *)output->data;
        for (int i = 0; i < size; i++) {
            output_data[i] = (int64_t)(input_data[i]);
        }
    } else {
        shl_debug_error("Unsupport destination type of bool input\n");
    }
    return CSINN_TRUE;
}

int shl_ref_cast_i64(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_cast_params *params)
{
    int64_t *input_data = (int64_t *)input->data;
    int size = csinn_tensor_size(input);
    if (params->dtype == CSINN_DTYPE_BOOL) {
        bool *output_data = (bool *)output->data;
        for (int i = 0; i < size; i++) {
            output_data[i] = (bool)(input_data[i]);
        }
    } else if (params->dtype == CSINN_DTYPE_INT8) {
        int8_t *output_data = (int8_t *)output->data;
        for (int i = 0; i < size; i++) {
            output_data[i] = (int8_t)(input_data[i]);
        }
    } else {
        shl_debug_error("Unsupport destination type of int64 input\n");
    }
    return CSINN_TRUE;
}