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

int shl_ref_ndarray_size_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_ndarray_size_params *params)
{
    float *output_data = output->data;
    output_data[0] = csinn_tensor_size(input);
    return CSINN_TRUE;
}

int shl_ref_ndarray_size_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_ndarray_size_params *params)
{
    uint8_t *output_data = output->data;
    output_data[0] = csinn_tensor_size(input);
    return CSINN_TRUE;
}

int shl_ref_ndarray_size_i8(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_ndarray_size_params *params)
{
    int8_t *output_data = output->data;
    output_data[0] = csinn_tensor_size(input);
    return CSINN_TRUE;
}

int shl_ref_ndarray_size_i32(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_ndarray_size_params *params)
{
    int32_t *output_data = output->data;
    output_data[0] = csinn_tensor_size(input);
    return CSINN_TRUE;
}
