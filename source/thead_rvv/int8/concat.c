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
#include "shl_thead_rvv.h"

int shl_rvv_concat_int8(struct csinn_tensor **input, struct csinn_tensor *output,
                        struct csinn_concat_params *params)
{
    int64_t outer_size = 1;
    for (int i = 0; i < params->axis; ++i) {
        outer_size *= output->dim[i];
    }
    int64_t base_inner_size = 1;
    for (int i = params->axis + 1; i < output->dim_count; ++i) {
        base_inner_size *= output->dim[i];
    }
    int vl;
    int8_t *output_ptr = (int8_t *)output->data;
    for (int k = 0; k < outer_size; k++) {
        for (int i = 0; i < params->inputs_count; ++i) {
            struct csinn_tensor *input_item = input[i];
            int8_t *input_item_data = (int8_t *)input_item->data;
            int copy_size = input_item->dim[params->axis] * base_inner_size;
            const int8_t *input_ptr = input_item_data + k * copy_size;
            while (copy_size > 0) {
                vl = vsetvl_e8m2(copy_size);
                vint8m2_t _input = vle8_v_i8m2(input_ptr, vl);
                input_ptr += vl;
                vse8_v_i8m2(output_ptr, _input, vl);
                output_ptr += vl;
                copy_size -= vl;
            }
        }
    }
    return CSINN_TRUE;
}
