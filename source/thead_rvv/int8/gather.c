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

#include "rvv/rvv.h"

static int shl_rvv_gather_in_i8_out_f16(struct csinn_tensor *input, struct csinn_tensor *indices,
                                        struct csinn_tensor *output,
                                        struct csinn_gather_params *params)
{
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(input);
    }

    int input_size = csinn_tensor_size(input);
    if (input_size == 0) {
        return CSINN_TRUE;
    }

    int8_t *input_data = (int8_t *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    int64_t *indices_data = (int64_t *)indices->data;

    int32_t zp = input->qinfo->zero_point;
    float scale = input->qinfo->scale;

    int inner_size = 1;
    for (int i = params->axis + 1; i < input->dim_count; i++) {
        inner_size *= input->dim[i];
    }
    int outer_size = 1;
    for (int i = 0; i < params->axis; i++) {
        outer_size *= input->dim[i];
    }
    int indices_size = 1;
    for (int i = 0; i < indices->dim_count; i++) {
        indices_size *= indices->dim[i];
    }
    int axis_shape = input->dim[params->axis];
    for (int i = 0; i < outer_size; i++) {
        for (int j = 0; j < indices_size; j++) {
            if ((indices_data[j] >= 0) && (indices_data[j] < axis_shape)) {
                shl_rvv_dequantize_i8_to_f16(input_data + indices_data[j] * inner_size, output_data,
                                             inner_size, zp, scale);
            } else if ((indices_data[j] < 0) && (indices_data[j] >= -axis_shape)) {
                shl_rvv_dequantize_i8_to_f16(
                    input_data + (indices_data[j] + axis_shape) * inner_size, output_data,
                    inner_size, zp, scale);
            } else {
                memset(output_data, 0, inner_size * sizeof(__fp16));
            }
            output_data += inner_size;
        }
        input_data += inner_size * axis_shape;
    }
    return CSINN_TRUE;
}

int shl_rvv_gather_int8(struct csinn_tensor *input, struct csinn_tensor *indices,
                        struct csinn_tensor *output, struct csinn_gather_params *params)
{
    if (input->dtype == CSINN_DTYPE_INT8 && output->dtype == CSINN_DTYPE_FLOAT16) {
        return shl_rvv_gather_in_i8_out_f16(input, indices, output, params);
    } else if (input->dtype == CSINN_DTYPE_INT8 && output->dtype == CSINN_DTYPE_INT8) {
        if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
            shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(input);
        }
        return shl_ref_gather_int8(input, indices, output, params);
    } else {
        return shl_ref_gather_quant(input, indices, output, params);
    }
}
