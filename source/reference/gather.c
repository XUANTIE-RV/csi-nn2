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

int shl_ref_gather_f32(struct csinn_tensor *input, struct csinn_tensor *indices,
                       struct csinn_tensor *output, struct csinn_gather_params *params)
{
    int input_size = csinn_tensor_size(input);
    if (input_size == 0) {
        return CSINN_TRUE;
    }

    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    assert(indices->dtype == CSINN_DTYPE_INT64);
    int64_t *indices_data = (int64_t *)indices->data;

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
                memcpy(output_data, input_data + indices_data[j] * inner_size,
                       inner_size * sizeof(float));
            } else if ((indices_data[j] < 0) && (indices_data[j] >= -axis_shape)) {
                memcpy(output_data, input_data + (indices_data[j] + axis_shape) * inner_size,
                       inner_size * sizeof(float));
            } else {
                memset(output_data, 0, inner_size * sizeof(float));
            }
            output_data += inner_size;
        }
        input_data += inner_size * axis_shape;
    }
    return CSINN_TRUE;
}

#if __riscv
int shl_ref_gather_f16(struct csinn_tensor *input, struct csinn_tensor *indices,
                       struct csinn_tensor *output, struct csinn_gather_params *params)
{
    int input_size = csinn_tensor_size(input);
    if (input_size == 0) {
        return CSINN_TRUE;
    }

    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    int64_t *indices_data = (int64_t *)indices->data;

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
                memcpy(output_data, input_data + indices_data[j] * inner_size,
                       inner_size * sizeof(__fp16));
            } else if ((indices_data[j] < 0) && (indices_data[j] >= -axis_shape)) {
                memcpy(output_data, input_data + (indices_data[j] + axis_shape) * inner_size,
                       inner_size * sizeof(__fp16));
            } else {
                memset(output_data, 0, inner_size * sizeof(__fp16));
            }
            output_data += inner_size;
        }
        input_data += inner_size * axis_shape;
    }
    return CSINN_TRUE;
}
#endif  // __riscv

// XXX: precision loss
int shl_ref_gather_int8(struct csinn_tensor *input, struct csinn_tensor *indices,
                        struct csinn_tensor *output, struct csinn_gather_params *params)
{
    int input_size = csinn_tensor_size(input);
    if (input_size == 0) {
        return CSINN_TRUE;
    }

    if (input->dtype == CSINN_DTYPE_INT8 && indices->dtype == CSINN_DTYPE_INT64 &&
        output->dtype == CSINN_DTYPE_INT8) {
        int8_t *input_data = (int8_t *)input->data;
        int8_t *output_data = (int8_t *)output->data;
        int64_t *indices_data = (int64_t *)indices->data;

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
                    memcpy(output_data, input_data + indices_data[j] * inner_size,
                           inner_size * sizeof(int8_t));
                } else if ((indices_data[j] < 0) && (indices_data[j] >= -axis_shape)) {
                    memcpy(output_data, input_data + (indices_data[j] + axis_shape) * inner_size,
                           inner_size * sizeof(int8_t));
                } else {
                    memset(output_data, 0, inner_size * sizeof(int8_t));  // xxx: input_zp?
                }
                output_data += inner_size;
            }
            input_data += inner_size * axis_shape;
        }
    } else {
        return shl_ref_gather_quant(input, indices, output, params);
    }

    return CSINN_TRUE;
}

int shl_ref_gather_quant(struct csinn_tensor *input, struct csinn_tensor *indices,
                         struct csinn_tensor *output, struct csinn_gather_params *params)
{
    int ret;
    // for dynamic shape
    int input_size = csinn_tensor_size(input);
    if (input_size == 0) {
        return CSINN_TRUE;
    }

    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    ret = shl_ref_gather_f32(finput, indices, foutput, params);
    csinn_tensor_data_convert(output, foutput);
    shl_ref_tensor_transform_free_f32(finput);
    shl_ref_tensor_transform_free_f32(foutput);
    return ret;
}
