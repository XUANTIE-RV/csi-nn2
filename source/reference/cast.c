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

#include "reference/ref.h"

int shl_bytes_for_dtype(enum csinn_dtype_enum dtype)
{
    switch (dtype) {
        case CSINN_DTYPE_BOOL:
        case CSINN_DTYPE_INT8:
        case CSINN_DTYPE_UINT8:
            return 1;
            break;
        case CSINN_DTYPE_INT16:
        case CSINN_DTYPE_UINT16:
        case CSINN_DTYPE_FLOAT16:
        case CSINN_DTYPE_BFLOAT16:
            return 2;
            break;
        case CSINN_DTYPE_INT32:
        case CSINN_DTYPE_UINT32:
        case CSINN_DTYPE_FLOAT32:
            return 4;
            break;
        case CSINN_DTYPE_INT64:
        case CSINN_DTYPE_FLOAT64:
            return 8;
            break;

        default:
            shl_debug_info("%s: Cannot find bits for dtype\n", __func__);
            return CSINN_FALSE;
            break;
    }
}

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
    } else if (params->dtype == CSINN_DTYPE_INT32) {
        int32_t *output_data = (int32_t *)output->data;
        for (int i = 0; i < size; i++) {
            output_data[i] = (int32_t)(input_data[i]);
        }
    } else if (params->dtype == CSINN_DTYPE_FLOAT32) {
        memcpy(output->data, input_data, sizeof(float) * size);
    } else {
        shl_debug_error("Unsupport destination type of float input\n");
    }
    return CSINN_TRUE;
}

int shl_ref_cast_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_cast_params *params)
{
    int ret;
    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *c_output = csinn_alloc_tensor(NULL);
    csinn_tensor_copy(c_output, output);
    if (c_output->qinfo != NULL) {
        shl_mem_free(c_output->qinfo);
        c_output->qinfo = NULL;
    }
    c_output->quant_channel = 0;
    c_output->dtype = params->dtype;
    int input_size = csinn_tensor_size(input);
    if (input_size == 0) {
        return CSINN_TRUE;
    }
    c_output->data = shl_mem_alloc(input_size * shl_bytes_for_dtype(params->dtype));

    ret = shl_ref_cast_f32(finput, c_output, params);
    if (params->dtype == CSINN_DTYPE_FLOAT32) {
        csinn_tensor_data_convert(output, c_output);
    } else {
        if (output->dtype != params->dtype) {
            shl_debug_error("%s: output's dtype and params's dtype are not equal.\n", __func__);
            ret = CSINN_FALSE;
        } else {
            memcpy(output->data, c_output->data, input_size * shl_bytes_for_dtype(params->dtype));
        }
    }
    shl_ref_tensor_transform_free_f32(finput);
    shl_ref_tensor_transform_free_f32(c_output);
    return ret;
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