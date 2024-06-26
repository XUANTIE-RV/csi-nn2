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

struct ArgPos {
    float value;
    int32_t index;
};

static struct ArgPos fargmax_stride(struct ArgPos lhs, struct ArgPos rhs)
{
    if (lhs.value < rhs.value) {
        return rhs;
    }
    return lhs;
}

int shl_ref_argmax_stride_i32_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_reduce_params *params)
{
    float *input_data = input->data;
    int32_t *output_data = output->data;

    int32_t inner_size = 1;
    int32_t out_size = 1;

    for (int32_t k = 0; k < params->n; k++) {
        out_size *= params->out_extents[k];
    }

    for (int32_t k = 0; k < params->m; k++) {
        inner_size *= params->inner_extents[k];
    }

    for (int32_t out = 0; out < out_size; out++) {
        struct ArgPos result = {-FLT_MAX, -1};
        int32_t out_index =
            shl_ref_get_reduction_index(out, params->out_strides, params->out_extents, params->n);
        for (int32_t inner = 0; inner < inner_size; inner++) {
            int32_t index =
                out_index + shl_ref_get_reduction_index(inner, params->inner_strides,
                                                        params->inner_extents, params->m);
            float val = input_data[index];
            struct ArgPos pos = {val, inner};
            result = fargmax_stride(result, pos);
        }

        output_data[out] = result.index;
    }
    return CSINN_TRUE;
}

int shl_ref_argmax_stride_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_reduce_params *params)
{
    int ret;
    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input);
    ret = shl_ref_argmax_stride_i32_f32(finput, output, params);
    shl_ref_tensor_transform_free_f32(finput);
    return ret;
}
