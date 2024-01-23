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

#include "rvv/rvv.h"

int shl_rvv_split_fp32(struct csinn_tensor *input, struct csinn_tensor **output,
                       struct csinn_split_params *params)
{
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(input);
    }
    int32_t inner_size = 1;
    int32_t out_size = 1;
    int32_t axis = params->axis < 0 ? (params->axis + input->dim_count) : params->axis;
    int32_t avg_dim = (input->dim[axis] + params->output_num - 1) / params->output_num;
    float *input_data = input->data;

    for (int i = 0; i < axis; i++) {
        out_size *= input->dim[i];
    }

    for (int i = axis + 1; i < input->dim_count; i++) {
        inner_size *= input->dim[i];
    }

    for (int i = 0; i < params->output_num; i++) {
        int p_size = 0;
        int s_index;
        if (params->split_index != NULL) {
            if (i == params->output_num - 1) {
                p_size = inner_size * (input->dim[axis] - params->split_index[i - 1]);
                s_index = params->split_index[i - 1];
            } else if (i == 0) {
                p_size = inner_size * params->split_index[0];
                s_index = 0;
            } else {
                p_size = inner_size * (params->split_index[i] - params->split_index[i - 1]);
                s_index = params->split_index[i - 1];
            }
        } else {
            if (i == params->output_num - 1) {
                p_size = inner_size * (input->dim[axis] - i * avg_dim);
            } else {
                p_size = inner_size * avg_dim;
            }
            s_index = i * avg_dim;
        }
        float *output_i_data = output[i]->data;

        for (int out = 0; out < out_size; out++) {
            int in_index = out * input->dim[axis] * inner_size + s_index * inner_size;
            memcpy(output_i_data, input_data + in_index, p_size * sizeof(float));
            output_i_data += p_size;
        }
    }

    return CSINN_TRUE;
}
