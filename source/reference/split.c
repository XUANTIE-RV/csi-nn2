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

int shl_ref_split_f32(struct csinn_tensor *input, struct csinn_tensor **output,
                      struct csinn_split_params *params)
{
    int32_t inner_size = 1;
    int32_t out_size = 1;
    float *input_data = input->data;

    for (int i = 0; i < params->axis; i++) {
        out_size *= input->dim[i];
    }

    for (int i = params->axis + 1; i < input->dim_count; i++) {
        inner_size *= input->dim[i];
    }

    for (int i = 0; i < params->output_num; i++) {
        int p_size = 0;
        int s_index;
        if (i == params->output_num - 1) {
            p_size = inner_size * (input->dim[params->axis] - params->split_index[i - 1]);
            s_index = params->split_index[i - 1];
        } else if (i == 0) {
            p_size = inner_size * params->split_index[0];
            s_index = 0;
        } else {
            p_size = inner_size * (params->split_index[i] - params->split_index[i - 1]);
            s_index = params->split_index[i - 1];
        }

        float *output_i_data = output[i]->data;

        for (int out = 0; out < out_size; out++) {
            int in_index = out * input->dim[params->axis] * inner_size + s_index * inner_size;
            int out_index = out * inner_size;
            memcpy(output_i_data + out_index, input_data + in_index, p_size * 4);
        }
    }

    return CSINN_TRUE;
}

int shl_ref_split_quant(struct csinn_tensor *input, struct csinn_tensor **output,
                        struct csinn_split_params *params)
{
    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input);

    struct csinn_tensor *foutput[params->output_num];
    for (int i = 0; i < params->output_num; i++) {
        foutput[i] = shl_ref_tensor_transform_f32(output[i]);
    }
    int ret = shl_ref_split_f32(finput, foutput, params);

    for (int i = 0; i < params->output_num; i++) {
        csinn_tensor_data_convert(output[i], foutput[i]);
        shl_ref_tensor_transform_free_f32(foutput[i]);
    }
    shl_ref_tensor_transform_free_f32(finput);

    return ret;
}