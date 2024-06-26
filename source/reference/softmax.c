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

int shl_ref_softmax_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_softmax_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int axis = params->axis;
    // FlatSize() = outer_size * inner_size * cnt;
    int64_t outer_size = 1;
    for (int i = 0; i < axis; i++) {
        outer_size *= input->dim[i];
    }

    int64_t inner_size = 1;
    for (int i = axis + 1; i < input->dim_count; i++) {
        inner_size *= input->dim[i];
    }

    int cnt = input->dim[axis];

    for (int i = 0; i < outer_size; i++) {
        for (int k = 0; k < inner_size; k++) {
            float acc_exp = 0.0f;
            float max = -FLT_MAX;
            // Find max element value which we'll use to ensure numerical stability
            // taking advantage of the following equality:
            // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
            for (int j = 0; j < cnt; j++) {
                max = fmax(max, *(input_data + j * inner_size + k));
            }

            // compute sum
            for (int j = 0; j < cnt; j++) {
                acc_exp += exp(*(input_data + j * inner_size + k) - max);
            }

            // compute final result
            for (int j = 0; j < cnt; j++) {
                *(output_data + j * inner_size + k) =
                    exp(*(input_data + j * inner_size + k) - max) / acc_exp;
            }
        }
        input_data += inner_size * cnt;
        output_data += inner_size * cnt;
    }
    return CSINN_TRUE;
}

int shl_ref_softmax_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_softmax_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_softmax_f32);
}
