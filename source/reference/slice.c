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

#include "reference/ref.h"

int shl_ref_slice_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_slice_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    if (input->dim_count == 4) {
        for (int b = params->begin[0]; b < params->end[0]; b++) {
            for (int c = params->begin[1]; c < params->end[1]; c++) {
                for (int h = params->begin[2]; h < params->end[2]; h++) {
                    for (int w = params->begin[3]; w < params->end[3]; w++) {
                        int32_t input_index = shl_ref_get_index(input->dim, b, c, h, w);
                        float out_val = input_data[input_index];
                        int32_t out_index = shl_ref_get_index(
                            output->dim, b - params->begin[0], c - params->begin[1],
                            h - params->begin[2], w - params->begin[3]);
                        output_data[out_index] = out_val;
                    }
                }
            }
        }
    } else if (input->dim_count == 5) {
        for (int i = params->begin[0]; i < params->end[0]; i++) {
            for (int j = params->begin[1]; j < params->end[1]; j++) {
                for (int k = params->begin[2]; k < params->end[2]; k++) {
                    for (int l = params->begin[3]; l < params->end[3]; l++) {
                        for (int m = params->begin[4]; m < params->end[4]; m++) {
                            int32_t input_index = shl_ref_get_index_5(input->dim, i, j, k, l, m);
                            float out_val = input_data[input_index];
                            int32_t out_index = shl_ref_get_index_5(
                                output->dim, i - params->begin[0], j - params->begin[1],
                                k - params->begin[2], l - params->begin[3], m - params->begin[4]);
                            output_data[out_index] = out_val;
                        }
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

int shl_ref_slice_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_slice_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_slice_f32);
}