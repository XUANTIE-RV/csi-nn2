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

/* CSI-NN2 version 1.13.x */

#include "csi_c906.h"

int csi_c906_lrn_fp16(struct csi_tensor *input, struct csi_tensor *output,
                      struct lrn_params *params)
{
    __fp16 *input_data = input->data;
    __fp16 *output_data = output->data;
    int inner_size = 1;
    const int depth = input->dim[1];
    int half_range = params->range / 2;

    /* inner_size = H * W */
    inner_size = input->dim[2] * input->dim[3];

    for (int j = 0; j < input->dim[0]; j++) {
        for (int c = 0; c < depth; ++c) {
            const int begin_input_c = csi_ref_max_internal_s32(0, c - half_range);
            const int end_input_c = csi_ref_min_internal_s32(depth, c + half_range + 1);
            for (int i = 0; i < inner_size; ++i) {
                float accum = 0.f;
                for (int input_c = begin_input_c; input_c < end_input_c; ++input_c) {
                    const float input_val =
                        input_data[j * depth * inner_size + input_c * inner_size + i];
                    accum += input_val * input_val;
                }
                const float multiplier =
                    pow(params->bias + params->alpha * accum / params->range, -params->beta);
                output_data[j * depth * inner_size + c * inner_size + i] =
                    input_data[j * depth * inner_size + c * inner_size + i] * multiplier;
            }
        }
    }
    return CSINN_TRUE;
}
