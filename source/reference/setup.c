/*
 * Copyright (C) 2016-2020 C-SKY Limited. All rights reserved.
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

#include "csi_nn.h"
#include "csi_utils.h"

void csi_nn_init(struct csi_tensor *input,
                 struct csi_tensor *output)
{
    float *input_data = input->data;
    uint8_t *output_data = output->data;
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }

    for (int i = 0; i < size; i++) {
        int32_t input_val = round(input_data[i] / output->scale) + output->zero_point;;
        if (input_val < 0) {
            input_val = 0;
        } else if (input_val > 255) {
            input_val = 255;
        }
        output_data[i] = input_val;
    }
}


void csi_nn_deinit(struct csi_tensor *input,
                   struct csi_tensor *output)
{
    uint8_t *input_data = input->data;
    float *output_data = output->data;
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }

    for (int i = 0; i < size; i++) {
        float input_val = csi_dequantize_f32(input_data[i], input->offset, input->multiplier, input->shift);
        output_data[i] = input_val;
    }
}
