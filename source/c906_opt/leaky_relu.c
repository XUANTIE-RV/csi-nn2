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

int csi_leaky_relu_f32_c906(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct relu_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }
    float alpha = params->n;
    float gata = 0.0f;
    asm volatile(
                "loop:\n\t"
                "vsetvli    t0, %3, e32, m1\n\t"
                "vlw.v      v2, (%2)\n\t"
                "sub        %3, %3, t0\n\t"
                "slli       t0, t0, 2\n\t"
                "add        %2, %2, t0\n\t"
                "vmflt.vf   v0, v2, %4\n\t"
                "vfmul.vf   v2, v2, %5, v0.t\n\t"
                "vsw.v      v2, (%0)\n\t"
                "add        %0, %0, t0\n\t"
                "bnez       %3, loop\n\t"

                :"=r"(output_data)  // %0
                :"0"(output_data),  // %1
                "r"(input_data),    // %2
                "r"(size),          // %3
                "f"(gata),          // %4
                "f"(alpha)          // %5
                : "v0", "v2", "v3", "v4", "v5", "t0"
    );

    // for (int i = 0; i < size; i++) {
    //     float val = input_data[i];
    //     output_data[i] = val > 0 ? val : val * params->n;
    // }
    return CSINN_TRUE;
}

int csi_leaky_relu_u8_c906(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct relu_params *params)
{
    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }

    float alpha_f = csi_dequantize_u8_to_f32(1, 0, params->n_multiplier, params->n_shift);
    for (int i = 0; i < size; i++) {
        float input_val = csi_dequantize_u8_to_f32(input_data[i], input->zero_point, input->multiplier,
                                             input->shift);
        float res = input_val > 0 ? input_val : input_val * alpha_f;

        output_data[i] = csi_quantize_f32_to_u8(res, output->zero_point, output->multiplier, output->shift);
    }
    return CSINN_TRUE;
}
