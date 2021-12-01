/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

#include "csi_c906.h"

static float relu1(float x){
	return fmin(x > 0 ? x : 0, 1);
}

int csi_c906_relu1_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct relu_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }

    float gata = 0.0f;
    float gata1 = 1.0f;
    asm volatile(
                "loop:\n\t"
                "vsetvli    t0, %3, e32, m2\n\t"
                "vlw.v      v2, (%2)\n\t"
                "sub        %3, %3, t0\n\t"
                "slli       t0, t0, 2\n\t"
                "add        %2, %2, t0\n\t"
                "vfmax.vf   v2, v2, %4\n\t"
                "vfmin.vf   v2, v2, %5\n\t"
                "vsw.v      v2, (%0)\n\t"
                "add        %0, %0, t0\n\t"
                "bnez       %3, loop\n\t"

                :"=r"(output_data)  // %0
                :"0"(output_data),  // %1
                "r"(input_data),    // %2
                "r"(size),          // %3
                "f"(gata),          // %4
                "f"(gata1)          // %5
                : "v0", "v2", "v3", "v4", "v5", "t0"
    );

    // for (int i = 0; i < size; i++) {
    //     output_data[i] = relu1(input_data[i]);
    // }
    return CSINN_TRUE;
}
