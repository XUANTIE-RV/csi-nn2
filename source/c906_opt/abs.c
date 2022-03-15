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

/* CSI-NN2 version 1.12.x */

#include "csi_c906.h"

int csi_c906_abs_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }

    asm volatile(
                "1:\n\t"
                "vsetvli        t0, %3, e32, m2\n\t"
                "vlw.v          v8, (%2)\n\t"
                "slli           t1, t0, 2\n\t"
                "add            %2, %2, t1\n\t"
                "vfsgnjx.vv     v8, v8, v8\n\t"
                "vsw.v          v8, (%0)\n\t"
                "add            %0, %0, t1\n\t"
                "sub            %3, %3, t0\n\t"
                "bnez           %3, 1b\n\t"

                :"=r"(output_data)  // %0
                :"0"(output_data),  // %1
                "r"(input_data),    // %2
                "r"(size)           // %3
                : "v8", "v9", "t0", "t1"
    );

    // for (int i = 0; i < size; i++) {
    //     output_data[i] = fabs(input_data[i]);
    // }
    return CSINN_TRUE;
}


int csi_c906_abs_fp16(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;

    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }

    asm volatile(
                "1:\n\t"
                "vsetvli        t0, %3, e16, m2\n\t"
                "vle.v          v8, (%2)\n\t"
                "slli           t1, t0, 1\n\t"
                "add            %2, %2, t1\n\t"
                "vfsgnjx.vv     v8, v8, v8\n\t"
                "vse.v          v8, (%0)\n\t"
                "add            %0, %0, t1\n\t"
                "sub            %3, %3, t0\n\t"
                "bnez           %3, 1b\n\t"

                :"=r"(output_data)  // %0
                :"0"(output_data),  // %1
                "r"(input_data),    // %2
                "r"(size)           // %3
                : "v8", "v9", "t0", "t1"
    );

    return CSINN_TRUE;
}
