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

/* CSI-NN2 version 1.8.x */

#include "csi_c906.h"

int csi_c906_add_f32(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params)
{
    int out_size = csi_tensor_size(output);
    float *in0_data_b = malloc(out_size * 4);
    float *in1_data_b = malloc(out_size * 4);

    struct csi_tensor *b_input0 = csi_alloc_tensor(NULL);
    struct csi_tensor *b_input1 = csi_alloc_tensor(NULL);
    csi_tensor_copy(b_input0, output);
    csi_tensor_copy(b_input1, output);
    b_input0->data = in0_data_b;
    b_input1->data = in1_data_b;

    csi_ref_broadcast_to_shape_f32(input0, b_input0, output->dim, output->dim_count);
    csi_ref_broadcast_to_shape_f32(input1, b_input1, output->dim, output->dim_count);

    float *input0_data = b_input0->data;
    float *input1_data = b_input1->data;
    float *output_data = output->data;
    int size0 = 1;
    for (int i = 0; i < b_input0->dim_count; i++) {
        size0 = size0 * b_input0->dim[i];
    }

    int size1 = 1;
    for (int i = 0; i < b_input1->dim_count; i++) {
        size1 = size1 * b_input1->dim[i];
    }
    if (size1 == 1){
        for (int i = 0; i< size0; i++){
            output_data[i] = input0_data[i] + input1_data[0];
        }
    } else if(size0 == size1){

        asm volatile(
                    "0:\n\t"
                    "vsetvli    t0, %4, e32, m2\n\t"
                    "vlw.v      v2, (%2)\n\t"
                    "sub        %4, %4, t0\n\t"
                    "slli       t0, t0, 2\n\t"
                    "add        %2, %2, t0\n\t"
                    "vlw.v      v4, (%3)\n\t"
                    "add        %3, %3, t0\n\t"
                    "vfadd.vv   v6, v2, v4\n\t"
                    "vsw.v      v6, (%0)\n\t"
                    "add        %0, %0, t0\n\t"
                    "bnez       %4, 0b\n\t"

                    :"=r"(output_data)  // %0
                    :"0"(output_data),  // %1
                    "r"(input0_data),   // %2
                    "r"(input1_data),   // %3
                    "r"(size0)          // %4
                    : "v2", "v3", "v4", "v5", "v6", "v7", "t0"
        );
    }
    else if(input1->dim[0] == input0->dim[3] && size1 == input1->dim[0]){

        int inner_size = input0->dim[3];
        int outer_size = input0->dim[0] * input0->dim[1] * input0->dim[2];

        asm volatile(
                    "outer_loop:\n\t"
                    "mv         a1, %4\n\t"
                    "inner_loop:\n\t"
                    "vsetvli    t0, a1, e32, m2\n\t"
                    "vlw.v      v2, (%2)\n\t"
                    "sub        a1, a1, t0\n\t"
                    "slli       t0, t0, 2\t\n"
                    "add        %2, %2, t0\n\t"
                    "vlw.v      v4, (%3)\n\t"
                    "add        %3, %3, t0\n\t"
                    "vfadd.vv   v6, v2, v4\n\t"
                    "vsw.v      v6, (%0)\n\t"
                    "add        %0, %0, t0\n\t"
                    "bnez       a1, inner_loop\n\t"
                    "slli       a2, %4, 2\n\t"
                    "sub        %3, %3, a2\n\t"
                    "addi       %5, %5, -1\n\t"
                    "bnez       %5, outer_loop\n\t"

                    :"=r"(output_data)  // %0
                    :"0"(output_data),  // %1
                    "r"(input0_data),   // %2
                    "r"(input1_data),   // %3
                    "r"(inner_size),    // %4
                    "r"(outer_size)     // %5
                    : "v2", "v3", "v4", "v5", "v6", "v7", "a1", "a2", "t0"
        );
    }

    return CSINN_TRUE;
}

