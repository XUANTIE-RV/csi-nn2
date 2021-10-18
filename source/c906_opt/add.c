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

int csi_add_f32_c906(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params)
{
    float *input0_data = input0->data;
    float *input1_data = input1->data;
    float *output_data = output->data;
    int size0 = 1;
    for (int i = 0; i < input0->dim_count; i++) {
        size0 = size0 * input0->dim[i];
    }

    int size1 = 1;
    for (int i = 0; i < input1->dim_count; i++) {
        size1 = size1 * input1->dim[i];
    }

    if(size0 == size1){

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

        // for (int i = 0; i < size0; i++) {
        //     output_data[i] = input0_data[i] + input1_data[i];
        // }
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
        // for(int n = 0; n < input0->dim[0]; n++){
        //     for(int h = 0; h < input0->dim[1]; h++){
        //         for(int w = 0; w < input0->dim[2]; w++){
        //             for(int c = 0; c < input0->dim[3]; c++){
        //                 int index = csi_get_index(input0->dim, n, h, w, c);
        //                 output_data[index] = input1_data[c] + input0_data[index];
        //             }
        //         }
        //     }
        // }
    }

    return CSINN_TRUE;
}

int csi_add_u8_c906(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params)
{
    uint8_t *input0_data = input0->data;
    uint8_t *input1_data = input1->data;
    uint8_t *output_data = output->data;

    int channel;
    if (params->layout == CSINN_NHWC){channel = input0->dim[3];}
    else if (params->layout == CSINN_NCHW){channel = input0->dim[1];}


    int size0 = 1;
    for (int i = 0; i < input0->dim_count; i++) {
        size0 = size0 * input0->dim[i];
    }

    int size1 = 1;
    int axis = 0;
    for (int i = 0; i < input1->dim_count; i++) {
        size1 = size1 * input1->dim[i];
        if (input1->dim[i] != 1){
            axis = i;
        }
    }

    if(size0 == size1){
        for (int i = 0; i < size0; i++) {
            float input0_val =
                csi_dequantize_u8_to_f32(input0_data[i], input0->zero_point, input0->multiplier, input0->shift);
            float input1_val =
                csi_dequantize_u8_to_f32(input1_data[i], input1->zero_point, input1->multiplier, input1->shift);
            float res = input0_val + input1_val;
            output_data[i] = csi_quantize_f32_to_u8(res, output->zero_point, output->multiplier, output->shift);
        }
    }
    else if(input1->dim[axis] == channel && size1 == input1->dim[axis]){
        for(int n = 0; n < input0->dim[0]; n++){
            for(int h = 0; h < input0->dim[1]; h++){
                for(int w = 0; w < input0->dim[2]; w++){
                    for(int c = 0; c < input0->dim[3]; c++){

                        if (params->layout == CSINN_NHWC){channel = c;}
                        else if (params->layout == CSINN_NCHW){channel = h;}

                        float input1_val =
                        csi_dequantize_u8_to_f32(input1_data[channel], input1->zero_point, input1->multiplier, input1->shift);

                        int index = csi_get_index(input0->dim, n, h, w, c);
                        float input0_val =
                        csi_dequantize_u8_to_f32(input0_data[index], input0->zero_point, input0->multiplier, input0->shift);
                        float res = input0_val + input1_val;
                        output_data[index] = csi_quantize_f32_to_u8(res, output->zero_point, output->multiplier, output->shift);
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}
