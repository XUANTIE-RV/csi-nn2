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

int csi_c906_fullyconnected_f32(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *weights,
                                struct csi_tensor *bias,
                                struct fc_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    float *weights_data = weights->data;
    float *bias_data = bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    const int batches = output->dim[0];
    const int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    const int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    float zero = 0.0f;
    asm volatile(
                "mv             a0, %5\n\t"
            "1:\n\t"
                "mv             a1, %6\n\t"
                "2:\n\t"
                    "mv             a2, %7\n\t"
                    "vxor.vv        v8, v8, v8\n\t"     // clear
                    "3:\n\t"
                        "vsetvli        t0, a2, e32, m1\n\t"
                        "vlw.v          v2, (%2)\n\t"       // load input_data
                        "sub            a2, a2, t0\n\t"
                        "slli           t0, t0, 2\n\t"
                        "add            %2, %2, t0\n\t"     // bump input_data pointer
                        "vlw.v          v4, (%3)\n\t"       // load weight_data
                        "add            %3, %3, t0\n\t"     // bump weight_data pointer
                        "vfsub.vv       v6, v6, v6\n\t"     // clear v6
                        "vfmacc.vv      v6, v2, v4\n\t"
                        "vfredsum.vs    v8, v6, v8\n\t"     // v8[0] = v8[0] + sum(v6[0..i])
                        "bnez           a2, 3b\n\t"

                    "vfmv.f.s       ft1, v8\n\t"
                    "beqz           %8, 4f\n\t"         // bias_dims_count = 0 (bias = NULL) ?

                    "flw            ft0, 0(%4)\n\t"     // load bias_data
                    "addi           %4, %4, 4\n\t"      // bump bias_data pointer
                    "fadd.s         ft1, ft1, ft0\n\t"

                "4:\n\t"
                    "fsw            ft1, 0(%0)\n\t"     // store output_data
                    "addi           %0, %0, 4\n\t"      // bump output_data pointer

                    "slli           a3, %7, 2\n\t"
                    "sub            %2, %2, a3\n\t"
                    "addi           a1, a1, -1\n\t"
                    "bnez           a1, 2b\n\t"

                "add            %2, %2, a3\n\t"
                "mul            t1, %6, %7\n\t"
                "slli           t1, t1, 2\n\t"
                "sub            %3, %3, t1\n\t"     // finish all output_nodes, jump weights_data pointer
                "slli           t2, %6, 2\n\t"
                "sub            %4, %4, t2\n\t"     // finish all output_nodes, jump bias_data pointer

                "addi           a0, a0, -1\n\t"
                "bnez           a0, 1b\n\t"

                :"=r"(output_data)  // %0
                :"0"(output_data),  // %1
                "r"(input_data),    // %2
                "r"(weights_data),  // %3
                "r"(bias_data),     // %4
                "r"(batches),       // %5
                "r"(output_depth),  // %6
                "r"(accum_depth),   // %7
                "r"(bias_dims_count)// %8
                : "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "a0", "a1", "a2", "a3", "t0", "t1", "t2", "ft0", "ft1"
    );

    return CSINN_TRUE;
}
