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

/* SHL version 2.1.x */

#include "shl_c906.h"

// nchw layout
int shl_c906_prelu_fp16(struct csinn_tensor *input, struct csinn_tensor *alpha,
                        struct csinn_tensor *output, struct csinn_prelu_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *alpha_data = (__fp16 *)alpha->data;

    int batch = output->dim[0];
    int channel = output->dim[1];
    int size = output->dim[2] * output->dim[3];

    asm volatile(
        "fmv.h.x    ft1, zero\n\t"
        "mv         t3, %4\n\t"     // t3 = batch
        "mv         a2, %2\n\t"     // a2 = input_data
        "mv         a0, %0\n\t"     // a0 = output_data
        "1:\n\t"                    // batch loop
        "mv         t2, %5\n\t"     // t2 = channel
        "mv         a1, %3\n\t"     // a1 = alpha_data
        "2:\n\t"                    // channel loop
        "mv         t1, %6\n\t"     // t1 = size;
        "flh        ft0, (a1)\n\t"  // load alpha
        "addi       a1, a1, 2\n\t"  // alpha_addr ++
        "3:\n\t"                    // size loop
        "vsetvli    t0, t1, e16, m2\n\t"
        "vle.v      v8, (a2)\n\t"
        "sub        t1, t1, t0\n\t"
        "slli       t0, t0, 1\n\t"
        "add        a2, a2, t0\n\t"
        "vmflt.vf   v0, v8, ft1\n\t"
        "vfmul.vf   v8, v8, ft0, v0.t\n\t"
        "vse.v      v8, (a0)\n\t"
        "add        a0, a0, t0\n\t"
        "bnez       t1, 3b\n\t"

        "addi       t2, t2, -1\n\t"
        "bnez       t2, 2b\n\t"

        "addi       t3, t3, -1\n\t"
        "bnez       t3, 1b\n\t"

        : "=r"(output_data)  // %0
        : "0"(output_data),  // %1
          "r"(input_data),   // %2
          "r"(alpha_data),   // %3
          "r"(batch),        // %4
          "r"(channel),      // %5
          "r"(size)          // %6
        : "v0", "v8", "v9", "t0", "t1", "t2", "t3", "a0", "a1", "a2", "ft0", "ft1");
    // requantize
    shl_rvv_siso_op_requantize_fp16(input, output);
    return CSINN_TRUE;
}