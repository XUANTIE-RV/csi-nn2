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

int shl_c906_global_avgpool2d_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_pool_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;

    int batch = input->dim[0];
    int in_c = input->dim[1];
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int in_hw = in_h * in_w;

    asm volatile(
        "vsetvli        zero, zero, e16, m2\n\t"
        "mv             t0, %4\n\t"
        "li             t1, 1\n\t"
        "fcvt.h.w       ft0, t0\n\t"
        "fcvt.h.w       ft1, t1\n\t"
        "fdiv.h         ft1, ft1, ft0\n\t"  // 1 / in_hw

        "1:\n\t"                     // batch_loop
        "mv             t1, %3\n\t"  // t1 = in_c

        "2:\n\t"                       // channel loop
        "mv             t2, %4\n\t"    // t2 = in_hw
        "vmv.v.x        v4, zero\n\t"  // clear v4

        "3:\n\t"
        "vsetvli        t0, t2, e16, m2\n\t"  // set vl = 8
        "vle.v          v2, (%0)\n\t"
        "sub            t2, t2, t0\n\t"
        "slli           t0, t0, 1\n\t"
        // "vfadd.vv       v4, v2, v4\n\t"
        "vfredsum.vs    v4, v2, v4\n\t"  // v4[0] = unorder_sum(v2[0..7]) + v4[0]

        "add            %0, %0, t0\n\t"
        "bnez           t2, 3b\n\t"

        "vfmv.f.s       ft0, v4\n\t"        // sum = v4[0]
        "fmul.h         ft0, ft0, ft1\n\t"  // sum / in_hw
        "fsh            ft0, 0(%1)\n\t"
        "addi           %1, %1, 2\n\t"

        "addi           t1, t1, -1\n\t"
        "bnez           t1, 2b\n\t"

        "addi           %2, %2, -1\n\t"
        "bnez           %2, 1b\n\t"

        : "=r"(input_data),   // %0
          "=r"(output_data),  // %1
          "=r"(batch),        // %2
          "=r"(in_c),         // %3
          "=r"(in_hw)         // %4
        : "0"(input_data), "1"(output_data), "2"(batch), "3"(in_c), "4"(in_hw)
        : "cc", "memory", "v2", "v3", "v4", "v5", "t0", "t1", "t2", "ft0", "ft1");
    // requantize
    shl_rvv_siso_op_requantize_fp16(input, output);
    return CSINN_TRUE;
}
