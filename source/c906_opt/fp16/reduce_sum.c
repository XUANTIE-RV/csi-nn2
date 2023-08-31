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

#include "c906/c906.h"

// reduce_sum
int shl_c906_reduce_sum_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_reduce_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    // axis=none
    if (*(params->axis) == -1) {
        int size = 1;
        for (int i = 0; i < input->dim_count; i++) {
            size = size * input->dim[i];
        }

        asm volatile(
            "vsetvli        zero, zero, e16, m2\n\t"
            "fmv.h.x        ft0, zero\n\t"   // clear
            "vfmv.s.f       v0, ft0\n\t"     // v6[0] = bias
            "srai           t0, %2, 4\n\t"   // t0 = size_16
            "andi           t1, %2, 15\n\t"  // size tail
            "vmv.v.x        v2, zero\n\t"    // clear
            "beqz           t0, 2f\n\t"
            "1:\n\t"
            "vle.v          v4, (%0)\n\t"
            "addi           %0, %0, 32\n\t"
            "vfadd.vv       v2, v2, v4\n\t"
            "addi           t0, t0, -1\n\t"
            "bnez           t0, 1b\n\t"
            "2:\n\t"
            "vsetvli        zero, t1, e16, m2\n\t"
            "vle.v          v4, (%0)\n\t"
            "vfadd.vv        v2, v2, v4\n\t"

            "3:\n\t"
            "vfredsum.vs    v0, v2, v0\n\t"  // v0[0] = v0[0] + sum(v2[0..7])
            "vfmv.f.s       ft0, v0\n\t"
            "fsh            ft0, 0(%1)\n\t"

            : "=r"(input_data),   // %0
              "=r"(output_data),  // %1
              "=r"(size)          // %2
            : "0"(input_data), "1"(output_data), "2"(size)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "t0", "t1", "ft0");

    } else {
        int axis = *(params->axis);
        int64_t outer_size = 1;
        for (int i = 0; i < axis; i++) {
            outer_size *= input->dim[i];
        }
        int64_t inner_size = 1;
        for (int i = axis + 1; i < input->dim_count; i++) {
            inner_size *= input->dim[i];
        }
        int cnt = input->dim[axis];

        asm volatile(
            "vsetvli        zero, zero, e16, m2\n\t"
            "mulw           t0, %3, %4\n\t"
            "slli           t0, t0, 1\n\t"  // inner_size * cnt * 2
            "slli           t1, %3, 1\n\t"  // inner_size * 2

            "0:\n\t"                         // outer_size loop
            "srai           t2, %3, 4\n\t"   // inner_size 16
            "andi           t3, %3, 15\n\t"  // inner_size tail
            "mv             a0, %0\n\t"
            "beqz           t2, 3f\n\t"

            "1:\n\t"  // inner_size_16 loop
            "mv             a1, a0\n\t"
            "vmv.v.x        v2, zero\n\t"
            "mv             t4, %4\n\t"  // t4 = cnt

            "2:\n\t"  // cnt loop
            "vle.v          v0, (a1)\n\t"
            "add            a1, a1, t1\n\t"
            "vfadd.vv       v2, v2, v0\n\t"

            "addi           t4, t4, -1\n\t"
            "bnez           t4, 2b\n\t"

            "vse.v          v2, (%1)\n\t"
            "addi           %1, %1, 32\n\t"
            "addi           a0, a0, 32\n\t"

            "addi           t2, t2, -1\n\t"
            "bnez           t2, 1b\n\t"

            "3:\n\t"  // inner_size tail
            "vsetvli        zero, t3, e16, m2\n\t"
            "vmv.v.x        v2, zero\n\t"
            "mv             t4, %4\n\t"  // t4 = cnt

            "4:\n\t"  // cnt loop
            "vle.v          v0, (a0)\n\t"
            "add            a0, a0, t1\n\t"
            "vfadd.vv       v2, v2, v0\n\t"

            "addi           t4, t4, -1\n\t"
            "bnez           t4, 4b\n\t"

            "vse.v          v2, (%1)\n\t"
            "add            %1, %1, t3\n\t"
            "add            %1, %1, t3\n\t"

            "add            %0, %0, t0\n\t"
            "addi           %2, %2, -1\n\t"
            "bnez           %2, 0b\n\t"

            : "=r"(input_data),   // %0
              "=r"(output_data),  // %1
              "=r"(outer_size),   // %2
              "=r"(inner_size),   // %3
              "=r"(cnt)           // %4
            : "0"(input_data), "1"(output_data), "2"(outer_size), "3"(inner_size), "4"(cnt)
            : "cc", "memory", "v0", "v1", "v2", "v3", "a0", "a1", "t0", "t1", "t2", "t3", "t4",
              "t5");
    }
    // requantize
    shl_rvv_siso_op_requantize_fp16(input, output);
    return CSINN_TRUE;
}
