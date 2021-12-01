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

/* CSI-NN2 version 1.10.x */

#include "csi_c906.h"


// constrain: only support pad on h and w dim
// pad_mode: constant
// layout: [n,c,h,w]
int csi_c906_pad_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct pad_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int32_t in_c = input->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t pad_top = params->pad_before[2];
    int32_t pad_down = params->pad_after[2];
    int32_t pad_left = params->pad_before[3];
    int32_t pad_right = params->pad_after[3];

    float pad_value = params->pad_value;

    asm volatile(
        "vsetvli        zero, zero, e32, m2\n\t"
        "vfmv.v.f       v2, %18\n\t"        // pad value
        "add            t0, %6, %8\n\t"     // pad_left + pad_right
        "add            t0, t0, %4\n\t"     // padded_w = in_w + pad_left + pad_right

        "mulw           t1, t0, %5\n\t"     // pad_top * padded_w
        "mulw           t2, t0, %7\n\t"     // pad_down * padded_w

    "1:\n\t"     // channel loop
        "mv             t5, %3\n\t"         // t5 = in_h
        "beqz           %5, 3f\n\t"         // if pad_top = 0
        "mv             t3, t1\n\t"         // t3 = num to memset 0

        "2:\n\t"    // pad h_top
            "vsetvli        t0, t3, e32, m2\n\t"
            "vsw.v          v2, (%1)\n\t"
            "sub            t3, t3, t0\n\t"
            "slli           t0, t0, 2\n\t"
            "add            %1, %1, t0\n\t"
            "bnez           t3, 2b\n\t"

        "3:\n\t"    // pad h_mid

            "mv             t4, %4\n\t"     // t4 = in_w
            "beqz           %6, 5f\n\t"     // if pad_left = 0
            "mv             t3, %6\n\t"     // t3 = pad_left

            "4:\n\t"    // pad w_left
                "vsetvli        t0, t3, e32, m2\n\t"
                "vsw.v          v2, (%1)\n\t"
                "sub            t3, t3, t0\n\t"
                "slli           t0, t0, 2\n\t"
                "add            %1, %1, t0\n\t"
                "bnez           t3, 4b\n\t"

            "5:\n\t"    // pad w_mid
                "vsetvli        t0, t4, e32, m2\n\t"
                "vlw.v          v4, (%0)\n\t"   // load from input_data
                "sub            t4, t4, t0\n\t"
                "slli           t0, t0, 2\n\t"
                "add            %0, %0, t0\n\t"
                "vsw.v          v4, (%1)\n\t"   // store to padded_buf
                "add            %1, %1, t0\n\t"
                "bnez           t4, 5b\n\t"

                "beqz           %8, 7f\n\t"    // if pad_right = 0
                "mv             t3, %8\n\t"

            "6:\n\t"    // pad w_right
                "vsetvli        t0, t3, e32, m2\n\t"
                "vsw.v          v2, (%1)\n\t"
                "sub            t3, t3, t0\n\t"
                "slli           t0, t0, 2\n\t"
                "add            %1, %1, t0\n\t"
                "bnez           t3, 6b\n\t"

        "7:\n\t"

            "addi           t5, t5, -1\n\t"
            "bnez           t5, 3b\n\t"

        "beqz           %7, 9f\n\t"     // if pad_down = 0
        "mv             t3, t2\n\t"     // t4 = num to memset 0

        "8:\n\t"     // pad h_down
            "vsetvli        t0, t3, e32, m2\n\t"
            "vsw.v          v2, (%1)\n\t"
            "sub            t3, t3, t0\n\t"
            "slli           t0, t0, 2\n\t"
            "add            %1, %1, t0\n\t"
            "bnez           t3, 8b\n\t"

    "9:\n\t"
        "addi           %2, %2, -1\n\t"
        "bnez           %2, 1b\n\t"

        :"=r"(input_data),  // %0
        "=r"(output_data),  // %1
        "=r"(in_c),         // %2
        "=r"(in_h),         // %3
        "=r"(in_w),         // %4
        "=r"(pad_top),      // %5
        "=r"(pad_left),     // %6
        "=r"(pad_down),     // %7
        "=r"(pad_right)     // %8
        :"0"(input_data),
        "1"(output_data),
        "2"(in_c),
        "3"(in_h),
        "4"(in_w),
        "5"(pad_top),
        "6"(pad_left),
        "7"(pad_down),
        "8"(pad_right),
        "f"(pad_value)      // %18
        :"cc", "memory", "v2", "v3", "v4", "v5",
         "t0", "t1", "t2", "t3", "t4", "t5"

    );

    return CSINN_TRUE;
}


int csi_c906_pad_fp16(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct pad_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;

    int32_t in_c = input->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t pad_top = params->pad_before[2];
    int32_t pad_down = params->pad_after[2];
    int32_t pad_left = params->pad_before[3];
    int32_t pad_right = params->pad_after[3];

    __fp16 pad_value = params->pad_value;

    asm volatile(
        "vsetvli        zero, zero, e16, m2\n\t"
        "vfmv.v.f       v2, %18\n\t"        // pad value
        "add            t0, %6, %8\n\t"     // pad_left + pad_right
        "add            t0, t0, %4\n\t"     // padded_w = in_w + pad_left + pad_right

        "mulw           t1, t0, %5\n\t"     // pad_top * padded_w
        "mulw           t2, t0, %7\n\t"     // pad_down * padded_w

    "1:\n\t"     // channel loop
        "mv             t5, %3\n\t"         // t5 = in_h
        "beqz           %5, 3f\n\t"         // if pad_top = 0
        "mv             t3, t1\n\t"         // t3 = num to memset 0

        "2:\n\t"    // pad h_top
            "vsetvli        t0, t3, e16, m2\n\t"
            "vse.v          v2, (%1)\n\t"
            "sub            t3, t3, t0\n\t"
            "slli           t0, t0, 1\n\t"
            "add            %1, %1, t0\n\t"
            "bnez           t3, 2b\n\t"

        "3:\n\t"    // pad h_mid

            "mv             t4, %4\n\t"     // t4 = in_w
            "beqz           %6, 5f\n\t"     // if pad_left = 0
            "mv             t3, %6\n\t"     // t3 = pad_left

            "4:\n\t"    // pad w_left
                "vsetvli        t0, t3, e16, m2\n\t"
                "vse.v          v2, (%1)\n\t"
                "sub            t3, t3, t0\n\t"
                "slli           t0, t0, 1\n\t"
                "add            %1, %1, t0\n\t"
                "bnez           t3, 4b\n\t"

            "5:\n\t"    // pad w_mid
                "vsetvli        t0, t4, e16, m2\n\t"
                "vle.v          v4, (%0)\n\t"   // load from input_data
                "sub            t4, t4, t0\n\t"
                "slli           t0, t0, 1\n\t"
                "add            %0, %0, t0\n\t"
                "vse.v          v4, (%1)\n\t"   // store to padded_buf
                "add            %1, %1, t0\n\t"
                "bnez           t4, 5b\n\t"

                "beqz           %8, 7f\n\t"    // if pad_right = 0
                "mv             t3, %8\n\t"

            "6:\n\t"    // pad w_right
                "vsetvli        t0, t3, e16, m2\n\t"
                "vse.v          v2, (%1)\n\t"
                "sub            t3, t3, t0\n\t"
                "slli           t0, t0, 1\n\t"
                "add            %1, %1, t0\n\t"
                "bnez           t3, 6b\n\t"

        "7:\n\t"

            "addi           t5, t5, -1\n\t"
            "bnez           t5, 3b\n\t"

        "beqz           %7, 9f\n\t"     // if pad_down = 0
        "mv             t3, t2\n\t"     // t4 = num to memset 0

        "8:\n\t"     // pad h_down
            "vsetvli        t0, t3, e16, m2\n\t"
            "vse.v          v2, (%1)\n\t"
            "sub            t3, t3, t0\n\t"
            "slli           t0, t0, 1\n\t"
            "add            %1, %1, t0\n\t"
            "bnez           t3, 8b\n\t"

    "9:\n\t"
        "addi           %2, %2, -1\n\t"
        "bnez           %2, 1b\n\t"

        :"=r"(input_data),  // %0
        "=r"(output_data),  // %1
        "=r"(in_c),         // %2
        "=r"(in_h),         // %3
        "=r"(in_w),         // %4
        "=r"(pad_top),      // %5
        "=r"(pad_left),     // %6
        "=r"(pad_down),     // %7
        "=r"(pad_right)     // %8
        :"0"(input_data),
        "1"(output_data),
        "2"(in_c),
        "3"(in_h),
        "4"(in_w),
        "5"(pad_top),
        "6"(pad_left),
        "7"(pad_down),
        "8"(pad_right),
        "f"(pad_value)      // %18
        :"cc", "memory", "v2", "v3", "v4", "v5",
         "t0", "t1", "t2", "t3", "t4", "t5"

    );

    return CSINN_TRUE;
}
