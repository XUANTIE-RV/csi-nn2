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

/* SHL version 2.1.x */

#include "shl_c906.h"

static void element_sub_f32(float *input0, float *input1, float *output, int size)
{
    asm volatile(
        "1:\n\t"
        "vsetvli    t0, %3, e32, m2\n\t"
        "vle.v      v8, (%1)\n\t"
        "sub        %3, %3, t0\n\t"
        "slli       t0, t0, 2\n\t"  // element: 4 bytes
        "add        %1, %1, t0\n\t"
        "vle.v      v12, (%2)\n\t"
        "add        %2, %2, t0\n\t"
        "vfsub.vv   v16, v8, v12\n\t"
        "vse.v      v16, (%0)\n\t"
        "add        %0, %0, t0\n\t"
        "bnez       %3, 1b\n\t"

        : "=r"(output),  // %0
          "=r"(input0),  // %1
          "=r"(input1),  // %2
          "=r"(size)     // %3
        : "0"(output), "1"(input0), "2"(input1), "3"(size)
        : "v8", "v9", "v12", "v13", "v16", "v17", "t0");
}

int shl_c906_sub_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    float *input0_data = (float *)input0->data;
    float *input1_data = (float *)input1->data;
    float *output_data = (float *)output->data;

    int in_size0 = csinn_tensor_size(input0);
    int in_size1 = csinn_tensor_size(input1);
    int out_size = csinn_tensor_size(output);

    // HACK: special case
    // example: [1, 64, 55, 55] + [1, 64, 1, 1] = [1, 64, 55, 55]
    if ((input1->dim[2] == 1) && (input1->dim[3] == 1) && (input1->dim[1] == input0->dim[1])) {
        int inner_size = input0->dim[2] * input0->dim[3];
        int outer_size = input0->dim[1];
        asm volatile(
            "1:\n\t"
            "flw        ft0, 0(%2)\n\t"
            "mv         t1, %4\n\t"
            "2:\n\t"
            "vsetvli    t0, t1, e32, m2\n\t"
            "vle.v      v8, (%1)\n\t"
            "sub        t1, t1, t0\n\t"
            "slli       t0, t0, 2\n\t"  // element: 4 bytes
            "add        %1, %1, t0\n\t"
            "vfsub.vf   v16, v8, ft0\n\t"
            "vse.v      v16, (%0)\n\t"
            "add        %0, %0, t0\n\t"
            "bnez       t1, 2b\n\t"
            "addi       %3, %3, -1\n\t"
            "addi       %2, %2, 4\n\t"
            "bnez       %3, 1b\n\t"

            : "=r"(output_data),  // %0
              "=r"(input0_data),  // %1
              "=r"(input1_data),  // %2
              "=r"(outer_size),   // %3
              "=r"(inner_size)    // %4
            : "0"(output_data), "1"(input0_data), "2"(input1_data), "3"(outer_size), "4"(inner_size)
            : "v8", "v9", "v16", "v17", "t0", "t1", "ft0");
        return CSINN_TRUE;
    }

    // example: [1, 3, 224, 224] + [1] = [1, 3, 224, 224]
    if (in_size1 == 1) {
        asm volatile(
            "flw        ft0, 0(%2)\n\t"
            "1:\n\t"
            "vsetvli    t0, %3, e32, m2\n\t"
            "vle.v      v8, (%1)\n\t"
            "sub        %3, %3, t0\n\t"
            "slli       t0, t0, 2\n\t"  // element: 4 bytes
            "add        %1, %1, t0\n\t"
            "vfsub.vf   v16, v8, ft0\n\t"
            "vse.v      v16, (%0)\n\t"
            "add        %0, %0, t0\n\t"
            "bnez       %3, 1b\n\t"

            : "=r"(output_data),  // %0
              "=r"(input0_data),  // %1
              "=r"(input1_data),  // %2
              "=r"(out_size)      // %3
            : "0"(output_data), "1"(input0_data), "2"(input1_data), "3"(out_size)
            : "v8", "v9", "v16", "v17", "t0", "ft0");
    }
    // example: [1, 3, 224, 224] + [1, 3, 224, 224] = [1, 3, 224, 224]
    else if (in_size0 == in_size1) {
        element_sub_f32(input0_data, input1_data, output_data, out_size);
    }

    else {
        int flag = 1;
        for (int i = input1->dim_count - 1, j = input0->dim_count - 1; i >= 0; i--, j--) {
            if (input0->dim[j] != input1->dim[i]) {
                flag = 0;
            }
        }
        // example: [1, 3, 224, 224] + [3, 224, 1] or [1, 3, 224, 224] + [3, 1, 224]
        if (!flag) {
            float *in0_data_b = shl_mem_alloc(out_size * 4);
            float *in1_data_b = shl_mem_alloc(out_size * 4);

            struct csinn_tensor *b_input0 = csinn_alloc_tensor(NULL);
            struct csinn_tensor *b_input1 = csinn_alloc_tensor(NULL);
            csinn_tensor_copy(b_input0, output);
            csinn_tensor_copy(b_input1, output);
            b_input0->data = in0_data_b;
            b_input1->data = in1_data_b;

            shl_ref_broadcast_to_shape_f32(input0, b_input0, output->dim, output->dim_count);
            shl_ref_broadcast_to_shape_f32(input1, b_input1, output->dim, output->dim_count);

            input0_data = b_input0->data;
            input1_data = b_input1->data;

            element_sub_f32(input0_data, input1_data, output_data, out_size);

            shl_mem_free(in0_data_b);
            shl_mem_free(in1_data_b);
            shl_mem_free(b_input0);
            shl_mem_free(b_input1);
        }
        // example: [1, 3, 224, 224] + [224] = [1, 3, 224, 224]  or  [1, 3, 224, 224] + [224, 224] =
        // [1, 3, 224, 224]
        else {
            int inner_size = in_size1;
            int outer_size = out_size / in_size1;
            for (int i = 0; i < outer_size; i++) {
                element_sub_f32(input0_data, input1_data, output_data, inner_size);
                input0_data += inner_size;
                output_data += inner_size;
            }
        }
    }
    return CSINN_TRUE;
}

static void element_sub_fp16(__fp16 *input0, __fp16 *input1, __fp16 *output, int size)
{
    asm volatile(
        "1:\n\t"
        "vsetvli    t0, %3, e16, m2\n\t"
        "vle.v      v8, (%1)\n\t"
        "sub        %3, %3, t0\n\t"
        "slli       t0, t0, 1\n\t"  // element: 2 bytes
        "add        %1, %1, t0\n\t"
        "vle.v      v12, (%2)\n\t"
        "add        %2, %2, t0\n\t"
        "vfsub.vv   v16, v8, v12\n\t"
        "vse.v      v16, (%0)\n\t"
        "add        %0, %0, t0\n\t"
        "bnez       %3, 1b\n\t"

        : "=r"(output),  // %0
          "=r"(input0),  // %1
          "=r"(input1),  // %2
          "=r"(size)     // %3
        : "0"(output), "1"(input0), "2"(input1), "3"(size)
        : "v8", "v9", "v12", "v13", "v16", "v17", "t0");
}

int shl_c906_sub_fp16(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params)
{
    __fp16 *input0_data = (__fp16 *)input0->data;
    __fp16 *input1_data = (__fp16 *)input1->data;
    __fp16 *output_data = (__fp16 *)output->data;

    int in_size0 = csinn_tensor_size(input0);
    int in_size1 = csinn_tensor_size(input1);
    int out_size = csinn_tensor_size(output);

    if ((input1->dim[2] == 1) && (input1->dim[3] == 1) && (input1->dim[1] == input0->dim[1])) {
        int inner_size = input0->dim[2] * input0->dim[3];
        int outer_size = input0->dim[1];
        asm volatile(
            "1:\n\t"
            "flh        ft0, 0(%2)\n\t"
            "mv         t1, %4\n\t"
            "2:\n\t"
            "vsetvli    t0, t1, e16, m2\n\t"
            "vle.v      v8, (%1)\n\t"
            "sub        t1, t1, t0\n\t"
            "slli       t0, t0, 1\n\t"
            "add        %1, %1, t0\n\t"
            "vfsub.vf   v16, v8, ft0\n\t"
            "vse.v      v16, (%0)\n\t"
            "add        %0, %0, t0\n\t"
            "bnez       t1, 2b\n\t"
            "addi       %3, %3, -1\n\t"
            "addi       %2, %2, 2\n\t"
            "bnez       %3, 1b\n\t"

            : "=r"(output_data),  // %0
              "=r"(input0_data),  // %1
              "=r"(input1_data),  // %2
              "=r"(outer_size),   // %3
              "=r"(inner_size)    // %4
            : "0"(output_data), "1"(input0_data), "2"(input1_data), "3"(outer_size), "4"(inner_size)
            : "v8", "v9", "v16", "v17", "t0", "t1", "ft0");
        return CSINN_TRUE;
    }

    if (in_size1 == 1) {
        asm volatile(
            "flh        ft0, 0(%2)\n\t"
            "1:\n\t"
            "vsetvli    t0, %3, e16, m2\n\t"
            "vle.v      v8, (%1)\n\t"
            "sub        %3, %3, t0\n\t"
            "slli       t0, t0, 1\n\t"  // element: 4 bytes
            "add        %1, %1, t0\n\t"
            "vfsub.vf   v16, v8, ft0\n\t"
            "vse.v      v16, (%0)\n\t"
            "add        %0, %0, t0\n\t"
            "bnez       %3, 1b\n\t"

            : "=r"(output_data),  // %0
              "=r"(input0_data),  // %1
              "=r"(input1_data),  // %2
              "=r"(out_size)      // %3
            : "0"(output_data), "1"(input0_data), "2"(input1_data), "3"(out_size)
            : "v8", "v9", "v16", "v17", "t0", "ft0");
    } else if (in_size0 == in_size1) {
        element_sub_fp16(input0_data, input1_data, output_data, out_size);
    } else {
        int flag = 1;
        for (int i = input1->dim_count - 1, j = input0->dim_count - 1; i >= 0; i--, j--) {
            if (input0->dim[j] != input1->dim[i]) {
                flag = 0;
            }
        }
        if (!flag) {
            __fp16 *in0_data_b = shl_mem_alloc(out_size * 2);
            __fp16 *in1_data_b = shl_mem_alloc(out_size * 2);

            struct csinn_tensor *b_input0 = csinn_alloc_tensor(NULL);
            struct csinn_tensor *b_input1 = csinn_alloc_tensor(NULL);
            csinn_tensor_copy(b_input0, output);
            csinn_tensor_copy(b_input1, output);
            b_input0->data = in0_data_b;
            b_input1->data = in1_data_b;

            shl_ref_broadcast_to_shape_quant(input0, b_input0, output->dim, output->dim_count);
            shl_ref_broadcast_to_shape_quant(input1, b_input1, output->dim, output->dim_count);

            input0_data = b_input0->data;
            input1_data = b_input1->data;

            element_sub_fp16(input0_data, input1_data, output_data, out_size);

            shl_mem_free(in0_data_b);
            shl_mem_free(in1_data_b);
            shl_mem_free(b_input0);
            shl_mem_free(b_input1);
        } else {
            int inner_size = in_size1;
            int outer_size = out_size / in_size1;
            for (int i = 0; i < outer_size; i++) {
                element_sub_fp16(input0_data, input1_data, output_data, inner_size);
                input0_data += inner_size;
                output_data += inner_size;
            }
        }
    }
    return CSINN_TRUE;
}
