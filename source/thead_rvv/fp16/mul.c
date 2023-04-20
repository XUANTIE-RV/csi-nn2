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

#include "shl_thead_rvv.h"

static int tail_coincide(struct csinn_tensor *input0, struct csinn_tensor *input1)
{
    int flag = 1;
    int i = 0, j = 0;
    for (i = input1->dim_count - 1, j = input0->dim_count - 1; i >= 0; i--, j--) {
        if (input0->dim[j] != input1->dim[i]) {
            flag = 0;
            break;
        }
    }
    flag = 1;
    for (; i >= 0; i--) {
        if (input1->dim[i] != 1) {
            flag = 0;
            break;
        }
    }
    return flag;
}

static void element_mul_fp16(__fp16 *input0, __fp16 *input1, __fp16 *output, int size)
{
    while (size > 0) {
        int vl = vsetvl_e16m2(size);
        vfloat16m2_t _in0 = vle16_v_f16m2(input0, vl);
        vfloat16m2_t _in1 = vle16_v_f16m2(input1, vl);
        vfloat16m2_t _sum = vfmul_vv_f16m2(_in0, _in1, vl);
        vse16_v_f16m2(output, _sum, vl);
        input0 += vl;
        input1 += vl;
        output += vl;
        size -= vl;
    }
}

int shl_rvv_mul_fp16(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    __fp16 *input0_data = (__fp16 *)input0->data;
    __fp16 *input1_data = (__fp16 *)input1->data;
    __fp16 *output_data = (__fp16 *)output->data;

    int in_size0 = csinn_tensor_size(input0);
    int in_size1 = csinn_tensor_size(input1);
    int out_size = csinn_tensor_size(output);

    if (in_size1 == 1) {
        int size = out_size;
        while (size > 0) {
            int vl = vsetvl_e16m2(size);
            vfloat16m2_t _in0 = vle16_v_f16m2(input0_data, vl);
            vfloat16m2_t _sum = vfmul_vf_f16m2(_in0, input1_data[0], vl);
            vse16_v_f16m2(output_data, _sum, vl);
            input0 += vl;
            output += vl;
            size -= vl;
        }
    } else if (in_size0 == in_size1) {
        element_mul_fp16(input0_data, input1_data, output_data, out_size);
    } else if (tail_coincide(input0, input1)) {
        int inner_size = in_size1;
        int outer_size = out_size / in_size1;
        for (int i = 0; i < outer_size; i++) {
            element_mul_fp16(input0_data, input1_data, output_data, inner_size);
            input0_data += inner_size;
            output_data += inner_size;
        }
    } else {
        __fp16 *in0_data_b = shl_mem_alloc(out_size * sizeof(__fp16));
        __fp16 *in1_data_b = shl_mem_alloc(out_size * sizeof(__fp16));

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

        element_mul_fp16(input0_data, input1_data, output_data, out_size);

        shl_mem_free(in0_data_b);
        shl_mem_free(in1_data_b);
        csinn_free_tensor(b_input0);
        csinn_free_tensor(b_input1);
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input0, output, input1);
    return CSINN_TRUE;
}
