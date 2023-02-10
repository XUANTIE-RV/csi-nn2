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

#include "shl_thead_rvv.h"

static void elementwise_mul_fp16(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                 struct csinn_tensor *output)
{
    __fp16 *input0_data = (__fp16 *)input0->data;
    __fp16 *input1_data = (__fp16 *)input1->data;
    __fp16 *output_data = (__fp16 *)output->data;

    int64_t size = csinn_tensor_size(output);
    while (size > 0) {
        int vl = vsetvl_e16m2(size);
        vfloat16m2_t _in0 = vle16_v_f16m2(input0_data, vl);
        vfloat16m2_t _in1 = vle16_v_f16m2(input1_data, vl);
        vfloat16m2_t _sum = vfmul_vv_f16m2(_in0, _in1, vl);
        vse16_v_f16m2(output_data, _sum, vl);
        input0_data += vl;
        input1_data += vl;
        output_data += vl;
        size -= vl;
    }
}

static void broadcast_single_1_mul_fp16(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                        struct csinn_tensor *output)
{
    __fp16 *input0_data = (__fp16 *)input0->data;
    __fp16 *input1_data = (__fp16 *)input1->data;
    __fp16 *output_data = (__fp16 *)output->data;

    int64_t size = csinn_tensor_size(output);
    while (size > 0) {
        int vl = vsetvl_e16m2(size);
        vfloat16m2_t _in0 = vle16_v_f16m2(input0_data, vl);
        vfloat16m2_t _sum = vfmul_vf_f16m2(_in0, input1_data[0], vl);
        vse16_v_f16m2(output_data, _sum, vl);
        input0_data += vl;
        output_data += vl;
        size -= vl;
    }
}

int shl_rvv_mul_fp16(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int64_t in_size0 = csinn_tensor_size(input0);
    int64_t in_size1 = csinn_tensor_size(input1);
    int64_t out_size = csinn_tensor_size(output);

    bool is_elementwise =
        (in_size0 == out_size) && (in_size1 == out_size) && (input0->layout == input1->layout);

    if (is_elementwise) {
        output->layout = input0->layout;
        output->dim_count = input0->dim_count;
        for (int i = 0; i < output->dim_count; i++) {
            output->dim[i] = input0->dim[i];
        }
        elementwise_mul_fp16(input0, input1, output);
        // requantize
        shl_rvv_sidcso_op_requantize_fp16(input0, output, input1);
    } else if (in_size1 == 1) {
        output->layout = input0->layout;
        output->dim_count = input0->dim_count;
        for (int i = 0; i < output->dim_count; i++) {
            output->dim[i] = input0->dim[i];
        }
        broadcast_single_1_mul_fp16(input0, input1, output);
        // requantize
        shl_rvv_sidcso_op_requantize_fp16(input0, output, input1);
    } else {
        /* TODO: recursive opt */
        return shl_ref_mul_quant(input0, input1, output, params);
    }
    return CSINN_TRUE;
}
