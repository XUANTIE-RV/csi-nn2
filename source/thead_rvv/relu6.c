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

/* CSI-NN2 version 2.0.x */

#include "shl_thead_rvv.h"

/*************************************************************
    note: VLEN = 128/256 ...
*************************************************************/
int shl_rvv_relu6_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int size = csinn_tensor_size(input);
    while (size > 0) {
        int vl = vsetvl_e32m2(size);
        vfloat32m2_t _input = vle32_v_f32m2(input_data, vl);
        input_data += vl;
        vfloat32m2_t _output = vfmin_vf_f32m2(vfmax_vf_f32m2(_input, 0.0f, vl), 6.0f, vl);
        vse32_v_f32m2(output_data, _output, vl);
        output_data += vl;
        size -= vl;
    }
    return CSINN_TRUE;
}

int shl_rvv_relu6_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;

    int size = csinn_tensor_size(input);
    while (size > 0) {
        int vl = vsetvl_e16m2(size);
        vfloat16m2_t _input = vle16_v_f16m2(input_data, vl);
        input_data += vl;
        vfloat16m2_t _output = vfmin_vf_f16m2(vfmax_vf_f16m2(_input, 0.0f, vl), 6.0f, vl);
        vse16_v_f16m2(output_data, _output, vl);
        output_data += vl;
        size -= vl;
    }
    return CSINN_TRUE;
}

/************************************************************************************
 * s2(q2 - z2) = relu6{ s1(q1 - z1) }
 * q2 = (q1 - z1) * s1/s2 + z2
 *
 * note：relu6 一般接在全连接/卷积后面，可以直接和全连接/卷积 融合
 ************************************************************************************/
int shl_rvv_relu6_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params)
{
    // refer to relu
    return CSINN_FALSE;
}
