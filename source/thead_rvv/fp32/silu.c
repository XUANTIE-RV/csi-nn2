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

#include "rvv/rvv.h"
#include "rvv_mathfun_fp32.h"

/*************************************************************************************
 * silu(x) = x * sigmoid(x)
 *         = x / (1 + exp(-x))
 **************************************************************************************/
int shl_rvv_silu_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_sigmoid_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int size = csinn_tensor_size(input);
    int i = 0;
    while (i < size) {
        size_t vl = vsetvl_e32m2(size - i);
        vfloat32m2_t _x = vle32_v_f32m2(input_data + i, vl);
        vfloat32m2_t _x_neg = vfmul_vf_f32m2(_x, -1.0f, vl);
        vfloat32m2_t _res = exp_ps_vfloat32m2(_x_neg, vl);
        _res = vfadd_vf_f32m2(_res, 1.0f, vl);
        _res = vfdiv_vv_f32m2(_x, _res, vl);
        vse32_v_f32m2(output_data + i, _res, vl);
        i += vl;
    }

    output->layout = input->layout;
    output->dim_count = input->dim_count;
    for (int i = 0; i < output->dim_count; i++) {
        output->dim[i] = input->dim[i];
    }
    return CSINN_TRUE;
}
