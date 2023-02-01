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

#include "shl_e907.h"

/************************************************************************************
 * s2(q2 - z2) = relu{ s1(q1 - z1) }
 * q2 = (q1 - z1) * s1/s2 + z2
 ************************************************************************************/
int shl_e907_relu_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    float real_scale = input->qinfo->scale / output->qinfo->scale;
    shl_quantize_multiplier(real_scale, &output->qinfo->multiplier, &output->qinfo->shift);

    int16_t z1 = input->qinfo->zero_point;
    int32_t multiplier = output->qinfo->multiplier;
    int16_t shift = output->qinfo->shift + 1;
    int16_t z2 = output->qinfo->zero_point;

    int size = csinn_tensor_size(input);
    const int xlenb = shl_rvp_get_xlenb();
    int i = 0;
    for (; i < size; i++) {
        int32_t tmp_i32 = (int32_t)input_data[i] - z1;
        tmp_i32 = tmp_i32 > 0 ? tmp_i32 : 0;
        tmp_i32 = __rv__smmul_u((tmp_i32 << shift), multiplier);
        tmp_i32 += z2;
        output_data[i] = (int8_t)__rv__sclip32(tmp_i32, 7);
    }

    return CSINN_TRUE;
}
