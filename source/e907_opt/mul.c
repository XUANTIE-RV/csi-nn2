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

#include "shl_e907.h"

/************************************************************************************
 * s3*(q3-z3) = s1*(q1-z1) * s2*(q2-z2)
 * q3 = [ (q1-z1) * (q2-z2) * (s1*s2/s3) ] + z3
 ************************************************************************************/
int shl_e907_mul_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params)
{
    int8_t *input0_data = (int8_t *)input0->data;
    int8_t *input1_data = (int8_t *)input1->data;
    int8_t *output_data = (int8_t *)output->data;

    int in_size0 = csinn_tensor_size(input0);
    int in_size1 = csinn_tensor_size(input1);
    int out_size = csinn_tensor_size(output);

    // TODO: move to init api
    for (int q = 0; q < input1->quant_channel; q++) {
        float real_scale = input0->qinfo->scale * input1->qinfo[q].scale / output->qinfo->scale;
        shl_quantize_multiplier(real_scale, &input1->qinfo[q].multiplier, &input1->qinfo[q].shift);
    }
    int32_t z1 = input0->qinfo->zero_point;
    int32_t z3 = output->qinfo->zero_point;

    if (in_size0 == in_size1) {
        int outer_size = input1->quant_channel;
        int inner_size = in_size1 / outer_size;
        for (int c = 0; c < outer_size; c++) {
            int8_t *in0_ptr = input0_data + c * inner_size;
            int8_t *in1_ptr = input1_data + c * inner_size;
            int8_t *out_ptr = output_data + c * inner_size;

            int32_t z2 = input1->qinfo[c].zero_point;
            int32_t multiplier = input1->qinfo[c].multiplier;
            int32_t shift = input1->qinfo[c].shift;

            int i = 0;
            for (; i < inner_size; i++) {
                int32_t res = (in0_ptr[i] - z1) * (in1_ptr[i] - z2);
                // FIXME: precision error
                res = shl_rvp_mulh(res, multiplier);
                if (shift < 0) {
                    res >>= -shift - 1;
                } else {
                    res <<= shift + 1;
                }
                res += z3;
                out_ptr[i] = shl_rvp_clip_i8(res);
            }
        }
    } else {
        shl_debug_error("Only support elementwise mul on RVP CPU\n");
    }

    return CSINN_TRUE;
}
