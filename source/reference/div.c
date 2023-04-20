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

#include "shl_ref.h"

static void element_div_f32(float *src0, float *src1, float *dest, int input_idx, int output_idx)
{
    dest[output_idx] = src0[output_idx] / src1[input_idx];
}

int shl_ref_div_f32(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params)
{
    struct shl_ref_diso_callback cb;

    cb.bc = element_div_f32;
    shl_ref_diso_broadcast_base(input0, input1, output, params, &cb);
    return CSINN_TRUE;
}

int shl_ref_div_quant(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params)
{
    return shl_ref_diso_callback_base(input0, input1, output, params, shl_ref_div_f32);
}
