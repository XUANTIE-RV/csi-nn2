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

/* CSI-NN2 version 1.8.x */

#include "csi_ref.h"
#include "csi_utils.h"

static void element_pow_f32(float *src0, float *src1, float *dest,
                            int input_idx, int output_idx)
{
    dest[output_idx] = powf(src0[output_idx], src1[input_idx]);
}

int csi_ref_power_f32(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params)
{
    struct csi_ref_diso_callback cb;

    cb.bc = element_pow_f32;
    csi_ref_diso_broadcast_base(input0, input1, output, params, &cb);
    return CSINN_TRUE;
}

int csi_ref_power_quant(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params)
{
    return csi_ref_diso_callback_base(input0, input1, output, params, csi_ref_power_f32);
}
