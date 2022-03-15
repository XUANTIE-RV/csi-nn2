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

/* CSI-NN2 version 1.12.x */

#include "csi_i805.h"


int csi_i805_relu_q7(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct relu_params *params)
{
    q7_t *input_data = (q7_t *)input->data;
    int size = csi_tensor_size(input);
    csky_vdsp2_relu_q7(input_data, size);   // FIXME: unified func name - csi_relu_q7?
    output->data = input_data;
    return CSINN_TRUE;
}

int csi_i805_relu_q15(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct relu_params *params)
{
    q15_t *input_data = (q15_t *)input->data;
    int size = csi_tensor_size(input);
    csky_vdsp2_relu_q15(input_data, size);
    output->data = input_data;
    return CSINN_TRUE;
}


int csi_i805_relu_init_u8(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct relu_params *params)
{
    // compute out multiplier and shift for scale_in/scale_out
    float real_multiplier = input->qinfo->scale / output->qinfo->scale;
    csi_quantize_multiplier(real_multiplier, &output->qinfo->multiplier, &output->qinfo->shift);
    params->base.bc = csi_i805_relu_u8;
    return CSINN_TRUE;
}

int csi_i805_relu_u8(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct relu_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    int32_t size = csi_tensor_size(input);

    csi_i805_relu_opt_u8(input_data, size, input->qinfo->zero_point, output->qinfo->multiplier, output->qinfo->shift);
    output->data = input_data;
    return CSINN_TRUE;
}
