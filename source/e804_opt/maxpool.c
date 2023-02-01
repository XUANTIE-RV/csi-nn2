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

#include "e804_function.h"
#include "shl_e804.h"

static int shl_e804_maxpool2d_q7(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params)
{
    q7_t *input_data = (q7_t *)input->data;
    q7_t *output_data = (q7_t *)output->data;

    uint16_t batch = input->dim[0];
    uint16_t in_hw = input->dim[1];  // e.g. in_hw = input->dim[2];
    uint16_t in_c = input->dim[3];

    uint16_t out_hw = output->dim[1];  // e.g. out_hw = output->dim[2]

    q7_t buffer_tmp[out_hw * out_hw * in_c];  // buffer_size = out_h * out_w * channel

    csky_dsp2_maxpool2d_q7_HWC(input_data, in_hw, in_c, params->filter_height, params->pad_top,
                               params->stride_height, out_hw, buffer_tmp, output_data);

    return CSINN_TRUE;
}

int shl_e804_maxpool2d_init_q7(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_pool_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    uint8_t flag = 0;
    if ((params->pad_top != params->pad_down) || (params->pad_left != params->pad_right) ||
        (params->pad_top != params->pad_left)) {
        flag |= 0x01;
    }
    if (input->dim[1] != input->dim[2]) {
        flag |= 0x02;
    }
    if (params->filter_height != params->filter_width) {
        flag |= 0x04;
    }
    if (params->stride_height != params->stride_width) {
        flag |= 0x08;
    }
    if (flag > 0) {
        shl_debug_warning(
            "maxpool q7 is not optimized to achieve under this condition on e804, call reference "
            "func replaced.\n");
        cb->exec = shl_ref_maxpool2d_quant;
    } else {
        cb->exec = shl_e804_maxpool2d_q7;
    }
    return CSINN_TRUE;
}