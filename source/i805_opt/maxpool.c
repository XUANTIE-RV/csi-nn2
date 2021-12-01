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

/* CSI-NN2 version 1.10.x */

#include "csi_i805.h"


static int csi_i805_maxpool2d_q7(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct pool_params *params)
{
    q7_t *input_data  = (q7_t *)input->data;
    q7_t *output_data = (q7_t *)output->data;

    uint16_t batch = input->dim[0];
    uint16_t in_hw = input->dim[1]; // e.g. in_hw = input->dim[2];
    uint16_t in_c = input->dim[3];

    uint16_t out_hw = output->dim[1]; // e.g. out_hw = output->dim[2]

    q7_t buffer_tmp[out_hw * out_hw * in_c];  // buffer_size = out_h * out_w * channel

    csky_vdsp2_maxpool2d_q7_HWC(input_data, in_hw, in_c, params->filter_height, params->pad_top,
                              params->stride_height, out_hw, buffer_tmp, output_data);

    return CSINN_TRUE;
}

int csi_i805_maxpool2d_init_q7(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct pool_params *params)
{
    uint8_t flag = 0;
    if ( (params->pad_top != params->pad_down) || (params->pad_left != params->pad_right) ||
         (params->pad_top != params->pad_left) ) {
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
        csi_debug_warning("maxpool q7 is not optimized to achieve under this condition on i805, call reference func replaced.\n");
        params->base.bc = csi_ref_maxpool2d_quant;
    } else {
        params->base.bc = csi_i805_maxpool2d_q7;
    }
    return CSINN_TRUE;
}


int csi_i805_maxpool2d_u8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct pool_params *params)
{
    uint8_t *input_data  = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;

    uint16_t batch = input->dim[0];
    uint16_t in_h  = input->dim[1];
    uint16_t in_w  = input->dim[2];
    uint16_t in_c  = input->dim[3];

    uint16_t out_h = output->dim[1];
    uint16_t out_w = output->dim[2];

    int32_t ker_h = params->filter_height;
    int32_t ker_w = params->filter_width;
    int32_t pad_h = params->pad_top;
    int32_t pad_w = params->pad_left;
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;

    csi_i805_maxpool2d_opt_u8(input_data, output_data, in_h, in_w, in_c, ker_h, ker_w,
                            pad_h, pad_w, stride_h, stride_w, out_h, out_w);

    return CSINN_TRUE;
}