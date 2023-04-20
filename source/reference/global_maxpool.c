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

int shl_ref_global_maxpool2d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params)
{
    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 0;
    params->pad_back = 0;
    params->pad_down = 0;
    params->pad_front = 0;
    params->pad_left = 0;
    params->pad_right = 0;
    params->pad_top = 0;
    if (params->base.layout == CSINN_LAYOUT_NCHW) {
        params->filter_height = input->dim[2];
        params->filter_width = input->dim[3];
    } else if (params->base.layout == CSINN_LAYOUT_NHWC) {
        params->filter_height = input->dim[1];
        params->filter_width = input->dim[2];
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    shl_ref_maxpool2d_f32(input, output, params);
    return CSINN_TRUE;
}

int shl_ref_global_maxpool2d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_pool_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_global_maxpool2d_f32);
}