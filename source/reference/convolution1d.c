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

#include "shl_ref.h"

int shl_ref_conv1d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                       struct csinn_conv1d_params *params)
{
    struct csinn_conv2d_params params_conv2d;
    params_conv2d.base = params->base;
    params_conv2d.group = params->group;
    params_conv2d.stride_height = 1;
    params_conv2d.stride_width = params->stride_width;
    params_conv2d.pad_top = 0;
    params_conv2d.pad_left = params->pad_left;
    params_conv2d.pad_right = params->pad_right;
    params_conv2d.dilation_height = 1;
    params_conv2d.dilation_width = params->dilation_width;
    params_conv2d.conv_extra.kernel_tm = NULL;
    params_conv2d.conv_extra.conv_mode = 0;
    params_conv2d.conv_extra.fuse_zp2bias = 0;
    kernel->dim_count = 4;
    kernel->dim[3] = 1;
    input->dim_count = 4;
    input->dim[3] = 1;
    output->dim_count = 4;
    output->dim[3] = 1;
    shl_ref_conv2d_f32(input, output, kernel, bias, &params_conv2d);

    return CSINN_TRUE;
}

int shl_ref_conv1d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv1d_params *params)
{
    struct csinn_conv2d_params params_conv2d;
    params_conv2d.base = params->base;
    params_conv2d.group = params->group;
    params_conv2d.stride_height = 1;
    params_conv2d.stride_width = params->stride_width;
    params_conv2d.pad_top = 0;
    params_conv2d.pad_left = params->pad_left;
    params_conv2d.pad_right = params->pad_right;
    params_conv2d.dilation_height = 1;
    params_conv2d.dilation_width = params->dilation_width;
    params_conv2d.conv_extra.kernel_tm = NULL;
    params_conv2d.conv_extra.conv_mode = 0;
    params_conv2d.conv_extra.fuse_zp2bias = 0;
    kernel->dim_count = 4;
    kernel->dim[3] = 1;
    input->dim_count = 4;
    input->dim[3] = 1;
    output->dim_count = 4;
    output->dim[3] = 1;
    shl_ref_conv2d_quant(input, output, kernel, bias, &params_conv2d);

    return CSINN_TRUE;
}
