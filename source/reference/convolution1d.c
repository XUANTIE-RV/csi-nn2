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

#include "shl_ref.h"

/* TODO: direct conv1d calculation */
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
    params_conv2d.pad_down = 0;
    params_conv2d.pad_left = params->pad_left;
    params_conv2d.pad_right = params->pad_right;
    params_conv2d.dilation_height = 1;
    params_conv2d.dilation_width = params->dilation_width;
    params_conv2d.conv_extra.kernel_tm = NULL;
    params_conv2d.conv_extra.conv_mode = 0;
    params_conv2d.conv_extra.fuse_zp2bias = 0;

    int h, w;
    if (input->layout == CSINN_LAYOUT_NCW) {
        params_conv2d.base.layout = CSINN_LAYOUT_NCHW;
        h = 2;
        w = 3;
    } else if (input->layout == CSINN_LAYOUT_NWC) {
        params_conv2d.base.layout = CSINN_LAYOUT_NHWC;
        h = 1;
        w = 2;
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    kernel->dim_count = 4;
    kernel->dim[w] = kernel->dim[h];
    kernel->dim[h] = 1;
    input->dim_count = 4;
    input->dim[w] = input->dim[h];
    input->dim[h] = 1;
    output->dim_count = 4;
    output->dim[w] = output->dim[h];
    output->dim[h] = 1;
    int ret;
    if (params->group == 1) {
        ret = shl_ref_conv2d_f32(input, output, kernel, bias, &params_conv2d);
    } else if (params->group == input->dim[1] &&
               (kernel->dim[1] == 1 && params_conv2d.base.layout == CSINN_LAYOUT_NCHW ||
                kernel->dim[0] == 1 && params_conv2d.base.layout == CSINN_LAYOUT_NHWC)) {
        ret = shl_ref_depthwise_conv2d_f32(input, output, kernel, bias, &params_conv2d);
    } else {
        ret = shl_ref_group_conv2d_f32(input, output, kernel, bias, &params_conv2d);
    }
    kernel->dim[h] = kernel->dim[w];
    kernel->dim_count = 3;
    input->dim[h] = input->dim[w];
    input->dim_count = 3;
    output->dim[h] = output->dim[w];
    output->dim_count = 3;
    return ret;
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
    params_conv2d.pad_down = 0;
    params_conv2d.pad_left = params->pad_left;
    params_conv2d.pad_right = params->pad_right;
    params_conv2d.dilation_height = 1;
    params_conv2d.dilation_width = params->dilation_width;
    params_conv2d.conv_extra.kernel_tm = NULL;
    params_conv2d.conv_extra.conv_mode = 0;
    params_conv2d.conv_extra.fuse_zp2bias = 0;

    int h, w;
    if (input->layout == CSINN_LAYOUT_NCW) {
        params_conv2d.base.layout = CSINN_LAYOUT_NCHW;
        h = 2;
        w = 3;
    } else if (input->layout == CSINN_LAYOUT_NWC) {
        params_conv2d.base.layout = CSINN_LAYOUT_NHWC;
        h = 1;
        w = 2;
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    kernel->dim_count = 4;
    kernel->dim[w] = kernel->dim[h];
    kernel->dim[h] = 1;
    input->dim_count = 4;
    input->dim[w] = input->dim[h];
    input->dim[h] = 1;
    output->dim_count = 4;
    output->dim[w] = output->dim[h];
    output->dim[h] = 1;
    int ret;
    if (params->group == 1) {
        ret = shl_ref_conv2d_quant(input, output, kernel, bias, &params_conv2d);
    } else if (params->group == input->dim[1] &&
               (kernel->dim[1] == 1 && params_conv2d.base.layout == CSINN_LAYOUT_NCHW ||
                kernel->dim[0] == 1 && params_conv2d.base.layout == CSINN_LAYOUT_NHWC)) {
        ret = shl_ref_depthwise_conv2d_quant(input, output, kernel, bias, &params_conv2d);
    } else {
        ret = shl_ref_group_conv2d_quant(input, output, kernel, bias, &params_conv2d);
    }
    kernel->dim[h] = kernel->dim[w];
    kernel->dim_count = 3;
    input->dim[h] = input->dim[w];
    input->dim_count = 3;
    output->dim[h] = output->dim[w];
    output->dim_count = 3;
    return ret;
}
