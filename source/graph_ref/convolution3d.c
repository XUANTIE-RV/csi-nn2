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

#include "shl_gref.h"

int shl_gref_conv3d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                    struct csinn_conv3d_params *params)
{
    shl_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_CONV3D, params);
    return CSINN_TRUE;
}

int shl_gref_conv3d_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv3d_params *params)
{
    int d, h, w;
    if (output->layout == CSINN_LAYOUT_NCDHW) {
        d = 2;
        h = 3;
        w = 4;
    } else if (output->layout == CSINN_LAYOUT_NDHWC) {
        d = 1;
        h = 2;
        w = 3;
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }

    int32_t in_d = input->dim[d];
    int32_t in_h = input->dim[h];
    int32_t in_w = input->dim[w];
    int32_t kernel_d = kernel->dim[d];
    int32_t kernel_h = kernel->dim[h];
    int32_t kernel_w = kernel->dim[w];
    int32_t padding_d = params->pad_front + params->pad_back;
    int32_t padding_h = params->pad_top + params->pad_down;
    int32_t padding_w = params->pad_left + params->pad_right;
    int32_t stride_d = params->stride_depth;
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t dilation_d = params->dilation_depth;
    int32_t dilation_h = params->dilation_height;
    int32_t dilation_w = params->dilation_width;

    output->dim_count = input->dim_count;
    output->dim[d] = (in_d + padding_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    output->dim[h] = (in_h + padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    output->dim[w] = (in_w + padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    return CSINN_TRUE;
}
