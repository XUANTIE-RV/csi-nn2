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

#include "shl_gref.h"

int shl_gref_im2col(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_im2col_params *params)
{
    shl_gref_siso_op(input, output, CSINN_OP_IM2COL, params);
    return CSINN_TRUE;
}

/* Only support NCHW/NHWC layout */
int shl_gref_im2col_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_im2col_params *params)
{
    int c, h, w;
    if (output->layout == CSINN_LAYOUT_NCHW) {
        c = 1;
        h = 2;
        w = 3;
    } else if (output->layout == CSINN_LAYOUT_NHWC) {
        h = 1;
        w = 2;
        c = 3;
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[c];
    int32_t in_h = input->dim[h];
    int32_t in_w = input->dim[w];
    int32_t kernel_h = params->kernel_h;
    int32_t kernel_w = params->kernel_w;
    int32_t padding_h = params->pad_top + params->pad_down;
    int32_t padding_w = params->pad_left + params->pad_right;
    int32_t stride_h = params->stride_h;
    int32_t stride_w = params->stride_w;

    int32_t out_h = (in_h + padding_h - kernel_h) / stride_h + 1;
    int32_t out_w = (in_w + padding_w - kernel_w) / stride_w + 1;

    output->dim_count = 2;
    if (output->layout == CSINN_LAYOUT_NCHW) {
        output->dim[0] = in_c * kernel_h * kernel_w;
        output->dim[1] = batch * out_h * out_w;
    } else if (output->layout == CSINN_LAYOUT_NHWC) {
        output->dim[0] = batch * out_h * out_w;
        output->dim[1] = in_c * kernel_h * kernel_w;
    }

    return CSINN_TRUE;
}
