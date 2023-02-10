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

int shl_gref_conv1d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                    struct csinn_conv1d_params *params)
{
    shl_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_CONV1D, params);
    return CSINN_TRUE;
}

int shl_gref_conv1d_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv1d_params *params)
{
    int c, w, kernel_c;
    if (output->layout == CSINN_LAYOUT_NCW) {
        c = 1;
        w = 2;
        kernel_c = kernel->dim[0];
    } else if (output->layout == CSINN_LAYOUT_NWC) {
        w = 1;
        c = 2;
        kernel_c = kernel->dim[2];
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }

    int32_t in_w = input->dim[w];
    int32_t kernel_w = kernel->dim[w];
    int32_t padding_w = params->pad_left + params->pad_right;
    int32_t stride_w = params->stride_width;
    int32_t dalition_w = params->dilation_width;

    output->dim_count = input->dim_count;
    output->dim[0] = input->dim[0];  // N
    output->dim[c] = kernel_c;
    output->dim[w] = (in_w + padding_w - dalition_w * (kernel_w - 1) - 1) / stride_w + 1;
    return CSINN_TRUE;
}

int shl_gref_depthwise_conv1d(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv1d_params *params)
{
    shl_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_DEPTHWISE_CONV1D, params);
    return CSINN_TRUE;
}

int shl_gref_depthwise_conv1d_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                          struct csinn_conv1d_params *params)
{
    return shl_gref_conv1d_infer_shape(input, output, kernel, bias, params);
}
