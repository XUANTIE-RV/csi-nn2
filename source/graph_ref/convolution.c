/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

int shl_gref_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                    struct csinn_conv2d_params *params)
{
    shl_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_CONV2D, params);
    return CSINN_TRUE;
}

int shl_gref_conv2d_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params)
{
    int c, h, w, kernel_oc;
    shl_tensor_try_nc1xc0_to_ndarray_shape(input);
    if (input->layout == CSINN_LAYOUT_NCHW) {
        c = 1;
        h = 2;
        w = 3;
        kernel_oc = kernel->dim[0];
    } else if (input->layout == CSINN_LAYOUT_NHWC) {
        h = 1;
        w = 2;
        c = 3;
        kernel_oc = params->group == input->dim[c] ? kernel->dim[3] : kernel->dim[0];
    } else {
        shl_debug_error("%s: Invalid input tensor layout!\n", __func__);
        return CSINN_UNSUPPORT_LAYOUT;
    }

    int32_t in_h = input->dim[h];
    int32_t in_w = input->dim[w];
    int32_t kernel_h = kernel->dim[h];
    int32_t kernel_w = kernel->dim[w];
    int32_t padding_h = params->pad_top + params->pad_down;
    int32_t padding_w = params->pad_left + params->pad_right;
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t dilation_h = params->dilation_height;
    int32_t dilation_w = params->dilation_width;

    output->layout = input->layout;
    output->dim_count = 4;
    output->dim[0] = input->dim[0];  // N
    output->dim[c] = kernel_oc;
    output->dim[h] = (in_h + padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    output->dim[w] = (in_w + padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    return CSINN_TRUE;
}

int shl_gref_conv2d_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv2d_params *params)
{
    shl_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_CONV2D_RELU, params);
    return CSINN_TRUE;
}

int shl_gref_conv2d_relu_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                     struct csinn_conv2d_params *params)
{
    return shl_gref_conv2d_infer_shape(input, output, kernel, bias, params);
}

int shl_gref_conv2d_relu6(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv2d_params *params)
{
    shl_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_CONV2D_RELU6, params);
    return CSINN_TRUE;
}

int shl_gref_conv2d_relu6_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params)
{
    return shl_gref_conv2d_infer_shape(input, output, kernel, bias, params);
}

int shl_gref_depthwise_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params)
{
    shl_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_DEPTHWISE_CONV2D, params);
    return CSINN_TRUE;
}

int shl_gref_depthwise_conv2d_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                          struct csinn_conv2d_params *params)
{
    return shl_gref_conv2d_infer_shape(input, output, kernel, bias, params);
}

int shl_gref_depthwise_conv2d_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params)
{
    shl_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_DEPTHWISE_CONV2D_RELU, params);
    return CSINN_TRUE;
}

int shl_gref_depthwise_conv2d_relu_infer_shape(struct csinn_tensor *input,
                                               struct csinn_tensor *output,
                                               struct csinn_tensor *kernel,
                                               struct csinn_tensor *bias,
                                               struct csinn_conv2d_params *params)
{
    return shl_gref_conv2d_infer_shape(input, output, kernel, bias, params);
}

int shl_gref_depthwise_conv2d_relu6(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                    struct csinn_conv2d_params *params)
{
    shl_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_DEPTHWISE_CONV2D_RELU6, params);
    return CSINN_TRUE;
}

int shl_gref_depthwise_conv2d_relu6_infer_shape(struct csinn_tensor *input,
                                                struct csinn_tensor *output,
                                                struct csinn_tensor *kernel,
                                                struct csinn_tensor *bias,
                                                struct csinn_conv2d_params *params)
{
    return shl_gref_conv2d_infer_shape(input, output, kernel, bias, params);
}

int shl_gref_group_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv2d_params *params)
{
    shl_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_GROUP_CONV2D, params);
    return CSINN_TRUE;
}

int shl_gref_group_conv2d_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params)
{
    return shl_gref_conv2d_infer_shape(input, output, kernel, bias, params);
}

int shl_gref_group_conv2d_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *kernel, struct csinn_tensor *bias,
                               struct csinn_conv2d_params *params)
{
    shl_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_GROUP_CONV2D, params);
    return CSINN_TRUE;
}

int shl_gref_group_conv2d_relu_infer_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params)
{
    return shl_gref_conv2d_infer_shape(input, output, kernel, bias, params);
}
