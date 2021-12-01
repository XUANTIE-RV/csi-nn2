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

#include "csi_gref.h"

int csi_gref_conv2d(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct csi_tensor *bias,
                    struct conv2d_params *params)
{
    csi_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_CONV2D, params);
    return CSINN_TRUE;
}

int csi_gref_conv2d_relu(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct csi_tensor *kernel,
                         struct csi_tensor *bias,
                         struct conv2d_params *params)
{
    csi_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_CONV2D_RELU, params);
    return CSINN_TRUE;
}

int csi_gref_conv2d_relu6(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct csi_tensor *kernel,
                          struct csi_tensor *bias,
                          struct conv2d_params *params)
{
    csi_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_CONV2D_RELU6, params);
    return CSINN_TRUE;
}

int csi_gref_depthwise_conv2d(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct csi_tensor *kernel,
                              struct csi_tensor *bias,
                              struct conv2d_params *params)
{
    csi_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_DEPTHWISE_CONV2D, params);
    return CSINN_TRUE;
}

int csi_gref_depthwise_conv2d_relu(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct csi_tensor *kernel,
                                   struct csi_tensor *bias,
                                   struct conv2d_params *params)
{
    csi_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_DEPTHWISE_CONV2D_RELU, params);
    return CSINN_TRUE;
}

int csi_gref_depthwise_conv2d_relu6(struct csi_tensor *input,
                                    struct csi_tensor *output,
                                    struct csi_tensor *kernel,
                                    struct csi_tensor *bias,
                                    struct conv2d_params *params)
{
    csi_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_DEPTHWISE_CONV2D_RELU6, params);
    return CSINN_TRUE;
}

int csi_gref_group_conv2d(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct csi_tensor *kernel,
                          struct csi_tensor *bias,
                          struct conv2d_params *params)
{
    csi_gref_sidcso_op(input, output, kernel, bias, CSINN_OP_GROUP_CONV2D, params);
    return CSINN_TRUE;
}

