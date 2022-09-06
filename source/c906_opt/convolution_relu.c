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

/* CSI-NN2 version 2.0.x */

#include "shl_c906.h"

/*
   only support layout:NCHW
   input layout:  N C H W
   kernel layout: O I h w
   output layout: N O H W
*/
int shl_c906_conv2d_relu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params)
{
    int32_t out_c = kernel->dim[0];
    int32_t in_c = kernel->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];
    int32_t kernel_h = kernel->dim[2];
    int32_t kernel_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t dalition_h = params->dilation_height;
    int32_t dalition_w = params->dilation_width;
    struct csinn_callback *cb = params->base.cb;

    if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
        dalition_w == 1) {
        shl_c906_conv1x1s1_sgemm_transform_kernel(kernel, params);
        params->conv_extra.conv_mode = CSINN_GEMM;
        cb->exec = shl_c906_conv1x1s1_sgemm_fuse_relu;
    } else {
        shl_c906_conv_im2col_sgemm_transform_kernel(kernel, params);
        params->conv_extra.conv_mode = CSINN_GEMM;
        cb->exec = shl_c906_conv_im2col_sgemm_fuse_relu;
    }

    return CSINN_TRUE;
}

int shl_c906_depthwise_conv2d_relu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params)
{
    int32_t batch = input->dim[0];
    int32_t in_ch = input->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_ch = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int32_t kernel_h = kernel->dim[2];
    int32_t kernel_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    struct csinn_callback *cb = params->base.cb;

    if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
        cb->exec = shl_c906_dwconv3x3s1_fuse_relu;

    } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2) {
        cb->exec = shl_c906_dwconv3x3s2_fuse_relu;

    } else if (kernel_h == 5 && kernel_w == 5 && stride_h == 1 && stride_w == 1) {
        cb->exec = shl_c906_dwconv5x5s1_fuse_relu;

    } else if (kernel_h == 5 && kernel_w == 5 && stride_h == 2 && stride_w == 2) {
        cb->exec = shl_c906_dwconv5x5s2_fuse_relu;

    } else {
        cb->exec = shl_ref_depthwise_conv2d_relu_f32;
    }

    return CSINN_TRUE;
}