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

/* CSI-NN2 version 1.8.x */

#include "csi_c906.h"

#include "convolution_1x1.h"
#include "convolution_3x3.h"
#include "convolution_sgemm.h"

/*
   only support layout:NCHW
   input layout:  N C H W
   kernel layout: O I h w
   output layout: N O H W
*/
int csi_c906_conv2d_init(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct csi_tensor *kernel,
                         struct csi_tensor *bias,
                         struct conv2d_params *params)
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

    if(kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 && dalition_w == 1) {

        conv1x1s1_sgemm_transform_kernel(kernel);
        params->conv_extra.conv_mode = CSINN_GEMM;
        params->base.bc = conv1x1s1_sgemm;

    // } else if(kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 && dalition_h == 1 && dalition_w == 1) {

    //     struct csi_tensor *t_kernel = csi_alloc_tensor(NULL);
    //     conv3x3s1_winograd64_transform_kernel_1(kernel, t_kernel);
    //     params->conv_extra.kernel_tm = t_kernel;
    //     params->conv_extra.conv_mode = CSINN_WINOGRAD;
    //     params->base.bc = conv3x3s1_winograd64_1;

    } else {

        conv_im2col_sgemm_transform_kernel(kernel);
        params->conv_extra.conv_mode = CSINN_GEMM;
        params->base.bc = conv_im2col_sgemm;
    }
    return CSINN_TRUE;
}
