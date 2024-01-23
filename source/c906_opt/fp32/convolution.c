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

#include "c906/c906.h"

/*
   only support layout:NCHW
   input layout:  N C H W
   kernel layout: O I h w
   output layout: N O H W
*/
int shl_c906_conv2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params)
{
    int32_t out_c = kernel->dim[0] / params->group;
    int32_t in_c = kernel->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];
    int32_t kernel_h = kernel->dim[2];
    int32_t kernel_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t dilation_h = params->dilation_height;
    int32_t dilation_w = params->dilation_width;
    struct csinn_callback *cb = params->base.cb;

    if (params->base.sess->base_run_mode == CSINN_RM_CPU_GRAPH) {
        struct shl_c906_option *option = shl_c906_get_graph_option(params->base.sess);
        if (option && option->base.use_packn_layout) {
            shl_debug_error("%s: unsupport packn\n", __func__);
            return CSINN_UNSUPPORT_LAYOUT;
        }
    }

    bool binary_model_op_init = shl_c906_get_binary_model_op_init(params->base.sess);

    // check
    int out_height = (in_h + params->pad_top + params->pad_down - kernel_h) / stride_h + 1;
    int out_width = (in_w + params->pad_left + params->pad_right - kernel_w) / stride_w + 1;
    if (out_height != output->dim[2] || out_width != output->dim[3]) {
        printf("output dim don't match.\n");
        return CSINN_FALSE;
    }

    if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
        dilation_w == 1) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (!binary_model_op_init) {
            shl_c906_conv1x1s1_sgemm_transform_kernel(kernel, params);
        }
        cb->exec = shl_c906_conv1x1s1_sgemm;
        // winograd convolution condition:
    } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
               dilation_h == 1 && dilation_w == 1) {
        if (params->group > 1) {
            params->conv_extra.conv_mode = CSINN_GEMM;
            if (!binary_model_op_init) {
                shl_c906_conv_im2col_sgemm_transform_kernel(kernel, params);
            }
            cb->exec = shl_c906_conv_im2col_sgemm;
            return CSINN_TRUE;
        }
        // pack4 for winograd convolution
        if ((out_c % 4 == 0) && (in_c % 4 == 0)) {
            params->conv_extra.conv_mode = CSINN_WINOGRAD;
            if (!binary_model_op_init) {
                struct csinn_tensor *t_kernel = csinn_alloc_tensor(NULL);
                shl_c906_conv3x3s1_winograd64_transform_kernel_pack4(kernel, t_kernel);
                params->conv_extra.kernel_tm = t_kernel;
            }
            cb->exec = shl_c906_conv3x3s1_winograd64_pack4;
        } else {
            params->conv_extra.conv_mode = CSINN_GEMM;
            if (!binary_model_op_init) {
                shl_c906_conv_im2col_sgemm_transform_kernel(kernel, params);
            }
            cb->exec = shl_c906_conv_im2col_sgemm;
        }
    } else {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (!binary_model_op_init) {
            shl_c906_conv_im2col_sgemm_transform_kernel(kernel, params);
        }
        cb->exec = shl_c906_conv_im2col_sgemm;
    }
    return CSINN_TRUE;
}
