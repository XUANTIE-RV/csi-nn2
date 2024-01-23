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

int shl_c906_depthwise_conv2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
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

    if (params->base.sess->base_run_mode == CSINN_RM_CPU_GRAPH) {
        struct shl_c906_option *option = shl_c906_get_graph_option(params->base.sess);
        if (option && option->base.use_packn_layout) {
            shl_debug_error("%s: unsupport packn\n", __func__);
            return CSINN_UNSUPPORT_LAYOUT;
        }
    }

    if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
        cb->exec = shl_c906_dwconv3x3s1;
    } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2) {
        cb->exec = shl_c906_dwconv3x3s2;
    } else if (kernel_h == 5 && kernel_w == 5 && stride_h == 1 && stride_w == 1) {
        cb->exec = shl_c906_dwconv5x5s1;
    } else if (kernel_h == 5 && kernel_w == 5 && stride_h == 2 && stride_w == 2) {
        cb->exec = shl_c906_dwconv5x5s2;
    } else {
        cb->exec = shl_ref_depthwise_conv2d_f32;
    }
    return CSINN_TRUE;
}
