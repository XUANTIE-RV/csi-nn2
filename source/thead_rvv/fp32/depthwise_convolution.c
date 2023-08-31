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

#include "rvv/rvv.h"

int shl_rvv_depthwise_conv2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params)
{
    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];
    int32_t out_c = output->dim[1];
    int32_t kernel_h = kernel->dim[2];
    int32_t kernel_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    struct csinn_callback *cb = params->base.cb;

    const int packn = csrr_vlenb() / sizeof(float);
    int in_elempack = 1;
    int out_elempack = 1;
    struct csinn_session *sess = params->base.sess;
    if (sess->base_run_mode == CSINN_RM_CPU_GRAPH) {
        struct shl_rvv_option *option = shl_rvv_get_graph_option(sess);
        if (option && option->use_packn_layout) {
            in_elempack = in_c % packn == 0 ? packn : 1;
            out_elempack = out_c % packn == 0 ? packn : 1;
        }

        /* first layer do not convert input layout */
        if (shl_is_first_layer_input(input, sess)) {
            in_elempack = 1;
            out_elempack = 1;  // dwconv2d out_channel pack is same as in_channel
        }
    } else if (sess->base_run_mode == CSINN_RM_LAYER) {
        in_elempack = in_c % packn == 0 ? packn : 1;
        out_elempack = out_c % packn == 0 ? packn : 1;
    }
    bool binary_model_op_init = shl_rvv_get_binary_model_op_init(sess);
    if (in_elempack % packn == 0 && out_elempack % packn == 0) {
        if (!binary_model_op_init) {
            shl_rvv_dwconv_reorder_kernel_packn_fp32(kernel, params);
        }
        if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
            cb->exec = shl_rvv_dwconv3x3s1_packn_fp32;
        } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2) {
            cb->exec = shl_rvv_dwconv3x3s2_packn_fp32;
        } else {
            cb->exec = shl_rvv_dwconv_packn_fp32;
        }
    }

    if (in_elempack % packn != 0 && out_elempack % packn != 0) {
        if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
            cb->exec = shl_rvv_dwconv3x3s1_fp32;
        } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2) {
            cb->exec = shl_rvv_dwconv3x3s2_fp32;
        } else {
            cb->exec = shl_ref_depthwise_conv2d_f32;
        }
    }
    return CSINN_TRUE;
}
