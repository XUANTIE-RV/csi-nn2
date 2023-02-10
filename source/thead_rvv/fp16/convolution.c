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

#include "shl_thead_rvv.h"

int shl_rvv_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
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
    int32_t dalition_h = params->dilation_height;
    int32_t dalition_w = params->dilation_width;
    struct csinn_callback *cb = params->base.cb;

    const int packn = csrr_vlenb() / sizeof(__fp16);
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
        }
    }

    bool binary_model_op_init = shl_rvv_get_binary_model_op_init(sess);

    if (input->layout == CSINN_LAYOUT_NHWC) {
        if (params->group == 1 && kernel_h == 3 && kernel_w == 3 && stride_h == 1 &&
            stride_w == 1 && dalition_h == 1 && dalition_w == 1) {
            params->conv_extra.conv_mode = CSINN_DIRECT;
            shl_rvv_conv3x3s1_direct_reorder_kernel_pack4n_fp16(kernel, params);
            cb->exec = shl_rvv_conv3x3s1_direct_fp16_nhwc;
            return CSINN_TRUE;
        }
    }

    // packn
    if (in_elempack % packn == 0 && out_elempack % packn == 0) {
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            params->conv_extra.conv_mode = CSINN_GEMM;
            if (!binary_model_op_init) {
                shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_fp16(kernel, params);
            }
            cb->exec = shl_rvv_conv1x1s1_gemm_packn_fp16;
        } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
                   dalition_h == 1 && dalition_w == 1) {
            if (params->group > 1) {
                params->conv_extra.conv_mode = CSINN_GEMM;
                if (!binary_model_op_init) {
                    shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp16(kernel, params);
                }
                cb->exec = shl_rvv_conv_im2col_gemm_packn_fp16;
                return CSINN_TRUE;
            } else {
                params->conv_extra.conv_mode = CSINN_WINOGRAD;
                // TODO: params->conv_extra.kernel_tm in binary model
                struct csinn_tensor *t_kernel = csinn_alloc_tensor(NULL);
                if ((in_h < 13) && (in_w < 13)) {
                    shl_rvv_wg_b4f3s1_trans_kernel_packn_fp16(kernel, t_kernel);
                    cb->exec = shl_rvv_wg_b4f3s1_packn_fp16;
                } else {
                    shl_rvv_wg_b6f3s1_trans_kernel_packn_fp16(kernel, t_kernel);
                    cb->exec = shl_rvv_wg_b6f3s1_packn_fp16;
                }
                params->conv_extra.kernel_tm = t_kernel;
            }
        } else {
            params->conv_extra.conv_mode = CSINN_GEMM;
            if (!binary_model_op_init) {
                shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp16(kernel, params);
            }
            cb->exec = shl_rvv_conv_im2col_gemm_packn_fp16;
        }
    }

    // pack1ton
    if (in_elempack % packn != 0 && out_elempack % packn == 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            if (!binary_model_op_init) {
                shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp16(kernel, params);
            }
            cb->exec = shl_rvv_conv1x1s1_gemm_pack1ton_fp16;
        } else {
            if (!binary_model_op_init) {
                shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp16(kernel, params);
            }
            cb->exec = shl_rvv_conv_im2col_gemm_pack1ton_fp16;
        }
    }

    // packnto1
    if (in_elempack % packn == 0 && out_elempack % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            if (!binary_model_op_init) {
                shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_fp16(kernel, params);
            }
            cb->exec = shl_rvv_conv1x1s1_gemm_packnto1_fp16;
        } else {
            if (!binary_model_op_init) {
                shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_fp16(kernel, params);
            }
            cb->exec = shl_rvv_conv_im2col_gemm_packnto1_fp16;
        }
    }

    // pack1
    if (in_elempack % packn != 0 && out_elempack % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            if (!binary_model_op_init) {
                shl_rvv_conv1x1s1_gemm_reorder_kernel_fp16(kernel, params);
            }
            cb->exec = shl_rvv_conv1x1s1_gemm_fp16;
        } else {
            if (!binary_model_op_init) {
                shl_rvv_conv_im2col_gemm_reorder_kernel_fp16(kernel, params);
            }
            cb->exec = shl_rvv_conv_im2col_gemm_fp16;
        }
    }
    return CSINN_TRUE;
}
