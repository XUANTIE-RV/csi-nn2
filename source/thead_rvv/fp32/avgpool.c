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

int shl_rvv_avgpool2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params)
{
    int32_t in_c = input->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];
    int32_t kernel_h = params->filter_height;
    int32_t kernel_w = params->filter_width;
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t pad_left = params->pad_left;
    int32_t pad_right = params->pad_right;
    int32_t pad_top = params->pad_top;
    int32_t pad_down = params->pad_down;
    struct csinn_callback *cb = params->base.cb;
    cb->exec = NULL;

    const int packn = csrr_vlenb() / sizeof(float);
    int elempack = 1;
    struct csinn_session *sess = params->base.sess;
    if (sess->base_run_mode == CSINN_RM_CPU_GRAPH) {
        struct shl_rvv_option *option = shl_rvv_get_graph_option(sess);
        if (option && option->use_packn_layout) {
            elempack = in_c % packn == 0 ? packn : 1;
        }
        /* first layer do not convert input layout */
        if (shl_is_first_layer_input(input, sess)) {
            elempack = 1;
        }
    }

    // global avgpool2d
    if (in_h == kernel_h && in_w == kernel_w) {
        cb->exec = (elempack % packn == 0) ? shl_rvv_global_avgpool2d_packn_fp32
                                           : shl_rvv_global_avgpool2d_fp32;
        return CSINN_TRUE;
    }

    if (elempack % packn == 0) {
        cb->exec = shl_rvv_avgpool_packn_fp32;
    } else {
        if (stride_h == 2 && stride_w == 2) {
            if (kernel_h == 2 && kernel_w == 2) {
                if (pad_left == 0 && pad_top == 0) {
                    // adjust pad according to ceil_mode (ceil mode on caffe pytorch..)
                    if (in_h % 2 == 1 && params->ceil_mode == 1) {
                        if (params->pad_down) params->pad_down++;
                    }
                    if (in_w % 2 == 1 && params->ceil_mode == 1) {
                        if (params->pad_right) params->pad_right++;
                    }
                    // end consider ceil_mode 2x2s2p0
                    cb->exec = shl_rvv_avgpool2x2s2_fp32;
                } else if (pad_left == 1 && pad_top == 1) {
                    cb->exec = shl_rvv_avgpool2x2s2_p1_fp32;
                }
            } else if (kernel_h == 3 && kernel_w == 3) {
                if (pad_left == 0 && pad_top == 0) {
                    // adjust pad according to ceil_mode (ceil mode on caffe pytorch..)
                    if (in_h % 2 == 0 && params->ceil_mode == 1) {
                        if (params->pad_down == 0)
                            params->pad_down++;  // origin pad_down mast be equal to zero ?
                    }
                    if (in_w % 2 == 0 && params->ceil_mode == 1) {
                        if (params->pad_right == 0) params->pad_right++;
                    }
                    // end consider ceil_mode 3x3s2p0
                    cb->exec = shl_rvv_avgpool3x3s2_fp32;
                } else if (pad_left == 1 && pad_top == 1) {
                    if (params->ceil_mode == 0) {
                        cb->exec = shl_rvv_avgpool3x3s2_p1_fp32;
                    } else {
                        if ((in_w % 2 == 0 && pad_right == 1) || (in_h % 2 == 0 && pad_down == 1)) {
                            cb->exec = shl_ref_avgpool2d_f32;
                        } else {
                            cb->exec = shl_rvv_avgpool3x3s2_p1_fp32;
                        }
                    }
                }
            }
        } else if (stride_h == 1 && stride_w == 1) {
            if (kernel_h == 3 && kernel_w == 3) {
                if (pad_left == 1 && pad_top == 1 && pad_right == 1 && pad_down == 1) {
                    cb->exec = shl_rvv_avgpool3x3s1_p1_fp32;
                }
            }
        }
        if (cb->exec == NULL) {
            shl_debug_warning(
                "avgpool is not optimized to achieve under this condition on rvv, call reference "
                "func replaced.\n");
            cb->exec = shl_ref_avgpool2d_f32;
        }
    }
    return CSINN_TRUE;
}

int shl_rvv_global_avgpool2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_pool_params *params)
{
    int32_t in_c = input->dim[1];
    struct csinn_callback *cb = params->base.cb;
    cb->exec = NULL;

    int packn = csrr_vlenb() / sizeof(float);
    int elempack = 1;
    struct csinn_session *sess = params->base.sess;
    if (sess->base_run_mode == CSINN_RM_CPU_GRAPH) {
        struct shl_rvv_option *option = shl_rvv_get_graph_option(sess);
        if (option && option->use_packn_layout) {
            elempack = in_c % packn == 0 ? packn : 1;
        }
        /* first layer do not convert input layout */
        if (shl_is_first_layer_input(input, sess)) {
            elempack = 1;
        }
    }

    cb->exec = (elempack % packn == 0) ? shl_rvv_global_avgpool2d_packn_fp32
                                       : shl_rvv_global_avgpool2d_fp32;
    return CSINN_TRUE;
}
