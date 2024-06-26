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

#include "c908/c908.h"

int shl_c908_maxpool2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
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

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    int elempack = 1;
    struct csinn_session *sess = params->base.sess;
    if (sess->base_run_mode == CSINN_RM_CPU_GRAPH) {
        struct shl_c908_option *option = shl_c908_get_graph_option(sess);
        if (option && option->base.use_packn_layout) {
            elempack = in_c % packn == 0 ? packn : 1;
        }
        /* first layer do not convert input layout */
        if (shl_is_first_layer_input(input, sess)) {
            elempack = 1;
        }
    } else if (sess->base_run_mode == CSINN_RM_LAYER) {
        elempack = in_c % packn == 0 ? packn : 1;
    }

    // global maxpool2d // TODO: remove
    if (in_h == kernel_h && in_w == kernel_w) {
        cb->exec = (elempack % packn == 0) ? shl_rvv_global_maxpool2d_packn_int8
                                           : shl_ref_global_maxpool2d_quant;
        return CSINN_TRUE;
    }

    if (stride_h == 2 && stride_w == 2) {
        if (kernel_h == 2 && kernel_w == 2) {  // 2x2s2
            if (pad_left == 0 && pad_top == 0) {
                // adjust pad according to ceil_mode (ceil mode on caffe pytorch..)
                if (in_h % 2 == 1 && params->ceil_mode == 1) {
                    if (params->pad_down == 0) params->pad_down++;
                }
                if (in_w % 2 == 1 && params->ceil_mode == 1) {
                    if (params->pad_right == 0) params->pad_right++;
                }
                // end consider ceil_mode 2x2s2p0
                cb->exec = (elempack % packn == 0) ? shl_rvv_maxpool2x2s2_packn_int8
                                                   : shl_rvv_maxpool2x2s2_int8;
            } else if (pad_left == 1 && pad_top == 1) {
                cb->exec = (elempack % packn == 0) ? shl_rvv_maxpool2x2s2_packn_int8
                                                   : shl_rvv_maxpool2x2s2_p1_int8;
            }
        } else if (kernel_h == 3 && kernel_w == 3) {  // 3x3s2
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
                cb->exec = (elempack % packn == 0) ? shl_rvv_maxpool3x3s2_packn_int8
                                                   : shl_rvv_maxpool3x3s2_int8;
            } else if (pad_left == 1 && pad_top == 1) {
                cb->exec = (elempack % packn == 0) ? shl_rvv_maxpool3x3s2_packn_int8
                                                   : shl_rvv_maxpool3x3s2_p1_int8;
            }
        }
    } else if (stride_h == 1 && stride_w == 1) {
        if (kernel_h == 3 && kernel_w == 3) {
            if (pad_left == 1 && pad_top == 1 && pad_right == 1 && pad_down == 1) {
                cb->exec = (elempack % packn == 0) ? shl_rvv_maxpool3x3s1_packn_int8
                                                   : shl_rvv_maxpool3x3s1_p1_int8;
            }
        }
    }
    if (cb->exec == NULL) {
        if (elempack % packn == 0) {
            cb->exec = shl_rvv_maxpool_packn_int8;
        } else {
            shl_debug_warning(
                "maxpool is not optimized to achieve under this condition on C908, call reference "
                "func replaced.\n");
            cb->exec = shl_ref_maxpool2d_quant;
        }
    }
    return CSINN_TRUE;
}
