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

int shl_rvv_avgpool2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
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
        struct shl_rvv_option *option = shl_rvv_get_graph_option(sess);
        if (option && option->use_packn_layout) {
            elempack = in_c % packn == 0 ? packn : 1;
        }
        /* first layer do not convert input layout */
        if (shl_is_first_layer_input(input, sess)) {
            elempack = 1;
        }
    } else if (sess->base_run_mode == CSINN_RM_LAYER) {
        elempack = in_c % packn == 0 ? packn : 1;
    }

    // global avgpool2d
    if (in_h == kernel_h && in_w == kernel_w) {
        cb->exec = (elempack % packn == 0) ? shl_rvv_global_avgpool2d_packn_int8
                                           : shl_ref_global_avgpool2d_quant;
        return CSINN_TRUE;
    }
    if (cb->exec == NULL) {
        if (elempack % packn == 0) {
            cb->exec = shl_rvv_avgpool_packn_int8;
        } else {
            shl_debug_warning(
                "avgpool is not optimized to achieve under this condition on rvv, call reference "
                "func replaced.\n");
            cb->exec = shl_ref_avgpool2d_quant;
        }
    }
}

int shl_rvv_global_avgpool2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_pool_params *params)
{
    int32_t in_c = input->dim[1];
    struct csinn_callback *cb = params->base.cb;
    cb->exec = NULL;

    int packn = csrr_vlenb() / sizeof(int8_t) / 2;
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

    cb->exec = (elempack % packn == 0) ? shl_rvv_global_avgpool2d_packn_int8
                                       : shl_ref_global_avgpool2d_quant;
}
