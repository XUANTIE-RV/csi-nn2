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

int shl_rvv_conv1d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv1d_params *params)
{
    int32_t out_c = kernel->dim[0] / params->group;
    int32_t in_c = kernel->dim[1];
    int32_t in_w = input->dim[2];
    int32_t kernel_w = kernel->dim[2];
    int32_t stride_w = params->stride_width;
    int32_t dalition_w = params->dilation_width;

    struct csinn_callback *cb = params->base.cb;

    struct csinn_session *sess = params->base.sess;
    if (sess->base_run_mode == CSINN_RM_CPU_GRAPH) {
        struct shl_rvv_option *option = shl_rvv_get_graph_option(sess);
    }

    bool binary_model_op_init = shl_rvv_get_binary_model_op_init(sess);

    // pack1
    if (!binary_model_op_init) {
        if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
            shl_rvv_conv1d_im2col_gemm_reorder_kernel_fp16_w_int8(kernel, params);
        } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
            shl_rvv_conv1d_im2col_gemm_reorder_kernel_fp16(kernel, params);
        }
    }
    cb->exec = shl_rvv_conv1d_im2col_gemm_fp16;
    return CSINN_TRUE;
}
