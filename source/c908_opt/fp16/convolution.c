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

int shl_c908_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
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

    const int packn = csrr_vlenb() / sizeof(__fp16);
    int in_elempack = 1;
    int out_elempack = 1;
    struct csinn_session *sess = params->base.sess;
    if (sess->base_run_mode == CSINN_RM_CPU_GRAPH) {
        struct shl_c908_option *option = shl_c908_get_graph_option(sess);
        if (option && option->base.use_packn_layout) {
            in_elempack = in_c % packn == 0 ? packn : 1;
            out_elempack = out_c % packn == 0 ? packn : 1;
        }
        /* first layer do not convert input layout */
        if (shl_is_first_layer_input(input, sess)) {
            in_elempack = 1;
        }
    } else if (sess->base_run_mode == CSINN_RM_LAYER) {
        in_elempack = in_c % packn == 0 ? packn : 1;
        out_elempack = out_c % packn == 0 ? packn : 1;
    }

    bool binary_model_op_init = shl_c908_get_binary_model_op_init(sess);

    // packn
    if (in_elempack % packn == 0 && out_elempack % packn == 0) {
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
            dilation_w == 1) {
            params->conv_extra.conv_mode = CSINN_GEMM;
            if (!binary_model_op_init) {
                if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
                    shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_fp16_w_int8(kernel, params);
                } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
                    shl_c908_conv1x1s1_gemm_reorder_kernel_packn_fp16(kernel, params);
                }
            }
            cb->exec = shl_c908_conv1x1s1_gemm_packn_fp16;
        } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
                   dilation_h == 1 && dilation_w == 1) {
            if (params->group > 1 || (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8)) {
                params->conv_extra.conv_mode = CSINN_GEMM;
                if (!binary_model_op_init) {
                    if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
                        shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp16_w_int8(kernel, params);
                    } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
                        shl_c908_conv_im2col_gemm_reorder_kernel_packn_fp16(kernel, params);
                    }
                }
                cb->exec = shl_c908_conv_im2col_gemm_packn_fp16;
                return CSINN_TRUE;
            } else {
                params->conv_extra.conv_mode = CSINN_WINOGRAD;
                if (!binary_model_op_init) {
                    struct csinn_tensor *t_kernel = csinn_alloc_tensor(NULL);
                    if ((in_h < 13) && (in_w < 13)) {
                        shl_c908_ncxhwx_wg_b4f3s1_trans_kernel_packn_fp16(kernel, t_kernel);
                    } else {
                        shl_c908_ncxhwx_wg_b6f3s1_trans_kernel_packn_fp16(kernel, t_kernel);
                    }
                    params->conv_extra.kernel_tm = t_kernel;
                }
                if ((in_h < 13) && (in_w < 13)) {
                    cb->exec = shl_c908_ncxhwx_wg_b4f3s1_packn_fp16;
                } else {
                    cb->exec = shl_c908_ncxhwx_wg_b6f3s1_packn_fp16;
                }
            }
        } else {
            params->conv_extra.conv_mode = CSINN_GEMM;
            if (!binary_model_op_init) {
                if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
                    shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp16_w_int8(kernel, params);
                } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
                    shl_c908_conv_im2col_gemm_reorder_kernel_packn_fp16(kernel, params);
                }
            }
            cb->exec = shl_c908_conv_im2col_gemm_packn_fp16;
        }
    }

    // pack1ton
    if (in_elempack % packn != 0 && out_elempack % packn == 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
            dilation_w == 1) {
            if (!binary_model_op_init) {
                if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
                    shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp16_w_int8(kernel, params);
                } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
                    shl_c908_conv1x1s1_gemm_reorder_kernel_pack1ton_fp16(kernel, params);
                }
            }
            cb->exec = shl_c908_conv1x1s1_gemm_pack1ton_fp16;
        } else {
            if (!binary_model_op_init) {
                if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
                    shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp16_w_int8(kernel, params);
                } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
                    shl_c908_conv_im2col_gemm_reorder_kernel_pack1ton_fp16(kernel, params);
                }
            }
            cb->exec = shl_c908_conv_im2col_gemm_pack1ton_fp16;
        }
    }

    // packnto1
    if (in_elempack % packn == 0 && out_elempack % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
            dilation_w == 1) {
            if (!binary_model_op_init) {
                if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
                    shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_fp16_w_int8(kernel, params);
                } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
                    shl_c908_conv1x1s1_gemm_reorder_kernel_packnto1_fp16(kernel, params);
                }
            }
            cb->exec = shl_c908_conv1x1s1_gemm_packnto1_fp16;
        } else {
            if (!binary_model_op_init) {
                if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
                    shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_fp16_w_int8(kernel, params);
                } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
                    shl_c908_conv_im2col_gemm_reorder_kernel_packnto1_fp16(kernel, params);
                }
            }
            cb->exec = shl_c908_conv_im2col_gemm_packnto1_fp16;
        }
    }

    // pack1
    if (in_elempack % packn != 0 && out_elempack % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
            dilation_w == 1) {
            if (!binary_model_op_init) {
                if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
                    shl_rvv_conv1x1s1_gemm_reorder_kernel_fp16_w_int8(kernel, params);
                } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
                    shl_c908_conv1x1s1_gemm_reorder_kernel_fp16(kernel, params);
                }
            }
            cb->exec = shl_c908_conv1x1s1_gemm_fp16;
        } else {
            if (!binary_model_op_init) {
                if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
                    shl_rvv_conv_im2col_gemm_reorder_kernel_fp16_w_int8(kernel, params);
                } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
                    shl_c908_conv_im2col_gemm_reorder_kernel_fp16(kernel, params);
                }
            }
            cb->exec = shl_c908_conv_im2col_gemm_fp16;
        }
    }
    return CSINN_TRUE;
}
