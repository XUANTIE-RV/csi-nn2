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

/* SHL version 2.1.x */

#include "shl_c920.h"

int shl_c920_conv2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params)
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
    struct csinn_callback *cb = params->base.cb;

    const int packn = csrr_vlenb() / sizeof(float);

    // packn
    if (in_c % packn == 0 && out_c % packn == 0) {
        output->layout = CSINN_LAYOUT_NC1HWC0;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            params->conv_extra.conv_mode = CSINN_GEMM;
            shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_fp32(kernel, params);
            cb->exec = shl_c920_conv1x1s1_gemm_packn_fp32;
        } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
                   dalition_h == 1 && dalition_w == 1) {
            if (params->group > 1) {
                params->conv_extra.conv_mode = CSINN_GEMM;
                shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp32(kernel, params);
                cb->exec = shl_rvv_conv_im2col_gemm_packn_fp32;
                return CSINN_TRUE;
            } else {
                params->conv_extra.conv_mode = CSINN_WINOGRAD;
                struct csinn_tensor *t_kernel = csinn_alloc_tensor(NULL);
                if ((in_h < 13) && (in_w < 13)) {
                    shl_rvv_wg_b4f3s1_trans_kernel_packn_fp32(kernel, t_kernel);
                    cb->exec = shl_c920_wg_b4f3s1_packn_fp32;
                } else {
                    shl_rvv_wg_b6f3s1_trans_kernel_packn_fp32(kernel, t_kernel);
                    cb->exec = shl_c920_wg_b6f3s1_packn_fp32;
                }
                params->conv_extra.kernel_tm = t_kernel;
            }
        } else {
            params->conv_extra.conv_mode = CSINN_GEMM;
            shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp32(kernel, params);
            cb->exec = shl_c920_conv_im2col_gemm_packn_fp32;
        }
    }

    // pack1ton
    if (in_c % packn != 0 && out_c % packn == 0) {
        output->layout = CSINN_LAYOUT_NC1HWC0;
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp32(kernel, params);
            cb->exec = shl_rvv_conv1x1s1_gemm_pack1ton_fp32;
        } else {
            shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp32(kernel, params);
            cb->exec = shl_rvv_conv_im2col_gemm_pack1ton_fp32;
        }
    }

    // packnto1
    if (in_c % packn == 0 && out_c % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_fp32(kernel, params);
            cb->exec = shl_rvv_conv1x1s1_gemm_packnto1_fp32;
        } else {
            shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_fp32(kernel, params);
            cb->exec = shl_rvv_conv_im2col_gemm_packnto1_fp32;
        }
    }

    // pack1
    if (in_c % packn != 0 && out_c % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_rvv_conv1x1s1_gemm_reorder_kernel_fp32(kernel, params);
            cb->exec = shl_rvv_conv1x1s1_gemm_fp32;
        } else {
            shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(kernel, params);
            cb->exec = shl_rvv_conv_im2col_gemm_fp32;
        }
    }
    return CSINN_TRUE;
}

int shl_c920_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params)
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
    struct csinn_callback *cb = params->base.cb;

    const int packn = csrr_vlenb() / sizeof(__fp16);

    // packn
    if (in_c % packn == 0 && out_c % packn == 0) {
        output->layout = CSINN_LAYOUT_NC1HWC0;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            params->conv_extra.conv_mode = CSINN_GEMM;
            shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_fp16(kernel, params);
            cb->exec = shl_c920_conv1x1s1_gemm_packn_fp16;
        } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
                   dalition_h == 1 && dalition_w == 1) {
            if (params->group > 1) {
                params->conv_extra.conv_mode = CSINN_GEMM;
                shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp16(kernel, params);
                cb->exec = shl_rvv_conv_im2col_gemm_packn_fp16;
                return CSINN_TRUE;
            } else {
                params->conv_extra.conv_mode = CSINN_WINOGRAD;
                struct csinn_tensor *t_kernel = csinn_alloc_tensor(NULL);
                if ((in_h < 13) && (in_w < 13)) {
                    shl_rvv_wg_b4f3s1_trans_kernel_packn_fp16(kernel, t_kernel);
                    cb->exec = shl_c920_wg_b4f3s1_packn_fp16;
                } else {
                    shl_rvv_wg_b6f3s1_trans_kernel_packn_fp16(kernel, t_kernel);
                    cb->exec = shl_c920_wg_b6f3s1_packn_fp16;
                }
                params->conv_extra.kernel_tm = t_kernel;
            }
        } else {
            params->conv_extra.conv_mode = CSINN_GEMM;
            shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp16(kernel, params);
            cb->exec = shl_c920_conv_im2col_gemm_packn_fp16;
        }
    }

    // pack1ton
    if (in_c % packn != 0 && out_c % packn == 0) {
        output->layout = CSINN_LAYOUT_NC1HWC0;
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp16(kernel, params);
            cb->exec = shl_rvv_conv1x1s1_gemm_pack1ton_fp16;
        } else {
            shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp16(kernel, params);
            cb->exec = shl_rvv_conv_im2col_gemm_pack1ton_fp16;
        }
    }

    // packnto1
    if (in_c % packn == 0 && out_c % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_fp16(kernel, params);
            cb->exec = shl_rvv_conv1x1s1_gemm_packnto1_fp16;
        } else {
            shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_fp16(kernel, params);
            cb->exec = shl_rvv_conv_im2col_gemm_packnto1_fp16;
        }
    }

    // pack1
    if (in_c % packn != 0 && out_c % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_rvv_conv1x1s1_gemm_reorder_kernel_fp16(kernel, params);
            cb->exec = shl_rvv_conv1x1s1_gemm_fp16;
        } else {
            shl_rvv_conv_im2col_gemm_reorder_kernel_fp16(kernel, params);
            cb->exec = shl_rvv_conv_im2col_gemm_fp16;
        }
    }
    return CSINN_TRUE;
}