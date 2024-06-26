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

#include "rvm/rvm.h"

int shl_rvm_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params)
{
    int32_t out_c = kernel->dim[0];
    int32_t in_c = kernel->dim[3];
    int32_t in_h = input->dim[1];
    int32_t in_w = input->dim[2];
    int32_t kernel_h = kernel->dim[1];
    int32_t kernel_w = kernel->dim[2];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t dilation_h = params->dilation_height;
    int32_t dilation_w = params->dilation_width;
    int32_t group = params->group;
    const int mcols = csrr_xrlenb() / 2;
    struct csinn_callback *cb = params->base.cb;
    bool has_reordered = false;

    if (params->conv_extra.conv_mode == CSINN_GEMM && kernel->mtype == CSINN_MEM_TYPE_CPU_ALIGNED) {
        has_reordered = true;
    }

    if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
        dilation_w == 1) {
        if (!has_reordered) {
            shl_rvm_conv1x1s1_gemm_reorder_kernel_fp16(kernel, params);
        }
        cb->exec = shl_rvm_conv1x1s1_gemm_fp16;
    } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
               dilation_h == 1 && dilation_w == 1 && group == 1) {
        params->conv_extra.conv_mode = CSINN_WINOGRAD;
        struct csinn_tensor *t_kernel = csinn_alloc_tensor(NULL);
        if ((in_h < 13) && (in_w < 13)) {
            shl_rvm_wg_b4f3s1_trans_kernel_nhwc_fp16(kernel, t_kernel);
            cb->exec = shl_rvm_wg_b4f3s1_nhwc_fp16;
        } else {
            shl_rvm_wg_b6f3s1_trans_kernel_nhwc_fp16(kernel, t_kernel);
            cb->exec = shl_rvm_wg_b6f3s1_nhwc_fp16;
        }
        params->conv_extra.kernel_tm = t_kernel;
    } else {
        if (!has_reordered) {
            shl_rvm_conv_im2col_gemm_reorder_kernel_fp16(kernel, params);
        }
        cb->exec = group == 1 ? shl_rvm_conv_im2col_gemm_fp16 : shl_rvm_group_conv_im2col_gemm_fp16;
    }
    return CSINN_TRUE;
}

int shl_rvm_conv2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params)
{
    int32_t out_c = kernel->dim[0];
    int32_t in_c = kernel->dim[3];
    int32_t in_h = input->dim[1];
    int32_t in_w = input->dim[2];
    int32_t kernel_h = kernel->dim[1];
    int32_t kernel_w = kernel->dim[2];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t dilation_h = params->dilation_height;
    int32_t dilation_w = params->dilation_width;
    int32_t group = params->group;
    struct csinn_callback *cb = params->base.cb;
    bool has_reordered = false;

    if (params->conv_extra.conv_mode == CSINN_GEMM && kernel->mtype == CSINN_MEM_TYPE_CPU_ALIGNED) {
        has_reordered = true;
    }

    if (!params->conv_extra.fuse_zp2bias) {
        params->conv_extra.fuse_zp2bias = true;
        int32_t *bias_data = (int32_t *)bias->data;
        int8_t *kernel_data = (int8_t *)kernel->data;
        int32_t input_zp = input->qinfo->zero_point;

        if (bias_data == NULL) {
            // XXX: memory leak
            bias_data = (int32_t *)shl_mem_alloc(out_c * sizeof(int32_t));
            bias->data = bias_data;
        }
        int kernel_inner = in_c * kernel_h * kernel_w;
        for (int oc = 0; oc < out_c; oc++) {
            int32_t tmp = 0;
            for (int j = 0; j < kernel_inner; j++) {
                tmp += kernel_data[oc * kernel_inner + j] * input_zp;
            }
            bias_data[oc] -= tmp;
        }
    }

    // support channel quantization
    for (int i = 0; i < kernel->quant_channel; i++) {
        float real_scale = input->qinfo->scale * kernel->qinfo[i].scale / output->qinfo->scale;
        shl_quantize_multiplier(real_scale, &(kernel->qinfo[i].multiplier),
                                &(kernel->qinfo[i].shift));
    }

    if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
        dilation_w == 1) {
        if (!has_reordered) {
            shl_rvm_conv1x1s1_gemm_reorder_kernel_int8(kernel, params);
        }
        cb->exec = group == 1 ? shl_rvm_conv1x1s1_gemm_int8 : shl_rvm_group_conv1x1s1_gemm_int8;
    } else {
        if (!has_reordered) {
            shl_rvm_conv_im2col_gemm_reorder_kernel_int8(kernel, params);
        }
        cb->exec = group == 1 ? shl_rvm_conv_im2col_gemm_int8 : shl_rvm_group_conv_im2col_gemm_int8;
    }
    return CSINN_TRUE;
}