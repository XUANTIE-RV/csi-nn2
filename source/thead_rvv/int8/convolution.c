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

int shl_rvv_conv2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
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

    if (params->base.quant_type != CSINN_QUANT_INT8_ASYM_W_SYM) {
        cb->exec = shl_ref_conv2d_quant;
        return CSINN_TRUE;
    }

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
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
    } else if (sess->base_run_mode == CSINN_RM_LAYER) {
        in_elempack = in_c % packn == 0 ? packn : 1;
        out_elempack = out_c % packn == 0 ? packn : 1;
    }

    // packn
    if (in_elempack % packn == 0 && out_elempack % packn == 0) {
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
            dilation_w == 1) {
            params->conv_extra.conv_mode = CSINN_GEMM;
            params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
            shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_int8(kernel, params);
            cb->exec = shl_rvv_conv1x1s1_gemm_packn_int8;
        } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
                   dilation_h == 1 && dilation_w == 1) {
            if (params->group > 1) {
                params->conv_extra.conv_mode = CSINN_GEMM;
                params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
                shl_rvv_conv_im2col_gemm_reorder_kernel_packn_int8(kernel, params);
                cb->exec = shl_rvv_conv_im2col_gemm_packn_int8;
            } else {
                params->conv_extra.conv_mode = CSINN_WINOGRAD;
                struct csinn_tensor *t_kernel = csinn_alloc_tensor(NULL);
                shl_rvv_wg_b4f3s1_trans_kernel_packn_int8(kernel, t_kernel);
                cb->exec = shl_rvv_wg_b4f3s1_packn_int8;
                params->conv_extra.kernel_tm = t_kernel;
            }
        } else {
            params->conv_extra.conv_mode = CSINN_GEMM;
            params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
            shl_rvv_conv_im2col_gemm_reorder_kernel_packn_int8(kernel, params);
            cb->exec = shl_rvv_conv_im2col_gemm_packn_int8;
        }
    }

    // pack1ton
    if (in_elempack % packn != 0 && out_elempack % packn == 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
            dilation_w == 1) {
            shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_int8(kernel, params);
            cb->exec = shl_rvv_conv1x1s1_gemm_pack1ton_int8;
        } else {
            shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_int8(kernel, params);
            cb->exec = shl_rvv_conv_im2col_gemm_pack1ton_int8;
        }
    }

    // packnto1
    if (in_elempack % packn == 0 && out_elempack % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
            dilation_w == 1) {
            shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_int8(kernel, params);
            cb->exec = shl_rvv_conv1x1s1_gemm_packnto1_int8;
        } else {
            shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_int8(kernel, params);
            cb->exec = shl_rvv_conv_im2col_gemm_packnto1_int8;
        }
    }

    // pack1
    if (in_elempack % packn != 0 && out_elempack % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dilation_h == 1 &&
            dilation_w == 1) {
            shl_rvv_conv1x1s1_gemm_reorder_kernel_int8(kernel, params);
            cb->exec = shl_rvv_conv1x1s1_gemm_int8;
        } else {
            shl_rvv_conv_im2col_gemm_reorder_kernel_int8(kernel, params);
            cb->exec = shl_rvv_conv_im2col_gemm_int8;
        }
    }

    // support channel quantization
    for (int i = 0; i < kernel->quant_channel; i++) {
        float real_scale = input->qinfo->scale * kernel->qinfo[i].scale / output->qinfo->scale;
        // trick for winograd b4f3
        if (params->conv_extra.conv_mode == CSINN_WINOGRAD) {
            real_scale = real_scale / 576.0f;
        }
        shl_quantize_multiplier(real_scale, &(kernel->qinfo[i].multiplier),
                                &(kernel->qinfo[i].shift));
    }

    // enable fuse zeropoint to bias for gemm
    if (params->conv_extra.conv_mode == CSINN_GEMM) {
        if (!params->conv_extra.fuse_zp2bias) {
            params->conv_extra.fuse_zp2bias = true;
            int32_t *bias_data = (int32_t *)bias->data;
            int8_t *kernel_data = (int8_t *)kernel->data;
            int32_t input_zp = input->qinfo->zero_point;

            if (bias_data == NULL) {
                // XXX: memory leak
                bias_data = (int32_t *)shl_mem_alloc(out_c * params->group * sizeof(int32_t));
                bias->data = bias_data;
            }
            int kernel_inner = in_c * kernel_h * kernel_w;
            for (int oc = 0; oc < out_c * params->group; oc++) {
                int32_t tmp = 0;
                for (int j = 0; j < kernel_inner; j++) {
                    tmp += kernel_data[oc * kernel_inner + j] * input_zp;
                }
                bias_data[oc] -= tmp;
            }
        }
    }

    // recover fuse zeropoint to bias for winograd
    if (params->conv_extra.conv_mode == CSINN_WINOGRAD) {
        if (params->conv_extra.fuse_zp2bias) {
            int32_t *bias_data = (int32_t *)bias->data;
            int8_t *kernel_data = (int8_t *)kernel->data;
            int32_t input_zp = input->qinfo->zero_point;

            int kernel_inner = in_c * kernel_h * kernel_w;
            for (int oc = 0; oc < out_c * params->group; oc++) {
                int32_t tmp = 0;
                for (int j = 0; j < kernel_inner; j++) {
                    tmp += kernel_data[oc * kernel_inner + j] * input_zp;
                }
                bias_data[oc] += tmp;
            }
        }
    }
    return CSINN_TRUE;
}
