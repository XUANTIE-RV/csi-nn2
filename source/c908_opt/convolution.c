/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

/* CSI-NN2 version 2.0.x */

#include "shl_c908.h"

int shl_c908_conv2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
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
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            params->conv_extra.conv_mode = CSINN_GEMM;
            shl_c908_conv1x1s1_gemm_reorder_kernel_packn_fp32(kernel, params);
            cb->exec = shl_c908_conv1x1s1_gemm_packn_fp32;
        } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
                   dalition_h == 1 && dalition_w == 1) {
            if (params->group > 1) {
                params->conv_extra.conv_mode = CSINN_GEMM;
                shl_c908_conv_im2col_gemm_reorder_kernel_packn_fp32(kernel, params);
                cb->exec = shl_c908_conv_im2col_gemm_packn_fp32;
                return CSINN_TRUE;
            } else {
                params->conv_extra.conv_mode = CSINN_WINOGRAD;
                struct csinn_tensor *t_kernel = csinn_alloc_tensor(NULL);
                if ((in_h < 13) && (in_w < 13)) {
                    shl_c908_ncxhwx_wg_b4f3s1_trans_kernel_packn_fp32(kernel, t_kernel);
                    cb->exec = shl_c908_ncxhwx_wg_b4f3s1_packn_fp32;
                } else {
                    shl_c908_ncxhwx_wg_b6f3s1_trans_kernel_packn_fp32(kernel, t_kernel);
                    cb->exec = shl_c908_ncxhwx_wg_b6f3s1_packn_fp32;
                }
                params->conv_extra.kernel_tm = t_kernel;
            }
        } else {
            params->conv_extra.conv_mode = CSINN_GEMM;
            shl_c908_conv_im2col_gemm_reorder_kernel_packn_fp32(kernel, params);
            cb->exec = shl_c908_conv_im2col_gemm_packn_fp32;
        }
    }

    // pack1ton
    if (in_c % packn != 0 && out_c % packn == 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_c908_conv1x1s1_gemm_reorder_kernel_pack1ton_fp32(kernel, params);
            cb->exec = shl_c908_conv1x1s1_gemm_pack1ton_fp32;
        } else {
            shl_c908_conv_im2col_gemm_reorder_kernel_pack1ton_fp32(kernel, params);
            cb->exec = shl_c908_conv_im2col_gemm_pack1ton_fp32;
        }
    }

    // packnto1
    if (in_c % packn == 0 && out_c % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_c908_conv1x1s1_gemm_reorder_kernel_packnto1_fp32(kernel, params);
            cb->exec = shl_c908_conv1x1s1_gemm_packnto1_fp32;
        } else {
            shl_c908_conv_im2col_gemm_reorder_kernel_packnto1_fp32(kernel, params);
            cb->exec = shl_c908_conv_im2col_gemm_packnto1_fp32;
        }
    }

    // pack1
    if (in_c % packn != 0 && out_c % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_c908_conv1x1s1_gemm_reorder_kernel_fp32(kernel, params);
            cb->exec = shl_c908_conv1x1s1_gemm_fp32;
        } else {
            shl_c908_conv_im2col_gemm_reorder_kernel_fp32(kernel, params);
            cb->exec = shl_c908_conv_im2col_gemm_fp32;
        }
    }
    return CSINN_TRUE;
}

int shl_c908_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
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
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            params->conv_extra.conv_mode = CSINN_GEMM;
            shl_c908_conv1x1s1_gemm_reorder_kernel_packn_fp16(kernel, params);
            cb->exec = shl_c908_conv1x1s1_gemm_packn_fp16;
        } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
                   dalition_h == 1 && dalition_w == 1) {
            if (params->group > 1) {
                params->conv_extra.conv_mode = CSINN_GEMM;
                shl_c908_conv_im2col_gemm_reorder_kernel_packn_fp16(kernel, params);
                cb->exec = shl_c908_conv_im2col_gemm_packn_fp16;
                return CSINN_TRUE;
            } else {
                params->conv_extra.conv_mode = CSINN_WINOGRAD;
                struct csinn_tensor *t_kernel = csinn_alloc_tensor(NULL);
                if ((in_h < 13) && (in_w < 13)) {
                    shl_c908_ncxhwx_wg_b4f3s1_trans_kernel_packn_fp16(kernel, t_kernel);
                    cb->exec = shl_c908_ncxhwx_wg_b4f3s1_packn_fp16;
                } else {
                    shl_c908_ncxhwx_wg_b6f3s1_trans_kernel_packn_fp16(kernel, t_kernel);
                    cb->exec = shl_c908_ncxhwx_wg_b6f3s1_packn_fp16;
                }
                params->conv_extra.kernel_tm = t_kernel;
            }
        } else {
            params->conv_extra.conv_mode = CSINN_GEMM;
            shl_c908_conv_im2col_gemm_reorder_kernel_packn_fp16(kernel, params);
            cb->exec = shl_c908_conv_im2col_gemm_packn_fp16;
        }
    }

    // pack1ton
    if (in_c % packn != 0 && out_c % packn == 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_c908_conv1x1s1_gemm_reorder_kernel_pack1ton_fp16(kernel, params);
            cb->exec = shl_c908_conv1x1s1_gemm_pack1ton_fp16;
        } else {
            shl_c908_conv_im2col_gemm_reorder_kernel_pack1ton_fp16(kernel, params);
            cb->exec = shl_c908_conv_im2col_gemm_pack1ton_fp16;
        }
    }

    // packnto1
    if (in_c % packn == 0 && out_c % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_c908_conv1x1s1_gemm_reorder_kernel_packnto1_fp16(kernel, params);
            cb->exec = shl_c908_conv1x1s1_gemm_packnto1_fp16;
        } else {
            shl_c908_conv_im2col_gemm_reorder_kernel_packnto1_fp16(kernel, params);
            cb->exec = shl_c908_conv_im2col_gemm_packnto1_fp16;
        }
    }

    // pack1
    if (in_c % packn != 0 && out_c % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_c908_conv1x1s1_gemm_reorder_kernel_fp16(kernel, params);
            cb->exec = shl_c908_conv1x1s1_gemm_fp16;
        } else {
            shl_c908_conv_im2col_gemm_reorder_kernel_fp16(kernel, params);
            cb->exec = shl_c908_conv_im2col_gemm_fp16;
        }
    }
    return CSINN_TRUE;
}

int shl_c908_conv2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
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

    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;

    // packn
    if (in_c % packn == 0 && out_c % packn == 0) {
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            params->conv_extra.conv_mode = CSINN_GEMM;
            params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
            shl_c908_conv1x1s1_gemm_reorder_kernel_packn_int8(kernel, params);
            cb->exec = shl_c908_conv1x1s1_gemm_packn_int8;
        } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
                   dalition_h == 1 && dalition_w == 1) {
            if (params->group > 1) {
                params->conv_extra.conv_mode = CSINN_GEMM;
                params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
                shl_c908_conv_im2col_gemm_reorder_kernel_packn_int8(kernel, params);
                cb->exec = shl_c908_conv_im2col_gemm_packn_int8;
                return CSINN_TRUE;
            } else {
                params->conv_extra.conv_mode = CSINN_WINOGRAD;
                struct csinn_tensor *t_kernel = csinn_alloc_tensor(NULL);
                shl_c908_ncxhwx_wg_b4f3s1_trans_kernel_packn_int8(kernel, t_kernel);
                cb->exec = shl_c908_ncxhwx_wg_b4f3s1_packn_int8;
                params->conv_extra.kernel_tm = t_kernel;
            }
        } else {
            params->conv_extra.conv_mode = CSINN_GEMM;
            params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
            shl_c908_conv_im2col_gemm_reorder_kernel_packn_int8(kernel, params);
            cb->exec = shl_c908_conv_im2col_gemm_packn_int8;
        }
    }

    // pack1ton
    if (in_c % packn != 0 && out_c % packn == 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_c908_conv1x1s1_gemm_reorder_kernel_pack1ton_int8(kernel, params);
            cb->exec = shl_c908_conv1x1s1_gemm_pack1ton_int8;
        } else {
            shl_c908_conv_im2col_gemm_reorder_kernel_pack1ton_int8(kernel, params);
            cb->exec = shl_c908_conv_im2col_gemm_pack1ton_int8;
        }
    }

    // packnto1
    if (in_c % packn == 0 && out_c % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_c908_conv1x1s1_gemm_reorder_kernel_packnto1_int8(kernel, params);
            cb->exec = shl_c908_conv1x1s1_gemm_packnto1_int8;
        } else {
            shl_c908_conv_im2col_gemm_reorder_kernel_packnto1_int8(kernel, params);
            cb->exec = shl_c908_conv_im2col_gemm_packnto1_int8;
        }
    }

    // pack1
    if (in_c % packn != 0 && out_c % packn != 0) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            shl_c908_conv1x1s1_gemm_reorder_kernel_int8(kernel, params);
            cb->exec = shl_c908_conv1x1s1_gemm_int8;
        } else {
            shl_c908_conv_im2col_gemm_reorder_kernel_int8(kernel, params);
            cb->exec = shl_c908_conv_im2col_gemm_int8;
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
    }

    // recover fuse zeropoint to bias for winograd
    if (params->conv_extra.conv_mode == CSINN_WINOGRAD) {
        if (params->conv_extra.fuse_zp2bias) {
            int32_t *bias_data = (int32_t *)bias->data;
            int8_t *kernel_data = (int8_t *)kernel->data;
            int32_t input_zp = input->qinfo->zero_point;

            int kernel_inner = in_c * kernel_h * kernel_w;
            for (int oc = 0; oc < out_c; oc++) {
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

int shl_c908_conv2d_init_int4(struct csinn_tensor *input, struct csinn_tensor *output,
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

    // xxx: only int4 support nhwc layout now
    if (input->layout == CSINN_LAYOUT_NHWC) {
        out_c = kernel->dim[0];
        in_c = kernel->dim[3];
        in_h = input->dim[1];
        in_w = input->dim[2];
        kernel_h = kernel->dim[1];
        kernel_w = kernel->dim[2];
        if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
            dalition_w == 1) {
            params->conv_extra.conv_mode = CSINN_GEMM;
            if (input->dtype == CSINN_DTYPE_INT4) {
                params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
                shl_rvv_conv1x1s1_gemm_reorder_kernel_int4(kernel, params);
                // support channel quantization
                for (int i = 0; i < kernel->quant_channel; i++) {
                    float real_scale =
                        input->qinfo->scale * kernel->qinfo[i].scale / output->qinfo->scale;
                    shl_quantize_multiplier(real_scale, &(kernel->qinfo[i].multiplier),
                                            &(kernel->qinfo[i].shift));
                }
                cb->exec = shl_rvv_conv1x1s1_gemm_int4;
            }
        } else {
            params->conv_extra.conv_mode = CSINN_GEMM;
            if (input->dtype == CSINN_DTYPE_INT4) {
                params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
                shl_rvv_conv_im2col_gemm_reorder_kernel_int4(kernel, params);
                for (int i = 0; i < kernel->quant_channel; i++) {
                    float real_scale =
                        input->qinfo->scale * kernel->qinfo[i].scale / output->qinfo->scale;
                    shl_quantize_multiplier(real_scale, &(kernel->qinfo[i].multiplier),
                                            &(kernel->qinfo[i].shift));
                }
                cb->exec = shl_rvv_conv_im2col_gemm_int4;
            }
        }
        return CSINN_TRUE;
    }
    return CSINN_FALSE;
}
