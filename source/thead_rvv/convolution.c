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

/* CSI-NN2 version 1.13.x */

#include "csi_thead_rvv.h"

/*
   only support layout:NCHW
   input layout:  N C H W
   kernel layout: O I h w
   output layout: N O H W
*/
int csi_nn_rvv_conv2d_init(struct csi_tensor *input, struct csi_tensor *output,
                           struct csi_tensor *kernel, struct csi_tensor *bias,
                           struct conv2d_params *params)
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

    // check
    int out_height = (in_h + params->pad_top + params->pad_down - kernel_h) / stride_h + 1;
    int out_width = (in_w + params->pad_left + params->pad_right - kernel_w) / stride_w + 1;
    if (out_height != output->dim[2] || out_width != output->dim[3]) {
        printf("output dim don't match.\n");
        return CSINN_FALSE;
    }

    if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && dalition_h == 1 &&
        dalition_w == 1) {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (input->dtype == CSINN_DTYPE_FLOAT32) {
            csi_nn_rvv_conv1x1s1_gemm_transform_kernel_fp32(kernel, params);
            params->base.bc = csi_nn_rvv_conv1x1s1_gemm_fp32;
        } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
            csi_nn_rvv_conv1x1s1_gemm_transform_kernel_fp16(kernel, params);
            params->base.bc = csi_nn_rvv_conv1x1s1_gemm_fp16;
        } else if (input->dtype == CSINN_DTYPE_INT8) {
#ifdef __riscv_xtheadv
            params->conv_extra.kernel_tm = csi_alloc_tensor(NULL);
            csi_nn_rvv_conv1x1s1_gemm_transform_kernel_int8(kernel, params);
            // support channel quantization
            for (int i = 0; i < kernel->quant_channel; i++) {
                float real_scale =
                    input->qinfo->scale * kernel->qinfo[i].scale / output->qinfo->scale;
                csi_quantize_multiplier(real_scale, &(kernel->qinfo[i].multiplier),
                                        &(kernel->qinfo[i].shift));
            }
            params->base.bc = csi_nn_rvv_conv1x1s1_gemm_int8;
#endif
        }
        // winograd convolution condition:
    } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
               dalition_h == 1 && dalition_w == 1) {
        if (input->dtype == CSINN_DTYPE_FLOAT32) {
            if (params->group > 1) {
                params->conv_extra.conv_mode = CSINN_GEMM;
                csi_nn_rvv_conv_im2col_sgemm_transform_kernel_fp32(kernel, params);
                params->base.bc = csi_nn_rvv_conv_im2col_gemm_fp32;
                return CSINN_TRUE;
            }

            // pack4 for winograd convolution
            if ((out_c % 4 == 0) && (in_c % 4 == 0)) {
                params->conv_extra.conv_mode = CSINN_WINOGRAD;
                struct csi_tensor *t_kernel = csi_alloc_tensor(NULL);
                csi_nn_rvv_conv3x3s1_winograd64_transform_kernel_packn_fp32(kernel, t_kernel);
                params->conv_extra.kernel_tm = t_kernel;
                params->base.bc = csi_nn_rvv_conv3x3s1_winograd64_packn_fp32;
            } else {
                params->conv_extra.conv_mode = CSINN_GEMM;
                csi_nn_rvv_conv_im2col_sgemm_transform_kernel_fp32(kernel, params);
                params->base.bc = csi_nn_rvv_conv_im2col_gemm_fp32;
            }

        } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
            if (params->group > 1) {
                params->conv_extra.conv_mode = CSINN_GEMM;
                csi_nn_rvv_conv_im2col_sgemm_transform_kernel_fp16(kernel, params);
                params->base.bc = csi_nn_rvv_conv_im2col_gemm_fp16;
                return CSINN_TRUE;
            }

            // pack8 for winograd convolution
            if ((out_c % 8 == 0) && (in_c % 8 == 0)) {
                params->conv_extra.conv_mode = CSINN_WINOGRAD;
                struct csi_tensor *t_kernel = csi_alloc_tensor(NULL);
                csi_nn_rvv_conv3x3s1_winograd64_transform_kernel_packn_fp16(kernel, t_kernel);
                params->conv_extra.kernel_tm = t_kernel;
                params->base.bc = csi_nn_rvv_conv3x3s1_winograd64_packn_fp16;
            } else {
                params->conv_extra.conv_mode = CSINN_GEMM;
                csi_nn_rvv_conv_im2col_sgemm_transform_kernel_fp16(kernel, params);
                params->base.bc = csi_nn_rvv_conv_im2col_gemm_fp16;
            }
        } else if (input->dtype == CSINN_DTYPE_INT8) {
#ifdef __riscv_xtheadv
            params->conv_extra.conv_mode = CSINN_GEMM;
            params->conv_extra.kernel_tm = csi_alloc_tensor(NULL);
            csi_nn_rvv_conv_im2col_sgemm_transform_kernel_int8(kernel, params);
            // support channel quantization
            for (int i = 0; i < kernel->quant_channel; i++) {
                float real_scale =
                    input->qinfo->scale * kernel->qinfo[i].scale / output->qinfo->scale;
                csi_quantize_multiplier(real_scale, &(kernel->qinfo[i].multiplier),
                                        &(kernel->qinfo[i].shift));
            }
            params->base.bc = csi_nn_rvv_conv_im2col_gemm_int8;
#endif
        }

    } else {
        params->conv_extra.conv_mode = CSINN_GEMM;
        if (input->dtype == CSINN_DTYPE_FLOAT32) {
            csi_nn_rvv_conv_im2col_sgemm_transform_kernel_fp32(kernel, params);
            params->base.bc = csi_nn_rvv_conv_im2col_gemm_fp32;
        } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
            csi_nn_rvv_conv_im2col_sgemm_transform_kernel_fp16(kernel, params);
            params->base.bc = csi_nn_rvv_conv_im2col_gemm_fp16;
        } else if (input->dtype == CSINN_DTYPE_INT8) {
#ifdef __riscv_xtheadv
            params->conv_extra.conv_mode = CSINN_GEMM;
            params->conv_extra.kernel_tm = csi_alloc_tensor(NULL);
            csi_nn_rvv_conv_im2col_sgemm_transform_kernel_int8(kernel, params);
            // support channel quantization
            for (int i = 0; i < kernel->quant_channel; i++) {
                float real_scale =
                    input->qinfo->scale * kernel->qinfo[i].scale / output->qinfo->scale;
                csi_quantize_multiplier(real_scale, &(kernel->qinfo[i].multiplier),
                                        &(kernel->qinfo[i].shift));
            }
            params->base.bc = csi_nn_rvv_conv_im2col_gemm_int8;
#endif
        }
    }
    return CSINN_TRUE;
}

int csi_nn_rvv_depthwise_conv2d_init(struct csi_tensor *input, struct csi_tensor *output,
                                     struct csi_tensor *kernel, struct csi_tensor *bias,
                                     struct conv2d_params *params)
{
    int32_t batch = input->dim[0];
    int32_t in_ch = input->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_ch = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int32_t kernel_h = kernel->dim[2];
    int32_t kernel_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;

    if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
        if (input->dtype == CSINN_DTYPE_FLOAT32) {
            params->base.bc = csi_nn_rvv_dwconv3x3s1_fp32;
        } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
            params->base.bc = csi_nn_rvv_dwconv3x3s1_fp16;
        } else if (input->dtype == CSINN_DTYPE_INT8) {
            // support channel quantization
            for (int i = 0; i < kernel->quant_channel; i++) {
                float real_scale =
                    input->qinfo->scale * kernel->qinfo[i].scale / output->qinfo->scale;
                csi_quantize_multiplier(real_scale, &(kernel->qinfo[i].multiplier),
                                        &(kernel->qinfo[i].shift));
            }
            params->base.bc = csi_nn_rvv_dwconv3x3s1_int8;
        }
    } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2) {
        if (input->dtype == CSINN_DTYPE_FLOAT32) {
            params->base.bc = csi_nn_rvv_dwconv3x3s2_fp32;
        } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
            params->base.bc = csi_nn_rvv_dwconv3x3s2_fp16;
        } else if (input->dtype == CSINN_DTYPE_INT8) {
            // support channel quantization
            for (int i = 0; i < kernel->quant_channel; i++) {
                float real_scale =
                    input->qinfo->scale * kernel->qinfo[i].scale / output->qinfo->scale;
                csi_quantize_multiplier(real_scale, &(kernel->qinfo[i].multiplier),
                                        &(kernel->qinfo[i].shift));
            }
            params->base.bc = csi_nn_rvv_dwconv3x3s2_int8;
        }
    } else {
        if (input->dtype == CSINN_DTYPE_FLOAT32) {
            params->base.bc = csi_ref_depthwise_conv2d_f32;
        } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
            params->base.bc = csi_ref_depthwise_conv2d_quant;
        }
    }
    return CSINN_TRUE;
}
