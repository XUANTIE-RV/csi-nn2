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

/* CSI-NN2 version 1.12.x */

#ifndef INCLUDE_CSI_C908_H_
#define INCLUDE_CSI_C908_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "csi_internal.h"
#include "csi_ref.h"
#include "csi_thead_rvv.h"
#include "csi_utils.h"

int csi_nn_c908_conv2d_init(struct csi_tensor *input, struct csi_tensor *output,
                            struct csi_tensor *kernel, struct csi_tensor *bias,
                            struct conv2d_params *params);

int csi_nn_c908_depthwise_conv2d_init(struct csi_tensor *input, struct csi_tensor *output,
                                      struct csi_tensor *kernel, struct csi_tensor *bias,
                                      struct conv2d_params *params);

int csi_nn_c908_avgpool2d_init(struct csi_tensor *input, struct csi_tensor *output,
                               struct pool_params *params);

int csi_nn_c908_maxpool2d_init(struct csi_tensor *input, struct csi_tensor *output,
                               struct pool_params *params);

int csi_nn_c908_fullyconnected_init(struct csi_tensor *input, struct csi_tensor *output,
                                    struct csi_tensor *weights, struct csi_tensor *bias,
                                    struct fc_params *params);

void csi_nn_c908_conv_im2col_sgemm_transform_kernel_fp32(struct csi_tensor *kernel,
                                                         struct conv2d_params *params);

int csi_nn_c908_conv_im2col_gemm_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                      struct csi_tensor *kernel, struct csi_tensor *bias,
                                      struct conv2d_params *params);

void csi_nn_c908_conv_im2col_sgemm_transform_kernel_fp16(struct csi_tensor *kernel,
                                                         struct conv2d_params *params);

int csi_nn_c908_conv_im2col_gemm_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                      struct csi_tensor *kernel, struct csi_tensor *bias,
                                      struct conv2d_params *params);

void csi_nn_c908_conv1x1s1_gemm_transform_kernel_fp32(struct csi_tensor *kernel,
                                                      struct conv2d_params *params);

int csi_nn_c908_conv1x1s1_gemm_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                    struct csi_tensor *kernel, struct csi_tensor *bias,
                                    struct conv2d_params *params);

void csi_nn_c908_conv1x1s1_gemm_transform_kernel_fp16(struct csi_tensor *kernel,
                                                      struct conv2d_params *params);

int csi_nn_c908_conv1x1s1_gemm_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                    struct csi_tensor *kernel, struct csi_tensor *bias,
                                    struct conv2d_params *params);

void csi_nn_c908_reorder_kernel_n8_fp32(float *src, float *dst, int m, int k, int ldc);
void csi_nn_c908_reorder_input_z12_fp32(float *src, float *dst, int k, int n, int ldc);
void csi_nn_c908_gemm_8x12_fp32(float *dst, const float *sa, const float *sb, int m, int k, int n,
                                int ldc, float *bias);
void csi_nn_c908_reorder_input_z8_fp32(float *src, float *dst, int k, int n, int ldc);
void csi_nn_c908_gemm_8x8_fp32(float *dst, const float *sa, const float *sb, int m, int k, int n,
                               int ldc, float *bias);

void csi_nn_c908_reorder_kernel_n8_fp16(__fp16 *src, __fp16 *dst, int m, int k, int ldc);
void csi_nn_c908_reorder_input_z24_fp16(__fp16 *src, __fp16 *dst, int k, int n, int ldc);
void csi_nn_c908_gemm_8x24_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, int m, int k,
                                int n, int ldc, __fp16 *bias);
void csi_nn_c908_reorder_input_z16_fp16(__fp16 *src, __fp16 *dst, int k, int n, int ldc);
void csi_nn_c908_gemm_8x16_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, int m, int k,
                                int n, int ldc, __fp16 *bias);

#endif  // INCLUDE_CSI_C908_H_
