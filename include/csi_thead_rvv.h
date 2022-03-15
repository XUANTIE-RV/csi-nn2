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

#ifndef INCLUDE_CSI_THEAD_RVV_H_
#define INCLUDE_CSI_THEAD_RVV_H_

#include <riscv_vector.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "csi_internal.h"
#include "csi_ref.h"
#include "csi_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

int csi_nn_rvv_conv2d_init(struct csi_tensor *input, struct csi_tensor *output,
                           struct csi_tensor *kernel, struct csi_tensor *bias,
                           struct conv2d_params *params);

int csi_nn_rvv_depthwise_conv2d_init(struct csi_tensor *input, struct csi_tensor *output,
                                     struct csi_tensor *kernel, struct csi_tensor *bias,
                                     struct conv2d_params *params);

int csi_nn_rvv_avgpool2d_init(struct csi_tensor *input, struct csi_tensor *output,
                              struct pool_params *params);

int csi_nn_rvv_maxpool2d_init(struct csi_tensor *input, struct csi_tensor *output,
                              struct pool_params *params);

int csi_nn_rvv_fullyconnected_init(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *weights, struct csi_tensor *bias,
                                   struct fc_params *params);

/************************************ convolution *********************************/
void csi_nn_rvv_conv_im2col_sgemm_transform_kernel_fp32(struct csi_tensor *kernel,
                                                        struct conv2d_params *params);

int csi_nn_rvv_conv_im2col_gemm_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                     struct csi_tensor *kernel, struct csi_tensor *bias,
                                     struct conv2d_params *params);

void csi_nn_rvv_conv_im2col_sgemm_transform_kernel_fp16(struct csi_tensor *kernel,
                                                        struct conv2d_params *params);

int csi_nn_rvv_conv_im2col_gemm_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                     struct csi_tensor *kernel, struct csi_tensor *bias,
                                     struct conv2d_params *params);

void csi_nn_rvv_conv_im2col_sgemm_transform_kernel_int8(struct csi_tensor *kernel,
                                                        struct conv2d_params *params);

int csi_nn_rvv_conv_im2col_gemm_int8(struct csi_tensor *input, struct csi_tensor *output,
                                     struct csi_tensor *kernel, struct csi_tensor *bias,
                                     struct conv2d_params *params);

void csi_nn_rvv_conv_im2col_sgemm_transform_kernel_int4(struct csi_tensor *kernel,
                                                        struct conv2d_params *params);

int csi_nn_rvv_conv_im2col_gemm_int4(struct csi_tensor *input, struct csi_tensor *output,
                                     struct csi_tensor *kernel, struct csi_tensor *bias,
                                     struct conv2d_params *params);

void csi_nn_rvv_conv1x1s1_gemm_transform_kernel_fp32(struct csi_tensor *kernel,
                                                     struct conv2d_params *params);

int csi_nn_rvv_conv1x1s1_gemm_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *kernel, struct csi_tensor *bias,
                                   struct conv2d_params *params);

void csi_nn_rvv_conv1x1s1_gemm_transform_kernel_fp16(struct csi_tensor *kernel,
                                                     struct conv2d_params *params);

int csi_nn_rvv_conv1x1s1_gemm_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *kernel, struct csi_tensor *bias,
                                   struct conv2d_params *params);

void csi_nn_rvv_conv1x1s1_gemm_transform_kernel_int8(struct csi_tensor *kernel,
                                                     struct conv2d_params *params);

int csi_nn_rvv_conv1x1s1_gemm_int8(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *kernel, struct csi_tensor *bias,
                                   struct conv2d_params *params);

void csi_nn_rvv_conv1x1s1_gemm_transform_kernel_int4(struct csi_tensor *kernel,
                                                     struct conv2d_params *params);

int csi_nn_rvv_conv1x1s1_gemm_int4(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *kernel, struct csi_tensor *bias,
                                   struct conv2d_params *params);

void csi_nn_rvv_conv3x3s1_winograd64_transform_kernel_packn_fp32(struct csi_tensor *o_kernel,
                                                                 struct csi_tensor *t_kernel);

int csi_nn_rvv_conv3x3s1_winograd64_packn_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                               struct csi_tensor *kernel, struct csi_tensor *bias,
                                               struct conv2d_params *params);

void csi_nn_rvv_conv3x3s1_winograd64_transform_kernel_packn_fp16(struct csi_tensor *o_kernel,
                                                                 struct csi_tensor *t_kernel);

int csi_nn_rvv_conv3x3s1_winograd64_packn_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                               struct csi_tensor *kernel, struct csi_tensor *bias,
                                               struct conv2d_params *params);

int csi_nn_rvv_dwconv3x3s1_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *kernel, struct csi_tensor *bias,
                                struct conv2d_params *params);

int csi_nn_rvv_dwconv3x3s2_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *kernel, struct csi_tensor *bias,
                                struct conv2d_params *params);

int csi_nn_rvv_dwconv3x3s1_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *kernel, struct csi_tensor *bias,
                                struct conv2d_params *params);

int csi_nn_rvv_dwconv3x3s2_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *kernel, struct csi_tensor *bias,
                                struct conv2d_params *params);

int csi_nn_rvv_dwconv3x3s1_int8(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *kernel, struct csi_tensor *bias,
                                struct conv2d_params *params);

int csi_nn_rvv_dwconv3x3s2_int8(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *kernel, struct csi_tensor *bias,
                                struct conv2d_params *params);

int csi_nn_rvv_dwconv3x3s1_int4(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *kernel, struct csi_tensor *bias,
                                struct conv2d_params *params);

int csi_nn_rvv_dwconv3x3s2_int4(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *kernel, struct csi_tensor *bias,
                                struct conv2d_params *params);

void csi_nn_rvv_reorder_kernel_n8_fp32(float *a, float *sa, int m, int k, int ldx);
void csi_nn_rvv_reorder_input_z8_fp32(float *b, float *sb, int k, int n, int ldx);
void csi_nn_rvv_gemm_8x8_fp32(float *dst, const float *sa, const float *sb, int m, int k, int n,
                              int ldc, float *bias);

void csi_nn_rvv256_reorder_input_z16_fp32(float *b, float *sb, int k, int n, int ldx);
void csi_nn_rvv256_gemm_8x16_fp32(float *dst, const float *sa, const float *sb, int m, int k, int n,
                                  int ldc, float *bias);

void csi_nn_rvv_reorder_kernel_n8_fp16(__fp16 *a, __fp16 *sa, int m, int k, int ldx);
void csi_nn_rvv_reorder_input_z16_fp16(__fp16 *b, __fp16 *sb, int k, int n, int ldx);
void csi_nn_rvv_gemm_8x16_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, int m, int k, int n,
                               int ldc, __fp16 *bias);

void csi_nn_rvv256_reorder_kernel_n16_fp16(__fp16 *a, __fp16 *sa, int m, int k, int ldx);
void csi_nn_rvv256_reorder_input_z16_fp16(__fp16 *b, __fp16 *sb, int k, int n, int ldx);
void csi_nn_rvv256_gemm_16x16_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, int m, int k,
                                   int n, int ldc, __fp16 *bias);

void csi_nn_rvv_reorder_kernel_n8_int8(int8_t *a, int8_t *sa, int m, int k, int ldx);
void csi_nn_rvv_reorder_input_z8_int8(int8_t *b, int8_t *sb, int k, int n, int ldx);
void csi_nn_rvv_gemm_8x8_int32(int32_t *dst, const int8_t *sa, const int8_t *sb, int m, int k,
                               int n, int ldc, int32_t *bias);
void csi_nn_rvv_gemm_8x8_int8(int8_t *dst, const int8_t *sa, const int8_t *sb, int m, int k, int n,
                              int ldc, int32_t *bias, int32_t out_zp, int32_t *mult,
                              int32_t *shift);

void csi_nn_rvv256_reorder_input_z16_int8(int8_t *b, int8_t *sb, int k, int n, int ldx);
void csi_nn_rvv256_gemm_8x16_int32(int32_t *dst, const int8_t *sa, const int8_t *sb, int m, int k,
                                   int n, int ldc, int32_t *bias);

void csi_nn_rvv_reorder_input_n8_int4(int8_t *a, int8_t *sa, int m, int k, int ldx);
void csi_nn_rvv_reorder_kernel_n8_int4(int8_t *b, int8_t *sb, int n, int k, int ldx);
void csi_nn_rvv_gemm_8x8_int4(int8_t *dst, const int8_t *sa, const int8_t *sb, int m, int k, int n,
                              int ldc, int32_t *bias, int32_t out_zp, int32_t *mult,
                              int32_t *shift);

/************************************ pooling *********************************/
int csi_nn_rvv_avgpool2x2s2_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                 struct pool_params *params);

int csi_nn_rvv_avgpool2x2s2_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                 struct pool_params *params);

int csi_nn_rvv_avgpool2x2s2_p1_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_avgpool2x2s2_p1_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_avgpool3x3s2_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                 struct pool_params *params);

int csi_nn_rvv_avgpool3x3s2_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                 struct pool_params *params);

int csi_nn_rvv_avgpool3x3s2_p1_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_avgpool3x3s2_p1_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_avgpool3x3s1_p1_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_avgpool3x3s1_p1_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_maxpool2x2s2_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                 struct pool_params *params);

int csi_nn_rvv_maxpool2x2s2_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                 struct pool_params *params);

int csi_nn_rvv_maxpool2x2s2_int8(struct csi_tensor *input, struct csi_tensor *output,
                                 struct pool_params *params);

int csi_nn_rvv_maxpool2x2s2_p1_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_maxpool2x2s2_p1_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_maxpool2x2s2_p1_int8(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_maxpool3x3s2_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                 struct pool_params *params);

int csi_nn_rvv_maxpool3x3s2_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                 struct pool_params *params);

int csi_nn_rvv_maxpool3x3s2_int8(struct csi_tensor *input, struct csi_tensor *output,
                                 struct pool_params *params);

int csi_nn_rvv_maxpool3x3s2_p1_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_maxpool3x3s2_p1_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_maxpool3x3s2_p1_int8(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_maxpool3x3s1_p1_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_maxpool3x3s1_p1_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_maxpool3x3s1_p1_int8(struct csi_tensor *input, struct csi_tensor *output,
                                    struct pool_params *params);

int csi_nn_rvv_global_avgpool2d_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                     struct pool_params *params);

int csi_nn_rvv_global_avgpool2d_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                     struct pool_params *params);

int csi_nn_rvv_global_maxpool2d_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                     struct pool_params *params);

int csi_nn_rvv_global_maxpool2d_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                     struct pool_params *params);

/************************************ fullyconnected *********************************/
void csi_nn_rvv_fc_gemv_transform_weight_fp32(struct csi_tensor *weights);

int csi_nn_rvv_fullyconnected_packn_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                         struct csi_tensor *weights, struct csi_tensor *bias,
                                         struct fc_params *params);

void csi_nn_rvv_fc_gemv_transform_weight_fp16(struct csi_tensor *weights);

int csi_nn_rvv_fullyconnected_packn_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                         struct csi_tensor *weights, struct csi_tensor *bias,
                                         struct fc_params *params);

void csi_nn_rvv_fc_gemv_transform_weight_int8(struct csi_tensor *weights);

int csi_nn_rvv_fullyconnected_packn_int8(struct csi_tensor *input, struct csi_tensor *output,
                                         struct csi_tensor *weights, struct csi_tensor *bias,
                                         struct fc_params *params);

/************************************ activation *********************************/
int csi_nn_rvv_relu_fp32(struct csi_tensor *input, struct csi_tensor *output,
                         struct relu_params *params);

int csi_nn_rvv_relu_fp16(struct csi_tensor *input, struct csi_tensor *output,
                         struct relu_params *params);

int csi_nn_rvv_relu_int8(struct csi_tensor *input, struct csi_tensor *output,
                         struct relu_params *params);

int csi_nn_rvv_leaky_relu_fp32(struct csi_tensor *input, struct csi_tensor *output,
                               struct relu_params *params);

int csi_nn_rvv_leaky_relu_fp16(struct csi_tensor *input, struct csi_tensor *output,
                               struct relu_params *params);

int csi_nn_rvv_leaky_relu_int8(struct csi_tensor *input, struct csi_tensor *output,
                               struct relu_params *params);

int csi_nn_rvv_sigmoid_fp16(struct csi_tensor *input, struct csi_tensor *output,
                            struct sigmoid_params *params);

int csi_nn_rvv_softmax_fp16(struct csi_tensor *input, struct csi_tensor *output,
                            struct softmax_params *params);

/************************************ layout/memory transform *********************************/
int csi_nn_rvv_concat_fp32(struct csi_tensor **input, struct csi_tensor *output,
                           struct concat_params *params);

int csi_nn_rvv_concat_fp16(struct csi_tensor **input, struct csi_tensor *output,
                           struct concat_params *params);

int csi_nn_rvv_concat_int8(struct csi_tensor **input, struct csi_tensor *output,
                           struct concat_params *params);

/************************************ basic math *********************************/
int csi_nn_rvv_add_fp32(struct csi_tensor *input0, struct csi_tensor *input1,
                        struct csi_tensor *output, struct diso_params *params);

int csi_nn_rvv_add_fp16(struct csi_tensor *input0, struct csi_tensor *input1,
                        struct csi_tensor *output, struct diso_params *params);

int csi_nn_rvv_add_int8(struct csi_tensor *input0, struct csi_tensor *input1,
                        struct csi_tensor *output, struct diso_params *params);

int csi_nn_rvv_mul_fp32(struct csi_tensor *input0, struct csi_tensor *input1,
                        struct csi_tensor *output, struct diso_params *params);

int csi_nn_rvv_mul_fp16(struct csi_tensor *input0, struct csi_tensor *input1,
                        struct csi_tensor *output, struct diso_params *params);

int csi_nn_rvv_mul_int8(struct csi_tensor *input0, struct csi_tensor *input1,
                        struct csi_tensor *output, struct diso_params *params);

int csi_nn_rvv_sum_stride_int8(struct csi_tensor *input, struct csi_tensor *output,
                               struct reduce_params *params);

/************************************ utils *********************************/
void csi_nn_rvv_pad_input_fp32(const float *input, float *input_padded, int inc, int inh, int inw,
                               int padded_h, int padded_w, int pad_top, int pad_left);

void csi_nn_rvv_pad_input_fp16(const __fp16 *input, __fp16 *input_padded, int inc, int inh, int inw,
                               int padded_h, int padded_w, int pad_top, int pad_left);

void csi_nn_rvv_pad_input_int8(const int8_t *input, int8_t *input_padded, int inc, int inh, int inw,
                               int padded_h, int padded_w, int pad_top, int pad_left,
                               int8_t pad_value);

void csi_nn_rvv_saturated_int8(int32_t *src, int8_t *dst, int32_t out_zp, int size);

void csi_nn_rvv_requantize(int32_t *src, int32_t multiplier, int32_t shift, int channel_size);

void csi_nn_rvv_pad_input_int4_trans_int8(const int8_t *input, int8_t *input_padded, int inc,
                                          int inh, int inw, int padded_h, int padded_w, int pad_top,
                                          int pad_left, int8_t pad_value);
void csi_nn_rvv_int4_to_int8(int8_t *src, int8_t *dst, int size);
void csi_nn_rvv_int8_to_int4(int8_t *src, int8_t *dst, int size);
void csi_nn_rvv_int4_trans_int8(int8_t *src, int8_t *dst, int size);

int csrr_vl();
int csrr_vlenb();

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_CSI_THEAD_RVV_H_
