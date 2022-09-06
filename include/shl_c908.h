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

#ifndef INCLUDE_SHL_C908_H_
#define INCLUDE_SHL_C908_H_

#include "csi_nn.h"
#include "shl_gref.h"
#include "shl_ref.h"
#include "shl_thead_rvv.h"

/*********************************** initialization ***********************************/
int shl_c908_conv2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params);
int shl_c908_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params);
int shl_c908_conv2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params);
int shl_c908_conv2d_init_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params);

int shl_c908_depthwise_conv2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params);
int shl_c908_depthwise_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params);
int shl_c908_depthwise_conv2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params);
int shl_c908_depthwise_conv2d_init_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params);

int shl_c908_avgpool2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_c908_avgpool2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_c908_avgpool2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_c908_avgpool2d_init_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);

int shl_c908_maxpool2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_c908_maxpool2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_c908_maxpool2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);
int shl_c908_maxpool2d_init_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);

int shl_c908_fullyconnected_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *weights, struct csinn_tensor *bias,
                                 struct csinn_fc_params *params);

/************************************ convolution *********************************/
/*********************************** im2col + gemm ********************************/
void shl_c908_conv_im2col_gemm_reorder_kernel_fp32(struct csinn_tensor *kernel,
                                                   struct csinn_conv2d_params *params);
void shl_c908_conv_im2col_gemm_reorder_kernel_fp16(struct csinn_tensor *kernel,
                                                   struct csinn_conv2d_params *params);
void shl_c908_conv_im2col_gemm_reorder_kernel_int8(struct csinn_tensor *kernel,
                                                   struct csinn_conv2d_params *params);

int shl_c908_conv_im2col_gemm_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params);
int shl_c908_conv_im2col_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params);
int shl_c908_conv_im2col_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params);

void shl_c908_conv_im2col_gemm_reorder_kernel_packn_fp32(struct csinn_tensor *kernel,
                                                         struct csinn_conv2d_params *params);
void shl_c908_conv_im2col_gemm_reorder_kernel_packn_fp16(struct csinn_tensor *kernel,
                                                         struct csinn_conv2d_params *params);
void shl_c908_conv_im2col_gemm_reorder_kernel_packn_int8(struct csinn_tensor *kernel,
                                                         struct csinn_conv2d_params *params);

int shl_c908_conv_im2col_gemm_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);
int shl_c908_conv_im2col_gemm_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);
int shl_c908_conv_im2col_gemm_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);

void shl_c908_conv_im2col_gemm_reorder_kernel_pack1ton_fp32(struct csinn_tensor *kernel,
                                                            struct csinn_conv2d_params *params);
void shl_c908_conv_im2col_gemm_reorder_kernel_pack1ton_fp16(struct csinn_tensor *kernel,
                                                            struct csinn_conv2d_params *params);
void shl_c908_conv_im2col_gemm_reorder_kernel_pack1ton_int8(struct csinn_tensor *kernel,
                                                            struct csinn_conv2d_params *params);

int shl_c908_conv_im2col_gemm_pack1ton_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params);
int shl_c908_conv_im2col_gemm_pack1ton_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params);
int shl_c908_conv_im2col_gemm_pack1ton_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params);

void shl_c908_conv_im2col_gemm_reorder_kernel_packnto1_fp32(struct csinn_tensor *kernel,
                                                            struct csinn_conv2d_params *params);
void shl_c908_conv_im2col_gemm_reorder_kernel_packnto1_fp16(struct csinn_tensor *kernel,
                                                            struct csinn_conv2d_params *params);
void shl_c908_conv_im2col_gemm_reorder_kernel_packnto1_int8(struct csinn_tensor *kernel,
                                                            struct csinn_conv2d_params *params);

int shl_c908_conv_im2col_gemm_packnto1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params);
int shl_c908_conv_im2col_gemm_packnto1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params);
int shl_c908_conv_im2col_gemm_packnto1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params);

/******************************** conv2d1x1s1 + gemm ******************************/
void shl_c908_conv1x1s1_gemm_reorder_kernel_fp32(struct csinn_tensor *kernel,
                                                 struct csinn_conv2d_params *params);
void shl_c908_conv1x1s1_gemm_reorder_kernel_fp16(struct csinn_tensor *kernel,
                                                 struct csinn_conv2d_params *params);
void shl_c908_conv1x1s1_gemm_reorder_kernel_int8(struct csinn_tensor *kernel,
                                                 struct csinn_conv2d_params *params);

int shl_c908_conv1x1s1_gemm_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params);
int shl_c908_conv1x1s1_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params);
int shl_c908_conv1x1s1_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params);

void shl_c908_conv1x1s1_gemm_reorder_kernel_packn_fp32(struct csinn_tensor *kernel,
                                                       struct csinn_conv2d_params *params);
void shl_c908_conv1x1s1_gemm_reorder_kernel_packn_fp16(struct csinn_tensor *kernel,
                                                       struct csinn_conv2d_params *params);
void shl_c908_conv1x1s1_gemm_reorder_kernel_packn_int8(struct csinn_tensor *kernel,
                                                       struct csinn_conv2d_params *params);

int shl_c908_conv1x1s1_gemm_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);
int shl_c908_conv1x1s1_gemm_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);
int shl_c908_conv1x1s1_gemm_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);

void shl_c908_conv1x1s1_gemm_reorder_kernel_pack1ton_fp32(struct csinn_tensor *kernel,
                                                          struct csinn_conv2d_params *params);
void shl_c908_conv1x1s1_gemm_reorder_kernel_pack1ton_fp16(struct csinn_tensor *kernel,
                                                          struct csinn_conv2d_params *params);
void shl_c908_conv1x1s1_gemm_reorder_kernel_pack1ton_int8(struct csinn_tensor *kernel,
                                                          struct csinn_conv2d_params *params);

int shl_c908_conv1x1s1_gemm_pack1ton_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                          struct csinn_conv2d_params *params);
int shl_c908_conv1x1s1_gemm_pack1ton_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                          struct csinn_conv2d_params *params);
int shl_c908_conv1x1s1_gemm_pack1ton_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                          struct csinn_conv2d_params *params);

void shl_c908_conv1x1s1_gemm_reorder_kernel_packnto1_fp32(struct csinn_tensor *kernel,
                                                          struct csinn_conv2d_params *params);
void shl_c908_conv1x1s1_gemm_reorder_kernel_packnto1_fp16(struct csinn_tensor *kernel,
                                                          struct csinn_conv2d_params *params);
void shl_c908_conv1x1s1_gemm_reorder_kernel_packnto1_int8(struct csinn_tensor *kernel,
                                                          struct csinn_conv2d_params *params);

int shl_c908_conv1x1s1_gemm_packnto1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                          struct csinn_conv2d_params *params);
int shl_c908_conv1x1s1_gemm_packnto1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                          struct csinn_conv2d_params *params);
int shl_c908_conv1x1s1_gemm_packnto1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                          struct csinn_conv2d_params *params);

/*********************************** winograd ***********************************/
void shl_c908_wg_b6f3s1_trans_kernel_pack8_fp32(struct csinn_tensor *src_kernel,
                                                struct csinn_tensor *dst_kernel);
void shl_c908_wg_b6f3s1_trans_kernel_pack8_fp16(struct csinn_tensor *src_kernel,
                                                struct csinn_tensor *dst_kernel);
void shl_c908_wg_b6f3s1_trans_kernel_pack16_fp16(struct csinn_tensor *src_kernel,
                                                 struct csinn_tensor *dst_kernel);

void shl_c908_wg_b4f3s1_trans_kernel_pack8_fp32(struct csinn_tensor *src_kernel,
                                                struct csinn_tensor *dst_kernel);
void shl_c908_wg_b4f3s1_trans_kernel_pack8_fp16(struct csinn_tensor *src_kernel,
                                                struct csinn_tensor *dst_kernel);
void shl_c908_wg_b4f3s1_trans_kernel_pack16_fp16(struct csinn_tensor *src_kernel,
                                                 struct csinn_tensor *dst_kernel);
void shl_c908_wg_b4f3s1_trans_kernel_pack8_int8(struct csinn_tensor *src_kernel,
                                                struct csinn_tensor *dst_kernel);

int shl_c908_wg_b6f3s1_pack8_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);
int shl_c908_wg_b6f3s1_pack8_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);
int shl_c908_wg_b6f3s1_pack16_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params);

int shl_c908_wg_b4f3s1_pack8_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);
int shl_c908_wg_b4f3s1_pack8_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);
int shl_c908_wg_b4f3s1_pack16_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params);
int shl_c908_wg_b4f3s1_pack8_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);

void shl_c908_ncxhwx_wg_b6f3s1_trans_kernel_packn_fp32(struct csinn_tensor *src_kernel,
                                                       struct csinn_tensor *dst_kernel);
void shl_c908_ncxhwx_wg_b6f3s1_trans_kernel_packn_fp16(struct csinn_tensor *src_kernel,
                                                       struct csinn_tensor *dst_kernel);

int shl_c908_ncxhwx_wg_b6f3s1_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);
int shl_c908_ncxhwx_wg_b6f3s1_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);

void shl_c908_ncxhwx_wg_b4f3s1_trans_kernel_packn_fp32(struct csinn_tensor *src_kernel,
                                                       struct csinn_tensor *dst_kernel);
void shl_c908_ncxhwx_wg_b4f3s1_trans_kernel_packn_fp16(struct csinn_tensor *src_kernel,
                                                       struct csinn_tensor *dst_kernel);
void shl_c908_ncxhwx_wg_b4f3s1_trans_kernel_packn_int8(struct csinn_tensor *src_kernel,
                                                       struct csinn_tensor *dst_kernel);

int shl_c908_ncxhwx_wg_b4f3s1_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);
int shl_c908_ncxhwx_wg_b4f3s1_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);
int shl_c908_ncxhwx_wg_b4f3s1_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);

/*********************************** gemm ncxhwx kernel ***********************************/
void shl_c908_ncxhwx_gemm_12xpack2n_fp32(float *dst, const float *sa, const float *sb,
                                         const float *bias, int m, int k, int n, bool fuse_relu);
void shl_c908_ncxhwx_gemm_12xpack2n_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb,
                                         const __fp16 *bias, int m, int k, int n, bool fuse_relu);

void shl_c908_ncxhwx_gemm_12xpackn_int8(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                        const int32_t *bias, int m, int k, int n, int32_t out_zp,
                                        int32_t *mult, int32_t *shift);

void shl_c908_ncxhwx_gemm_12xpackn_int16(int32_t *dst, const int16_t *sa, const int16_t *sb, int m,
                                         int k, int n);
/*********************************** gemm kernel ***********************************/
void shl_c908_reorder_kernel_n8_fp32(float *src, float *dst, int m, int k, int ldc);
void shl_c908_reorder_input_z12_fp32(float *src, float *dst, int k, int n, int ldc);
void shl_c908_gemm_8x12_fp32(float *dst, const float *sa, const float *sb, float *bias, int m,
                             int k, int n, int ldc);
void shl_c908_reorder_input_z8_fp32(float *src, float *dst, int k, int n, int ldc);
void shl_c908_gemm_8x8_fp32(float *dst, const float *sa, const float *sb, float *bias, int m, int k,
                            int n, int ldc);

void shl_c908_reorder_kernel_n8_fp16(__fp16 *src, __fp16 *dst, int m, int k, int ldc);
void shl_c908_reorder_input_z24_fp16(__fp16 *src, __fp16 *dst, int k, int n, int ldc);
void shl_c908_gemm_8x24_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, __fp16 *bias, int m,
                             int k, int n, int ldc);
void shl_c908_reorder_input_z16_fp16(__fp16 *src, __fp16 *dst, int k, int n, int ldc);
void shl_c908_gemm_8x16_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, __fp16 *bias, int m,
                             int k, int n, int ldc);

void shl_c908_reorder_kernel_n8_int8(int8_t *src, int8_t *dst, int m, int k, int ldc);
void shl_c908_reorder_input_z8_int8(int8_t *src, int8_t *dst, int k, int n, int ldc);
void shl_c908_gemm_8x8_int8(int8_t *dst, const int8_t *sa, const int8_t *sb, int32_t *bias, int m,
                            int k, int n, int ldc, int32_t out_zp, int32_t *mult, int32_t *shift);
void shl_c908_reorder_input_z12_int8(int8_t *src, int8_t *dst, int k, int n, int ldc);

/*********************************** VLEN = 256 ***********************************/
/*********************************** VLEN = 256 ***********************************/
/*********************************** VLEN = 256 ***********************************/

void shl_c908_reorder_input_z16_fp32_v256(float *src, float *dst, int k, int n, int ldc);
void shl_c908_gemm_8x16_fp32_v256(float *dst, const float *sa, const float *sb, float *bias, int m,
                                  int k, int n, int ldc);

void shl_c908_reorder_input_z32_fp16_v256(__fp16 *src, __fp16 *dst, int k, int n, int ldc);
void shl_c908_gemm_8x32_fp16_v256(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, __fp16 *bias,
                                  int m, int k, int n, int ldc);

void shl_c908_reorder_input_z16_int8_v256(int8_t *src, int8_t *dst, int k, int n, int ldc);
void shl_c908_gemm_8x16_int8_v256(int8_t *dst, const int8_t *sa, const int8_t *sb, int32_t *bias,
                                  int m, int k, int n, int ldc, int32_t out_zp, int32_t *mult,
                                  int32_t *shift);

#endif  // INCLUDE_SHL_C908_H_
