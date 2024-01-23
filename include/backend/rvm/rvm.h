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

#ifndef INCLUDE_SHL_THEAD_RVM_H_
#define INCLUDE_SHL_THEAD_RVM_H_

#include "rvv/rvv.h"

#ifdef __riscv_xtheadmatrix
#include <riscv_matrix.h>
#define MATRIX_PW_I32  // requantize inst: enable
#endif                 // __riscv_xtheadmatrix

/********************************** initialization ******************************/
int shl_rvm_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);
int shl_rvm_conv2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);

int shl_rvm_depthwise_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);
int shl_rvm_depthwise_conv2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);

int shl_rvm_fullyconnected_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params);
int shl_rvm_fullyconnected_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params);

int shl_rvm_matmul_init_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                             struct csinn_tensor *output, struct csinn_matmul_params *params);
int shl_rvm_matmul_init_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                             struct csinn_tensor *output, struct csinn_matmul_params *params);

/************************************ convolution *********************************/
/*********************************** im2col + gemm ********************************/
void shl_rvm_conv1x1s1_gemm_reorder_kernel_int8(struct csinn_tensor *kernel,
                                                struct csinn_conv2d_params *params);
int shl_rvm_conv1x1s1_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);
int shl_rvm_group_conv1x1s1_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params);

void shl_rvm_conv_im2col_gemm_reorder_kernel_int8(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params);
int shl_rvm_conv_im2col_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);
int shl_rvm_group_conv_im2col_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params);

void shl_rvm_conv1x1s1_gemm_reorder_kernel_fp16(struct csinn_tensor *kernel,
                                                struct csinn_conv2d_params *params);
int shl_rvm_conv1x1s1_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);
int shl_rvm_group_conv1x1s1_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params);

void shl_rvm_conv_im2col_gemm_reorder_kernel_fp16(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params);
int shl_rvm_conv_im2col_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);
int shl_rvm_group_conv_im2col_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params);

/************************************* winograd ***********************************/
void shl_rvm_wg_b4f3s1_trans_kernel_nhwc_fp16(struct csinn_tensor *src_kernel,
                                              struct csinn_tensor *dst_kernel);
int shl_rvm_wg_b4f3s1_nhwc_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);
void shl_rvm_wg_b6f3s1_trans_kernel_nhwc_fp16(struct csinn_tensor *src_kernel,
                                              struct csinn_tensor *dst_kernel);
int shl_rvm_wg_b6f3s1_nhwc_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);

/*************************************** gemm *************************************/
void shl_rvm_nhwc_gemm_int8(int8_t *dst, const int8_t *sa, const int8_t *sb, const int32_t *bias,
                            int m, int k, int n, int32_t out_zp, int32_t *mult, int32_t *shift);
void shl_rvm_nhwc_gemm_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, const __fp16 *bias,
                            int m, int k, int n);

void shl_rvm_gemm_a0b1_fp16(__fp16 *dst, __fp16 *sa, __fp16 *sb, __fp16 *bias, int M, int K, int N);
void shl_rvm_gemm_a0b1_int8_pw_i32(int8_t *dst, int8_t *sa, int8_t *sb, int32_t *bias, int M, int K,
                                   int N, int32_t out_zp, int32_t *mult, int32_t *shift);
void shl_rvm_gemm_a0b1_int8_to_int32(int8_t *dst, int8_t *sa, int8_t *sb, int32_t *bias, int M,
                                     int K, int N, int32_t out_zp, int32_t *mult, int32_t *shift);

/************************************ fullyconnected *********************************/
void shl_rvm_fc_gemm_reorder_weight_fp16(struct csinn_tensor *weights);
void shl_rvm_fc_gemm_reorder_weight_fp16_w_int8(struct csinn_tensor *weights);
void shl_rvm_fc_gemm_reorder_weight_int8(struct csinn_tensor *weights);

void shl_rvm_fc_dequantize_per_channel_i8_to_f16(struct csinn_tensor *weights,
                                                 struct csinn_fc_params *params,
                                                 __fp16 *weights_fp16);

int shl_rvm_fullyconnected_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params);
int shl_rvm_fullyconnected_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params);

/*********************************** matmul *********************************/
void shl_rvm_matmul_reorder_weight_fp16(struct csinn_tensor *mat1);
void shl_rvm_matmul_reorder_weight_fp16_w_int8(struct csinn_tensor *mat1);
void shl_rvm_matmul_reorder_weight_int8(struct csinn_tensor *mat1);

void shl_rvm_matmul_a0b0_fp16(__fp16 *dst, __fp16 *sa, __fp16 *sb, int M, int K, int N);
void shl_rvm_matmul_a0b0_int8_pw_i32(int8_t *dst, int8_t *sa, int8_t *sb, int M, int K, int N,
                                     int32_t z1, int32_t z2, int32_t z3, int32_t mult,
                                     int32_t shift);
void shl_rvm_matmul_a0b0_int8_to_int32(int8_t *dst, int8_t *sa, int8_t *sb, int M, int K, int N,
                                       int32_t z1, int32_t z2, int32_t z3, int32_t mult,
                                       int32_t shift);

int shl_rvm_matmul_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                        struct csinn_tensor *output, struct csinn_matmul_params *params);
int shl_rvm_matmul_fp16_w_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                               struct csinn_tensor *output, struct csinn_matmul_params *params);
int shl_rvm_matmul_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                        struct csinn_tensor *output, struct csinn_matmul_params *params);

int csrr_xrlenb();
struct shl_rvm_option {
    bool binary_model_op_init;
};

struct shl_rvm_option *shl_rvm_get_graph_option(struct csinn_session *sess);
bool shl_rvm_get_binary_model_op_init(struct csinn_session *sess);
void shl_rvm_set_binary_model_op_init(struct csinn_session *sess, bool value);

#endif  // INCLUDE_SHL_THEAD_RVM_H_
