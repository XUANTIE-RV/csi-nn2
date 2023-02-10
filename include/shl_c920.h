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

#ifndef INCLUDE_SHL_C920_H_
#define INCLUDE_SHL_C920_H_

#include "csi_nn.h"
#include "shl_gref.h"
#include "shl_ref.h"
#include "shl_thead_rvv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*********************************** initialization ***********************************/
int shl_c920_conv2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params);
int shl_c920_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params);

int shl_c920_matmul_init_fp32(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                              struct csinn_tensor *output, struct csinn_matmul_params *params);
int shl_c920_matmul_init_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                              struct csinn_tensor *output, struct csinn_matmul_params *params);

/************************************* convolution ************************************/
int shl_c920_conv_im2col_gemm_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);
int shl_c920_conv_im2col_gemm_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);

int shl_c920_conv1x1s1_gemm_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);
int shl_c920_conv1x1s1_gemm_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params);

int shl_c920_wg_b4f3s1_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);
int shl_c920_wg_b6f3s1_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);
int shl_c920_wg_b4f3s1_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);
int shl_c920_wg_b6f3s1_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params);

/************************************* gemm ncxhwx ************************************/
void shl_c920_ncxhwx_gemm_8xpack2n_fp32(float *dst, const float *sa, const float *sb, float *bias,
                                        int m, int k, int n, int ldc);
void shl_c920_ncxhwx_gemm_8xpack2n_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb,
                                        __fp16 *bias, int m, int k, int n, int ldc);

/************************************* gemm block *************************************/
void shl_c920_reorder_kernel_block_8xk_fp32(float *src, float *dst, int m, int k, const int M_BLK,
                                            const int K_BLK);
void shl_c920_gemm_block_8xpack2n_fp32(float *dst, const float *sa, const float *sb, float *bias,
                                       int m, int k, int n, const int M_BLK, const int K_BLK,
                                       const int N_BLK);

void shl_c920_reorder_kernel_block_8xk_fp16(__fp16 *src, __fp16 *dst, int m, int k, const int M_BLK,
                                            const int K_BLK);
void shl_c920_gemm_block_8xpack2n_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb,
                                       __fp16 *bias, int m, int k, int n, const int M_BLK,
                                       const int K_BLK, const int N_BLK);

/*************************************** matmul ***************************************/
int shl_c920_matmul_fp32(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                         struct csinn_tensor *output, struct csinn_matmul_params *params);
int shl_c920_matmul_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                         struct csinn_tensor *output, struct csinn_matmul_params *params);
int shl_c920_matmul_fp16_w_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                                struct csinn_tensor *output, struct csinn_matmul_params *params);

void shl_c920_u8_to_f32(const uint8_t *input, float *output, int32_t offset, float *scale,
                        uint32_t length);
void shl_c920_i8_to_f32(const int8_t *input, float *output, int32_t offset, float *scale,
                        uint32_t length);
void shl_c920_f32_to_u8(const float *input, uint8_t *output, int32_t offset, float *scale,
                        uint32_t length);
void shl_c920_f32_to_i8(const float *input, int8_t *output, int32_t offset, float *scale,
                        uint32_t length);

void *shl_c920_f32_to_input_dtype(uint32_t index, float *data, struct csinn_session *sess);
float *shl_c920_output_to_f32_dtype(uint32_t index, void *data, struct csinn_session *sess);
struct shl_c920_option {
    struct shl_rvv_option base;
};

int shl_c920_set_packn_layout(struct csinn_session *sess, bool packn_layout);
struct shl_c920_option *shl_c920_get_graph_option(struct csinn_session *sess);
bool shl_c920_get_binary_model_op_init(struct csinn_session *sess);
void shl_c920_set_binary_model_op_init(struct csinn_session *sess, bool value);

int shl_c920_detect_yolov5_postprocess(struct csinn_tensor **input_tensors,
                                       struct shl_yolov5_box *out,
                                       struct shl_yolov5_params *params);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SHL_C920_H_
