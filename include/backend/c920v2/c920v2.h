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

#ifndef INCLUDE_SHL_C920V2_H_
#define INCLUDE_SHL_C920V2_H_

#include "csi_nn.h"
#include "reference/ref.h"
#include "rvv/rvv.h"
#include "shl_gref.h"

#ifdef __cplusplus
extern "C" {
#endif

/*********************************** initialization ***********************************/
int shl_c920v2_conv2d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);
int shl_c920v2_conv2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);
int shl_c920v2_conv2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);

/************************************* convolution ************************************/
/*********************************** im2col + gemm ********************************/
int shl_c920v2_conv_im2col_gemm_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params);
int shl_c920v2_conv_im2col_gemm_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params);

int shl_c920v2_conv_im2col_gemm_pack1ton_fp32(struct csinn_tensor *input,
                                              struct csinn_tensor *output,
                                              struct csinn_tensor *kernel,
                                              struct csinn_tensor *bias,
                                              struct csinn_conv2d_params *params);
int shl_c920v2_conv_im2col_gemm_pack1ton_fp16(struct csinn_tensor *input,
                                              struct csinn_tensor *output,
                                              struct csinn_tensor *kernel,
                                              struct csinn_tensor *bias,
                                              struct csinn_conv2d_params *params);

int shl_c920v2_conv_im2col_gemm_packnto1_fp32(struct csinn_tensor *input,
                                              struct csinn_tensor *output,
                                              struct csinn_tensor *kernel,
                                              struct csinn_tensor *bias,
                                              struct csinn_conv2d_params *params);
int shl_c920v2_conv_im2col_gemm_packnto1_fp16(struct csinn_tensor *input,
                                              struct csinn_tensor *output,
                                              struct csinn_tensor *kernel,
                                              struct csinn_tensor *bias,
                                              struct csinn_conv2d_params *params);

/******************************** conv2d1x1s1 + gemm ******************************/
int shl_c920v2_conv1x1s1_gemm_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);
int shl_c920v2_conv1x1s1_gemm_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);
int shl_c920v2_conv1x1s1_gemm_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params);

int shl_c920v2_conv1x1s1_gemm_pack1ton_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params);
int shl_c920v2_conv1x1s1_gemm_pack1ton_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params);
int shl_c920v2_conv1x1s1_gemm_pack1ton_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params);

int shl_c920v2_conv1x1s1_gemm_packnto1_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params);
int shl_c920v2_conv1x1s1_gemm_packnto1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params);
int shl_c920v2_conv1x1s1_gemm_packnto1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params);

/************************************* gemm ncxhwx ************************************/
void shl_c920v2_ncxhwx_gemm_12xpack2n_fp32(float *dst, const float *sa, const float *sb,
                                           float *bias, int m, int k, int n, bool fuse_relu);
void shl_c920v2_ncxhwx_gemm_12xpack2n_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb,
                                           __fp16 *bias, int m, int k, int n, bool fuse_relu);
void shl_c920v2_ncxhwx_gemm_12xpackn_int8_dot(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                              int32_t *bias, int m, int k, int n, int32_t out_zp,
                                              int32_t *mult, int32_t *shift);
void shl_c920v2_ncxhwx_gemm_4xpack2n_int8(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                          int32_t *bias, int m, int k, int n, int32_t out_zp,
                                          int32_t *mult, int32_t *shift);

struct shl_c920v2_option {
    struct shl_rvv_option base;
};

int shl_c920v2_set_packn_layout(struct csinn_session *sess, bool packn_layout);
struct shl_c920v2_option *shl_c920v2_get_graph_option(struct csinn_session *sess);
bool shl_c920v2_get_binary_model_op_init(struct csinn_session *sess);
void shl_c920v2_set_binary_model_op_init(struct csinn_session *sess, bool value);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SHL_C920V2_H_
