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

#ifndef INCLUDE_CSI_C906_H_
#define INCLUDE_CSI_C906_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "csi_internal.h"
#include "csi_ref.h"
#include "csi_thead_rvv.h"
#include "csi_utils.h"

/************************** f32 func declaration ***************************/
int csi_c906_abs_f32(struct csi_tensor *input, struct csi_tensor *output,
                     struct siso_params *params);

int csi_c906_add_f32(struct csi_tensor *input0, struct csi_tensor *input1,
                     struct csi_tensor *output, struct diso_params *params);

int csi_c906_sub_f32(struct csi_tensor *input0, struct csi_tensor *input1,
                     struct csi_tensor *output, struct diso_params *params);

int csi_c906_mul_f32(struct csi_tensor *input0, struct csi_tensor *input1,
                     struct csi_tensor *output, struct diso_params *params);

int csi_c906_minimum_f32(struct csi_tensor *input0, struct csi_tensor *input1,
                         struct csi_tensor *output, struct diso_params *params);

int csi_c906_broadcast_to_f32(struct csi_tensor *input, struct csi_tensor *output,
                              struct broadcast_to_params *params);

int csi_c906_clip_f32(struct csi_tensor *input, struct csi_tensor *output,
                      struct clip_params *params);

int csi_c906_concat_f32(struct csi_tensor **input, struct csi_tensor *output,
                        struct concat_params *params);

int csi_c906_split_f32(struct csi_tensor *input, struct csi_tensor **output,
                       struct split_params *params);

int csi_c906_fullyconnected_init(struct csi_tensor *input, struct csi_tensor *output,
                                 struct csi_tensor *weights, struct csi_tensor *bias,
                                 struct fc_params *params);

int csi_c906_fullyconnected_f32(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *weights, struct csi_tensor *bias,
                                struct fc_params *params);

int csi_c906_pad_f32(struct csi_tensor *input, struct csi_tensor *output,
                     struct pad_params *params);

int csi_c906_prelu_f32(struct csi_tensor *input, struct csi_tensor *alpha,
                       struct csi_tensor *output, struct prelu_params *params);

int csi_c906_relu_f32(struct csi_tensor *input, struct csi_tensor *output,
                      struct relu_params *params);

int csi_c906_relu1_f32(struct csi_tensor *input, struct csi_tensor *output,
                       struct relu_params *params);

int csi_c906_relu6_f32(struct csi_tensor *input, struct csi_tensor *output,
                       struct relu_params *params);

int csi_c906_leaky_relu_f32(struct csi_tensor *input, struct csi_tensor *output,
                            struct relu_params *params);

int csi_c906_conv1d_init(struct csi_tensor *input, struct csi_tensor *output,
                         struct csi_tensor *kernel, struct csi_tensor *bias,
                         struct conv1d_params *params);

int csi_c906_conv2d_init(struct csi_tensor *input, struct csi_tensor *output,
                         struct csi_tensor *kernel, struct csi_tensor *bias,
                         struct conv2d_params *params);

int csi_c906_conv2d_relu_init(struct csi_tensor *input, struct csi_tensor *output,
                              struct csi_tensor *kernel, struct csi_tensor *bias,
                              struct conv2d_params *params);

int csi_c906_depthwise_conv2d_init(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *kernel, struct csi_tensor *bias,
                                   struct conv2d_params *params);

int csi_c906_depthwise_conv2d_relu_init(struct csi_tensor *input, struct csi_tensor *output,
                                        struct csi_tensor *kernel, struct csi_tensor *bias,
                                        struct conv2d_params *params);

int csi_c906_maxpool2d_init(struct csi_tensor *input, struct csi_tensor *output,
                            struct pool_params *params);

int csi_c906_global_maxpool2d_f32(struct csi_tensor *input, struct csi_tensor *output,
                                  struct pool_params *params);

int csi_c906_avgpool2d_init(struct csi_tensor *input, struct csi_tensor *output,
                            struct pool_params *params);

int csi_c906_global_avgpool2d_f32(struct csi_tensor *input, struct csi_tensor *output,
                                  struct pool_params *params);

int csi_c906_div_init(struct csi_tensor *input0, struct csi_tensor *input1,
                      struct csi_tensor *output, struct diso_params *params);

/* pack */
void csi_c906_reorder_kernel(float *a, float *sa, int m, int k, int ldx);

void csi_c906_reorder_input(float *b, float *sb, int k, int n, int ldx);

void csi_c906_reorder_input_1(float *b, float *sb, int k, int n, int ldx);

/* gemm */
void csi_c906_sgemm_kernel_f32(float *dst, const float *sa, const float *sb, int m, int k, int n,
                               int ldc, float *bias, bool fuse_relu);

/* kernel transform */
void csi_c906_conv1x1s1_sgemm_transform_kernel(struct csi_tensor *kernel,
                                               struct conv2d_params *params);

void csi_c906_conv_im2col_sgemm_transform_kernel(struct csi_tensor *kernel,
                                                 struct conv2d_params *params);

void csi_c906_conv3x3s1_winograd23_transform_kernel(struct csi_tensor *o_kernel,
                                                    struct csi_tensor *t_kernel);

void csi_c906_conv3x3s1_winograd43_transform_kernel(struct csi_tensor *o_kernel,
                                                    struct csi_tensor *t_kernel);

void csi_c906_conv3x3s1_winograd64_transform_kernel(struct csi_tensor *o_kernel,
                                                    struct csi_tensor *t_kernel);

void csi_c906_conv3x3s1_winograd64_transform_kernel_1(struct csi_tensor *o_kernel,
                                                      struct csi_tensor *t_kernel);

void csi_c906_conv3x3s1_winograd64_transform_kernel_pack4(struct csi_tensor *o_kernel,
                                                          struct csi_tensor *t_kernel);

void csi_c906_conv3x3s1_winograd43_transform_kernel_pack4(struct csi_tensor *o_kernel,
                                                          struct csi_tensor *t_kernel);

/* convolution optimization */
int csi_c906_conv1x1s1_sgemm(struct csi_tensor *input, struct csi_tensor *output,
                             struct csi_tensor *kernel, struct csi_tensor *bias,
                             struct conv2d_params *params);

int csi_c906_conv1x1s1_sgemm_fuse_relu(struct csi_tensor *input, struct csi_tensor *output,
                                       struct csi_tensor *kernel, struct csi_tensor *bias,
                                       struct conv2d_params *params);

int csi_c906_conv_im2col_sgemm(struct csi_tensor *input, struct csi_tensor *output,
                               struct csi_tensor *kernel, struct csi_tensor *bias,
                               struct conv2d_params *params);

int csi_c906_conv_im2col_sgemm_fuse_relu(struct csi_tensor *input, struct csi_tensor *output,
                                         struct csi_tensor *kernel, struct csi_tensor *bias,
                                         struct conv2d_params *params);

int csi_c906_conv3x3s1_winograd23(struct csi_tensor *input, struct csi_tensor *output,
                                  struct csi_tensor *kernel, struct csi_tensor *bias,
                                  struct conv2d_params *params);

int csi_c906_conv3x3s1_winograd43(struct csi_tensor *input, struct csi_tensor *output,
                                  struct csi_tensor *kernel, struct csi_tensor *bias,
                                  struct conv2d_params *params);

int csi_c906_conv3x3s1_winograd64(struct csi_tensor *input, struct csi_tensor *output,
                                  struct csi_tensor *kernel, struct csi_tensor *bias,
                                  struct conv2d_params *params);

int csi_c906_conv3x3s1_winograd64_1(struct csi_tensor *input, struct csi_tensor *output,
                                    struct csi_tensor *kernel, struct csi_tensor *bias,
                                    struct conv2d_params *params);

int csi_c906_conv3x3s1_winograd64_pack4(struct csi_tensor *input, struct csi_tensor *output,
                                        struct csi_tensor *kernel, struct csi_tensor *bias,
                                        struct conv2d_params *params);

int csi_c906_conv3x3s1_winograd43_pack4(struct csi_tensor *input, struct csi_tensor *output,
                                        struct csi_tensor *kernel, struct csi_tensor *bias,
                                        struct conv2d_params *params);

void csi_c906_conv3x3s1(struct csi_tensor *input, struct csi_tensor *output,
                        struct csi_tensor *kernel, struct csi_tensor *bias,
                        struct conv2d_params *params);

void csi_c906_conv3x3s2(struct csi_tensor *input, struct csi_tensor *output,
                        struct csi_tensor *kernel, struct csi_tensor *bias,
                        struct conv2d_params *params);

/* depthwise convolution optimization */
int csi_c906_dwconv3x3s1(struct csi_tensor *input, struct csi_tensor *output,
                         struct csi_tensor *kernel, struct csi_tensor *bias,
                         struct conv2d_params *params);

int csi_c906_dwconv3x3s2(struct csi_tensor *input, struct csi_tensor *output,
                         struct csi_tensor *kernel, struct csi_tensor *bias,
                         struct conv2d_params *params);

int csi_c906_dwconv5x5s1(struct csi_tensor *input, struct csi_tensor *output,
                         struct csi_tensor *kernel, struct csi_tensor *bias,
                         struct conv2d_params *params);

int csi_c906_dwconv5x5s2(struct csi_tensor *input, struct csi_tensor *output,
                         struct csi_tensor *kernel, struct csi_tensor *bias,
                         struct conv2d_params *params);

int csi_c906_dwconv3x3s1_pack4(struct csi_tensor *input, struct csi_tensor *output,
                               struct csi_tensor *kernel, struct csi_tensor *bias,
                               struct conv2d_params *params);

int csi_c906_dwconv3x3s2_pack4(struct csi_tensor *input, struct csi_tensor *output,
                               struct csi_tensor *kernel, struct csi_tensor *bias,
                               struct conv2d_params *params);

/* depthwise convolution fuse relu */
int csi_c906_dwconv3x3s1_fuse_relu(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *kernel, struct csi_tensor *bias,
                                   struct conv2d_params *params);

int csi_c906_dwconv3x3s2_fuse_relu(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *kernel, struct csi_tensor *bias,
                                   struct conv2d_params *params);

int csi_c906_dwconv5x5s1_fuse_relu(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *kernel, struct csi_tensor *bias,
                                   struct conv2d_params *params);

int csi_c906_dwconv5x5s2_fuse_relu(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *kernel, struct csi_tensor *bias,
                                   struct conv2d_params *params);

int csi_c906_dwconv3x3s1_pack4_fuse_relu(struct csi_tensor *input, struct csi_tensor *output,
                                         struct csi_tensor *kernel, struct csi_tensor *bias,
                                         struct conv2d_params *params);

int csi_c906_dwconv3x3s2_pack4_fuse_relu(struct csi_tensor *input, struct csi_tensor *output,
                                         struct csi_tensor *kernel, struct csi_tensor *bias,
                                         struct conv2d_params *params);

int csi_c906_dwconv2d_s1_pad0_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *kernel, struct csi_tensor *bias,
                                   struct conv2d_params *params);

/************************** fp16 func declaration ***************************/
int csi_c906_add_fp16(struct csi_tensor *input0, struct csi_tensor *input1,
                      struct csi_tensor *output, struct diso_params *params);

int csi_c906_sub_fp16(struct csi_tensor *input0, struct csi_tensor *input1,
                      struct csi_tensor *output, struct diso_params *params);

int csi_c906_mul_fp16(struct csi_tensor *input0, struct csi_tensor *input1,
                      struct csi_tensor *output, struct diso_params *params);

int csi_c906_minimum_fp16(struct csi_tensor *input0, struct csi_tensor *input1,
                          struct csi_tensor *output, struct diso_params *params);

int csi_c906_global_avgpool2d_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                   struct pool_params *params);

int csi_c906_global_maxpool2d_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                   struct pool_params *params);

int csi_c906_pad_fp16(struct csi_tensor *input, struct csi_tensor *output,
                      struct pad_params *params);

int csi_c906_relu_fp16(struct csi_tensor *input, struct csi_tensor *output,
                       struct relu_params *params);

int csi_c906_relu1_fp16(struct csi_tensor *input, struct csi_tensor *output,
                        struct relu_params *params);

int csi_c906_relu6_fp16(struct csi_tensor *input, struct csi_tensor *output,
                        struct relu_params *params);

int csi_c906_prelu_fp16(struct csi_tensor *input, struct csi_tensor *alpha,
                        struct csi_tensor *output, struct prelu_params *params);

int csi_c906_leaky_relu_fp16(struct csi_tensor *input, struct csi_tensor *output,
                             struct relu_params *params);

int csi_c906_abs_fp16(struct csi_tensor *input, struct csi_tensor *output,
                      struct siso_params *params);

int csi_c906_clip_fp16(struct csi_tensor *input, struct csi_tensor *output,
                       struct clip_params *params);

int csi_c906_concat_fp16(struct csi_tensor **input, struct csi_tensor *output,
                         struct concat_params *params);

int csi_c906_split_fp16(struct csi_tensor *input, struct csi_tensor **output,
                        struct split_params *params);

int csi_c906_fullyconnected_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                 struct csi_tensor *weights, struct csi_tensor *bias,
                                 struct fc_params *params);

int csi_c906_fullyconnected_pack8_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                       struct csi_tensor *weights, struct csi_tensor *bias,
                                       struct fc_params *params);

int csi_c906_fullyconnected_pack8_fp16_1(struct csi_tensor *input, struct csi_tensor *output,
                                         struct csi_tensor *weights, struct csi_tensor *bias,
                                         struct fc_params *params);

int csi_c906_fullyconnected_pack16_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                        struct csi_tensor *weights, struct csi_tensor *bias,
                                        struct fc_params *params);

int csi_c906_fullyconnected_pack16_output16_fp16(struct csi_tensor *input,
                                                 struct csi_tensor *output,
                                                 struct csi_tensor *weights,
                                                 struct csi_tensor *bias, struct fc_params *params);

void csi_c906_reorder_weight_n8_fp16(__fp16 *src, __fp16 *dst, int m, int k, int ldx);

void csi_c906_reorder_weight_n16_fp16(__fp16 *src, __fp16 *dst, int m, int k, int ldx);

/* pack fp16 */
void csi_c906_reorder_kernel_fp16(__fp16 *a, __fp16 *sa, int m, int k, int ldx);
void csi_c906_reorder_input_fp16(__fp16 *b, __fp16 *sb, int k, int n, int ldx);

void csi_c906_reorder_input_fp16_1(__fp16 *b, __fp16 *sb, int k, int n, int ldx);

void csi_c906_reorder_matrix_z8_fp16(__fp16 *src, __fp16 *dst, int k, int n, int ldx);
void csi_c906_reorder_matrix_z16_fp16(__fp16 *src, __fp16 *dst, int k, int n, int ldx);

/* gemm fp16 */
void csi_c906_sgemm_kernel_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, int m, int k,
                                int n, int ldc, __fp16 *bias);
void csi_c906_sgemm_kernel_fp16_1(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, int m, int k,
                                  int n, int ldc, __fp16 *bias);

/* gemv fp16 */
void csi_c906_gemv_pack8_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, int k, int n,
                              int ldc, __fp16 *bias);
void csi_c906_gemv_pack16_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, int k, int n,
                               int ldc, __fp16 *bias);

void csi_c906_gemv_trans_pack8_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, int k, int n,
                                    int ldc, __fp16 *bias);
void csi_c906_gemv_trans_pack16_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, int k, int n,
                                     int ldc, __fp16 *bias);

/* kernel transform fp16 */
void csi_c906_conv1x1s1_sgemm_transform_kernel_fp16(struct csi_tensor *kernel,
                                                    struct conv2d_params *params);
void csi_c906_conv_im2col_sgemm_transform_kernel_fp16(struct csi_tensor *kernel,
                                                      struct conv2d_params *params);

void csi_c906_conv3x3s1_winograd43_transform_kernel_pack8_fp16(struct csi_tensor *o_kernel,
                                                               struct csi_tensor *t_kernel);

void csi_c906_conv3x3s1_winograd64_transform_kernel_pack8_fp16(struct csi_tensor *o_kernel,
                                                               struct csi_tensor *t_kernel);

/* convolution optimization fp16 */
int csi_c906_conv1x1s1_sgemm_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                  struct csi_tensor *kernel, struct csi_tensor *bias,
                                  struct conv2d_params *params);

int csi_c906_conv1x1s1_batch_gemv_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                       struct csi_tensor *kernel, struct csi_tensor *bias,
                                       struct conv2d_params *params);

int csi_c906_conv_im2col_sgemm_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                    struct csi_tensor *kernel, struct csi_tensor *bias,
                                    struct conv2d_params *params);

int csi_c906_conv3x3s1_winograd43_pack8_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                             struct csi_tensor *kernel, struct csi_tensor *bias,
                                             struct conv2d_params *params);

int csi_c906_conv3x3s1_winograd64_pack8_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                             struct csi_tensor *kernel, struct csi_tensor *bias,
                                             struct conv2d_params *params);

void csi_c906_conv3x3s1_fp16(struct csi_tensor *input, struct csi_tensor *output,
                             struct csi_tensor *kernel, struct csi_tensor *bias,
                             struct conv2d_params *params);

void csi_c906_conv3x3s2_fp16(struct csi_tensor *input, struct csi_tensor *output,
                             struct csi_tensor *kernel, struct csi_tensor *bias,
                             struct conv2d_params *params);

/* depthwise convolution optimization for fp16*/
int csi_c906_dwconv3x3s1_fp16(struct csi_tensor *input, struct csi_tensor *output,
                              struct csi_tensor *kernel, struct csi_tensor *bias,
                              struct conv2d_params *params);

int csi_c906_dwconv3x3s2_fp16(struct csi_tensor *input, struct csi_tensor *output,
                              struct csi_tensor *kernel, struct csi_tensor *bias,
                              struct conv2d_params *params);

int csi_c906_dwconv3x3s1_pack8_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                    struct csi_tensor *kernel, struct csi_tensor *bias,
                                    struct conv2d_params *params);

int csi_c906_dwconv3x3s2_pack8_fp16(struct csi_tensor *input, struct csi_tensor *output,
                                    struct csi_tensor *kernel, struct csi_tensor *bias,
                                    struct conv2d_params *params);

/* utils */
void csi_c906_memcpy(void *dst, const void *src, size_t n);

void csi_c906_pad_input(const float *input, float *input_padded, int inc, int inh, int inw,
                        int padded_h, int padded_w, int pad_top, int pad_left);

void csi_c906_crop_output(float *output_trans, float *output, int out_c, int out_h, int out_w,
                          int wino_h, int wino_w);

void csi_c906_pad_input_fp16(const __fp16 *input, __fp16 *input_padded, int inc, int inh, int inw,
                             int padded_h, int padded_w, int pad_top, int pad_left);

void csi_c906_crop_output_fp16(__fp16 *output_trans, __fp16 *output, int out_c, int out_h,
                               int out_w, int wino_h, int wino_w);

/*asr related fuctions*/
int csi_c906_cache_matmul_init(struct csi_tensor *input, struct csi_tensor *output,
                               struct csi_tensor *weight, struct csi_tensor *bias,
                               struct cache_matmul_params *params);

int csi_c906_cache_matmul_fp16(struct csi_tensor *input, struct csi_tensor *output,
                               struct csi_tensor *weight, struct csi_tensor *bias,
                               struct cache_matmul_params *params);

int csi_c906_matmul_fp16(struct csi_tensor *mat0, struct csi_tensor *mat1,
                         struct csi_tensor *output, struct matmul_params *params);

int csi_c906_layer_norm_fp16(struct csi_tensor *input, struct csi_tensor *output,
                             struct csi_tensor *gamma, struct csi_tensor *beta,
                             struct layer_norm_params *params);

int csi_c906_reshape_fp16(struct csi_tensor *input, struct csi_tensor *output,
                          struct reshape_params *params);

int csi_c906_transpose_fp16(struct csi_tensor *input, struct csi_tensor *output,
                            struct transpose_params *params);

int csi_c906_gather_fp16(struct csi_tensor *input, struct csi_tensor *indices,
                         struct csi_tensor *output, struct gather_params *params);

int csi_c906_cache_conv1d_init(struct csi_tensor *input, struct csi_tensor *output,
                               struct csi_tensor *weight, struct csi_tensor *bias,
                               struct cache_conv1d_params *params);

int csi_c906_cache_conv1d_fp16(struct csi_tensor *input, struct csi_tensor *output,
                               struct csi_tensor *weight, struct csi_tensor *bias,
                               struct cache_conv1d_params *params);

int csi_c906_lrn_fp16(struct csi_tensor *input, struct csi_tensor *output,
                      struct lrn_params *params);

void asr_buffer_init_c906(struct asr_buffer_t *buffer, size_t buffer_size, size_t data_lenth);

void *asr_buffer_insert_c906_front(struct asr_buffer_t *buffer, void *input, size_t len);

void *asr_buffer_insert_c906_back(struct asr_buffer_t *buffer, void *input, size_t len);

void *asr_buffer_get_buffer_c906(struct asr_buffer_t *buffer);

void asr_buffer_reset_c906(struct asr_buffer_t *buffer);

void csi_c906_reset_fcsr();
int csi_c906_get_fcsr();

/* hardware performance */
struct csi_c906_hpm {
    size_t inst;
    size_t cycle;
    size_t l1_icache_access;
    size_t l1_icache_miss;
    size_t store_inst;
    size_t l1_dcache_raccess;
    size_t l1_dcache_rmiss;
    size_t l1_dcache_waccess;
    size_t l1_dcache_wmiss;
};

uint64_t csi_c906_get_inst();
uint64_t csi_c906_get_cycle();
uint64_t csi_c906_get_l1_icache_access();
uint64_t csi_c906_get_l1_icache_miss();
uint64_t csi_c906_get_cb_miss();
uint64_t csi_c906_get_cb_inst();
uint64_t csi_c906_get_store_inst();
uint64_t csi_c906_get_l1_dcache_raccess();
uint64_t csi_c906_get_l1_dcache_rmiss();
uint64_t csi_c906_get_l1_dcache_waccess();
uint64_t csi_c906_get_l1_dcache_wmiss();

struct csi_c906_hpm csi_c906_get_hw_perf();

int csi_c906_sum_stride_fp16(struct csi_tensor *input, struct csi_tensor *output,
                             struct reduce_params *params);

int csi_nn_c906_register_op_init(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *bc);
int csi_nn_c906_register_op(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *bc);

void csi_nn_c906_bc_init_reg();
void csi_nn_c906_bc_reg();

#endif  // INCLUDE_CSI_C906_H_
