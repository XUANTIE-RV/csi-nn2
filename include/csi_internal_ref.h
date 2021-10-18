/*
 * Copyright (C) 2016-2020 C-SKY Limited. All rights reserved.
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

#ifndef _CSI_INTERNAL_REF_H
#define _CSI_INTERNAL_REF_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "csi_internal.h"
#include "csi_utils.h"

int csi_abs_f32(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_abs_u8(struct csi_tensor *input,
               struct csi_tensor *output,
               struct siso_params *params);

int csi_acos_f32(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_acos_u8(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_acosh_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_acosh_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_add_f32(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_add_u8(struct csi_tensor *input0,
               struct csi_tensor *input1,
               struct csi_tensor *output,
               struct diso_params *params);

int csi_and_u32(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_and_u8(struct csi_tensor *input0,
               struct csi_tensor *input1,
               struct csi_tensor *output,
               struct diso_params *params);

int csi_arange_f32(struct csi_tensor *output,
                   struct arange_params *params);

int csi_arange_u8(struct csi_tensor *output,
                  struct arange_params *params);

int csi_argmax_stride_i32_f32(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct reduce_params *params);

int csi_argmax_stride_i32_u8(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct reduce_params *params);

int csi_argmin_stride_i32_f32(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct reduce_params *params);

int csi_argmin_stride_i32_u8(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct reduce_params *params);

int csi_asin_f32(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_asin_u8(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_asinh_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_asinh_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_atan_f32(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_atan_u8(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_atanh_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_atanh_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_averagepool_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct pool_params *params);

int csi_averagepool_u8(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct pool_params *params);

int csi_averagepool3d_f32(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct pool_params *params);

int csi_averagepool3d_u8(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct pool_params *params);

int csi_batch_normalization_f32(struct csi_tensor *input,
                                struct csi_tensor *mean,
                                struct csi_tensor *variance,
                                struct csi_tensor *gamma,
                                struct csi_tensor *beta,
                                struct csi_tensor *output,
                                struct bn_params *params);

int csi_batch_normalization_u8(struct csi_tensor *input,
                               struct csi_tensor *mean,
                               struct csi_tensor *variance,
                               struct csi_tensor *gamma,
                               struct csi_tensor *beta,
                               struct csi_tensor *output,
                               struct bn_params *params);

int csi_batch_to_space_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct batch_to_space_params *params);

int csi_batch_to_space_u8(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct batch_to_space_params *params);

int csi_broadcast_to_f32(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct broadcast_to_params *params);

int csi_broadcast_to_u8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct broadcast_to_params *params);

int csi_ceil_f32(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_ceil_u8(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_clip_f32(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct clip_params *params);

int csi_clip_u8(struct csi_tensor *input,
                struct csi_tensor *output,
                struct clip_params *params);

int csi_col2im_f32(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct csi_tensor *kernel,
                   struct col2im_params *params);

int csi_concat_f32(struct csi_tensor **input,
                   struct csi_tensor *output,
                   struct concat_params *params);

int csi_concat_u8(struct csi_tensor **input,
                  struct csi_tensor *output,
                  struct concat_params *params);

int csi_conv2d_f32(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct csi_tensor *kernel,
                   struct csi_tensor *bias,
                   struct conv2d_params *params);

int csi_conv2d_u8(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct csi_tensor *kernel,
                  struct csi_tensor *bias,
                  struct conv2d_params *params);

int csi_conv2d_i8(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct csi_tensor *kernel,
                  struct csi_tensor *bias,
                  struct conv2d_params *params);

int csi_conv2d_channel_u8(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct csi_tensor *kernel,
                  struct csi_tensor *bias,
                  struct conv2d_params *params);

int csi_conv2d_relu_u8(struct csi_tensor *o_input,
                       struct csi_tensor *o_output,
                       struct csi_tensor *o_kernel,
                       struct csi_tensor *o_bias,
                       struct conv2d_params *params);
int csi_conv2d_relu_i8(struct csi_tensor *o_input,
                       struct csi_tensor *o_output,
                       struct csi_tensor *o_kernel,
                       struct csi_tensor *o_bias,
                       struct conv2d_params *params);

int csi_conv2d_channel_relu_u8(struct csi_tensor *o_input,
                       struct csi_tensor *o_output,
                       struct csi_tensor *o_kernel,
                       struct csi_tensor *o_bias,
                       struct conv2d_params *params);

int csi_conv2d_relu6_u8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct csi_tensor *kernel,
                        struct csi_tensor *bias,
                        struct conv2d_params *params);

int csi_conv2d_relu6_i8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct csi_tensor *kernel,
                        struct csi_tensor *bias,
                        struct conv2d_params *params);

int csi_conv2d_channel_relu6_u8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct csi_tensor *kernel,
                        struct csi_tensor *bias,
                        struct conv2d_params *params);

int csi_depthwise_conv2d_f32(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct csi_tensor *kernel,
                             struct csi_tensor *bias,
                             struct conv2d_params *params);

int csi_depthwise_conv2d_u8(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct csi_tensor *kernel,
                            struct csi_tensor *bias,
                            struct conv2d_params *params);

int csi_depthwise_conv2d_i8(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct csi_tensor *kernel,
                            struct csi_tensor *bias,
                            struct conv2d_params *params);

int csi_depthwise_conv2d_channel_u8(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct csi_tensor *kernel,
                            struct csi_tensor *bias,
                            struct conv2d_params *params);

int csi_depthwise_conv2d_relu_u8(struct csi_tensor *o_input,
                                 struct csi_tensor *o_output,
                                 struct csi_tensor *o_kernel,
                                 struct csi_tensor *o_bias,
                                 struct conv2d_params *params);

int csi_depthwise_conv2d_relu_i8(struct csi_tensor *o_input,
                                 struct csi_tensor *o_output,
                                 struct csi_tensor *o_kernel,
                                 struct csi_tensor *o_bias,
                                 struct conv2d_params *params);

int csi_depthwise_conv2d_channel_relu_u8(struct csi_tensor *o_input,
                                 struct csi_tensor *o_output,
                                 struct csi_tensor *o_kernel,
                                 struct csi_tensor *o_bias,
                                 struct conv2d_params *params);

int csi_depthwise_conv2d_relu6_u8(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct csi_tensor *kernel,
                                  struct csi_tensor *bias,
                                  struct conv2d_params *params);

int csi_depthwise_conv2d_relu6_i8(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct csi_tensor *kernel,
                                  struct csi_tensor *bias,
                                  struct conv2d_params *params);

int csi_depthwise_conv2d_channel_relu6_u8(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct csi_tensor *kernel,
                                  struct csi_tensor *bias,
                                  struct conv2d_params *params);

int csi_group_conv2d_u8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct csi_tensor *kernel,
                        struct csi_tensor *bias,
                        struct conv2d_params *params);

int csi_group_conv2d_i8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct csi_tensor *kernel,
                        struct csi_tensor *bias,
                        struct conv2d_params *params);

int csi_group_conv2d_channel_u8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct csi_tensor *kernel,
                        struct csi_tensor *bias,
                        struct conv2d_params *params);

int csi_group_conv2d_relu_u8(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct csi_tensor *kernel,
                             struct csi_tensor *bias,
                             struct conv2d_params *params);

int csi_group_conv2d_relu_i8(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct csi_tensor *kernel,
                             struct csi_tensor *bias,
                             struct conv2d_params *params);

int csi_group_conv2d_channel_relu_u8(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct csi_tensor *kernel,
                             struct csi_tensor *bias,
                             struct conv2d_params *params);

int csi_conv3d_f32(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct csi_tensor *kernel,
                         struct csi_tensor *bias,
                         struct conv3d_params *params);

int csi_conv3d_u8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct csi_tensor *kernel,
                        struct csi_tensor *bias,
                        struct conv3d_params *params);

int csi_cos_f32(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_cos_u8(struct csi_tensor *input,
               struct csi_tensor *output,
               struct siso_params *params);

int csi_cosh_f32(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_cosh_u8(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_cumprod_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct cumprod_params *params);

int csi_cumprod_u8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct cumprod_params *params);

int csi_cumsum_f32(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct cumsum_params *params);

int csi_cumsum_u8(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct cumsum_params *params);

int csi_deconv2d_u8(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct csi_tensor *bias,
                    struct conv2d_params *params);

int csi_depthwise_deconv2d_u8(struct csi_tensor *o_input,
                              struct csi_tensor *o_output,
                              struct csi_tensor *o_kernel,
                              struct csi_tensor *o_bias,
                              struct conv2d_params *params);

int csi_deconv3d_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct csi_tensor *kernel,
                     struct csi_tensor *bias,
                     struct conv3d_params *params);

int csi_depth_to_space_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct depth_to_space_params *params);

int csi_depth_to_space_u8(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct depth_to_space_params *params);

int csi_div_f32(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_div_u8(struct csi_tensor *input0,
               struct csi_tensor *input1,
               struct csi_tensor *output,
               struct diso_params *params);

int csi_elu_f32(struct csi_tensor *input,
                struct csi_tensor *output,
                struct relu_params *params);

int csi_elu_u8(struct csi_tensor *input,
               struct csi_tensor *output,
               struct relu_params *params);

int csi_equal_f32(struct csi_tensor *input0,
                  struct csi_tensor *input1,
                  struct csi_tensor *output,
                  struct diso_params *params);

int csi_equal_u8(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_erf_f32(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_erf_u8(struct csi_tensor *input,
               struct csi_tensor *output,
               struct siso_params *params);

int csi_exp_f32(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_exp_u8(struct csi_tensor *input,
               struct csi_tensor *output,
               struct siso_params *params);

int csi_expand_dims_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct expand_dims_params *params);

int csi_expand_dims_u8(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct expand_dims_params *params);

int csi_expm1_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_expm1_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_flatten_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct flatten_params *params);

int csi_flatten_u8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct flatten_params *params);

int csi_floor_divide_f32(struct csi_tensor *input0,
                         struct csi_tensor *input1,
                         struct csi_tensor *output,
                         struct diso_params *params);

int csi_floor_divide_u8(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params);

int csi_floor_mod_f32(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params);

int csi_floor_mod_u8(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params);

int csi_floor_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_floor_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_fullyconnected_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct csi_tensor *weights,
                           struct csi_tensor *bias,
                           struct fc_params *params);

int csi_fullyconnected_u8(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct csi_tensor *weights,
                          struct csi_tensor *bias,
                          struct fc_params *params);

int csi_gather_nd_f32(struct csi_tensor *input,
                      struct csi_tensor *indices,
                      struct csi_tensor *output,
                      struct gather_nd_params *params);

int csi_gather_nd_u8(struct csi_tensor *input,
                     struct csi_tensor *indices,
                     struct csi_tensor *output,
                     struct gather_nd_params *params);

int csi_gather_f32(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct gather_params *params);

int csi_gather_u8(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct gather_params *params);

int csi_global_averagepool_u8(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct pool_params *params);

int csi_global_averagepool_i8(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct pool_params *params);

int csi_global_maxpool_u8(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct pool_params *params);

int csi_greater_equal_f32(struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct diso_params *params);

int csi_greater_equal_u8(struct csi_tensor *input0,
                         struct csi_tensor *input1,
                         struct csi_tensor *output,
                         struct diso_params *params);

int csi_greater_f32(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_greater_u8(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_hard_sigmoid_f32(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct sigmoid_params *params);

int csi_hard_sigmoid_u8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct sigmoid_params *params);

int csi_im2col_f32(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct im2col_params *params);

int csi_im2col_u8(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct im2col_params *params);

int csi_isnan_bool_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct siso_params *params);

int csi_isnan_bool_u8(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_l2_normalization_f32(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct l2n_params *params);

int csi_l2_normalization_u8(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct l2n_params *params);

int csi_l2pool_f32(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct pool_params *params);

int csi_leaky_relu_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct relu_params *params);

int csi_leaky_relu_u8(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct relu_params *params);

int csi_less_equal_f32(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct diso_params *params);

int csi_less_equal_u8(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params);

int csi_less_f32(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_less_u8(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_log_softmax_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct softmax_params *params);

int csi_log_softmax_u8(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct softmax_params *params);

int csi_log_f32(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_log_u8(struct csi_tensor *input,
               struct csi_tensor *output,
               struct siso_params *params);

int csi_log1p_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_log1p_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_logical_and_f32(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params);

int csi_logical_and_u8(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct diso_params *params);

int csi_logical_not_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_logical_not_u8(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct siso_params *params);

int csi_logical_or_f32(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct diso_params *params);

int csi_logical_or_u8(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params);

int csi_logical_xor_f32(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params);

int csi_logical_xor_u8(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct diso_params *params);

int csi_lrn_f32(struct csi_tensor *input,
                struct csi_tensor *output,
                struct lrn_params *params);

int csi_lrn_u8(struct csi_tensor *input,
               struct csi_tensor *output,
               struct lrn_params *params);

int csi_matmul_f32(struct csi_tensor *mat0,
                   struct csi_tensor *mat1,
                   struct csi_tensor *output,
                   struct matmul_params *params);

int csi_matmul_u8(struct csi_tensor *mat0,
                  struct csi_tensor *mat1,
                  struct csi_tensor *output,
                  struct matmul_params *params);

int csi_max_stride_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct reduce_params *params);

int csi_max_stride_u8(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct reduce_params *params);

int csi_maximum_f32(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_maximum_u8(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_maxpool_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct pool_params *params);

int csi_maxpool_u8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct pool_params *params);

int csi_maxpool2d_locat_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct pool_params *params);

int csi_maxpool2d_locat_i32_u8(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct pool_params *params);

int csi_maxpool3d_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct pool_params *params);

int csi_maxpool3d_u8(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct pool_params *params);

int csi_mean_stride_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reduce_params *params);

int csi_mean_stride_u8(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct reduce_params *params);

int csi_min_stride_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct reduce_params *params);

int csi_min_stride_u8(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct reduce_params *params);

int csi_minimum_f32(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_minimum_u8(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_mod_f32(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_mod_u8(struct csi_tensor *input0,
               struct csi_tensor *input1,
               struct csi_tensor *output,
               struct diso_params *params);

int csi_mul_f32(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_mul_u8(struct csi_tensor *input0,
               struct csi_tensor *input1,
               struct csi_tensor *output,
               struct diso_params *params);

int csi_ndarray_size_f32(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct ndarray_size_params *params);

int csi_ndarray_size_u8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct ndarray_size_params *params);

int csi_ndarray_size_i32(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct ndarray_size_params *params);

int csi_negative_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_negative_u8(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_non_max_suppression_std(struct csi_tensor *input0,
                                struct csi_tensor *input1,
                                struct csi_tensor *output,
                                struct non_max_suppression_params *params);

int csi_not_equal_f32(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params);

int csi_not_equal_u8(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params);

int csi_not_u32(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_not_u8(struct csi_tensor *input,
               struct csi_tensor *output,
               struct siso_params *params);

int csi_or_u32(struct csi_tensor *input0,
               struct csi_tensor *input1,
               struct csi_tensor *output,
               struct diso_params *params);

int csi_or_u8(struct csi_tensor *input0,
              struct csi_tensor *input1,
              struct csi_tensor *output,
              struct diso_params *params);

int csi_pad_f32(struct csi_tensor *input,
                struct csi_tensor *output,
                struct pad_params *params);

int csi_pad_u8(struct csi_tensor *input,
               struct csi_tensor *output,
               struct pad_params *params);

int csi_power_f32(struct csi_tensor *input0,
                  struct csi_tensor *input1,
                  struct csi_tensor *output,
                  struct diso_params *params);

int csi_power_u8(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_prelu_f32(struct csi_tensor *input,
                  struct csi_tensor *alpha,
                  struct csi_tensor *output,
                  struct prelu_params *params);

int csi_prelu_u8(struct csi_tensor *input,
                 struct csi_tensor *alpha,
                 struct csi_tensor *output,
                 struct prelu_params *params);

int csi_prod_stride_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reduce_params *params);

int csi_prod_stride_u8(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct reduce_params *params);

int csi_proposal_f32(struct csi_tensor *cls_prob,
                     struct csi_tensor *bbox_pred,
                     struct csi_tensor *im_info,
                     struct csi_tensor *output,
                     struct proposal_params *params);

int csi_proposal_u8(struct csi_tensor *cls_prob,
                    struct csi_tensor *bbox_pred,
                    struct csi_tensor *im_info,
                    struct csi_tensor *output,
                    struct proposal_params *params);

int csi_psroipooling_f32(struct csi_tensor *data,
                         struct csi_tensor *rois,
                         struct csi_tensor *output,
                         struct psroipooling_params *params);

int csi_psroipooling_u8(struct csi_tensor *data,
                        struct csi_tensor *rois,
                        struct csi_tensor *output,
                        struct psroipooling_params *params);

int csi_reduce_logsumexp_f32(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct reduce_params *params);

int csi_reduce_logsumexp_u8(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct reduce_params *params);

int csi_reduce_max_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct reduce_params *params);

int csi_reduce_max_u8(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct reduce_params *params);

int csi_reduce_mean_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reduce_params *params);

int csi_reduce_mean_u8(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct reduce_params *params);

int csi_reduce_min_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct reduce_params *params);

int csi_reduce_min_u8(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct reduce_params *params);

int csi_reduce_prod_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reduce_params *params);

int csi_reduce_prod_u8(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct reduce_params *params);

int csi_reduce_sum_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct reduce_params *params);

int csi_reduce_sum_u8(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct reduce_params *params);

int csi_relu_f32(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct relu_params *params);

int csi_relu_u8(struct csi_tensor *input,
                struct csi_tensor *output,
                struct relu_params *params);

int csi_relu1_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct relu_params *params);

int csi_relu1_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct relu_params *params);

int csi_relu6_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct relu_params *params);

int csi_relu6_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct relu_params *params);

int csi_relun_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct relu_params *params);

int csi_relun_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct relu_params *params);

int csi_reshape_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reshape_params *params);

int csi_reshape_u8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct reshape_params *params);

int csi_resize_f32(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct resize_params *params);

int csi_resize_u8(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct resize_params *params);

int csi_reverse_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reverse_params *params);

int csi_reverse_u8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct reverse_params *params);

int csi_roi_align_f32(struct csi_tensor *data,
                      struct csi_tensor *rois,
                      struct csi_tensor *output,
                      struct roi_align_params *params);

int csi_roipool_f32(struct csi_tensor *data,
                    struct csi_tensor *rois,
                    struct csi_tensor *output,
                    struct roi_pool_params *params);

int csi_roipool_u8(struct csi_tensor *data,
                   struct csi_tensor *rois,
                   struct csi_tensor *output,
                   struct roi_pool_params *params);

int csi_round_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_round_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_rsqrt_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_rsqrt_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_unsorted_segment_max_f32(struct csi_tensor *input,
                                 struct csi_tensor *segment_ids,
                                 struct csi_tensor *output,
                                 struct segment_params *params);

int csi_segment_max_f32(struct csi_tensor *input,
                        struct csi_tensor *segment_ids,
                        struct csi_tensor *output,
                        struct segment_params *params);

int csi_unsorted_segment_max_u8(struct csi_tensor *input,
                                struct csi_tensor *segment_ids,
                                struct csi_tensor *output,
                                struct segment_params *params);

int csi_segment_max_u8(struct csi_tensor *input,
                       struct csi_tensor *segment_ids,
                       struct csi_tensor *output,
                       struct segment_params *params);

int csi_unsorted_segment_mean_f32(struct csi_tensor *input,
                                  struct csi_tensor *segment_ids,
                                  struct csi_tensor *output,
                                  struct segment_params *params);

int csi_segment_mean_f32(struct csi_tensor *input,
                         struct csi_tensor *segment_ids,
                         struct csi_tensor *output,
                         struct segment_params *params);

int csi_unsorted_segment_mean_u8(struct csi_tensor *input,
                                 struct csi_tensor *segment_ids,
                                 struct csi_tensor *output,
                                 struct segment_params *params);

int csi_segment_mean_u8(struct csi_tensor *input,
                        struct csi_tensor *segment_ids,
                        struct csi_tensor *output,
                        struct segment_params *params);

int csi_unsorted_segment_min_f32(struct csi_tensor *input,
                                 struct csi_tensor *segment_ids,
                                 struct csi_tensor *output,
                                 struct segment_params *params);

int csi_segment_min_f32(struct csi_tensor *input,
                        struct csi_tensor *segment_ids,
                        struct csi_tensor *output,
                        struct segment_params *params);

int csi_unsorted_segment_min_u8(struct csi_tensor *input,
                                struct csi_tensor *segment_ids,
                                struct csi_tensor *output,
                                struct segment_params *params);

int csi_segment_min_u8(struct csi_tensor *input,
                       struct csi_tensor *segment_ids,
                       struct csi_tensor *output,
                       struct segment_params *params);

int csi_unsorted_segment_prod_f32(struct csi_tensor *input,
                                  struct csi_tensor *segment_ids,
                                  struct csi_tensor *output,
                                  struct segment_params *params);

int csi_segment_prod_f32(struct csi_tensor *input,
                         struct csi_tensor *segment_ids,
                         struct csi_tensor *output,
                         struct segment_params *params);

int csi_unsorted_segment_prod_u8(struct csi_tensor *input,
                                 struct csi_tensor *segment_ids,
                                 struct csi_tensor *output,
                                 struct segment_params *params);

int csi_segment_prod_u8(struct csi_tensor *input,
                        struct csi_tensor *segment_ids,
                        struct csi_tensor *output,
                        struct segment_params *params);

int csi_unsorted_segment_sum_f32(struct csi_tensor *input,
                                 struct csi_tensor *segment_ids,
                                 struct csi_tensor *output,
                                 struct segment_params *params);

int csi_segment_sum_f32(struct csi_tensor *input,
                        struct csi_tensor *segment_ids,
                        struct csi_tensor *output,
                        struct segment_params *params);

int csi_unsorted_segment_sum_u8(struct csi_tensor *input,
                                struct csi_tensor *segment_ids,
                                struct csi_tensor *output,
                                struct segment_params *params);

int csi_segment_sum_u8(struct csi_tensor *input,
                       struct csi_tensor *segment_ids,
                       struct csi_tensor *output,
                       struct segment_params *params);

int csi_select_f32(struct csi_tensor *condition,
                   struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct select_params *params);

int csi_select_u8(struct csi_tensor *condition,
                  struct csi_tensor *input0,
                  struct csi_tensor *input1,
                  struct csi_tensor *output,
                  struct select_params *params);

int csi_shape_i32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct shape_params *params);

int csi_shape_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct shape_params *params);

int csi_shuffle_channel_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct shuffle_channel_params *params);

int csi_shuffle_channel_u8(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct shuffle_channel_params *params);

int csi_sigmoid_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct sigmoid_params *params);

int csi_sigmoid_u8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct sigmoid_params *params);

int csi_sign_f32(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_sign_u8(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_sin_f32(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_sin_u8(struct csi_tensor *input,
               struct csi_tensor *output,
               struct siso_params *params);

int csi_sinh_f32(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_sinh_u8(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_slice_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct slice_params *params);

int csi_slice_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct slice_params *params);

int csi_softmax_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct softmax_params *params);

int csi_softmax_u8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct softmax_params *params);

int csi_softmax_i8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct softmax_params *params);

int csi_softplus_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_softplus_u8(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_softrelu_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct relu_params *params);

int csi_softrelu_u8(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct relu_params *params);

int csi_softsign_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_softsign_u8(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_space_to_batch_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct space_to_batch_params *params);

int csi_space_to_batch_u8(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct space_to_batch_params *params);

int csi_space_to_depth_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct space_to_depth_params *params);

int csi_space_to_depth_u8(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct space_to_depth_params *params);

int csi_split_u8(struct csi_tensor *input,
                 struct csi_tensor **output,
                 struct split_params *params);

int csi_sqrt_f32(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_sqrt_u8(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_square_f32(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_squeeze_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct squeeze_params *params);

int csi_squeeze_u8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct squeeze_params *params);

int csi_stack_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct stack_params *params);

int csi_stack_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct stack_params *params);

int csi_strided_slice_f32(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct strided_slice_params *params);

int csi_strided_slice_u8(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct strided_slice_params *params);

int csi_sub_f32(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_sub_u8(struct csi_tensor *input0,
               struct csi_tensor *input1,
               struct csi_tensor *output,
               struct diso_params *params);

int csi_sum_stride_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct reduce_params *params);

int csi_sum_stride_u8(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct reduce_params *params);

int csi_tan_f32(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_tan_u8(struct csi_tensor *input,
               struct csi_tensor *output,
               struct siso_params *params);

int csi_tanh_f32(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_tanh_f64(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_tanh_u8(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_threshold_relu_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct relu_params *params);

int csi_threshold_relu_u8(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct relu_params *params);

int csi_tile_f32(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct tile_params *params);

int csi_tile_u8(struct csi_tensor *input,
                struct csi_tensor *output,
                struct tile_params *params);

int csi_topk_f32(struct csi_tensor *input,
                 struct csi_tensor *output1,
                 struct csi_tensor *output2,
                 struct topk_params *params);

int csi_topk_u8(struct csi_tensor *input,
                struct csi_tensor *output1,
                struct csi_tensor *output2,
                struct topk_params *params);

int csi_transpose_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct transpose_params *params);

int csi_transpose_u8(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct transpose_params *params);

int csi_transpose_i8(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct transpose_params *params);

int csi_trunc_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_trunc_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_unpooling_f32(struct csi_tensor *input,
                      struct csi_tensor *mask,
                      struct csi_tensor *output,
                      struct unpooling_params *params);

int csi_unpooling_u8(struct csi_tensor *input,
                     struct csi_tensor *mask,
                     struct csi_tensor *output,
                     struct unpooling_params *params);

int csi_unstack_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct unstack_params *params);

int csi_unstack_u8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct unstack_params *params);

int csi_xor_u32(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_xor_u8(struct csi_tensor *input0,
               struct csi_tensor *input1,
               struct csi_tensor *output,
               struct diso_params *params);

int csi_yuv_rgb_scale_f32(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct siso_params *params);

int csi_yuv_rgb_scale_u8(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct siso_params *params);

#endif
