/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

/* CSI-NN2 version 1.10.x */

#ifndef _CSI_INTERNAL_REF_H
#define _CSI_INTERNAL_REF_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "csi_nn.h"
#include "csi_internal.h"
#include "csi_utils.h"

int csi_ref_abs_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_ref_abs_quant(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_acos_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_ref_acos_quant(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct siso_params *params);

int csi_ref_acosh_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_acosh_quant(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_ref_add_f32(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_ref_add_u8(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_ref_add_f32(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_ref_add_quant(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params);

int csi_ref_and_u32(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_ref_and_u8(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_ref_and_i8(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_ref_arange_f32(struct csi_tensor *output,
                       struct arange_params *params);

int csi_ref_arange_quant(struct csi_tensor *output,
                         struct arange_params *params);

int csi_ref_argmax_stride_i32_f32(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct reduce_params *params);

int csi_ref_argmax_stride_quant(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct reduce_params *params);

int csi_ref_argmin_stride_i32_f32(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct reduce_params *params);

int csi_ref_argmin_stride_quant(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct reduce_params *params);

int csi_ref_asin_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_ref_asin_quant(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct siso_params *params);

int csi_ref_asinh_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_asinh_quant(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_ref_atan_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_ref_atan_quant(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct siso_params *params);

int csi_ref_atanh_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_atanh_quant(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_ref_avgpool2d_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct pool_params *params);

int csi_ref_avgpool2d_quant(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct pool_params *params);

int csi_ref_avgpool3d_f32(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct pool_params *params);

int csi_ref_avgpool3d_quant(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct pool_params *params);

int csi_ref_batch_normalization_f32(struct csi_tensor *input,
                                    struct csi_tensor *mean,
                                    struct csi_tensor *variance,
                                    struct csi_tensor *gamma,
                                    struct csi_tensor *beta,
                                    struct csi_tensor *output,
                                    struct bn_params *params);

int csi_ref_batch_normalization_quant(struct csi_tensor *input,
                                      struct csi_tensor *mean,
                                      struct csi_tensor *variance,
                                      struct csi_tensor *gamma,
                                      struct csi_tensor *beta,
                                      struct csi_tensor *output,
                                      struct bn_params *params);

int csi_ref_batch_to_space_f32(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct batch_to_space_params *params);

int csi_ref_batch_to_space_quant(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct batch_to_space_params *params);

int csi_ref_broadcast_to_f32(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct broadcast_to_params *params);

int csi_ref_broadcast_to_quant(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct broadcast_to_params *params);

int csi_ref_ceil_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_ref_ceil_quant(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct siso_params *params);

int csi_ref_clip_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct clip_params *params);

int csi_ref_clip_quant(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct clip_params *params);

int csi_ref_col2im_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct csi_tensor *kernel,
                       struct col2im_params *params);

int csi_ref_concat_f32(struct csi_tensor **input,
                       struct csi_tensor *output,
                       struct concat_params *params);

int csi_ref_concat_quant(struct csi_tensor **input,
                         struct csi_tensor *output,
                         struct concat_params *params);

int csi_ref_conv2d_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct csi_tensor *kernel,
                       struct csi_tensor *bias,
                       struct conv2d_params *params);

int csi_ref_conv2d_quant(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct csi_tensor *kernel,
                         struct csi_tensor *bias,
                         struct conv2d_params *params);

int csi_ref_conv2d_channel_quant(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct csi_tensor *kernel,
                                 struct csi_tensor *bias,
                                 struct conv2d_params *params);

int csi_ref_conv2d_relu_f32(struct csi_tensor *o_input,
                            struct csi_tensor *o_output,
                            struct csi_tensor *o_kernel,
                            struct csi_tensor *o_bias,
                            struct conv2d_params *params);

int csi_ref_conv2d_relu_quant(struct csi_tensor *o_input,
                              struct csi_tensor *o_output,
                              struct csi_tensor *o_kernel,
                              struct csi_tensor *o_bias,
                              struct conv2d_params *params);

int csi_ref_conv2d_channel_relu_quant(struct csi_tensor *o_input,
                                      struct csi_tensor *o_output,
                                      struct csi_tensor *o_kernel,
                                      struct csi_tensor *o_bias,
                                      struct conv2d_params *params);

int csi_ref_conv2d_relu6_quant(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct csi_tensor *kernel,
                               struct csi_tensor *bias,
                               struct conv2d_params *params);

int csi_ref_conv2d_channel_relu6_quant(struct csi_tensor *input,
                                       struct csi_tensor *output,
                                       struct csi_tensor *kernel,
                                       struct csi_tensor *bias,
                                       struct conv2d_params *params);

int csi_ref_depthwise_conv2d_f32(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct csi_tensor *kernel,
                                 struct csi_tensor *bias,
                                 struct conv2d_params *params);

int csi_ref_depthwise_conv2d_quant(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct csi_tensor *kernel,
                                   struct csi_tensor *bias,
                                   struct conv2d_params *params);

int csi_ref_depthwise_conv2d_channel_quant(struct csi_tensor *input,
                                           struct csi_tensor *output,
                                           struct csi_tensor *kernel,
                                           struct csi_tensor *bias,
                                           struct conv2d_params *params);

int csi_ref_depthwise_conv2d_relu_f32(struct csi_tensor *o_input,
                                      struct csi_tensor *o_output,
                                      struct csi_tensor *o_kernel,
                                      struct csi_tensor *o_bias,
                                      struct conv2d_params *params);

int csi_ref_depthwise_conv2d_relu_quant(struct csi_tensor *o_input,
                                        struct csi_tensor *o_output,
                                        struct csi_tensor *o_kernel,
                                        struct csi_tensor *o_bias,
                                        struct conv2d_params *params);

int csi_ref_depthwise_conv2d_channel_relu_quant(struct csi_tensor *o_input,
                                                struct csi_tensor *o_output,
                                                struct csi_tensor *o_kernel,
                                                struct csi_tensor *o_bias,
                                                struct conv2d_params *params);

int csi_ref_depthwise_conv2d_relu6_quant(struct csi_tensor *input,
                                         struct csi_tensor *output,
                                         struct csi_tensor *kernel,
                                         struct csi_tensor *bias,
                                         struct conv2d_params *params);

int csi_ref_depthwise_conv2d_channel_relu6_quant(struct csi_tensor *input,
                                                 struct csi_tensor *output,
                                                 struct csi_tensor *kernel,
                                                 struct csi_tensor *bias,
                                                 struct conv2d_params *params);

int csi_ref_group_conv2d_f32(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct csi_tensor *kernel,
                             struct csi_tensor *bias,
                             struct conv2d_params *params);

int csi_ref_group_conv2d_quant(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct csi_tensor *kernel,
                               struct csi_tensor *bias,
                               struct conv2d_params *params);

int csi_ref_group_conv2d_channel_quant(struct csi_tensor *input,
                                       struct csi_tensor *output,
                                       struct csi_tensor *kernel,
                                       struct csi_tensor *bias,
                                       struct conv2d_params *params);

int csi_ref_group_conv2d_relu_quant(struct csi_tensor *input,
                                    struct csi_tensor *output,
                                    struct csi_tensor *kernel,
                                    struct csi_tensor *bias,
                                    struct conv2d_params *params);

int csi_ref_group_conv2d_relu6_quant(struct csi_tensor *input,
                                     struct csi_tensor *output,
                                     struct csi_tensor *kernel,
                                     struct csi_tensor *bias,
                                     struct conv2d_params *params);

int csi_ref_group_conv2d_channel_relu_quant(struct csi_tensor *input,
                                            struct csi_tensor *output,
                                            struct csi_tensor *kernel,
                                            struct csi_tensor *bias,
                                            struct conv2d_params *params);

int csi_ref_conv3d_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct csi_tensor *kernel,
                       struct csi_tensor *bias,
                       struct conv3d_params *params);

int csi_ref_conv3d_quant(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct csi_tensor *kernel,
                         struct csi_tensor *bias,
                         struct conv3d_params *params);

int csi_ref_cos_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_ref_cos_quant(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_cosh_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_ref_cosh_quant(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct siso_params *params);

int csi_ref_cumprod_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct cumprod_params *params);

int csi_ref_cumprod_quant(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct cumprod_params *params);

int csi_ref_cumsum_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct cumsum_params *params);

int csi_ref_cumsum_quant(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct cumsum_params *params);

int csi_ref_deconv2d_f32(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct csi_tensor *kernel,
                         struct csi_tensor *bias,
                         struct conv2d_params *params);

int csi_ref_deconv2d_quant(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct csi_tensor *kernel,
                           struct csi_tensor *bias,
                           struct conv2d_params *params);

int csi_ref_depthwise_deconv2d_f32(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct csi_tensor *kernel,
                                   struct csi_tensor *bias,
                                   struct conv2d_params *params);

int csi_ref_depthwise_deconv2d_quant(struct csi_tensor *input,
                                     struct csi_tensor *output,
                                     struct csi_tensor *kernel,
                                     struct csi_tensor *bias,
                                     struct conv2d_params *params);

int csi_ref_deconv3d_f32(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct csi_tensor *kernel,
                         struct csi_tensor *bias,
                         struct conv3d_params *params);

int csi_ref_deconv3d_quant(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct csi_tensor *kernel,
                           struct csi_tensor *bias,
                           struct conv3d_params *params);

int csi_ref_depth_to_space_f32(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct depth_to_space_params *params);

int csi_ref_depth_to_space_quant(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct depth_to_space_params *params);

int csi_ref_div_f32(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_ref_div_quant(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params);

int csi_ref_elu_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct relu_params *params);

int csi_ref_elu_quant(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct relu_params *params);

int csi_ref_fsmn_f32(struct csi_tensor *frame,
                     struct csi_tensor *l_filter,
                     struct csi_tensor *r_filter,
                     struct csi_tensor *frame_sequence,
                     struct csi_tensor *frame_counter,
                     struct csi_tensor *output,
                     struct fsmn_params *params);

int csi_ref_fsmn_quant(struct csi_tensor *frame,
                     struct csi_tensor *l_filter,
                     struct csi_tensor *r_filter,
                     struct csi_tensor *frame_sequence,
                     struct csi_tensor *frame_counter,
                     struct csi_tensor *output,
                     struct fsmn_params *params);

int csi_ref_equal_f32(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params);

int csi_ref_equal_quant(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params);

int csi_ref_erf_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_ref_erf_quant(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_exp_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_ref_exp_quant(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_expand_dims_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct expand_dims_params *params);

int csi_ref_expand_dims_quant(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct expand_dims_params *params);

int csi_ref_expm1_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_expm1_quant(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_ref_flatten(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct flatten_params *params);

int csi_ref_flatten_requant(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct flatten_params *params);

int csi_ref_floor_divide_f32(struct csi_tensor *input0,
                             struct csi_tensor *input1,
                             struct csi_tensor *output,
                             struct diso_params *params);

int csi_ref_floor_divide_quant(struct csi_tensor *input0,
                               struct csi_tensor *input1,
                               struct csi_tensor *output,
                               struct diso_params *params);

int csi_ref_floor_mod_f32(struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct diso_params *params);

int csi_ref_floor_mod_quant(struct csi_tensor *input0,
                            struct csi_tensor *input1,
                            struct csi_tensor *output,
                            struct diso_params *params);

int csi_ref_floor_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_floor_quant(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_ref_fullyconnected_f32(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct csi_tensor *weights,
                               struct csi_tensor *bias,
                               struct fc_params *params);

int csi_ref_fullyconnected_quant(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct csi_tensor *weights,
                                 struct csi_tensor *bias,
                                 struct fc_params *params);

int csi_ref_gather_nd_f32(struct csi_tensor *input,
                          struct csi_tensor *indices,
                          struct csi_tensor *output,
                          struct gather_nd_params *params);

int csi_ref_gather_nd_quant(struct csi_tensor *input,
                            struct csi_tensor *indices,
                            struct csi_tensor *output,
                            struct gather_nd_params *params);

int csi_ref_gather_f32(struct csi_tensor *input,
                       struct csi_tensor *indices,
                       struct csi_tensor *output,
                       struct gather_params *params);

int csi_ref_gather_quant(struct csi_tensor *input,
                         struct csi_tensor *indices,
                         struct csi_tensor *output,
                         struct gather_params *params);

int csi_ref_global_avgpool2d_f32(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct pool_params *params);

int csi_ref_global_avgpool2d_quant(struct csi_tensor *input,
                                     struct csi_tensor *output,
                                     struct pool_params *params);

int csi_ref_global_maxpool2d_f32(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct pool_params *params);

int csi_ref_global_maxpool2d_quant(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct pool_params *params);

int csi_ref_greater_equal_f32(struct csi_tensor *input0,
                              struct csi_tensor *input1,
                              struct csi_tensor *output,
                              struct diso_params *params);

int csi_ref_greater_equal_quant(struct csi_tensor *input0,
                                struct csi_tensor *input1,
                                struct csi_tensor *output,
                                struct diso_params *params);

int csi_ref_greater_f32(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params);

int csi_ref_greater_quant(struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct diso_params *params);

int csi_ref_hard_sigmoid_f32(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct sigmoid_params *params);

int csi_ref_hard_sigmoid_quant(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct sigmoid_params *params);

int csi_ref_im2col_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct im2col_params *params);

int csi_ref_im2col_quant(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct im2col_params *params);

int csi_ref_isnan_bool_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct siso_params *params);

int csi_ref_l2_normalization_f32(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct l2n_params *params);

int csi_ref_l2_normalization_quant(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct l2n_params *params);

int csi_ref_l2pool_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct pool_params *params);

int csi_ref_leaky_relu_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct relu_params *params);

int csi_ref_leaky_relu_quant(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct relu_params *params);

int csi_ref_less_equal_f32(struct csi_tensor *input0,
                           struct csi_tensor *input1,
                           struct csi_tensor *output,
                           struct diso_params *params);

int csi_ref_less_equal_quant(struct csi_tensor *input0,
                             struct csi_tensor *input1,
                             struct csi_tensor *output,
                             struct diso_params *params);

int csi_ref_less_f32(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params);

int csi_ref_less_quant(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_ref_log_softmax_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct softmax_params *params);

int csi_ref_log_softmax_quant(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct softmax_params *params);

int csi_ref_log_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_ref_log_quant(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_log1p_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_log1p_quant(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_ref_logical_and_f32(struct csi_tensor *input0,
                            struct csi_tensor *input1,
                            struct csi_tensor *output,
                            struct diso_params *params);

int csi_ref_logical_and_quant(struct csi_tensor *input0,
                              struct csi_tensor *input1,
                              struct csi_tensor *output,
                              struct diso_params *params);

int csi_ref_logical_not_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct siso_params *params);

int csi_ref_logical_not_quant(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct siso_params *params);

int csi_ref_logical_or_f32(struct csi_tensor *input0,
                           struct csi_tensor *input1,
                           struct csi_tensor *output,
                           struct diso_params *params);

int csi_ref_logical_or_quant(struct csi_tensor *input0,
                             struct csi_tensor *input1,
                             struct csi_tensor *output,
                             struct diso_params *params);

int csi_ref_logical_xor_f32(struct csi_tensor *input0,
                            struct csi_tensor *input1,
                            struct csi_tensor *output,
                            struct diso_params *params);

int csi_ref_logical_xor_quant(struct csi_tensor *input0,
                              struct csi_tensor *input1,
                              struct csi_tensor *output,
                              struct diso_params *params);

int csi_ref_lrn_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct lrn_params *params);

int csi_ref_lrn_quant(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct lrn_params *params);

int csi_ref_matmul_f32(struct csi_tensor *mat0,
                       struct csi_tensor *mat1,
                       struct csi_tensor *output,
                       struct matmul_params *params);

int csi_ref_matmul_quant(struct csi_tensor *mat0,
                         struct csi_tensor *mat1,
                         struct csi_tensor *output,
                         struct matmul_params *params);

int csi_ref_max_stride_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct reduce_params *params);

int csi_ref_max_stride_quant(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct reduce_params *params);

int csi_ref_maximum_f32(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params);

int csi_ref_maximum_quant(struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct diso_params *params);

int csi_ref_maxpool2d_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct pool_params *params);

int csi_ref_maxpool2d_quant(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct pool_params *params);

int csi_ref_maxpool2d_locat_f32(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct pool_params *params);

int csi_ref_maxpool2d_locat_quant(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct pool_params *params);

int csi_ref_maxpool3d_f32(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct pool_params *params);

int csi_ref_maxpool3d_quant(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct pool_params *params);

int csi_ref_mean_stride_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct reduce_params *params);

int csi_ref_mean_stride_quant(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct reduce_params *params);

int csi_ref_mean_quant(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct reduce_params *params);

int csi_ref_min_stride_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct reduce_params *params);

int csi_ref_min_stride_quant(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct reduce_params *params);

int csi_ref_minimum_f32(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params);

int csi_ref_minimum_quant(struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct diso_params *params);

int csi_ref_mod_f32(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_ref_mod_quant(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params);

int csi_ref_mul_f32(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_ref_mul_quant(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params);

int csi_ref_ndarray_size_f32(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct ndarray_size_params *params);

int csi_ref_ndarray_size_u8(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct ndarray_size_params *params);

int csi_ref_ndarray_size_i8(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct ndarray_size_params *params);

int csi_ref_ndarray_size_i32(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct ndarray_size_params *params);

int csi_ref_negative_f32(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct siso_params *params);

int csi_ref_negative_quant(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct siso_params *params);

int csi_ref_non_max_suppression_std(struct csi_tensor *input0,
                                    struct csi_tensor *input1,
                                    struct csi_tensor *output,
                                    struct non_max_suppression_params *params);

int csi_ref_not_equal_f32(struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct diso_params *params);

int csi_ref_not_equal_quant(struct csi_tensor *input0,
                            struct csi_tensor *input1,
                            struct csi_tensor *output,
                            struct diso_params *params);

int csi_ref_not_u32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_ref_not_u8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_ref_not_i8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_ref_or_u32(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_ref_or_u8(struct csi_tensor *input0,
                  struct csi_tensor *input1,
                  struct csi_tensor *output,
                  struct diso_params *params);

int csi_ref_or_i8(struct csi_tensor *input0,
                  struct csi_tensor *input1,
                  struct csi_tensor *output,
                  struct diso_params *params);

int csi_ref_pad_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct pad_params *params);

int csi_ref_pad_quant(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct pad_params *params);

int csi_ref_power_f32(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params);

int csi_ref_power_quant(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params);

int csi_ref_prelu_f32(struct csi_tensor *input,
                      struct csi_tensor *alpha,
                      struct csi_tensor *output,
                      struct prelu_params *params);

int csi_ref_prelu_quant(struct csi_tensor *input,
                        struct csi_tensor *alpha,
                        struct csi_tensor *output,
                        struct prelu_params *params);

int csi_ref_prod_stride_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct reduce_params *params);

int csi_ref_prod_stride_quant(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct reduce_params *params);

int csi_ref_proposal_f32(struct csi_tensor *cls_prob,
                         struct csi_tensor *bbox_pred,
                         struct csi_tensor *im_info,
                         struct csi_tensor *output,
                         struct proposal_params *params);

int csi_ref_proposal_quant(struct csi_tensor *cls_prob,
                           struct csi_tensor *bbox_pred,
                           struct csi_tensor *im_info,
                           struct csi_tensor *output,
                           struct proposal_params *params);

int csi_ref_psroipooling_f32(struct csi_tensor *data,
                             struct csi_tensor *rois,
                             struct csi_tensor *output,
                             struct psroipooling_params *params);

int csi_ref_psroipooling_quant(struct csi_tensor *data,
                               struct csi_tensor *rois,
                               struct csi_tensor *output,
                               struct psroipooling_params *params);

int csi_ref_reduce_logsumexp_f32(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct reduce_params *params);

int csi_ref_reduce_logsumexp_quant(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct reduce_params *params);

int csi_ref_reduce_max_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct reduce_params *params);

int csi_ref_reduce_max_quant(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct reduce_params *params);

int csi_ref_reduce_mean_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct reduce_params *params);

int csi_ref_reduce_mean_quant(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct reduce_params *params);

int csi_ref_reduce_min_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct reduce_params *params);

int csi_ref_reduce_min_quant(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct reduce_params *params);

int csi_ref_reduce_prod_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct reduce_params *params);

int csi_ref_reduce_prod_quant(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct reduce_params *params);

int csi_ref_reduce_sum_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct reduce_params *params);

int csi_ref_reduce_sum_quant(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct reduce_params *params);

int csi_ref_relu_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct relu_params *params);

int csi_ref_relu_quant(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct relu_params *params);

int csi_ref_relu1_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct relu_params *params);

int csi_ref_relu1_quant(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct relu_params *params);

int csi_ref_relu6_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct relu_params *params);

int csi_ref_relu6_quant(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct relu_params *params);

int csi_ref_relun_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct relu_params *params);

int csi_ref_relun_quant(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct relu_params *params);

int csi_ref_reshape(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reshape_params *params);

int csi_ref_reshape_requant(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct reshape_params *params);

int csi_ref_resize_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct resize_params *params);

int csi_ref_resize_quant(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct resize_params *params);

int csi_ref_reverse_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reverse_params *params);

int csi_ref_reverse_quant(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct reverse_params *params);

int csi_ref_roi_align_f32(struct csi_tensor *data,
                          struct csi_tensor *rois,
                          struct csi_tensor *output,
                          struct roi_align_params *params);

int csi_ref_roipool_f32(struct csi_tensor *data,
                        struct csi_tensor *rois,
                        struct csi_tensor *output,
                        struct roi_pool_params *params);

int csi_ref_roipool_quant(struct csi_tensor *data,
                          struct csi_tensor *rois,
                          struct csi_tensor *output,
                          struct roi_pool_params *params);

int csi_ref_round_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_round_quant(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_ref_rsqrt_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_rsqrt_quant(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_ref_scatter_nd_f32(struct csi_tensor *input,
                           struct csi_tensor *indices,
                           struct csi_tensor *updates,
                           struct csi_tensor *output,
                           struct scatter_nd_params *params);

int csi_ref_scatter_nd_quant(struct csi_tensor *input,
                             struct csi_tensor *indices,
                             struct csi_tensor *updates,
                             struct csi_tensor *output,
                             struct scatter_nd_params *params);

int csi_ref_unsorted_segment_max_f32(struct csi_tensor *input,
                                     struct csi_tensor *segment_ids,
                                     struct csi_tensor *output,
                                     struct segment_params *params);

int csi_ref_segment_max_f32(struct csi_tensor *input,
                            struct csi_tensor *segment_ids,
                            struct csi_tensor *output,
                            struct segment_params *params);

int csi_ref_unsorted_segment_max_quant(struct csi_tensor *input,
                                       struct csi_tensor *segment_ids,
                                       struct csi_tensor *output,
                                       struct segment_params *params);

int csi_ref_segment_max_quant(struct csi_tensor *input,
                              struct csi_tensor *segment_ids,
                              struct csi_tensor *output,
                              struct segment_params *params);

int csi_ref_unsorted_segment_mean_f32(struct csi_tensor *input,
                                      struct csi_tensor *segment_ids,
                                      struct csi_tensor *output,
                                      struct segment_params *params);

int csi_ref_segment_mean_f32(struct csi_tensor *input,
                             struct csi_tensor *segment_ids,
                             struct csi_tensor *output,
                             struct segment_params *params);

int csi_ref_unsorted_segment_mean_quant(struct csi_tensor *input,
                                        struct csi_tensor *segment_ids,
                                        struct csi_tensor *output,
                                        struct segment_params *params);

int csi_ref_segment_mean_quant(struct csi_tensor *input,
                               struct csi_tensor *segment_ids,
                               struct csi_tensor *output,
                               struct segment_params *params);

int csi_ref_unsorted_segment_min_f32(struct csi_tensor *input,
                                     struct csi_tensor *segment_ids,
                                     struct csi_tensor *output,
                                     struct segment_params *params);

int csi_ref_segment_min_f32(struct csi_tensor *input,
                            struct csi_tensor *segment_ids,
                            struct csi_tensor *output,
                            struct segment_params *params);

int csi_ref_unsorted_segment_min_quant(struct csi_tensor *input,
                                       struct csi_tensor *segment_ids,
                                       struct csi_tensor *output,
                                       struct segment_params *params);

int csi_ref_segment_min_quant(struct csi_tensor *input,
                              struct csi_tensor *segment_ids,
                              struct csi_tensor *output,
                              struct segment_params *params);

int csi_ref_unsorted_segment_prod_f32(struct csi_tensor *input,
                                      struct csi_tensor *segment_ids,
                                      struct csi_tensor *output,
                                      struct segment_params *params);

int csi_ref_segment_prod_f32(struct csi_tensor *input,
                             struct csi_tensor *segment_ids,
                             struct csi_tensor *output,
                             struct segment_params *params);

int csi_ref_unsorted_segment_prod_quant(struct csi_tensor *input,
                                        struct csi_tensor *segment_ids,
                                        struct csi_tensor *output,
                                        struct segment_params *params);

int csi_ref_segment_prod_quant(struct csi_tensor *input,
                               struct csi_tensor *segment_ids,
                               struct csi_tensor *output,
                               struct segment_params *params);

int csi_ref_unsorted_segment_sum_f32(struct csi_tensor *input,
                                     struct csi_tensor *segment_ids,
                                     struct csi_tensor *output,
                                     struct segment_params *params);

int csi_ref_segment_sum_f32(struct csi_tensor *input,
                            struct csi_tensor *segment_ids,
                            struct csi_tensor *output,
                            struct segment_params *params);

int csi_ref_unsorted_segment_sum_quant(struct csi_tensor *input,
                                       struct csi_tensor *segment_ids,
                                       struct csi_tensor *output,
                                       struct segment_params *params);

int csi_ref_segment_sum_quant(struct csi_tensor *input,
                              struct csi_tensor *segment_ids,
                              struct csi_tensor *output,
                              struct segment_params *params);

int csi_ref_select_f32(struct csi_tensor *condition,
                       struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct select_params *params);

int csi_ref_select_u8(struct csi_tensor *condition,
                      struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct select_params *params);

int csi_ref_select_i8(struct csi_tensor *condition,
                      struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct select_params *params);

int csi_ref_shape_i32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct shape_params *params);

int csi_ref_shape_u8(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct shape_params *params);

int csi_ref_shape_i8(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct shape_params *params);

int csi_ref_shuffle_channel_f32(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct shuffle_channel_params *params);

int csi_ref_shuffle_channel_quant(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct shuffle_channel_params *params);

int csi_ref_sigmoid_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct sigmoid_params *params);

int csi_ref_sigmoid_quant(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct sigmoid_params *params);

int csi_ref_sign_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_ref_sign_quant(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct siso_params *params);

int csi_ref_sin_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_ref_sin_quant(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_sinh_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_ref_sinh_quant(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct siso_params *params);

int csi_ref_slice_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct slice_params *params);

int csi_ref_slice_quant(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct slice_params *params);

int csi_ref_softmax_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct softmax_params *params);

int csi_ref_softmax_quant(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct softmax_params *params);

int csi_ref_softplus_f32(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct siso_params *params);

int csi_ref_softplus_quant(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct siso_params *params);

int csi_ref_softrelu_f32(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct relu_params *params);

int csi_ref_softrelu_quant(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct relu_params *params);

int csi_ref_softsign_f32(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct siso_params *params);

int csi_ref_softsign_quant(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct siso_params *params);

int csi_ref_space_to_batch_f32(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct space_to_batch_params *params);

int csi_ref_space_to_batch_quant(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct space_to_batch_params *params);

int csi_ref_space_to_depth_f32(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct space_to_depth_params *params);

int csi_ref_space_to_depth_quant(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct space_to_depth_params *params);

int csi_ref_split_f32(struct csi_tensor *input,
                      struct csi_tensor **output,
                      struct split_params *params);

int csi_ref_split_quant(struct csi_tensor *input,
                        struct csi_tensor **output,
                        struct split_params *params);

int csi_ref_sqrt_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_ref_sqrt_quant(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct siso_params *params);

int csi_ref_square_f32(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct siso_params *params);

int csi_ref_squeeze(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct squeeze_params *params);

int csi_ref_stack_f32(struct csi_tensor **input,
                      struct csi_tensor *output,
                      struct stack_params *params);

int csi_ref_stack_quant(struct csi_tensor **input,
                        struct csi_tensor *output,
                        struct stack_params *params);

int csi_ref_strided_slice_f32(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct strided_slice_params *params);

int csi_ref_strided_slice_quant(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct strided_slice_params *params);

int csi_ref_sub_f32(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_ref_sub_quant(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params);

int csi_ref_sum_stride_f32(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct reduce_params *params);

int csi_ref_sum_stride_quant(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct reduce_params *params);

int csi_ref_tan_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_ref_tan_quant(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_tanh_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_ref_tanh_f64(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_ref_tanh_quant(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct siso_params *params);

int csi_ref_threshold_relu_f32(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct relu_params *params);

int csi_ref_threshold_relu_quant(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct relu_params *params);

int csi_ref_tile_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct tile_params *params);

int csi_ref_tile_quant(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct tile_params *params);

int csi_ref_topk_f32(struct csi_tensor *input,
                     struct csi_tensor *output1,
                     struct csi_tensor *output2,
                     struct topk_params *params);

int csi_ref_topk_quant(struct csi_tensor *input,
                       struct csi_tensor *output1,
                       struct csi_tensor *output2,
                       struct topk_params *params);

int csi_ref_transpose(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct transpose_params *params);

int csi_ref_transpose_requant(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct transpose_params *params);

int csi_ref_trunc_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ref_trunc_quant(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_ref_unpooling_f32(struct csi_tensor *input,
                          struct csi_tensor *mask,
                          struct csi_tensor *output,
                          struct unpooling_params *params);

int csi_ref_unpooling_quant(struct csi_tensor *input,
                            struct csi_tensor *mask,
                            struct csi_tensor *output,
                            struct unpooling_params *params);

int csi_ref_unstack_f32(struct csi_tensor *input,
                        struct csi_tensor **output,
                        struct unstack_params *params);

int csi_ref_unstack_qunat(struct csi_tensor *input,
                          struct csi_tensor **output,
                          struct unstack_params *params);

int csi_ref_xor_u32(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_ref_xor_u8(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_ref_xor_i8(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_ref_yuv_rgb_scale_f32(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct siso_params *params);

int csi_ref_yuv_rgb_scale_quant(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct siso_params *params);

int32_t csi_ref_max_internal_s32(int32_t a, int32_t b);
int32_t csi_ref_min_internal_s32(int32_t a, int32_t b);
int32_t csi_ref_get_index(int32_t *dim, int32_t index0, int32_t index1, int32_t index2, int32_t index3);
int32_t csi_ref_get_index_5(int32_t *dim, int32_t index0, int32_t index1, int32_t index2, int32_t index3, int32_t index4);
int32_t csi_ref_get_index_iter(int32_t *dim, int dim_count, int32_t *index);
float csi_ref_get_scale(int32_t multiplier, int32_t shift);
float csi_ref_dequantize_u8_to_f32(uint8_t input, struct csi_quant_info *qinfo);
float csi_ref_dequantize_i8_to_f32(int8_t input, struct csi_quant_info *qinfo);
uint8_t csi_ref_quantize_f32_to_u8(float input, struct csi_quant_info *qinfo);
int8_t csi_ref_quantize_f32_to_i8(float input, struct csi_quant_info *qinfo);
uint8_t csi_ref_quantize_channel_u8(int32_t data, struct csi_tensor *input, struct csi_tensor *output, float wscale);
int8_t csi_ref_quantize_channel_i8(int32_t data, struct csi_tensor *input, struct csi_tensor *output, float wscale);
float csi_ref_uint8_to_float(uint8_t i, struct csi_tensor *t);
float csi_ref_int8_to_float(int8_t i, struct csi_tensor *t);
int16_t csi_ref_float32_to_float16(float value);
float csi_ref_float16_to_float32(int16_t value);
struct csi_tensor *csi_ref_nchw_to_nhwc_8(struct csi_tensor *t);
void csi_ref_nhwc_to_nchw_8(struct csi_tensor *nt, struct csi_tensor *t);
struct csi_tensor *csi_ref_deconv_kernel_nchw_to_nhwc_f32(struct csi_tensor *t, int32_t permute[4]);
struct csi_tensor *csi_ref_nchw_to_nhwc_f32(struct csi_tensor *t);
void csi_ref_nhwc_to_nchw_f32(struct csi_tensor *nt, struct csi_tensor *t);
int32_t csi_ref_get_reduction_index(int32_t k, const int32_t *strides, const int32_t *extents, int32_t n);
struct csi_tensor *csi_ref_alloc_float_tensor(struct csi_tensor *src);
void csi_ref_free_float_tensor(struct csi_tensor *src);
struct csi_tensor *csi_ref_convert_float_tensor(struct csi_tensor *src);
void csi_ref_conv_free_float_tensor(struct csi_tensor *input, struct csi_tensor *output,
                                    struct csi_tensor *kernel, struct csi_tensor *bias);
struct csi_tensor *csi_ref_tensor_transform_f32(struct csi_tensor *input);
int csi_ref_tensor_transform_free_f32(struct csi_tensor *input);
uint8_t *csi_ref_f32_to_input_dtype(uint32_t index, float *data, struct csi_session *sess);

struct csi_ref_diso_callback
{
    void (*bc)();
    struct csi_tensor *input0;
    struct csi_tensor *input1;
    struct csi_tensor *output;
    int32_t *input_dim;
};

void *csi_init_map_ref(int op, int dtype);

int csi_ref_diso_broadcast_base(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
                                struct diso_params *params, struct csi_ref_diso_callback *cb);
int csi_ref_broadcast_to_shape(struct csi_tensor *input, struct csi_tensor *output, int32_t *shape, int32_t shape_count);
int csi_ref_broadcast_to_shape_f32(struct csi_tensor *input, struct csi_tensor *output, int32_t *shape, int32_t shape_count);
int csi_ref_broadcast_to_shape_quant(struct csi_tensor *input, struct csi_tensor *output, int32_t *shape, int32_t shape_count);

int csi_ref_siso_callback_base(struct csi_tensor *input, struct csi_tensor *output, void *params, void *cb);
int csi_ref_diso_callback_base(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output, void *params, void *cb);
int csi_ref_conv_callback_base(struct csi_tensor *input, struct csi_tensor *output, struct csi_tensor *kernel, struct csi_tensor *bias, void *params, void *cb);

void csi_ref_nn_init(struct csi_tensor *input,
                     struct csi_tensor *output);

void csi_ref_nn_deinit(struct csi_tensor *input,
                       struct csi_tensor *output);

int csi_ref_flatten_init(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct reshape_params *params);

int csi_ref_reshape_init(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct reshape_params *params);

int csi_ref_transpose_init(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct transpose_params *params);
#endif
