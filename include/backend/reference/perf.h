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

#ifndef INCLUDE_SHL_REF_PERF_H_
#define INCLUDE_SHL_REF_PERF_H_

#include "csi_nn.h"
#include "shl_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

int shl_ref_abs_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_acos_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_acosh_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_add_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params,
                     struct csinn_perf_info *perf_info);

int shl_ref_and_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params,
                     struct csinn_perf_info *perf_info);

int shl_ref_arange_perf(struct csinn_tensor *output, struct csinn_arange_params *params,
                        struct csinn_perf_info *perf_info);

int shl_ref_argmax_stride_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_reduce_params *params,
                               struct csinn_perf_info *perf_info);

int shl_ref_argmin_stride_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_reduce_params *params,
                               struct csinn_perf_info *perf_info);

int shl_ref_asin_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_asinh_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_atan_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_atanh_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_avgpool2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params, struct csinn_perf_info *perf_info);

int shl_ref_avgpool3d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params, struct csinn_perf_info *perf_info);

int shl_ref_batch_normalization_perf(struct csinn_tensor *input, struct csinn_tensor *mean,
                                     struct csinn_tensor *variance, struct csinn_tensor *gamma,
                                     struct csinn_tensor *beta, struct csinn_tensor *output,
                                     struct csinn_bn_params *params,
                                     struct csinn_perf_info *perf_info);

int shl_ref_batch_to_space_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_batch_to_space_params *params,
                                struct csinn_perf_info *perf_info);

int shl_ref_broadcast_to_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_broadcast_to_params *params,
                              struct csinn_perf_info *perf_info);

int shl_ref_ceil_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_clip_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_clip_params *params, struct csinn_perf_info *perf_info);

int shl_ref_col2im_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_col2im_params *params,
                        struct csinn_perf_info *perf_info);

int shl_ref_concat_perf(struct csinn_tensor **input, struct csinn_tensor *output,
                        struct csinn_concat_params *params, struct csinn_perf_info *perf_info);

int shl_ref_conv1d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                        struct csinn_conv1d_params *params, struct csinn_perf_info *perf_info);

int shl_ref_conv2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                        struct csinn_conv2d_params *params, struct csinn_perf_info *perf_info);

int shl_ref_conv2d_channel_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params,
                                struct csinn_perf_info *perf_info);

int shl_ref_conv2d_relu_perf(struct csinn_tensor *o_input, struct csinn_tensor *o_output,
                             struct csinn_tensor *o_kernel, struct csinn_tensor *o_bias,
                             struct csinn_conv2d_params *params, struct csinn_perf_info *perf_info);

int shl_ref_cache_matmul_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *weight, struct csinn_tensor *bias,
                              struct csinn_cache_matmul_params *params,
                              struct csinn_perf_info *perf_info);

int shl_ref_cache_conv1d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *weight, struct csinn_tensor *bias,
                              struct csinn_cache_conv1d_params *params,
                              struct csinn_perf_info *perf_info);

int shl_ref_conv2d_channel_relu_perf(struct csinn_tensor *o_input, struct csinn_tensor *o_output,
                                     struct csinn_tensor *o_kernel, struct csinn_tensor *o_bias,
                                     struct csinn_conv2d_params *params,
                                     struct csinn_perf_info *perf_info);

int shl_ref_conv2d_relu6_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params,
                              struct csinn_perf_info *perf_info);

int shl_ref_conv2d_channel_relu6_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params,
                                      struct csinn_perf_info *perf_info);

int shl_ref_depthwise_conv2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params,
                                  struct csinn_perf_info *perf_info);

int shl_ref_depthwise_conv2d_channel_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                          struct csinn_conv2d_params *params,
                                          struct csinn_perf_info *perf_info);

int shl_ref_depthwise_conv2d_relu_perf(struct csinn_tensor *o_input, struct csinn_tensor *o_output,
                                       struct csinn_tensor *o_kernel, struct csinn_tensor *o_bias,
                                       struct csinn_conv2d_params *params,
                                       struct csinn_perf_info *perf_info);

int shl_ref_depthwise_conv2d_channel_relu_perf(struct csinn_tensor *o_input,
                                               struct csinn_tensor *o_output,
                                               struct csinn_tensor *o_kernel,
                                               struct csinn_tensor *o_bias,
                                               struct csinn_conv2d_params *params,
                                               struct csinn_perf_info *perf_info);

int shl_ref_depthwise_conv2d_relu6_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params,
                                        struct csinn_perf_info *perf_info);

int shl_ref_depthwise_conv2d_channel_relu6_perf(struct csinn_tensor *input,
                                                struct csinn_tensor *output,
                                                struct csinn_tensor *kernel,
                                                struct csinn_tensor *bias,
                                                struct csinn_conv2d_params *params,
                                                struct csinn_perf_info *perf_info);

int shl_ref_group_conv2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params,
                              struct csinn_perf_info *perf_info);

int shl_ref_group_conv2d_channel_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params,
                                      struct csinn_perf_info *perf_info);

int shl_ref_group_conv2d_relu_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params,
                                   struct csinn_perf_info *perf_info);

int shl_ref_group_conv2d_relu6_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                    struct csinn_conv2d_params *params,
                                    struct csinn_perf_info *perf_info);

int shl_ref_group_conv2d_channel_relu_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                           struct csinn_conv2d_params *params,
                                           struct csinn_perf_info *perf_info);

int shl_ref_conv3d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                        struct csinn_conv3d_params *params, struct csinn_perf_info *perf_info);

int shl_ref_cos_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_cosh_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_cumprod_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_cumprod_params *params, struct csinn_perf_info *perf_info);

int shl_ref_cumsum_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_cumsum_params *params, struct csinn_perf_info *perf_info);

int shl_ref_data_convert_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_deconv2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv2d_params *params, struct csinn_perf_info *perf_info);

int shl_ref_depthwise_deconv2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                    struct csinn_conv2d_params *params,
                                    struct csinn_perf_info *perf_info);

int shl_ref_group_deconv2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params,
                                struct csinn_perf_info *perf_info);

int shl_ref_deconv3d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv3d_params *params, struct csinn_perf_info *perf_info);

int shl_ref_depth_to_space_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_depth_to_space_params *params,
                                struct csinn_perf_info *perf_info);

int shl_ref_div_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params,
                     struct csinn_perf_info *perf_info);

int shl_ref_elu_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_relu_params *params, struct csinn_perf_info *perf_info);

int shl_ref_fsmn_perf(struct csinn_tensor *frame, struct csinn_tensor *l_filter,
                      struct csinn_tensor *r_filter, struct csinn_tensor *frame_sequence,
                      struct csinn_tensor *frame_counter, struct csinn_tensor *output,
                      struct csinn_fsmn_params *params, struct csinn_perf_info *perf_info);

int shl_ref_equal_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params,
                       struct csinn_perf_info *perf_info);

int shl_ref_erf_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_exp_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_expand_dims_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_expand_dims_params *params,
                             struct csinn_perf_info *perf_info);

int shl_ref_expm1_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_flatten_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_flatten_params *params, struct csinn_perf_info *perf_info);

int shl_ref_floor_divide_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                              struct csinn_tensor *output, struct csinn_diso_params *params,
                              struct csinn_perf_info *perf_info);

int shl_ref_floor_mod_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_diso_params *params,
                           struct csinn_perf_info *perf_info);

int shl_ref_floor_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_fullyconnected_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *weights, struct csinn_tensor *bias,
                                struct csinn_fc_params *params, struct csinn_perf_info *perf_info);

int shl_ref_gather_nd_perf(struct csinn_tensor *input, struct csinn_tensor *indices,
                           struct csinn_tensor *output, struct csinn_gather_nd_params *params,
                           struct csinn_perf_info *perf_info);

int shl_ref_gather_perf(struct csinn_tensor *input, struct csinn_tensor *indices,
                        struct csinn_tensor *output, struct csinn_gather_params *params,
                        struct csinn_perf_info *perf_info);

int shl_ref_global_avgpool2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params,
                                  struct csinn_perf_info *perf_info);

int shl_ref_global_maxpool2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params,
                                  struct csinn_perf_info *perf_info);

int shl_ref_greater_equal_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                               struct csinn_tensor *output, struct csinn_diso_params *params,
                               struct csinn_perf_info *perf_info);

int shl_ref_greater_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params,
                         struct csinn_perf_info *perf_info);

int shl_ref_hard_sigmoid_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_sigmoid_params *params,
                              struct csinn_perf_info *perf_info);

int shl_ref_im2col_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_im2col_params *params, struct csinn_perf_info *perf_info);

int shl_ref_isnan_bool_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_l2_normalization_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_l2n_params *params,
                                  struct csinn_perf_info *perf_info);

int shl_ref_l2pool_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_pool_params *params, struct csinn_perf_info *perf_info);

int shl_ref_layer_norm_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *gamma, struct csinn_tensor *beta,
                            struct csinn_layer_norm_params *params,
                            struct csinn_perf_info *perf_info);

int shl_ref_leaky_relu_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_relu_params *params, struct csinn_perf_info *perf_info);

int shl_ref_less_equal_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                            struct csinn_tensor *output, struct csinn_diso_params *params,
                            struct csinn_perf_info *perf_info);

int shl_ref_less_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params,
                      struct csinn_perf_info *perf_info);

int shl_ref_log_softmax_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_softmax_params *params,
                             struct csinn_perf_info *perf_info);

int shl_ref_log_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_log1p_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_logical_and_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                             struct csinn_tensor *output, struct csinn_diso_params *params,
                             struct csinn_perf_info *perf_info);

int shl_ref_logical_not_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_logical_or_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                            struct csinn_tensor *output, struct csinn_diso_params *params,
                            struct csinn_perf_info *perf_info);

int shl_ref_logical_xor_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                             struct csinn_tensor *output, struct csinn_diso_params *params,
                             struct csinn_perf_info *perf_info);

int shl_ref_lrn_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_lrn_params *params, struct csinn_perf_info *perf_info);

int shl_ref_matmul_perf(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                        struct csinn_tensor *output, struct csinn_matmul_params *params,
                        struct csinn_perf_info *perf_info);

int shl_ref_max_stride_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params, struct csinn_perf_info *perf_info);

int shl_ref_maximum_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params,
                         struct csinn_perf_info *perf_info);

int shl_ref_maxpool2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params, struct csinn_perf_info *perf_info);

int shl_ref_maxpool2d_locat_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params,
                                 struct csinn_perf_info *perf_info);

int shl_ref_maxpool3d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params, struct csinn_perf_info *perf_info);

int shl_ref_mean_stride_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_reduce_params *params, struct csinn_perf_info *perf_info);

int shl_ref_mean_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_reduce_params *params, struct csinn_perf_info *perf_info);

int shl_ref_min_stride_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params, struct csinn_perf_info *perf_info);

int shl_ref_minimum_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params,
                         struct csinn_perf_info *perf_info);

int shl_ref_mod_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params,
                     struct csinn_perf_info *perf_info);

int shl_ref_mul_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params,
                     struct csinn_perf_info *perf_info);

int shl_ref_ndarray_size_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_ndarray_size_params *params,
                              struct csinn_perf_info *perf_info);

int shl_ref_negative_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_non_max_suppression_std_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                         struct csinn_tensor *output,
                                         struct csinn_non_max_suppression_params *params,
                                         struct csinn_perf_info *perf_info);

int shl_ref_not_equal_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_diso_params *params,
                           struct csinn_perf_info *perf_info);

int shl_ref_not_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_or_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params,
                    struct csinn_perf_info *perf_info);

int shl_ref_pad_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_pad_params *params, struct csinn_perf_info *perf_info);

int shl_ref_power_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params,
                       struct csinn_perf_info *perf_info);

int shl_ref_prelu_perf(struct csinn_tensor *input, struct csinn_tensor *alpha,
                       struct csinn_tensor *output, struct csinn_prelu_params *params,
                       struct csinn_perf_info *perf_info);

int shl_ref_prod_stride_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_reduce_params *params, struct csinn_perf_info *perf_info);

int shl_ref_proposal_perf(struct csinn_tensor *cls_prob, struct csinn_tensor *bbox_pred,
                          struct csinn_tensor *im_info, struct csinn_tensor *output,
                          struct csinn_proposal_params *params, struct csinn_perf_info *perf_info);

int shl_ref_psroipooling_perf(struct csinn_tensor *data, struct csinn_tensor *rois,
                              struct csinn_tensor *output, struct csinn_psroipooling_params *params,
                              struct csinn_perf_info *perf_info);

int shl_ref_reduce_logsumexp_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_reduce_params *params,
                                  struct csinn_perf_info *perf_info);

int shl_ref_reduce_max_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params, struct csinn_perf_info *perf_info);

int shl_ref_reduce_mean_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_reduce_params *params, struct csinn_perf_info *perf_info);

int shl_ref_reduce_min_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params, struct csinn_perf_info *perf_info);

int shl_ref_reduce_prod_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_reduce_params *params, struct csinn_perf_info *perf_info);

int shl_ref_reduce_sum_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params, struct csinn_perf_info *perf_info);

int shl_ref_relu_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params, struct csinn_perf_info *perf_info);

int shl_ref_relu1_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params, struct csinn_perf_info *perf_info);

int shl_ref_relu6_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params, struct csinn_perf_info *perf_info);

int shl_ref_relun_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params, struct csinn_perf_info *perf_info);

int shl_ref_reshape_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reshape_params *params, struct csinn_perf_info *perf_info);

int shl_ref_resize_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_resize_params *params, struct csinn_perf_info *perf_info);

int shl_ref_reverse_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reverse_params *params, struct csinn_perf_info *perf_info);

int shl_ref_roi_align_perf(struct csinn_tensor *data, struct csinn_tensor *rois,
                           struct csinn_tensor *output, struct csinn_roi_align_params *params,
                           struct csinn_perf_info *perf_info);

int shl_ref_roipool_perf(struct csinn_tensor *data, struct csinn_tensor *rois,
                         struct csinn_tensor *output, struct csinn_roi_pool_params *params,
                         struct csinn_perf_info *perf_info);

int shl_ref_round_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_rsqrt_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_scatter_nd_perf(struct csinn_tensor *input, struct csinn_tensor *indices,
                            struct csinn_tensor *updates, struct csinn_tensor *output,
                            struct csinn_scatter_nd_params *params,
                            struct csinn_perf_info *perf_info);

int shl_ref_unsorted_segment_max_perf(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                      struct csinn_tensor *output,
                                      struct csinn_segment_params *params,
                                      struct csinn_perf_info *perf_info);

int shl_ref_segment_max_perf(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                             struct csinn_tensor *output, struct csinn_segment_params *params,
                             struct csinn_perf_info *perf_info);

int shl_ref_unsorted_segment_mean_perf(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                       struct csinn_tensor *output,
                                       struct csinn_segment_params *params,
                                       struct csinn_perf_info *perf_info);

int shl_ref_segment_mean_perf(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                              struct csinn_tensor *output, struct csinn_segment_params *params,
                              struct csinn_perf_info *perf_info);

int shl_ref_unsorted_segment_min_perf(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                      struct csinn_tensor *output,
                                      struct csinn_segment_params *params,
                                      struct csinn_perf_info *perf_info);

int shl_ref_segment_min_perf(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                             struct csinn_tensor *output, struct csinn_segment_params *params,
                             struct csinn_perf_info *perf_info);

int shl_ref_unsorted_segment_prod_perf(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                       struct csinn_tensor *output,
                                       struct csinn_segment_params *params,
                                       struct csinn_perf_info *perf_info);

int shl_ref_segment_prod_perf(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                              struct csinn_tensor *output, struct csinn_segment_params *params,
                              struct csinn_perf_info *perf_info);

int shl_ref_unsorted_segment_sum_perf(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                      struct csinn_tensor *output,
                                      struct csinn_segment_params *params,
                                      struct csinn_perf_info *perf_info);

int shl_ref_segment_sum_perf(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                             struct csinn_tensor *output, struct csinn_segment_params *params,
                             struct csinn_perf_info *perf_info);

int shl_ref_select_perf(struct csinn_tensor *condition, struct csinn_tensor *input0,
                        struct csinn_tensor *input1, struct csinn_tensor *output,
                        struct csinn_select_params *params, struct csinn_perf_info *perf_info);

int shl_ref_shape_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_shape_params *params, struct csinn_perf_info *perf_info);

int shl_ref_shuffle_channel_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_shuffle_channel_params *params,
                                 struct csinn_perf_info *perf_info);

int shl_ref_sigmoid_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_sigmoid_params *params, struct csinn_perf_info *perf_info);

int shl_ref_silu_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_sigmoid_params *params, struct csinn_perf_info *perf_info);

int shl_ref_sign_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_sin_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_sinh_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_slice_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_slice_params *params, struct csinn_perf_info *perf_info);

int shl_ref_softmax_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_softmax_params *params, struct csinn_perf_info *perf_info);

int shl_ref_softplus_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_softrelu_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_relu_params *params, struct csinn_perf_info *perf_info);

int shl_ref_softsign_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_space_to_batch_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_space_to_batch_params *params,
                                struct csinn_perf_info *perf_info);

int shl_ref_space_to_depth_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_space_to_depth_params *params,
                                struct csinn_perf_info *perf_info);

int shl_ref_split_perf(struct csinn_tensor *input, struct csinn_tensor **output,
                       struct csinn_split_params *params, struct csinn_perf_info *perf_info);

int shl_ref_sqrt_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_square_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_squeeze_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_squeeze_params *params, struct csinn_perf_info *perf_info);

int shl_ref_stack_perf(struct csinn_tensor **input, struct csinn_tensor *output,
                       struct csinn_stack_params *params, struct csinn_perf_info *perf_info);

int shl_ref_strided_slice_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_strided_slice_params *params,
                               struct csinn_perf_info *perf_info);

int shl_ref_sub_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params,
                     struct csinn_perf_info *perf_info);

int shl_ref_sum_stride_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params, struct csinn_perf_info *perf_info);

int shl_ref_tan_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_tanh_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_threshold_relu_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_relu_params *params,
                                struct csinn_perf_info *perf_info);

int shl_ref_tile_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tile_params *params, struct csinn_perf_info *perf_info);

int shl_ref_topk_perf(struct csinn_tensor *input, struct csinn_tensor *output1,
                      struct csinn_tensor *output2, struct csinn_topk_params *params,
                      struct csinn_perf_info *perf_info);

int shl_ref_transpose_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_transpose_params *params,
                           struct csinn_perf_info *perf_info);

int shl_ref_trunc_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_unpooling_perf(struct csinn_tensor *input, struct csinn_tensor *mask,
                           struct csinn_tensor *output, struct csinn_unpooling_params *params,
                           struct csinn_perf_info *perf_info);

int shl_ref_unstack_perf(struct csinn_tensor *input, struct csinn_tensor **output,
                         struct csinn_unstack_params *params, struct csinn_perf_info *perf_info);

int shl_ref_xor_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params,
                     struct csinn_perf_info *perf_info);

int shl_ref_yuv_rgb_scale_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_siso_params *params, struct csinn_perf_info *perf_info);

int shl_ref_one_hot_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_one_hot_params *params, struct csinn_perf_info *perf_info);

int shl_ref_where_perf(struct csinn_tensor *condition, struct csinn_tensor *x,
                       struct csinn_tensor *y, struct csinn_tensor *output,
                       struct csinn_where_params *params, struct csinn_perf_info *perf_info);

int shl_ref_where_softmax_perf(struct csinn_tensor *condition, struct csinn_tensor *y,
                               struct csinn_tensor *output,
                               struct csinn_where_softmax_params *params,
                               struct csinn_perf_info *perf_info);

int shl_ref_cast_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_cast_params *params, struct csinn_perf_info *perf_info);

int shl_ref_instance_norm_perf(struct csinn_tensor *input, struct csinn_tensor *scales,
                               struct csinn_tensor *bias, struct csinn_tensor *output,
                               struct csinn_instance_norm_params *params,
                               struct csinn_perf_info *perf_info);

int shl_ref_rms_norm_perf(struct csinn_tensor *input, struct csinn_tensor *weight,
                          struct csinn_tensor *output, struct csinn_rms_norm_params *params,
                          struct csinn_perf_info *perf_info);

int shl_ref_rope_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_rope_params *params, struct csinn_perf_info *perf_info);

int shl_ref_llm_pos_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_llm_pos_params *params, struct csinn_perf_info *perf_info);

int shl_ref_embedding_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_diso_params *params,
                           struct csinn_perf_info *perf_info);

int shl_ref_scaled_dot_product_attention_perf(struct csinn_tensor *query, struct csinn_tensor *key,
                                              struct csinn_tensor *value,
                                              struct csinn_tensor *output,
                                              struct csinn_scale_dot_attention_params *params,
                                              struct csinn_perf_info *perf_info);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SHL_REF_PERF_H_
