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

#ifndef _CSI_NN_H
#define _CSI_NN_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "csi_internal.h"
#include "csi_utils.h"

int csi_conv2d_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct csi_tensor *bias,
                    struct conv2d_params *params);

int csi_conv2d(struct csi_tensor *input,
               struct csi_tensor *output,
               struct csi_tensor *kernel,
               struct csi_tensor *bias,
               struct conv2d_params *params);

int csi_conv2d_relu_init(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct csi_tensor *kernel,
                         struct csi_tensor *bias,
                         struct conv2d_params *params);

int csi_conv2d_relu(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct csi_tensor *bias,
                    struct conv2d_params *params);

int csi_conv2d_relu6_init(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct csi_tensor *kernel,
                          struct csi_tensor *bias,
                          struct conv2d_params *params);

int csi_conv2d_relu6(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct csi_tensor *kernel,
                     struct csi_tensor *bias,
                     struct conv2d_params *params);

int csi_deconv2d_init(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct csi_tensor *kernel,
                      struct csi_tensor *bias,
                      struct conv2d_params *params);

int csi_deconv2d(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct csi_tensor *kernel,
                 struct csi_tensor *bias,
                 struct conv2d_params *params);

int csi_conv3d_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct csi_tensor *bias,
                    struct conv3d_params *params);

int csi_conv3d(struct csi_tensor *input,
                struct csi_tensor *output,
                struct csi_tensor *kernel,
                struct csi_tensor *bias,
                struct conv3d_params *params);

int csi_deconv3d_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct csi_tensor *bias,
                    struct conv3d_params *params);

int csi_deconv3d(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct csi_tensor *kernel,
                 struct csi_tensor *bias,
                 struct conv3d_params *params);

int csi_fullyconnected_init(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct csi_tensor *weights,
                            struct csi_tensor *bias,
                            struct fc_params *params);

int csi_fullyconnected(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct csi_tensor *weights,
                       struct csi_tensor *bias,
                       struct fc_params *params);

int csi_fullyconnected_relu_init(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct csi_tensor *weights,
                                 struct csi_tensor *bias,
                                 struct fc_params *params);

int csi_fullyconnected_relu(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct csi_tensor *weights,
                            struct csi_tensor *bias,
                            struct fc_params *params);

int csi_maxpool_init(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct pool_params *params);

int csi_maxpool(struct csi_tensor *input,
                struct csi_tensor *output,
                struct pool_params *params);

int csi_maxpool3d_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct pool_params *params);

int csi_maxpool3d(struct csi_tensor *input,
                struct csi_tensor *output,
                struct pool_params *params);

int csi_global_maxpool_init(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct pool_params *params);

int csi_global_maxpool(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct pool_params *params);

int csi_averagepool_init(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct pool_params *params);

int csi_averagepool(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct pool_params *params);

int csi_averagepool3d_init(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct pool_params *params);

int csi_averagepool3d(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct pool_params *params);

int csi_global_averagepool_init(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct pool_params *params);

int csi_global_averagepool(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct pool_params *params);

int csi_global_maxpool_init(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct pool_params *params);

int csi_global_maxpool(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct pool_params *params);

int csi_l2pool_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct pool_params *params);

int csi_l2pool(struct csi_tensor *input,
               struct csi_tensor *output,
               struct pool_params *params);

int csi_pool_with_argmax_init(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct pool_params *params);

int csi_pool_with_argmax(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct pool_params *params);

int csi_maxpool2d_locat_init(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct pool_params *params);

int csi_maxpool2d_locat(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct pool_params *params);

int csi_unpooling_init(struct csi_tensor *input,
                       struct csi_tensor *mask,
                       struct csi_tensor *output,
                       struct unpooling_params *params);

int csi_unpooling(struct csi_tensor *input,
                  struct csi_tensor *mask,
                  struct csi_tensor *output,
                  struct unpooling_params *params);

int csi_roi_align_init(struct csi_tensor *data,
                       struct csi_tensor *rois,
                       struct csi_tensor *output,
                       struct roi_align_params *params);

int csi_roi_align(struct csi_tensor *data,
                  struct csi_tensor *rois,
                  struct csi_tensor *output,
                  struct roi_align_params *params);

int csi_roi_pool_init(struct csi_tensor *data,
                      struct csi_tensor *rois,
                      struct csi_tensor *output,
                      struct roi_pool_params *params);

int csi_roi_pool(struct csi_tensor *data,
                 struct csi_tensor *rois,
                 struct csi_tensor *output,
                 struct roi_pool_params *params);

int csi_negative_init(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_negative(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_floor_init(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_floor(struct csi_tensor *input,
              struct csi_tensor *output,
              struct siso_params *params);

int csi_ceil_init(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_ceil(struct csi_tensor *input,
             struct csi_tensor *output,
             struct siso_params *params);

int csi_sign_init(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_sign(struct csi_tensor *input,
             struct csi_tensor *output,
             struct siso_params *params);

int csi_trunc_init(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_trunc(struct csi_tensor *input,
              struct csi_tensor *output,
              struct siso_params *params);

int csi_round_init(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_round(struct csi_tensor *input,
              struct csi_tensor *output,
              struct siso_params *params);

int csi_abs_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_abs(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_isnan_bool_init(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_isnan_bool(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_exp_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_exp(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_expm1_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_expm1(struct csi_tensor *input,
             struct csi_tensor *output,
             struct siso_params *params);

int csi_sin_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_sin(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_cos_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_cos(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_tanh_init(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_tanh(struct csi_tensor *input,
             struct csi_tensor *output,
             struct siso_params *params);

int csi_log_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_log(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_sqrt_init(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_sqrt(struct csi_tensor *input,
             struct csi_tensor *output,
             struct siso_params *params);

int csi_rsqrt_init(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_rsqrt(struct csi_tensor *input,
              struct csi_tensor *output,
              struct siso_params *params);

int csi_square_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_square(struct csi_tensor *input,
               struct csi_tensor *output,
               struct siso_params *params);

int csi_sigmoid_init(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct sigmoid_params *params);

int csi_sigmoid(struct csi_tensor *input,
                struct csi_tensor *output,
                struct sigmoid_params *params);

int csi_hard_sigmoid_init(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct sigmoid_params *params);

int csi_hard_sigmoid(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct sigmoid_params *params);

int csi_elu_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct relu_params *params);

int csi_elu(struct csi_tensor *input,
            struct csi_tensor *output,
            struct relu_params *params);

int csi_relu_init(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct relu_params *params);

int csi_relu(struct csi_tensor *input,
             struct csi_tensor *output,
             struct relu_params *params);

int csi_relu1_init(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct relu_params *params);

int csi_relu1(struct csi_tensor *input,
              struct csi_tensor *output,
              struct relu_params *params);

int csi_relu6_init(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct relu_params *params);

int csi_relu6(struct csi_tensor *input,
              struct csi_tensor *output,
              struct relu_params *params);

int csi_relun_init(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct relu_params *params);

int csi_relun(struct csi_tensor *input,
              struct csi_tensor *output,
              struct relu_params *params);

int csi_leaky_relu_init(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct relu_params *params);

int csi_leaky_relu(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct relu_params *params);

int csi_softrelu_init(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct relu_params *params);

int csi_softrelu(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct relu_params *params);

int csi_prelu_init(struct csi_tensor *input,
                   struct csi_tensor *alpha,
                   struct csi_tensor *output,
                   struct prelu_params *params);

int csi_prelu(struct csi_tensor *input,
              struct csi_tensor *alpha,
              struct csi_tensor *output,
              struct prelu_params *params);

int csi_softplus_init(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_softplus(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_softmax_init(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct softmax_params *params);

int csi_softmax(struct csi_tensor *input,
                struct csi_tensor *output,
                struct softmax_params *params);

int csi_log_softmax_init(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct softmax_params *params);

int csi_log_softmax(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct softmax_params *params);

int csi_batch_normalization_init(struct csi_tensor *input,
                                 struct csi_tensor *mean,
                                 struct csi_tensor *variance,
                                 struct csi_tensor *gamma,
                                 struct csi_tensor *beta,
                                 struct csi_tensor *output,
                                 struct bn_params *params);

int csi_batch_normalization(struct csi_tensor *input,
                            struct csi_tensor *mean,
                            struct csi_tensor *variance,
                            struct csi_tensor *gamma,
                            struct csi_tensor *beta,
                            struct csi_tensor *output,
                            struct bn_params *params);

int csi_l2_normalization_init(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct l2n_params *params);

int csi_l2_normalization(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct l2n_params *params);

int csi_lrn_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct lrn_params *params);

int csi_lrn(struct csi_tensor *input,
            struct csi_tensor *output,
            struct lrn_params *params);

int csi_matmul_init(struct csi_tensor *mat0,
                    struct csi_tensor *mat1,
                    struct csi_tensor *output,
                    struct matmul_params *params);

int csi_matmul(struct csi_tensor *mat0,
               struct csi_tensor *mat1,
               struct csi_tensor *output,
               struct matmul_params *params);

int csi_add_init(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_add(struct csi_tensor *input0,
            struct csi_tensor *input1,
            struct csi_tensor *output,
            struct diso_params *params);

int csi_sub_init(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_sub(struct csi_tensor *input0,
            struct csi_tensor *input1,
            struct csi_tensor *output,
            struct diso_params *params);

int csi_mul_init(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_mul(struct csi_tensor *input0,
            struct csi_tensor *input1,
            struct csi_tensor *output,
            struct diso_params *params);

int csi_div_init(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_div(struct csi_tensor *input0,
            struct csi_tensor *input1,
            struct csi_tensor *output,
            struct diso_params *params);

int csi_floor_divide_init(struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct diso_params *params);

int csi_floor_divide(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params);

int csi_floor_mod_init(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct diso_params *params);

int csi_floor_mod(struct csi_tensor *input0,
                  struct csi_tensor *input1,
                  struct csi_tensor *output,
                  struct diso_params *params);

int csi_mod_init(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_mod(struct csi_tensor *input0,
            struct csi_tensor *input1,
            struct csi_tensor *output,
            struct diso_params *params);

int csi_maximum_init(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params);

int csi_maximum(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_minimum_init(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params);

int csi_minimum(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_power_init(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_power(struct csi_tensor *input0,
              struct csi_tensor *input1,
              struct csi_tensor *output,
              struct diso_params *params);

int csi_greater_init(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params);

int csi_greater(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_less_init(struct csi_tensor *input0,
                  struct csi_tensor *input1,
                  struct csi_tensor *output,
                  struct diso_params *params);

int csi_less(struct csi_tensor *input0,
             struct csi_tensor *input1,
             struct csi_tensor *output,
             struct diso_params *params);

int csi_logical_and_init(struct csi_tensor *input0,
                  struct csi_tensor *input1,
                  struct csi_tensor *output,
                  struct diso_params *params);

int csi_logical_and(struct csi_tensor *input0,
             struct csi_tensor *input1,
             struct csi_tensor *output,
             struct diso_params *params);

int csi_logical_or_init(struct csi_tensor *input0,
                  struct csi_tensor *input1,
                  struct csi_tensor *output,
                  struct diso_params *params);

int csi_logical_or(struct csi_tensor *input0,
             struct csi_tensor *input1,
             struct csi_tensor *output,
             struct diso_params *params);

int csi_logical_not_init(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_logical_not(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_logical_xor_init(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params);

int csi_logical_xor(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_equal_init(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_equal(struct csi_tensor *input0,
              struct csi_tensor *input1,
              struct csi_tensor *output,
              struct diso_params *params);

int csi_not_equal_init(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct diso_params *params);

int csi_not_equal(struct csi_tensor *input0,
                  struct csi_tensor *input1,
                  struct csi_tensor *output,
                  struct diso_params *params);

int csi_greater_equal_init(struct csi_tensor *input0,
                           struct csi_tensor *input1,
                           struct csi_tensor *output,
                           struct diso_params *params);

int csi_greater_equal(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params);

int csi_less_equal_init(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params);

int csi_less_equal(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_select_init(struct csi_tensor *condition,
                    struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct select_params *params);

int csi_select(struct csi_tensor *condition,
               struct csi_tensor *input0,
               struct csi_tensor *input1,
               struct csi_tensor *output,
               struct select_params *params);

int csi_and_init(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_and(struct csi_tensor *input0,
            struct csi_tensor *input1,
            struct csi_tensor *output,
            struct diso_params *params);

int csi_or_init(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_or(struct csi_tensor *input0,
           struct csi_tensor *input1,
           struct csi_tensor *output,
           struct diso_params *params);

int csi_xor_init(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_xor(struct csi_tensor *input0,
            struct csi_tensor *input1,
            struct csi_tensor *output,
            struct diso_params *params);

int csi_not_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_not(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_pad_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct pad_params *params);

int csi_pad(struct csi_tensor *input,
            struct csi_tensor *output,
            struct pad_params *params);

int csi_resize_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct resize_params *params);

int csi_resize(struct csi_tensor *input,
               struct csi_tensor *output,
               struct resize_params *params);

int csi_concat_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct concat_params *params);

int csi_concat(struct csi_tensor *input,
               struct csi_tensor *output,
               struct concat_params *params);

int csi_proposal_init(struct csi_tensor *cls_prob,
                      struct csi_tensor *bbox_pred,
                      struct csi_tensor *im_info,
                      struct csi_tensor *output,
                      struct proposal_params *params);

int csi_proposal(struct csi_tensor *cls_prob,
                 struct csi_tensor *bbox_pred,
                 struct csi_tensor *im_info,
                 struct csi_tensor *output,
                 struct proposal_params *params);

int csi_psroipooling_init(struct csi_tensor *data,
                          struct csi_tensor *rois,
                          struct csi_tensor *output,
                          struct psroipooling_params *params);

int csi_psroipooling(struct csi_tensor *data,
                     struct csi_tensor *rois,
                     struct csi_tensor *output,
                     struct psroipooling_params *params);

int csi_transpose_init(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct transpose_params *params);

int csi_transpose(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct transpose_params *params);

int csi_reshape_init(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct reshape_params *params);

int csi_reshape(struct csi_tensor *input,
                struct csi_tensor *output,
                struct reshape_params *params);

int csi_shape_init(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct shape_params *params);

int csi_shape(struct csi_tensor *input,
              struct csi_tensor *output,
              struct shape_params *params);

int csi_expand_dims_init(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct expand_dims_params *params);

int csi_expand_dims(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct expand_dims_params *params);

int csi_reverse_init(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct reverse_params *params);

int csi_reverse(struct csi_tensor *input,
                struct csi_tensor *output,
                struct reverse_params *params);

int csi_flatten_init(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct flatten_params *params);

int csi_flatten(struct csi_tensor *input,
                struct csi_tensor *output,
                struct flatten_params *params);

int csi_crop_init(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct crop_params *params);

int csi_crop(struct csi_tensor *input,
             struct csi_tensor *output,
             struct crop_params *params);

int csi_slice_init(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct slice_params *params);

int csi_slice(struct csi_tensor *input,
              struct csi_tensor *output,
              struct slice_params *params);

int csi_split_init(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct split_params *params);

int csi_split(struct csi_tensor *input,
              struct csi_tensor *output,
              struct split_params *params);

int csi_stack_init(struct csi_tensor *inputs,
                   struct csi_tensor *output,
                   struct stack_params *params);

int csi_stack(struct csi_tensor *inputs,
              struct csi_tensor *output,
              struct stack_params *params);

int csi_unstack_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct unstack_params *params);

int csi_unstack(struct csi_tensor *input,
                struct csi_tensor *output,
                struct unstack_params *params);

int csi_tile_init(struct csi_tensor *inputs,
                  struct csi_tensor *output,
                  struct tile_params *params);

int csi_tile(struct csi_tensor *inputs,
             struct csi_tensor *output,
             struct tile_params *params);

int csi_arange_init(struct csi_tensor *output,
                    struct arange_params *params);

int csi_arange(struct csi_tensor *output,
               struct arange_params *params);

int csi_where_init(struct csi_tensor *condition,
                   struct csi_tensor *x,
                   struct csi_tensor *y,
                   struct csi_tensor *output,
                   struct where_params *params);

int csi_where(struct csi_tensor *condition,
              struct csi_tensor *x,
              struct csi_tensor *y,
              struct csi_tensor *output,
              struct where_params *params);

int csi_unstack_init(struct csi_tensor *input,
                     struct csi_tensor *outputs,
                     struct unstack_params *params);

int csi_unstack(struct csi_tensor *input,
                struct csi_tensor *outputs,
                struct unstack_params *params);

int csi_gather_init(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct gather_params *params);

int csi_gather(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct gather_params *params);

int csi_gather_nd_init(struct csi_tensor *input,
                       struct csi_tensor *indices,
                       struct csi_tensor *output,
                       struct gather_nd_params *params);

int csi_gather_nd(struct csi_tensor *input,
                  struct csi_tensor *indices,
                  struct csi_tensor *output,
                  struct gather_nd_params *params);

int csi_squeeze_init(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct squeeze_params *params);

int csi_squeeze(struct csi_tensor *input,
                struct csi_tensor *output,
                struct squeeze_params *params);

int csi_ndarray_size_init(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct ndarray_size_params *params);

int csi_ndarray_size(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct ndarray_size_params *params);

int csi_space_to_batch_init(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct space_to_batch_params *params);

int csi_space_to_batch(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct space_to_batch_params *params);

int csi_batch_to_space_init(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct batch_to_space_params *params);

int csi_batch_to_space(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct batch_to_space_params *params);

int csi_space_to_depth_init(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct space_to_depth_params *params);

int csi_space_to_depth(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct space_to_depth_params *params);

int csi_depth_to_space_init(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct depth_to_space_params *params);

int csi_depth_to_space(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct depth_to_space_params *params);

int csi_one_hot_init(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct one_hot_params *params);

int csi_one_hot(struct csi_tensor *input,
                struct csi_tensor *output,
                struct one_hot_params *params);

int csi_sequence_mask_init(struct csi_tensor *input0,
                           struct csi_tensor *input1,
                           struct csi_tensor *output,
                           struct sequence_mask_params *params);

int csi_sequence_mask(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct sequence_mask_params *params);

int csi_im2col_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct im2col_params *params);

int csi_im2col(struct csi_tensor *input,
               struct csi_tensor *output,
               struct csi_tensor *kernel,
               struct im2col_params *params);

int csi_col2im_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct col2im_params *params);

int csi_col2im(struct csi_tensor *input,
               struct csi_tensor *output,
               struct csi_tensor *kernel,
               struct col2im_params *params);

int csi_sum_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct reduce_params *params);

int csi_sum(struct csi_tensor *input,
            struct csi_tensor *output,
            struct reduce_params *params);

int csi_mean_init(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct reduce_params *params);

int csi_mean(struct csi_tensor *input,
             struct csi_tensor *output,
             struct reduce_params *params);

int csi_max_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct reduce_params *params);

int csi_max(struct csi_tensor *input,
            struct csi_tensor *output,
            struct reduce_params *params);

int csi_min_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct reduce_params *params);

int csi_min(struct csi_tensor *input,
            struct csi_tensor *output,
            struct reduce_params *params);

int csi_prod_init(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct reduce_params *params);

int csi_prod(struct csi_tensor *input,
             struct csi_tensor *output,
             struct reduce_params *params);

int csi_argmin_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reduce_params *params);

int csi_argmin(struct csi_tensor *input,
               struct csi_tensor *output,
               struct reduce_params *params);

int csi_argmax_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reduce_params *params);

int csi_argmax(struct csi_tensor *input,
               struct csi_tensor *output,
               struct reduce_params *params);

int csi_all_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct reduce_params *params);

int csi_all(struct csi_tensor *input,
            struct csi_tensor *output,
            struct reduce_params *params);

int csi_any_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct reduce_params *params);

int csi_any(struct csi_tensor *input,
            struct csi_tensor *output,
            struct reduce_params *params);

int csi_reorg_init(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct reorg_params *params);

int csi_reorg(struct csi_tensor *input,
              struct csi_tensor *output,
              struct reorg_params *params);

int csi_yuv_rgb_scale_init(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct siso_params *params);

int csi_yuv_rgb_scale(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_segment_max_init(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct segment_params *params);

int csi_segment_max(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct segment_params *params);

int csi_segment_min_init(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct segment_params *params);

int csi_segment_min(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct segment_params *params);

int csi_segment_sum_init(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct segment_params *params);

int csi_segment_sum(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct segment_params *params);

int csi_segment_mean_init(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct segment_params *params);

int csi_segment_mean(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct segment_params *params);

int csi_segment_prod_init(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct segment_params *params);

int csi_segment_prod(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct segment_params *params);

int csi_threshold_relu_init(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct relu_params *params);

int csi_threshold_relu(struct csi_tensor *input,
                struct csi_tensor *output,
                struct relu_params *params);

int csi_acos_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);
int csi_acos(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_acosh_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_acosh(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_asin_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_asin(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_asinh_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_asinh(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_atan_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_atan(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_atanh_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_atanh(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_cosh_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_cosh(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_sinh_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_sinh(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_tan_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_tan(struct csi_tensor *input,
             struct csi_tensor *output,
             struct siso_params *params);

int csi_log1p_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_log1p(struct csi_tensor *input,
             struct csi_tensor *output,
             struct siso_params *params);

int csi_softsign_init(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_softsign(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_erf_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_erf(struct csi_tensor *input,
            struct csi_tensor *output,
            struct siso_params *params);

int csi_cumsum_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct cumsum_params *params);

int csi_cumsum(struct csi_tensor *input,
                struct csi_tensor *output,
                struct cumsum_params *params);

int csi_cumprod_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct cumprod_params *params);

int csi_cumprod(struct csi_tensor *input,
                struct csi_tensor *output,
                struct cumprod_params *params);

int csi_reduce_max_init(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reduce_params *params);

int csi_reduce_max(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reduce_params *params);

int csi_reduce_min_init(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reduce_params *params);

int csi_reduce_min(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reduce_params *params);

int csi_reduce_mean_init(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reduce_params *params);

int csi_reduce_mean(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reduce_params *params);

int csi_reduce_sum_init(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reduce_params *params);

int csi_reduce_sum(struct csi_tensor *input,
                    struct csi_tensor *output,
                struct reduce_params *params);

int csi_reduce_prod_init(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reduce_params *params);

int csi_reduce_prod(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reduce_params *params);

int csi_reduce_logsumexp_init(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct reduce_params *params);

int csi_reduce_logsumexp(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reduce_params *params);

int csi_broadcast_to_init(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct broadcast_to_params *params);

int csi_broadcast_to(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct broadcast_to_params *params);

int csi_clip_init(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct clip_params *params);

int csi_clip(struct csi_tensor *input,
             struct csi_tensor *output,
             struct clip_params *params);

int csi_strided_slice_init(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct strided_slice_params *params);

int csi_strided_slice(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct strided_slice_params *params);

#endif
