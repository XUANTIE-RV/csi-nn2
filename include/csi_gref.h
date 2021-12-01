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

#ifndef _CSI_NN_GREF_H
#define _CSI_NN_GREF_H
#include "csi_nn.h"
#include "csi_utils.h"
#include "csi_node.h"

int csi_gref_acos(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_gref_acosh(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_gref_cos(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_gref_cosh(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_gref_asin(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_gref_asinh(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_gref_tan(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_gref_atan(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_gref_atanh(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_gref_threshold_relu(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct relu_params *params);

int csi_gref_trunc(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_gref_topk(struct csi_tensor *input,
                  struct csi_tensor *output1,
                  struct csi_tensor *output2,
                  struct topk_params *params);

int csi_gref_cumprod(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct cumprod_params *params);

int csi_gref_cumsum(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct cumsum_params *params);

int csi_gref_conv2d(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct csi_tensor *bias,
                    struct conv2d_params *params);

int csi_gref_depthwise_conv2d(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct csi_tensor *kernel,
                              struct csi_tensor *bias,
                              struct conv2d_params *params);

int csi_gref_group_conv2d(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct csi_tensor *kernel,
                          struct csi_tensor *bias,
                          struct conv2d_params *params);

int csi_gref_conv2d_relu(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct csi_tensor *kernel,
                         struct csi_tensor *bias,
                         struct conv2d_params *params);

int csi_gref_conv2d_relu6(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct csi_tensor *kernel,
                         struct csi_tensor *bias,
                         struct conv2d_params *params);

int csi_gref_conv3d(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct csi_tensor *bias,
                    struct conv3d_params *params);

int csi_gref_deconv2d(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct csi_tensor *kernel,
                      struct csi_tensor *bias,
                      struct conv2d_params *params);

int csi_gref_deconv3d(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct csi_tensor *kernel,
                      struct csi_tensor *bias,
                      struct conv3d_params *params);

int csi_gref_depthwise_deconv2d(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *kernel,
                                struct csi_tensor *bias,
                                struct conv2d_params *params);

int csi_gref_depthwise_conv2d_relu(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct csi_tensor *kernel,
                                   struct csi_tensor *bias,
                                   struct conv2d_params *params);

int csi_gref_depthwise_conv2d_relu6(struct csi_tensor *input,
                                    struct csi_tensor *output,
                                    struct csi_tensor *kernel,
                                    struct csi_tensor *bias,
                                    struct conv2d_params *params);

int csi_gref_fsmn(struct csi_tensor *frame,
                  struct csi_tensor *l_filter,
                  struct csi_tensor *r_filter,
                  struct csi_tensor *frame_sequence,
                  struct csi_tensor *frame_counter,
                  struct csi_tensor *output,
                  struct fsmn_params *params);

int csi_gref_fullyconnected(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct csi_tensor *weights,
                            struct csi_tensor *bias,
                            struct fc_params *params);

int csi_gref_fullyconnected_relu(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct csi_tensor *weights,
                                 struct csi_tensor *bias,
                                 struct fc_params *params);

int csi_gref_maxpool2d(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct pool_params *params);

int csi_gref_maxpool3d(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct pool_params *params);

int csi_gref_avgpool2d(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct pool_params *params);

int csi_gref_avgpool3d(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct pool_params *params);

int csi_gref_global_avgpool3d(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct pool_params *params);

int csi_gref_global_avgpool2d(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct pool_params *params);

int csi_gref_global_maxpool2d(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct pool_params *params);

int csi_gref_l2pool(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct pool_params *params);

int csi_gref_pool_with_argmax(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct pool_params *params);

int csi_gref_maxpool2d_locat(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct pool_params *params);

int csi_gref_mod(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_gref_non_max_suppression(struct csi_tensor *input0,
                                 struct csi_tensor *input1,
                                 struct csi_tensor *output,
                                 struct non_max_suppression_params *params);

int csi_gref_unpooling(struct csi_tensor *input,
                       struct csi_tensor *mask,
                       struct csi_tensor *output,
                       struct unpooling_params *params);

int csi_gref_negative(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_gref_floor(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_gref_ceil(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_gref_clip(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_gref_abs(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_gref_exp(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_gref_sin(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_gref_sinh(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_gref_tanh(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_gref_sqrt(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_gref_rsqrt(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_gref_square(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_gref_sigmoid(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct sigmoid_params *params);

int csi_gref_softsign(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_gref_space_to_batch_nd(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct space_to_batch_nd_params *params);

int csi_gref_elu(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct relu_params *params);

int csi_gref_relu(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct relu_params *params);

int csi_gref_relu1(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct relu_params *params);

int csi_gref_relu6(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct relu_params *params);

int csi_gref_relun(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct relu_params *params);

int csi_gref_roi_align(struct csi_tensor *data,
                       struct csi_tensor *rois,
                       struct csi_tensor *output,
                       struct roi_align_params *params);

int csi_gref_roipool(struct csi_tensor *data,
                     struct csi_tensor *rois,
                     struct csi_tensor *output,
                     struct roi_pool_params *params);

int csi_gref_round(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_gref_leaky_relu(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct relu_params *params);

int csi_gref_softrelu(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct relu_params *params);

int csi_gref_prelu(struct csi_tensor *input,
                   struct csi_tensor *alpha,
                   struct csi_tensor *output,
                   struct prelu_params *params);

int csi_gref_softplus(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_gref_softmax(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct softmax_params *params);

int csi_gref_batch_normalization(struct csi_tensor *input,
                                 struct csi_tensor *mean,
                                 struct csi_tensor *variance,
                                 struct csi_tensor *gamma,
                                 struct csi_tensor *beta,
                                 struct csi_tensor *output,
                                 struct bn_params *params);

int csi_gref_l2_normalization(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct l2n_params *params);

int csi_gref_lrn(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct lrn_params *params);

int csi_gref_matmul(struct csi_tensor *mat0,
                    struct csi_tensor *mat1,
                    struct csi_tensor *output,
                    struct matmul_params *params);

int csi_gref_add(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_gref_sub(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_gref_mul(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_gref_div(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_gref_floor_divide(struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct diso_params *params);

int csi_gref_floor_mod(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct diso_params *params);

int csi_gref_maximum(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params);

int csi_gref_minimum(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params);

int csi_gref_power(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_gref_greater(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params);

int csi_gref_less(struct csi_tensor *input0,
                  struct csi_tensor *input1,
                  struct csi_tensor *output,
                  struct diso_params *params);

int csi_gref_log_softmax(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct softmax_params *params);

int csi_gref_log(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_gref_log1p(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_gref_equal(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_gref_not_equal(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct diso_params *params);

int csi_gref_not(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_gref_reduce_logsumexp(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct reduce_params *params);

int csi_gref_reduce_max(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reduce_params *params);

int csi_gref_reduce_mean(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct reduce_params *params);

int csi_gref_reduce_min(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reduce_params *params);

int csi_gref_reduce_prod(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct reduce_params *params);

int csi_gref_reduce_sum(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct reduce_params *params);

int csi_gref_greater_equal(struct csi_tensor *input0,
                           struct csi_tensor *input1,
                           struct csi_tensor *output,
                           struct diso_params *params);

int csi_gref_less_equal(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params);

int csi_gref_select(struct csi_tensor *condition,
                    struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct select_params *params);

int csi_gref_and(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_gref_or(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_gref_pad(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct pad_params *params);

int csi_gref_resize(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct resize_params *params);

int csi_gref_concat(struct csi_tensor **input,
                    struct csi_tensor *output,
                    struct concat_params *params);

int csi_gref_proposal(struct csi_tensor *cls_prob,
                      struct csi_tensor *bbox_pred,
                      struct csi_tensor *im_info,
                      struct csi_tensor *output,
                      struct proposal_params *params);

int csi_gref_psroipooling(struct csi_tensor *data,
                          struct csi_tensor *rois,
                          struct csi_tensor *output,
                          struct psroipooling_params *params);

int csi_gref_transpose(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct transpose_params *params);

int csi_gref_reshape(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct reshape_params *params);

int csi_gref_shape(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct shape_params *params);

int csi_gref_strided_slice(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct strided_slice_params *params);

int csi_gref_expand_dims(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct expand_dims_params *params);

int csi_gref_expm1(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_gref_reverse(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct reverse_params *params);

int csi_gref_flatten(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct flatten_params *params);

int csi_gref_crop(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct crop_params *params);

int csi_gref_slice(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct slice_params *params);

int csi_gref_split(struct csi_tensor *input,
                   struct csi_tensor **output,
                   struct split_params *params);

int csi_gref_stack(struct csi_tensor **input,
                   struct csi_tensor *output,
                   struct stack_params *params);

int csi_gref_tile(struct csi_tensor *inputs,
                  struct csi_tensor *output,
                  struct tile_params *params);

int csi_gref_arange(struct csi_tensor *output,
                    struct arange_params *params);

int csi_gref_where(struct csi_tensor *condition,
                   struct csi_tensor *x,
                   struct csi_tensor *y,
                   struct csi_tensor *output,
                   struct where_params *params);

int csi_gref_unstack(struct csi_tensor *input,
                     struct csi_tensor **output,
                     struct unstack_params *params);

int csi_gref_gather(struct csi_tensor *input,
                    struct csi_tensor *indices,
                    struct csi_tensor *output,
                    struct gather_params *params);

int csi_gref_gather_nd(struct csi_tensor *input,
                       struct csi_tensor *indices,
                       struct csi_tensor *output,
                       struct gather_nd_params *params);

int csi_gref_hard_sigmoid(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct sigmoid_params *params);

int csi_gref_isnan_bool(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_gref_logical_and(struct csi_tensor *input0,
                         struct csi_tensor *input1,
                         struct csi_tensor *output,
                         struct diso_params *params);

int csi_gref_logical_not(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct siso_params *params);

int csi_gref_logical_or(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params);

int csi_gref_logical_xor(struct csi_tensor *input0,
                         struct csi_tensor *input1,
                         struct csi_tensor *output,
                         struct diso_params *params);

int csi_gref_squeeze(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct squeeze_params *params);

int csi_gref_segment_max(struct csi_tensor *input0,
                         struct csi_tensor *input1,
                         struct csi_tensor *output,
                         struct segment_params *params);

int csi_gref_segment_mean(struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct segment_params *params);

int csi_gref_segment_min(struct csi_tensor *input0,
                         struct csi_tensor *input1,
                         struct csi_tensor *output,
                         struct segment_params *params);

int csi_gref_segment_prod(struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct segment_params *params);

int csi_gref_segment_sum(struct csi_tensor *input0,
                         struct csi_tensor *input1,
                         struct csi_tensor *output,
                         struct segment_params *params);

int csi_gref_scatter_nd(struct csi_tensor *input,
                        struct csi_tensor *indices,
                        struct csi_tensor *updates,
                        struct csi_tensor *output,
                        struct scatter_nd_params *params);

int csi_gref_shuffle_channel(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct shuffle_channel_params *params);

int csi_gref_sign(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_gref_ndarray_size(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct ndarray_size_params *params);

int csi_gref_space_to_batch(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct space_to_batch_params *params);

int csi_gref_batch_to_space(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct batch_to_space_params *params);

int csi_gref_batch_to_space_nd(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct batch_to_space_nd_params *params);

int csi_gref_space_to_depth(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct space_to_depth_params *params);

int csi_gref_depth_to_space(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct depth_to_space_params *params);

int csi_gref_broadcast_to(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct broadcast_to_params *params);

int csi_gref_one_hot(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct one_hot_params *params);

int csi_gref_sequence_mask(struct csi_tensor *input0,
                           struct csi_tensor *input1,
                           struct csi_tensor *output,
                           struct sequence_mask_params *params);

int csi_gref_im2col(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct im2col_params *params);

int csi_gref_col2im(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct col2im_params *params);

int csi_gref_sum(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct reduce_params *params);

int csi_gref_mean(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct reduce_params *params);

int csi_gref_max(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct reduce_params *params);

int csi_gref_min(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct reduce_params *params);

int csi_gref_prod(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct reduce_params *params);

int csi_gref_argmin(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reduce_params *params);

int csi_gref_argmax(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reduce_params *params);

int csi_gref_all(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct reduce_params *params);

int csi_gref_any(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct reduce_params *params);

int csi_gref_reorg(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct reorg_params *params);

int csi_gref_erf(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_gref_xor(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_gref_yuv_rgb_scale(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct siso_params *params);

struct csi_ref_graph
{
    struct csi_node **input;
    struct csi_node **output;
    int input_num;
    int output_num;
    struct csi_node **layer;
    int layer_size;
    int layer_index;
};

struct csi_gref_target_data
{
    struct csi_ref_graph *graph;
};

struct csi_ref_graph *csi_gref_get_graph(struct csi_session *sess);
int csi_gref_graph_insert(struct csi_node *node, struct csi_ref_graph *graph);
int csi_gref_siso_op(struct csi_tensor *input, struct csi_tensor *output,
                     int op, void *params);
int csi_gref_diso_op(struct csi_tensor *input0, struct csi_tensor *input1,
                     struct csi_tensor *output, int op, void *params);
int csi_gref_sidcso_op(struct csi_tensor *input, struct csi_tensor *output,
                       struct csi_tensor *const0, struct csi_tensor *const1,
                       int op, void *params);
void csi_gref_set_tensor(struct csi_tensor *tensor, struct csi_session *sess);
void csi_gref_set_const_tensor(struct csi_tensor *tensor, struct csi_session *sess);
int csi_gref_get_tensor(int index, struct csi_tensor *ret, struct csi_session *sess);
void csi_gref_nbg(struct csi_tensor **input, struct csi_tensor **output,
                  uint32_t inputs_count, uint32_t outputs_count, const char *url);

void csi_subgraph_alloc(struct csi_node *node, struct csi_ref_graph *ograph, struct csi_ref_graph *ggraph);
int csi_subgraph_init(struct csi_node *n);
int csi_subgraph_deinit(struct csi_node *n);
int csi_subgraph_run_init(struct csi_node *n);
int csi_subgraph_run(struct csi_node *n);
int csi_subgraph_run_deinit(struct csi_node *n);
#endif
