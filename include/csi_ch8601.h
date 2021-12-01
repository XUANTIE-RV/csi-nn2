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

#ifndef _CSI_NN_CH8601_H
#define _CSI_NN_CH8601_H
#include "csi_nn.h"
#include "csi_utils.h"
#include "csi_node.h"

int csi_ch8601_conv2d(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct csi_tensor *kernel,
                      struct csi_tensor *bias,
                      struct conv2d_params *params);

int csi_ch8601_depthwise_conv2d(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *kernel,
                                struct csi_tensor *bias,
                                struct conv2d_params *params);

int csi_ch8601_group_conv2d(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct csi_tensor *kernel,
                            struct csi_tensor *bias,
                            struct conv2d_params *params);

int csi_ch8601_conv2d_relu(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct csi_tensor *kernel,
                           struct csi_tensor *bias,
                           struct conv2d_params *params);

int csi_ch8601_deconv2d(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct csi_tensor *kernel,
                        struct csi_tensor *bias,
                        struct conv2d_params *params);

int csi_ch8601_depthwise_deconv2d(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct csi_tensor *kernel,
                                  struct csi_tensor *bias,
                                  struct conv2d_params *params);

int csi_ch8601_fullyconnected(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct csi_tensor *weights,
                              struct csi_tensor *bias,
                              struct fc_params *params);

int csi_ch8601_fullyconnected_relu(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct csi_tensor *weights,
                                   struct csi_tensor *bias,
                                   struct fc_params *params);

int csi_ch8601_maxpool2d(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct pool_params *params);

int csi_ch8601_global_maxpool2d(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct pool_params *params);

int csi_ch8601_avgpool2d(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct pool_params *params);

int csi_ch8601_global_avgpool2d(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct pool_params *params);

int csi_ch8601_l2pool(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct pool_params *params);

int csi_ch8601_pool_with_argmax(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct pool_params *params);

int csi_ch8601_maxpool2d_locat(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct pool_params *params);

int csi_ch8601_unpooling(struct csi_tensor *input,
                         struct csi_tensor *mask,
                         struct csi_tensor *output,
                         struct unpooling_params *params);

int csi_ch8601_negative(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_ch8601_floor(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_ch8601_ceil(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_ch8601_abs(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_ch8601_exp(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_ch8601_sin(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_ch8601_tanh(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_ch8601_sqrt(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_ch8601_rsqrt(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_ch8601_square(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct siso_params *params);

int csi_ch8601_sigmoid(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct sigmoid_params *params);

int csi_ch8601_elu(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct relu_params *params);

int csi_ch8601_relu(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct relu_params *params);

int csi_ch8601_relu1(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct relu_params *params);

int csi_ch8601_relu6(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct relu_params *params);

int csi_ch8601_relun(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct relu_params *params);

int csi_ch8601_leaky_relu(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct relu_params *params);

int csi_ch8601_softrelu(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct relu_params *params);

int csi_ch8601_prelu(struct csi_tensor *input,
                     struct csi_tensor *alpha,
                     struct csi_tensor *output,
                     struct prelu_params *params);

int csi_ch8601_softplus(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params);

int csi_ch8601_softmax(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct softmax_params *params);

int csi_ch8601_batch_normalization(struct csi_tensor *input,
                                   struct csi_tensor *mean,
                                   struct csi_tensor *variance,
                                   struct csi_tensor *gamma,
                                   struct csi_tensor *beta,
                                   struct csi_tensor *output,
                                   struct bn_params *params);

int csi_ch8601_l2_normalization(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct l2n_params *params);

int csi_ch8601_lrn(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct lrn_params *params);

int csi_ch8601_matmul(struct csi_tensor *mat0,
                      struct csi_tensor *mat1,
                      struct csi_tensor *output,
                      struct matmul_params *params);

int csi_ch8601_add(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_ch8601_sub(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_ch8601_mul(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_ch8601_div(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_ch8601_floor_divide(struct csi_tensor *input0,
                            struct csi_tensor *input1,
                            struct csi_tensor *output,
                            struct diso_params *params);

int csi_ch8601_maximum(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct diso_params *params);

int csi_ch8601_minimum(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct diso_params *params);

int csi_ch8601_power(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params);

int csi_ch8601_greater(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct diso_params *params);

int csi_ch8601_less(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_ch8601_equal(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params);

int csi_ch8601_not_equal(struct csi_tensor *input0,
                         struct csi_tensor *input1,
                         struct csi_tensor *output,
                         struct diso_params *params);

int csi_ch8601_greater_equal(struct csi_tensor *input0,
                             struct csi_tensor *input1,
                             struct csi_tensor *output,
                             struct diso_params *params);

int csi_ch8601_less_equal(struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct diso_params *params);

int csi_ch8601_select(struct csi_tensor *condition,
                      struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params);

int csi_ch8601_and(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_ch8601_or(struct csi_tensor *input0,
                  struct csi_tensor *input1,
                  struct csi_tensor *output,
                  struct diso_params *params);

int csi_ch8601_pad(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct pad_params *params);

int csi_ch8601_resize(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct resize_params *params);

int csi_ch8601_concat(struct csi_tensor **input,
                      struct csi_tensor *output,
                      struct concat_params *params);

int csi_ch8601_proposal(struct csi_tensor *cls_prob,
                        struct csi_tensor *bbox_pred,
                        struct csi_tensor *im_info,
                        struct csi_tensor *output,
                        struct proposal_params *params);

int csi_ch8601_psroipooling(struct csi_tensor *data,
                            struct csi_tensor *rois,
                            struct csi_tensor *output,
                            struct psroipooling_params *params);

int csi_ch8601_roipool(struct csi_tensor *data,
                       struct csi_tensor *rois,
                       struct csi_tensor *output,
                       struct roi_pool_params *params);

int csi_ch8601_transpose(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct transpose_params *params);

int csi_ch8601_reshape(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct reshape_params *params);

int csi_ch8601_reshape_tail(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct reshape_params *params);

int csi_ch8601_shape(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct shape_params *params);

int csi_ch8601_expand_dims_f32(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct expand_dims_params *params);

int csi_ch8601_expand_dims_u8(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct expand_dims_params *params);

int csi_ch8601_reverse(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct reverse_params *params);

int csi_ch8601_flatten(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct flatten_params *params);

int csi_ch8601_flatten_tail(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct flatten_params *params);

int csi_ch8601_crop(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct crop_params *params);

int csi_ch8601_slice(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct slice_params *params);

int csi_ch8601_slice_tail(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct slice_params *params);

int csi_ch8601_split(struct csi_tensor *input,
                     struct csi_tensor **output,
                     struct split_params *params);

int csi_ch8601_stack(struct csi_tensor *inputs,
                     struct csi_tensor *output,
                     struct stack_params *params);

int csi_ch8601_tile(struct csi_tensor *inputs,
                    struct csi_tensor *output,
                    struct tile_params *params);

int csi_ch8601_arange(struct csi_tensor *output,
                      struct arange_params *params);

int csi_ch8601_where(struct csi_tensor *condition,
                     struct csi_tensor *x,
                     struct csi_tensor *y,
                     struct csi_tensor *output,
                     struct where_params *params);

int csi_ch8601_unstack(struct csi_tensor *input,
                       struct csi_tensor *outputs,
                       struct unstack_params *params);

int csi_ch8601_gather_nd(struct csi_tensor *input,
                         struct csi_tensor *indices,
                         struct csi_tensor *output,
                         struct gather_nd_params *params);

int csi_ch8601_squeeze(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct squeeze_params *params);

int csi_ch8601_squeeze_tail(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct squeeze_params *params);

int csi_ch8601_ndarray_size(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct ndarray_size_params *params);

int csi_ch8601_space_to_batch(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct space_to_batch_params *params);

int csi_ch8601_batch_to_space(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct batch_to_space_params *params);

int csi_ch8601_space_to_depth(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct space_to_depth_params *params);

int csi_ch8601_depth_to_space(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct depth_to_space_params *params);

int csi_ch8601_one_hot(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct one_hot_params *params);

int csi_ch8601_sequence_mask(struct csi_tensor *input0,
                             struct csi_tensor *input1,
                             struct csi_tensor *output,
                             struct sequence_mask_params *params);

int csi_ch8601_im2col(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct csi_tensor *kernel,
                      struct im2col_params *params);

int csi_ch8601_col2im(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct csi_tensor *kernel,
                      struct col2im_params *params);

int csi_ch8601_sum(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct reduce_params *params);

int csi_ch8601_mean(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reduce_params *params);

int csi_ch8601_max(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct reduce_params *params);

int csi_ch8601_min(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct reduce_params *params);

int csi_ch8601_prod(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reduce_params *params);

int csi_ch8601_argmin(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct reduce_params *params);

int csi_ch8601_argmax(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct reduce_params *params);

int csi_ch8601_all(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct reduce_params *params);

int csi_ch8601_any(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct reduce_params *params);

struct csi_ch8601_graph
{
    struct csi_node *input[8];
    struct csi_node *output[8];
    int input_num;
    int output_num;
    struct csi_node **layer;
    int layer_size;
    int layer_index;
};

struct csi_ch8601_target_data {
    struct csi_ch8601_graph *graph;
};

struct csi_ch8601_graph *csi_ch8601_get_graph(struct csi_session *sess);

void csi_ch8601_set_tensor(struct csi_tensor *tensor, struct csi_session *sess);
void csi_ch8601_set_const_tensor(struct csi_tensor *tensor, struct csi_session *sess);
int csi_ch8601_get_tensor(int index, struct csi_tensor *ret, struct csi_session *sess);

void csi_ch8601_get_multiplier_and_shift(double double_multiplier, int16_t* multiplier, int16_t* shift);
int csi_ch8601_get_q1(struct csi_tensor *input, struct csi_tensor *kernel, struct csi_tensor *output);
int csi_ch8601_get_q2(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *input2, struct csi_tensor *output, int q1);
int csi_ch8601_get_q3(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output);
struct csi_ch8601_graph *csi_ch8601_get_graph(struct csi_session *sess);
int csi_ch8601_siso_op(struct csi_tensor *input,
                       struct csi_tensor *output,
                       int op,
                       void *params);
int csi_ch8601_diso_op(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       int op,
                       void *params);
int csi_ch8601_sidcso_op(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct csi_tensor *const0,
                         struct csi_tensor *const1,
                         int op,
                         void *params);

int csi_ch8601_conv2d_internel(struct csi_tensor *conv2d_input,
                               struct csi_tensor *conv2d_output,
                               struct csi_tensor *conv2d_kernel,
                               struct csi_tensor *conv2d_bias,
                               struct conv2d_params *conv2d_params,
                               struct csi_tensor *mul_rhs,
                               struct csi_tensor *mul_output,
                               struct csi_tensor *add_rhs,
                               struct csi_tensor *add_output);
#endif
