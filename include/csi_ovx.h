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

/* CSI-NN2 version 1.8.x */

#ifndef _CSI_NN_OVX_H
#define _CSI_NN_OVX_H
#include "csi_nn.h"
#include "csi_utils.h"

int csi_ovx_conv2d(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct csi_tensor *kernel,
                   struct csi_tensor *bias,
                   struct conv2d_params *params);

int csi_ovx_depthwise_conv2d(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct csi_tensor *kernel,
                             struct csi_tensor *bias,
                             struct conv2d_params *params);

int csi_ovx_group_conv2d(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct csi_tensor *kernel,
                         struct csi_tensor *bias,
                         struct conv2d_params *params);

int csi_ovx_conv2d_relu(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct csi_tensor *kernel,
                        struct csi_tensor *bias,
                        struct conv2d_params *params);

int csi_ovx_deconv2d(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct csi_tensor *kernel,
                     struct csi_tensor *bias,
                     struct conv2d_params *params);

int csi_ovx_depthwise_deconv2d(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *kernel,
                                struct csi_tensor *bias,
                                struct conv2d_params *params);

int csi_ovx_fullyconnected(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct csi_tensor *weights,
                           struct csi_tensor *bias,
                           struct fc_params *params);

int csi_ovx_fullyconnected_relu(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *weights,
                                struct csi_tensor *bias,
                                struct fc_params *params);

int csi_ovx_maxpool(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct pool_params *params);

int csi_ovx_global_maxpool(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct pool_params *params);

int csi_ovx_averagepool(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct pool_params *params);

int csi_ovx_global_averagepool(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct pool_params *params);

int csi_ovx_global_maxpool(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct pool_params *params);

int csi_ovx_l2pool(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct pool_params *params);

int csi_ovx_pool_with_argmax(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct pool_params *params);

int csi_ovx_maxpool2d_locat(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct pool_params *params);

int csi_ovx_unpooling(struct csi_tensor *input,
                      struct csi_tensor *mask,
                      struct csi_tensor *output,
                      struct unpooling_params *params);

int csi_ovx_negative(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_ovx_floor(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_ovx_ceil(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_ovx_abs(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_ovx_exp(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_ovx_log(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_ovx_sin(struct csi_tensor *input,
                struct csi_tensor *output,
                struct siso_params *params);

int csi_ovx_tanh(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_ovx_sqrt(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params);

int csi_ovx_rsqrt(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params);

int csi_ovx_square(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params);

int csi_ovx_sigmoid(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct sigmoid_params *params);

int csi_ovx_elu(struct csi_tensor *input,
                struct csi_tensor *output,
                struct relu_params *params);

int csi_ovx_relu(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct relu_params *params);

int csi_ovx_relu1(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct relu_params *params);

int csi_ovx_relu6(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct relu_params *params);

int csi_ovx_relun(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct relu_params *params);

int csi_ovx_leaky_relu(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct relu_params *params);

int csi_ovx_softrelu(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct relu_params *params);

int csi_ovx_prelu(struct csi_tensor *input,
                  struct csi_tensor *alpha,
                  struct csi_tensor *output,
                  struct prelu_params *params);

int csi_ovx_softplus(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_ovx_softmax(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct softmax_params *params);

int csi_ovx_log_softmax(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct softmax_params *params);

int csi_ovx_batch_normalization(struct csi_tensor *input,
                                struct csi_tensor *mean,
                                struct csi_tensor *variance,
                                struct csi_tensor *gamma,
                                struct csi_tensor *beta,
                                struct csi_tensor *output,
                                struct bn_params *params);

int csi_ovx_l2_normalization(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct l2n_params *params);

int csi_ovx_lrn(struct csi_tensor *input,
                struct csi_tensor *output,
                struct lrn_params *params);

int csi_ovx_matmul(struct csi_tensor *mat0,
                   struct csi_tensor *mat1,
                   struct csi_tensor *output,
                   struct matmul_params *params);

int csi_ovx_add(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_ovx_sub(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_ovx_mul(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_ovx_div(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_ovx_floor_divide(struct csi_tensor *input0,
                         struct csi_tensor *input1,
                         struct csi_tensor *output,
                         struct diso_params *params);

int csi_ovx_maximum(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_ovx_minimum(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_ovx_power(struct csi_tensor *input0,
                  struct csi_tensor *input1,
                  struct csi_tensor *output,
                  struct diso_params *params);

int csi_ovx_greater(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_ovx_less(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params);

int csi_ovx_equal(struct csi_tensor *input0,
                  struct csi_tensor *input1,
                  struct csi_tensor *output,
                  struct diso_params *params);

int csi_ovx_not_equal(struct csi_tensor *input0,
                      struct csi_tensor *input1,
                      struct csi_tensor *output,
                      struct diso_params *params);

int csi_ovx_greater_equal(struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct diso_params *params);

int csi_ovx_less_equal(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct diso_params *params);

int csi_ovx_select(struct csi_tensor *condition,
                   struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct diso_params *params);

int csi_ovx_and(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params);

int csi_ovx_or(struct csi_tensor *input0,
               struct csi_tensor *input1,
               struct csi_tensor *output,
               struct diso_params *params);

int csi_ovx_pad(struct csi_tensor *input,
                struct csi_tensor *output,
                struct pad_params *params);

int csi_ovx_resize(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct resize_params *params);

int csi_ovx_concat(struct csi_tensor **input,
                   struct csi_tensor *output,
                   struct concat_params *params);

int csi_ovx_proposal(struct csi_tensor *cls_prob,
                     struct csi_tensor *bbox_pred,
                     struct csi_tensor *im_info,
                     struct csi_tensor *output,
                     struct proposal_params *params);

int csi_ovx_psroipooling(struct csi_tensor *data,
                         struct csi_tensor *rois,
                         struct csi_tensor *output,
                         struct psroipooling_params *params);

int csi_ovx_roipool(struct csi_tensor *data,
                    struct csi_tensor *rois,
                    struct csi_tensor *output,
                    struct roi_pool_params *params);

int csi_ovx_roi_align(struct csi_tensor *input,
                      struct csi_tensor *rois,
                      struct csi_tensor *output,
                      struct roi_align_params *params);

int csi_ovx_transpose(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct transpose_params *params);

int csi_ovx_reshape(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reshape_params *params);

int csi_ovx_reshape_tail(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct reshape_params *params);

int csi_ovx_shape(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct shape_params *params);

int csi_ovx_expand_dims_f32(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct expand_dims_params *params);

int csi_ovx_expand_dims_u8(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct expand_dims_params *params);

int csi_ovx_reverse(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reverse_params *params);

int csi_ovx_flatten(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct flatten_params *params);

int csi_ovx_flatten_tail(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct flatten_params *params);

int csi_ovx_crop(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct crop_params *params);

int csi_ovx_slice(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct slice_params *params);

int csi_ovx_slice_tail(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct slice_params *params);

int csi_ovx_strided_slice(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct strided_slice_params *params);

int csi_ovx_split(struct csi_tensor *input,
                  struct csi_tensor **output,
                  struct split_params *params);

int csi_ovx_stack(struct csi_tensor **inputs,
                  struct csi_tensor *output,
                  struct stack_params *params);

int csi_ovx_tile(struct csi_tensor *inputs,
                 struct csi_tensor *output,
                 struct tile_params *params);

int csi_ovx_arange(struct csi_tensor *output,
                   struct arange_params *params);

int csi_ovx_where(struct csi_tensor *condition,
                  struct csi_tensor *x,
                  struct csi_tensor *y,
                  struct csi_tensor *output,
                  struct where_params *params);

int csi_ovx_unstack(struct csi_tensor *input,
                    struct csi_tensor **outputs,
                    struct unstack_params *params);

int csi_ovx_gather(struct csi_tensor *input,
                   struct csi_tensor *indices,
                   struct csi_tensor *output,
                   struct gather_params *params);

int csi_ovx_gather_nd(struct csi_tensor *input,
                      struct csi_tensor *indices,
                      struct csi_tensor *output,
                      struct gather_nd_params *params);

int csi_ovx_squeeze(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct squeeze_params *params);

int csi_ovx_squeeze_tail(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct squeeze_params *params);

int csi_ovx_ndarray_size(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct ndarray_size_params *params);

int csi_ovx_space_to_batch(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct space_to_batch_params *params);

int csi_ovx_batch_to_space(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct batch_to_space_params *params);

int csi_ovx_space_to_depth(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct space_to_depth_params *params);

int csi_ovx_depth_to_space(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct depth_to_space_params *params);

int csi_ovx_one_hot(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct one_hot_params *params);

int csi_ovx_sequence_mask(struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct sequence_mask_params *params);

int csi_ovx_im2col(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct csi_tensor *kernel,
                   struct im2col_params *params);

int csi_ovx_col2im(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct csi_tensor *kernel,
                   struct col2im_params *params);

int csi_ovx_sum(struct csi_tensor *input,
                struct csi_tensor *output,
                struct reduce_params *params);

int csi_ovx_mean(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct reduce_params *params);

int csi_ovx_max(struct csi_tensor *input,
                struct csi_tensor *output,
                struct reduce_params *params);

int csi_ovx_min(struct csi_tensor *input,
                struct csi_tensor *output,
                struct reduce_params *params);

int csi_ovx_prod(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct reduce_params *params);

int csi_ovx_argmin(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct reduce_params *params);

int csi_ovx_argmax(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct reduce_params *params);

int csi_ovx_all(struct csi_tensor *input,
                struct csi_tensor *output,
                struct reduce_params *params);

int csi_ovx_any(struct csi_tensor *input,
                struct csi_tensor *output,
                struct reduce_params *params);

int csi_ovx_reorg(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct reorg_params *params);

int csi_ovx_topk(struct csi_tensor *input,
                 struct csi_tensor *output0,
                 struct csi_tensor *output1,
                 struct topk_params *params);

int csi_ovx_clip(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct clip_params *params);

int csi_ovx_shuffle_channel(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct shuffle_channel_params *params);

int32_t csi_get_ceil_mode_fix(int32_t input, int32_t kernel, int32_t stride, int32_t pad);

struct csi_ovx_target_data {
    void *graph;
};

void *csi_ovx_get_graph(struct csi_session *sess);

uint8_t *csi_ovx_input_f32_to_u8(uint32_t idx, float *data, struct csi_session *sess);
int csi_ovx_get_tensor(int index, struct csi_tensor *ret, struct csi_session *sess);
void csi_ovx_save_output(int index, const char *filename, struct csi_session *sess);
void csi_ovx_show_top5(int index, struct csi_session *sess);
void csi_ovx_set_graph_attribute(struct csi_session *sess, int device_index);
int csi_ovx_get_device_number();

#endif
