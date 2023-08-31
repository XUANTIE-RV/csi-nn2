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

#ifndef INCLUDE_SHL_PNNA_H_
#define INCLUDE_SHL_PNNA_H_
#include "csi_nn.h"
#include "shl_utils.h"

int shl_pnna_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                    struct csinn_conv2d_params *params);

int shl_pnna_depthwise_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params);

int shl_pnna_group_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv2d_params *params);

int shl_pnna_deconv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                      struct csinn_conv2d_params *params);

int shl_pnna_depthwise_deconv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);

int shl_pnna_fullyconnected(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *weights, struct csinn_tensor *bias,
                            struct csinn_fc_params *params);

int shl_pnna_maxpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_pool_params *params);

int shl_pnna_avgpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_pool_params *params);

int shl_pnna_global_avgpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);

int shl_pnna_global_maxpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params);

int shl_pnna_negative(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

int shl_pnna_tanh(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_siso_params *params);

int shl_pnna_sigmoid(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_sigmoid_params *params);

int shl_pnna_elu(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_relu_params *params);

int shl_pnna_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_relu_params *params);

int shl_pnna_relu1(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_relu_params *params);

int shl_pnna_relu6(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_relu_params *params);

int shl_pnna_leaky_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_relu_params *params);

int shl_pnna_prelu(struct csinn_tensor *input, struct csinn_tensor *alpha,
                   struct csinn_tensor *output, struct csinn_prelu_params *params);

int shl_pnna_softmax(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_softmax_params *params);

int shl_pnna_batch_normalization(struct csinn_tensor *input, struct csinn_tensor *mean,
                                 struct csinn_tensor *variance, struct csinn_tensor *gamma,
                                 struct csinn_tensor *beta, struct csinn_tensor *output,
                                 struct csinn_bn_params *params);

int shl_pnna_l2_normalization(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_l2n_params *params);

int shl_pnna_lrn(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_lrn_params *params);

int shl_pnna_matmul(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                    struct csinn_tensor *output, struct csinn_matmul_params *params);

int shl_pnna_add(struct csinn_tensor *input0, struct csinn_tensor *input1,
                 struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_sub(struct csinn_tensor *input0, struct csinn_tensor *input1,
                 struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_mul(struct csinn_tensor *input0, struct csinn_tensor *input1,
                 struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_div(struct csinn_tensor *input0, struct csinn_tensor *input1,
                 struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_maximum(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_minimum(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_power(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_greater(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_less(struct csinn_tensor *input0, struct csinn_tensor *input1,
                  struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_not_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_greater_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_less_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_select(struct csinn_tensor *condition, struct csinn_tensor *input0,
                    struct csinn_tensor *input1, struct csinn_tensor *output,
                    struct csinn_diso_params *params);

int shl_pnna_and(struct csinn_tensor *input0, struct csinn_tensor *input1,
                 struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_or(struct csinn_tensor *input0, struct csinn_tensor *input1,
                struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_pnna_pad(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_pad_params *params);

int shl_pnna_resize(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_resize_params *params);

int shl_pnna_concat(struct csinn_tensor **input, struct csinn_tensor *output,
                    struct csinn_concat_params *params);

int shl_pnna_transpose(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_transpose_params *params);

int shl_pnna_reshape(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_reshape_params *params);

int shl_pnna_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_shape_params *params);

int shl_pnna_flatten(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_flatten_params *params);

int shl_pnna_crop(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_crop_params *params);

int shl_pnna_slice(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_slice_params *params);

int shl_pnna_split(struct csinn_tensor *input, struct csinn_tensor **output,
                   struct csinn_split_params *params);

int shl_pnna_squeeze(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_squeeze_params *params);

int shl_pnna_space_to_batch_nd(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_space_to_batch_nd_params *params);

int shl_pnna_batch_to_space_nd(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_batch_to_space_nd_params *params);

int shl_pnna_space_to_depth(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_space_to_depth_params *params);

int shl_pnna_depth_to_space(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_depth_to_space_params *params);

int shl_pnna_sum(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_reduce_params *params);

int shl_pnna_mean(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_reduce_params *params);

int shl_pnna_max(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_reduce_params *params);

int shl_pnna_min(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_reduce_params *params);

int shl_pnna_prod(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_reduce_params *params);

int shl_pnna_argmin(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_reduce_params *params);

int shl_pnna_argmax(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_reduce_params *params);

int shl_pnna_all(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_reduce_params *params);

int shl_pnna_any(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_reduce_params *params);

int shl_pnna_strided_slice(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_strided_slice_params *params);

int shl_pnna_roipool(struct csinn_tensor *data, struct csinn_tensor *rois,
                     struct csinn_tensor *output, struct csinn_roi_pool_params *params);

int shl_pnna_proposal(struct csinn_tensor *cls_prob, struct csinn_tensor *bbox_pred,
                      struct csinn_tensor *im_info, struct csinn_tensor *output,
                      struct csinn_proposal_params *params);

int shl_pnna_unpooling(struct csinn_tensor *input, struct csinn_tensor *mask,
                       struct csinn_tensor *output, struct csinn_unpooling_params *params);

int shl_pnna_maxpool2d_locat(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_pool_params *params);
int shl_pnna_sqrt(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_siso_params *params);
int shl_pnna_matmul(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_matmul_params *params);

int shl_pnna_data_covert(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_siso_params *params);

int shl_pnna_set_input_strides(struct csinn_session *sess, int input_byte_size, int input_fix_h,
                               int input_fix_w);

struct shl_pnna_tensor_fix {
    int height;
    int width;
};

struct shl_pnna_target_data {
    void *network;
    void *net_obj;
    void *context;
    void *binding;
    void *attrs;
    void *graph;
    void *nodes;
    void *in_buffers;
    void *out_buffers;
    void *th1520_hwconfig;
    void *th1520_mapconfig;
    void *to_free;
    int priority;
    struct shl_pnna_tensor_fix **input_fix;
    enum csinn_quant_enum quant_type;
};

#endif  // INCLUDE_SHL_PNNA_H_
