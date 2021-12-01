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

#ifndef _CSI_NN_PNNA_H
#define _CSI_NN_PNNA_H
#include "csi_nn.h"

int csi_pnna_conv2d(struct csi_tensor *input, struct csi_tensor *output, struct csi_tensor *kernel,
                    struct csi_tensor *bias, struct conv2d_params *params);

int csi_pnna_depthwise_conv2d(struct csi_tensor *input, struct csi_tensor *output,
                              struct csi_tensor *kernel, struct csi_tensor *bias,
                              struct conv2d_params *params);

int csi_pnna_group_conv2d(struct csi_tensor *input, struct csi_tensor *output,
                          struct csi_tensor *kernel, struct csi_tensor *bias,
                          struct conv2d_params *params);

int csi_pnna_deconv2d(struct csi_tensor *input, struct csi_tensor *output,
                      struct csi_tensor *kernel, struct csi_tensor *bias,
                      struct conv2d_params *params);

int csi_pnna_depthwise_deconv2d(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *kernel, struct csi_tensor *bias,
                                struct conv2d_params *params);

int csi_pnna_fullyconnected(struct csi_tensor *input, struct csi_tensor *output,
                            struct csi_tensor *weights, struct csi_tensor *bias,
                            struct fc_params *params);

int csi_pnna_maxpool2d(struct csi_tensor *input, struct csi_tensor *output,
                       struct pool_params *params);

int csi_pnna_avgpool2d(struct csi_tensor *input, struct csi_tensor *output,
                       struct pool_params *params);

int csi_pnna_global_avgpool2d(struct csi_tensor *input, struct csi_tensor *output,
                              struct pool_params *params);

int csi_pnna_global_maxpool2d(struct csi_tensor *input, struct csi_tensor *output,
                              struct pool_params *params);

int csi_pnna_negative(struct csi_tensor *input, struct csi_tensor *output,
                      struct siso_params *params);

int csi_pnna_tanh(struct csi_tensor *input, struct csi_tensor *output, struct siso_params *params);

int csi_pnna_sigmoid(struct csi_tensor *input, struct csi_tensor *output,
                     struct sigmoid_params *params);

int csi_pnna_elu(struct csi_tensor *input, struct csi_tensor *output, struct relu_params *params);

int csi_pnna_relu(struct csi_tensor *input, struct csi_tensor *output, struct relu_params *params);

int csi_pnna_relu1(struct csi_tensor *input, struct csi_tensor *output, struct relu_params *params);

int csi_pnna_relu6(struct csi_tensor *input, struct csi_tensor *output, struct relu_params *params);

int csi_pnna_leaky_relu(struct csi_tensor *input, struct csi_tensor *output,
                        struct relu_params *params);

int csi_pnna_prelu(struct csi_tensor *input, struct csi_tensor *alpha, struct csi_tensor *output,
                   struct prelu_params *params);

int csi_pnna_softmax(struct csi_tensor *input, struct csi_tensor *output,
                     struct softmax_params *params);

int csi_pnna_batch_normalization(struct csi_tensor *input, struct csi_tensor *mean,
                                 struct csi_tensor *variance, struct csi_tensor *gamma,
                                 struct csi_tensor *beta, struct csi_tensor *output,
                                 struct bn_params *params);

int csi_pnna_l2_normalization(struct csi_tensor *input, struct csi_tensor *output,
                              struct l2n_params *params);

int csi_pnna_lrn(struct csi_tensor *input, struct csi_tensor *output, struct lrn_params *params);

int csi_pnna_matmul(struct csi_tensor *mat0, struct csi_tensor *mat1, struct csi_tensor *output,
                    struct matmul_params *params);

int csi_pnna_add(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
                 struct diso_params *params);

int csi_pnna_sub(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
                 struct diso_params *params);

int csi_pnna_mul(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
                 struct diso_params *params);

int csi_pnna_div(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
                 struct diso_params *params);

int csi_pnna_maximum(struct csi_tensor *input0, struct csi_tensor *input1,
                     struct csi_tensor *output, struct diso_params *params);

int csi_pnna_minimum(struct csi_tensor *input0, struct csi_tensor *input1,
                     struct csi_tensor *output, struct diso_params *params);

int csi_pnna_power(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
                   struct diso_params *params);

int csi_pnna_greater(struct csi_tensor *input0, struct csi_tensor *input1,
                     struct csi_tensor *output, struct diso_params *params);

int csi_pnna_less(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
                  struct diso_params *params);

int csi_pnna_equal(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
                   struct diso_params *params);

int csi_pnna_not_equal(struct csi_tensor *input0, struct csi_tensor *input1,
                       struct csi_tensor *output, struct diso_params *params);

int csi_pnna_greater_equal(struct csi_tensor *input0, struct csi_tensor *input1,
                           struct csi_tensor *output, struct diso_params *params);

int csi_pnna_less_equal(struct csi_tensor *input0, struct csi_tensor *input1,
                        struct csi_tensor *output, struct diso_params *params);

int csi_pnna_select(struct csi_tensor *condition, struct csi_tensor *input0,
                    struct csi_tensor *input1, struct csi_tensor *output,
                    struct diso_params *params);

int csi_pnna_and(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
                 struct diso_params *params);

int csi_pnna_or(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
                struct diso_params *params);

int csi_pnna_pad(struct csi_tensor *input, struct csi_tensor *output, struct pad_params *params);

int csi_pnna_resize(struct csi_tensor *input, struct csi_tensor *output,
                    struct resize_params *params);

int csi_pnna_concat(struct csi_tensor **input, struct csi_tensor *output,
                    struct concat_params *params);

int csi_pnna_transpose(struct csi_tensor *input, struct csi_tensor *output,
                       struct transpose_params *params);

int csi_pnna_reshape(struct csi_tensor *input, struct csi_tensor *output,
                     struct reshape_params *params);

int csi_pnna_shape(struct csi_tensor *input, struct csi_tensor *output,
                   struct shape_params *params);

int csi_pnna_flatten(struct csi_tensor *input, struct csi_tensor *output,
                     struct flatten_params *params);

int csi_pnna_crop(struct csi_tensor *input, struct csi_tensor *output, struct crop_params *params);

int csi_pnna_slice(struct csi_tensor *input, struct csi_tensor *output,
                   struct slice_params *params);

int csi_pnna_split(struct csi_tensor *input, struct csi_tensor **output,
                   struct split_params *params);

int csi_pnna_squeeze(struct csi_tensor *input, struct csi_tensor *output,
                     struct squeeze_params *params);

int csi_pnna_space_to_batch_nd(struct csi_tensor *input, struct csi_tensor *output,
                               struct space_to_batch_nd_params *params);

int csi_pnna_batch_to_space_nd(struct csi_tensor *input, struct csi_tensor *output,
                               struct batch_to_space_nd_params *params);

int csi_pnna_space_to_depth(struct csi_tensor *input, struct csi_tensor *output,
                            struct space_to_depth_params *params);

int csi_pnna_depth_to_space(struct csi_tensor *input, struct csi_tensor *output,
                            struct depth_to_space_params *params);

int csi_pnna_sum(struct csi_tensor *input, struct csi_tensor *output, struct reduce_params *params);

int csi_pnna_mean(struct csi_tensor *input, struct csi_tensor *output,
                  struct reduce_params *params);

int csi_pnna_max(struct csi_tensor *input, struct csi_tensor *output, struct reduce_params *params);

int csi_pnna_min(struct csi_tensor *input, struct csi_tensor *output, struct reduce_params *params);

int csi_pnna_prod(struct csi_tensor *input, struct csi_tensor *output,
                  struct reduce_params *params);

int csi_pnna_argmin(struct csi_tensor *input, struct csi_tensor *output,
                    struct reduce_params *params);

int csi_pnna_argmax(struct csi_tensor *input, struct csi_tensor *output,
                    struct reduce_params *params);

int csi_pnna_all(struct csi_tensor *input, struct csi_tensor *output, struct reduce_params *params);

int csi_pnna_any(struct csi_tensor *input, struct csi_tensor *output, struct reduce_params *params);

int csi_pnna_strided_slice(struct csi_tensor *input, struct csi_tensor *output,
                           struct strided_slice_params *params);

int csi_pnna_roipool(struct csi_tensor *data, struct csi_tensor *rois, struct csi_tensor *output,
                     struct roi_pool_params *params);

int csi_pnna_proposal(struct csi_tensor *cls_prob, struct csi_tensor *bbox_pred,
                      struct csi_tensor *im_info, struct csi_tensor *output,
                      struct proposal_params *params);

int csi_pnna_unpooling(struct csi_tensor *input, struct csi_tensor *mask, struct csi_tensor *output,
                       struct unpooling_params *params);

int csi_pnna_maxpool2d_locat(struct csi_tensor *input, struct csi_tensor *output,
                             struct pool_params *params);
int csi_pnna_set_input_strides(struct csi_session *sess, int input_byte_size, int input_fix_h,
                               int input_fix_w);

struct csi_pnna_tensor_fix {
    int height;
    int width;
};

struct csi_pnna_target_data {
    void *network;
    void *net_obj;
    void *context;
    void *attrs;
    void *graph;
    void *nodes;
    void *out_buffers;
    void *light_hwconfig;
    void *light_mapconfig;
    struct csi_pnna_tensor_fix **input_fix;
    enum csinn_quant_enum quant_type;
};

#endif
