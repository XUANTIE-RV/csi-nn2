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

#ifndef INCLUDE_SHL_RVV_CAP_H_
#define INCLUDE_SHL_RVV_CAP_H_

#include "csi_nn.h"
#include "shl_utils.h"

int shl_rvv_conv2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                       struct csinn_conv2d_params *params);

int shl_rvv_depthwise_conv2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params);

int shl_rvv_conv1d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                       struct csinn_conv1d_params *params);

int shl_rvv_deconv2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv2d_params *params);

int shl_rvv_fullyconnected_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *weights, struct csinn_tensor *bias,
                               struct csinn_fc_params *params);

int shl_rvv_maxpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_pool_params *params);

int shl_rvv_avgpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_pool_params *params);

int shl_rvv_add_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_rvv_sub_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_rvv_mul_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_rvv_div_cap(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_rvv_concat_cap(struct csinn_tensor **input, struct csinn_tensor *output,
                       struct csinn_clip_params *params);

int shl_rvv_leaky_relu_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_relu_params *params);

int shl_rvv_relu_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_relu_params *params);

int shl_rvv_relu6_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params);

int shl_rvv_global_avgpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);

int shl_rvv_global_maxpool2d_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pool_params *params);

int shl_rvv_reshape_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_reshape_params *params);

int shl_rvv_sigmoid_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_sigmoid_params *params);

int shl_rvv_softmax_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_softmax_params *params);

int shl_rvv_reduce_sum_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reduce_params *params);

int shl_rvv_prelu_cap(struct csinn_tensor *input, struct csinn_tensor *alpha,
                      struct csinn_tensor *output, struct csinn_prelu_params *params);

int shl_rvv_layer_norm_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_tensor *gamma, struct csinn_tensor *beta,
                           struct csinn_layer_norm_params *params);

int shl_rvv_clip_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_clip_params *params);

int shl_rvv_transpose_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_transpose_params *params);

int shl_rvv_matmul_cap(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                       struct csinn_tensor *output, struct csinn_matmul_params *params);

int shl_rvv_gather_cap(struct csinn_tensor *input, struct csinn_tensor *indices,
                       struct csinn_tensor *output, struct csinn_gather_params *params);

int shl_rvv_erf_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_clip_params *params);

int shl_rvv_strided_slice_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_strided_slice_params *params);

int shl_rvv_split_cap(struct csinn_tensor *input, struct csinn_tensor **output,
                      struct csinn_split_params *params);

int shl_rvv_silu_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_sigmoid_params *params);

int shl_rvv_rms_norm_cap(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *weight, struct csinn_rms_norm_params *params);

#endif  // INCLUDE_SHL_RVV_CAP_H_
