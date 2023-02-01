/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

/* SHL version 2.1.x */

#ifndef INCLUDE_SHL_I805_H_
#define INCLUDE_SHL_I805_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "csi_nn.h"
#include "shl_ref.h"

int shl_i805_conv2d_init_q7(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                            struct csinn_conv2d_params *params);

int shl_i805_conv2d_init_q15(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params);

int shl_i805_depthwise_conv2d_init_q7(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params);

int shl_i805_avgpool2d_init_q7(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_pool_params *params);

int shl_i805_maxpool2d_init_q7(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_pool_params *params);

int shl_i805_fullyconnected_q7(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *weights, struct csinn_tensor *bias,
                               struct csinn_fc_params *params);

int shl_i805_fullyconnected_q15(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *weights, struct csinn_tensor *bias,
                                struct csinn_fc_params *params);

int shl_i805_softmax_q7(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_softmax_params *params);

int shl_i805_softmax_q15(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_softmax_params *params);

int shl_i805_relu_q7(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_relu_params *params);

int shl_i805_relu_q15(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params);

int shl_i805_sigmoid_q7(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_sigmoid_params *params);

int shl_i805_sigmoid_q15(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_sigmoid_params *params);

int shl_i805_tanh_q7(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

int shl_i805_tanh_q15(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

/*********************** u8 asym quant opt func *********************************/

int shl_i805_add_init_u8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_i805_add_u8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_i805_clip_init_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_clip_params *params);

int shl_i805_clip_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_clip_params *params);

int shl_i805_conv2d_init_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                            struct csinn_conv2d_params *params);

int shl_i805_conv2d_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                       struct csinn_conv2d_params *params);

int shl_i805_depthwise_conv2d_init_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params);

int shl_i805_depthwise_conv2d_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params);

int shl_i805_fullyconnected_init_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *weights, struct csinn_tensor *bias,
                                    struct csinn_fc_params *params);

int shl_i805_fullyconnected_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *weights, struct csinn_tensor *bias,
                               struct csinn_fc_params *params);

int shl_i805_maxpool2d_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_pool_params *params);

int shl_i805_mul_init_u8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_i805_mul_u8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

int shl_i805_relu_init_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_relu_params *params);

int shl_i805_relu_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_relu_params *params);

int shl_i805_relu6_init_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_relu_params *params);

int shl_i805_relu6_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params);

int shl_i805_reshape_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_reshape_params *params);

#endif  // INCLUDE_SHL_I805_H_
