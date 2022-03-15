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

/* CSI-NN2 version 1.12.x */

#ifndef INCLUDE_CSI_REF_I805_H_
#define INCLUDE_CSI_REF_I805_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "csi_internal.h"
#include "csi_nnfunctions.h"
#include "csi_ref.h"
#include "csi_utils.h"

int csi_ref_i805_conv2d_init_q7(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *kernel, struct csi_tensor *bias,
                                struct conv2d_params *params);

int csi_ref_i805_conv2d_init_q15(struct csi_tensor *input, struct csi_tensor *output,
                                 struct csi_tensor *kernel, struct csi_tensor *bias,
                                 struct conv2d_params *params);

int csi_ref_i805_depthwise_conv2d_init_q7(struct csi_tensor *input, struct csi_tensor *output,
                                          struct csi_tensor *kernel, struct csi_tensor *bias,
                                          struct conv2d_params *params);

int csi_ref_i805_avgpool2d_init_q7(struct csi_tensor *input, struct csi_tensor *output,
                                   struct pool_params *params);

int csi_ref_i805_maxpool2d_init_q7(struct csi_tensor *input, struct csi_tensor *output,
                                   struct pool_params *params);

int csi_ref_i805_fullyconnected_q7(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *weights, struct csi_tensor *bias,
                                   struct fc_params *params);

int csi_ref_i805_fullyconnected_q15(struct csi_tensor *input, struct csi_tensor *output,
                                    struct csi_tensor *weights, struct csi_tensor *bias,
                                    struct fc_params *params);

int csi_ref_i805_softmax_q7(struct csi_tensor *input, struct csi_tensor *output,
                            struct softmax_params *params);

int csi_ref_i805_softmax_q15(struct csi_tensor *input, struct csi_tensor *output,
                             struct softmax_params *params);

int csi_ref_i805_relu_q7(struct csi_tensor *input, struct csi_tensor *output,
                         struct relu_params *params);

int csi_ref_i805_relu_q15(struct csi_tensor *input, struct csi_tensor *output,
                          struct relu_params *params);

int csi_ref_i805_sigmoid_q7(struct csi_tensor *input, struct csi_tensor *output,
                            struct sigmoid_params *params);

int csi_ref_i805_sigmoid_q15(struct csi_tensor *input, struct csi_tensor *output,
                             struct sigmoid_params *params);

int csi_ref_i805_tanh_q7(struct csi_tensor *input, struct csi_tensor *output,
                         struct siso_params *params);

int csi_ref_i805_tanh_q15(struct csi_tensor *input, struct csi_tensor *output,
                          struct siso_params *params);

#endif  // INCLUDE_CSI_REF_I805_H_
