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

#ifndef _CSI_INTERNAL_C906_H
#define _CSI_INTERNAL_C906_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "csi_internal.h"
#include "csi_utils.h"

int csi_abs_f32_c906(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct siso_params *params);

int csi_abs_u8_c906(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params);

int csi_add_f32_c906(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     struct diso_params *params);

int csi_add_u8_c906(struct csi_tensor *input0,
                    struct csi_tensor *input1,
                    struct csi_tensor *output,
                    struct diso_params *params);

int csi_broadcast_to_f32_c906(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct broadcast_to_params *params);

int csi_broadcast_to_u8_c906(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct broadcast_to_params *params);

int csi_clip_f32_c906(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct clip_params *params);

int csi_clip_u8_c906(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct clip_params *params);

int csi_fullyconnected_f32_c906(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *weights,
                                struct csi_tensor *bias,
                                struct fc_params *params);

int csi_fullyconnected_u8_c906(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct csi_tensor *weights,
                               struct csi_tensor *bias,
                               struct fc_params *params);

int csi_prelu_f32_c906(struct csi_tensor *input,
                       struct csi_tensor *alpha,
                       struct csi_tensor *output,
                       struct prelu_params *params);

int csi_prelu_u8_c906(struct csi_tensor *input,
                      struct csi_tensor *alpha,
                      struct csi_tensor *output,
                      struct prelu_params *params);

int csi_relu_f32_c906(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct relu_params *params);

int csi_relu_u8_c906(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct relu_params *params);

int csi_relu1_f32_c906(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct relu_params *params);

int csi_relu1_u8_c906(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct relu_params *params);

int csi_relu6_f32_c906(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct relu_params *params);

int csi_relu6_u8_c906(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct relu_params *params);

int csi_leaky_relu_f32_c906(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct relu_params *params);

int csi_leaky_relu_u8_c906(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct relu_params *params);



#endif
