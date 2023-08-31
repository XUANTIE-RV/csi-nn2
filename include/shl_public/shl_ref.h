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

#ifndef INCLUDE_SHL_REF_H_
#define INCLUDE_SHL_REF_H_

#include "../csinn/csi_nn.h"
#include "../shl_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

void shl_ref_nn_init(struct csinn_tensor *input, struct csinn_tensor *output);
void shl_ref_nn_deinit(struct csinn_tensor *input, struct csinn_tensor *output);
struct csinn_tensor *shl_ref_alloc_float_tensor(struct csinn_tensor *src);
void shl_ref_free_float_tensor(struct csinn_tensor *src);
struct csinn_tensor *shl_ref_convert_float_tensor(struct csinn_tensor *src);
void shl_ref_conv_free_float_tensor(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *kernel, struct csinn_tensor *bias);
struct csinn_tensor *shl_ref_tensor_transform_f32(struct csinn_tensor *input);
struct csinn_tensor *shl_ref_tensor_transform_int64(struct csinn_tensor *input);
int shl_ref_tensor_transform_free_f32(struct csinn_tensor *input);
int shl_ref_tensor_transform_free_int64(struct csinn_tensor *input);
uint8_t *shl_ref_f32_to_input_dtype(uint32_t index, float *data, struct csinn_session *sess);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SHL_REF_H_
