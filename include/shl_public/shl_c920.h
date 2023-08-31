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

#ifndef INCLUDE_SHL_C920_H_
#define INCLUDE_SHL_C920_H_

#include "csi_nn.h"
#include "shl_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

void shl_c920_u8_to_f32(const uint8_t *input, float *output, int32_t offset, float *scale,
                        uint32_t length);
void shl_c920_i8_to_f32(const int8_t *input, float *output, int32_t offset, float *scale,
                        uint32_t length);
void shl_c920_f32_to_u8(const float *input, uint8_t *output, int32_t offset, float *scale,
                        uint32_t length);
void shl_c920_f32_to_i8(const float *input, int8_t *output, int32_t offset, float *scale,
                        uint32_t length);

void *shl_c920_f32_to_input_dtype(uint32_t index, float *data, struct csinn_session *sess);
float *shl_c920_output_to_f32_dtype(uint32_t index, void *data, struct csinn_session *sess);

int shl_c920_detect_yolov5_postprocess(struct csinn_tensor **input_tensors,
                                       struct shl_yolov5_box *out,
                                       struct shl_yolov5_params *params);
int shl_c920_yolox_preprocess(struct csinn_tensor *input, struct csinn_tensor *output);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SHL_C920_H_
