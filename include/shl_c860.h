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

/* CSI-NN2 version 2.0.x */

#ifndef INCLUDE_CSI_C860_H_
#define INCLUDE_CSI_C860_H_

#include "csi_nn.h"
#include "shl_ref.h"

void shl_c860_dequantize_f32(uint8_t *input, float *output, int32_t offset, int32_t multiplier,
                             int32_t shift, int32_t length);

#endif  // INCLUDE_CSI_C860_H_
