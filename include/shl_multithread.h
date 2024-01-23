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
#ifndef INCLUDE_SHL_MULTITHREAD_H_
#define INCLUDE_SHL_MULTITHREAD_H_

#if (!defined SHL_BUILD_RTOS)
#include <omp.h>
#endif
#include "csinn/csi_nn.h"

void shl_multithread_set_threads(int threads);

int shl_multithread_is_enable();

#endif  // INCLUDE_SHL_MULTITHREAD_H_
