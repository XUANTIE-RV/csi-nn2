/*
 * Copyright (C) 2016-2019 C-SKY Limited. All rights reserved.
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

/******************************************************************************
 * @file     math_snr.c
 * @brief    Definition of all helper functions required.
 * @version  V1.0
 * @date     05. Feb 2018
 ******************************************************************************/

#include <math.h>
#include <stdint.h>

#define MAX_SNR_VALUE 400.0
float csi_snr_f32(float *pRef, float *pTest, uint32_t buffSize);
float csi_snr_q31(int32_t *pRef, int32_t *pTest, uint32_t buffSize);
float csi_snr_q7(int8_t *pRef, int8_t *pTest, uint32_t buffSize);
