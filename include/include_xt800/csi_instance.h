/*
 * Copyright (C) 2016-2020 T-head Limited. All rights reserved.
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
 * @file     csi_instance.h
 * @brief    Some common define
 * @version  V1.0
 * @date     Feb. 2020
 ******************************************************************************/

#ifndef INCLUDE_INCLUDE_XT800_CSI_INSTANCE_H_
#define INCLUDE_INCLUDE_XT800_CSI_INSTANCE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <string.h>

/**
 * @brief 8-bit fractional data type in 1.7 format.
 */
typedef int8_t q7_t;

/**
 * @brief 16-bit fractional data type in 1.15 format.
 */
typedef int16_t q15_t;

/**
 * @brief 32-bit fractional data type in 1.31 format.
 */
typedef int32_t q31_t;

/**
 * @brief 64-bit fractional data type in 1.63 format.
 */
typedef int64_t q63_t;

/**
 * @brief 32-bit floating-point type definition.
 */
typedef float float32_t;

/**
 * @brief 64-bit floating-point type definition.
 */
typedef double float64_t;

/**
  @brief definition to read/write two 16 bit values.
  @deprecated
 */
#define __SIMD32_TYPE int32_t
#define __SIMD32(addr) (*(__SIMD32_TYPE **)&(addr))

/**
 * @brief definition to pack two 16 bit values.
 */
#define __PKHBT(ARG1, ARG2, ARG3)                     \
    ((((int32_t)(ARG1) << 0) & (int32_t)0x0000FFFF) | \
     (((int32_t)(ARG2) << ARG3) & (int32_t)0xFFFF0000))
#define __PKHTB(ARG1, ARG2, ARG3)                     \
    ((((int32_t)(ARG1) << 0) & (int32_t)0xFFFF0000) | \
     (((int32_t)(ARG2) >> ARG3) & (int32_t)0x0000FFFF))

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_INCLUDE_XT800_CSI_INSTANCE_H_
