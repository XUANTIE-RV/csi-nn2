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

/* ----------------------------------------------------------------------
 * Title:        csi_nnsupportfunctions.h
 * Description:  Public header file of support functions for CSI NN Library
 *
 * -------------------------------------------------------------------- */

#ifndef SOURCE_I805_REF_NN_SUPPORT_I805_REF_SUPPORT_H_
#define SOURCE_I805_REF_NN_SUPPORT_I805_REF_SUPPORT_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
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
 * @brief tables for various activation functions
 *
 */

extern const q15_t sigmoidTable_q15[256];
extern const q7_t sigmoidTable_q7[256];

extern const q7_t tanhTable_q7[256];
extern const q15_t tanhTable_q15[256];

int32_t __SSAT_8(int32_t x)
{
    int32_t res = x;
    if (x > 0x7f) {
        res = 0x7f;
    } else if (x < -128) {
        res = -128;
    }

    return res;
}

int32_t __SSAT(int32_t val, uint32_t sat)
{
    if ((sat >= 1U) && (sat <= 32U)) {
        const int32_t max = (int32_t)((1U << (sat - 1U)) - 1U);
        const int32_t min = -1 - max;

        if (val > max) {
            return max;

        } else if (val < min) {
            return min;
        }
    }

    return val;
}

uint32_t __USAT(int32_t val, uint32_t sat)
{
    if (sat <= 31U) {
        const uint32_t max = ((1U << sat) - 1U);

        if (val > (int32_t)max) {
            return max;

        } else if (val < 0) {
            return 0U;
        }
    }

    return (uint32_t)val;
}

/**
 * @brief defition to adding rouding offset
 */
#ifndef CSKY_NN_TRUNCATE
#define NN_ROUND(out_shift) (0x1 << (out_shift - 1))
#else
#define NN_ROUND(out_shift) 0
#endif

#endif  // SOURCE_I805_REF_NN_SUPPORT_I805_REF_SUPPORT_H_
