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

/* ----------------------------------------------------------------------
 * Title:        csky_dsp2_nnsupportfunctions.h
 * Description:  Public header file of support functions for CSI NN Library
 *
 * -------------------------------------------------------------------- */

#ifndef _CSKY_DSP2_NNSUPPORTFUNCTIONS_H_
#define _CSKY_DSP2_NNSUPPORTFUNCTIONS_H_

#include "csky_math.h"

#ifdef __cplusplus
extern    "C"
{
#endif

/**
 * @brief Union for SIMD access of Q31/Q15/Q7 types
 */
union csky_dsp2_nnword
{
    q31_t     word;             /**< Q31 type */
    q15_t     half_words[2];    /**< Q15 type */
    q7_t      bytes[4];         /**< Q7 type */
};

/**
 * @defgroup nndata_convert Neural Network Data Conversion Functions
 *
 * Perform data type conversion in-between neural network operations
 *
 */

/**
 * @brief Converts the elements of the Q7 vector to Q15 vector without left-shift
 * @param[in]       *pSrc points to the Q7 input vector
 * @param[out]      *pDst points to the Q15 output vector
 * @param[in]       blockSize length of the input vector
 * @return none.
 *
 */

void csky_dsp2_q7_to_q15_no_shift(const q7_t * pSrc, q15_t * pDst,
                             uint32_t blockSize);

/**
 * @brief  Converts the elements of the Q7 vector to reordered Q15 vector without left-shift
 * @param[in]       *pSrc points to the Q7 input vector
 * @param[out]      *pDst points to the Q15 output vector
 * @param[in]       blockSize length of the input vector
 * @return none.
 *
 */

void csky_dsp2_q7_to_q15_reordered_no_shift(const q7_t * pSrc, q15_t * pDst,
                                       uint32_t blockSize);

#if defined (CSKY_MATH_DSP)

/**
 * @brief read and expand one Q7 word into two Q15 words
 */

__ALWAYS_INLINE void *read_and_pad(void *source, q31_t *out1, q31_t *out2)
{
        q31_t     inA = *__SIMD32(source)++;
        q31_t     inAbuf1 = __SXTB16(__ROR(inA, 8));
        q31_t     inAbuf2 = __SXTB16(inA);

#ifndef CSKY_MATH_BIG_ENDIAN
        *out2 = __PKHTB(inAbuf1, inAbuf2, 16);
        *out1 = __PKHBT(inAbuf2, inAbuf1, 16);
#else
        *out1 = __PKHTB(inAbuf1, inAbuf2, 16);
        *out2 = __PKHBT(inAbuf2, inAbuf1, 16);
#endif

        return source;
}

/**
 * @brief read and expand one Q7 word into two Q15 words with reordering
 */

__ALWAYS_INLINE void *read_and_pad_reordered(void *source, q31_t * out1,
                                                  q31_t * out2)
{
        q31_t     inA = *__SIMD32(source)++;
#ifndef CSKY_MATH_BIG_ENDIAN
        *out2 = __SXTB16(__ROR(inA, 8));
        *out1 = __SXTB16(inA);
#else
        *out1 = __SXTB16(__ROR(inA, 8));
        *out2 = __SXTB16(inA);
#endif

        return source;
}
#endif

/**
 * @brief defition to adding rouding offset
 */
#ifndef CSKY_NN_TRUNCATE
    #define NN_ROUND(out_shift) ( 0x1 << (out_shift - 1) )
#else
    #define NN_ROUND(out_shift) 0
#endif

#ifdef __cplusplus
}
#endif

#endif
