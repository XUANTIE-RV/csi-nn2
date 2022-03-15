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

#ifndef INCLUDE_INCLUDE_XT800_CSI_NNSUPPORTFUNCTIONS_H_
#define INCLUDE_INCLUDE_XT800_CSI_NNSUPPORTFUNCTIONS_H_

#include "csi_instance.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Union for SIMD access of Q31/Q15/Q7 types
 */
union csi_nnword {
    q31_t word;          /**< Q31 type */
    q15_t half_words[2]; /**< Q15 type */
    q7_t bytes[4];       /**< Q7 type */
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

void csi_q7_to_q15_no_shift(const q7_t *pSrc, q15_t *pDst, uint32_t blockSize);

/**
 * @brief  Converts the elements of the Q7 vector to reordered Q15 vector without left-shift
 * @param[in]       *pSrc points to the Q7 input vector
 * @param[out]      *pDst points to the Q15 output vector
 * @param[in]       blockSize length of the input vector
 * @return none.
 *
 */

void csi_q7_to_q15_reordered_no_shift(const q7_t *pSrc, q15_t *pDst, uint32_t blockSize);

#if defined(CSI_MATH_DSP)

/*
 * @brief C custom defined SXTB16
 */
uint32_t __SXTB16(uint32_t x)
{
    return ((uint32_t)(((((q31_t)x << 24) >> 24) & (q31_t)0x0000FFFF) |
                       ((((q31_t)x << 8) >> 8) & (q31_t)0xFFFF0000)));
}

/**
  \brief   Rotate Right in unsigned value (32 bit)
  \details Rotate Right (immediate) provides the value of the contents of a register rotated by a
  variable number of bits. \param [in]    op1  Value to rotate \param [in]    op2  Number of Bits to
  rotate \return               Rotated value
 */
uint32_t __ROR(uint32_t op1, uint32_t op2) { return (op1 >> op2) | (op1 << (32U - op2)); }

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

/**
  \details This function saturates a signed value.
  \param [in]    x   Value to be saturated
  \param [in]    y   Bit position to saturate to [1..32]
  \return            Saturated value.
 */
int32_t __SSAT(int32_t x, uint32_t y)
{
    int32_t posMax, negMin;
    uint32_t i;

    posMax = 1;

    for (i = 0; i < (y - 1); i++) {
        posMax = posMax * 2;
    }

    if (x > 0) {
        posMax = (posMax - 1);

        if (x > posMax) {
            x = posMax;
        }

        //    x &= (posMax * 2 + 1);
    } else {
        negMin = -posMax;

        if (x < negMin) {
            x = negMin;
        }

        //    x &= (posMax * 2 - 1);
    }

    return (x);
}

/**
  \brief   Unsigned Saturate
  \details Saturates an unsigned value.
  \param [in]  value  Value to be saturated
  \param [in]    sat  Bit position to saturate to (0..31)
  \return             Saturated value
 */
uint32_t __USAT(uint32_t value, uint32_t sat)
{
    uint32_t result;

    if ((((0xFFFFFFFF >> sat) << sat) & value) != 0) {
        result = 0xFFFFFFFF >> (32 - sat);
    } else {
        result = value;
    }

    return (result);
}

/**
  \brief   Dual 16-bit saturating subtract.
  \details This function enables you to perform two 16-bit integer subtractions in parallel,
           saturating the results to the 16-bit signed integer range -2^15 <= x <= 2^15 - 1.
  \param [in]    x   first two 16-bit summands.
  \param [in]    y   second two 16-bit summands.
  \return        the saturated subtraction of the low halfwords, in the low halfword of the return
  value.\n the saturated subtraction of the high halfwords, in the high halfword of the return
  value.\n The returned results are saturated to the 16-bit signed integer range -2^15 <= x <= 2^15
  - 1. \remark res[15:0]  = val1[15:0]  - val2[15:0]        \n res[31:16] = val1[31:16] -
  val2[31:16]
 */
uint32_t __QSUB16(uint32_t x, uint32_t y)
{
    int32_t r, s;

    r = __SSAT(((((int32_t)x << 16) >> 16) - (((int32_t)y << 16) >> 16)), 16) & (int32_t)0x0000FFFF;
    s = __SSAT(((((int32_t)x) >> 16) - (((int32_t)y) >> 16)), 16) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r)));
}

/**
  \brief   Quad 8-bit saturating subtract.
  \details This function enables you to perform four 8-bit integer subtractions,
           saturating the results to the 8-bit signed integer range -2^7 <= x <= 2^7 - 1.
  \param [in]    x   first four 8-bit summands.
  \param [in]    y   second four 8-bit summands.
  \return        the subtraction of the first byte of each operand in the first byte of the return
  value.\n the subtraction of the second byte of each operand in the second byte of the return
  value.\n the subtraction of the third byte of each operand in the third byte of the return
  value.\n the subtraction of the fourth byte of each operand in the fourth byte of the return
  value.\n The returned results are saturated to the 8-bit signed integer range -2^7 <= x <= 2^7
  - 1. \remark res[7:0]   = val1[7:0]   - val2[7:0]        \n res[15:8]  = val1[15:8]  - val2[15:8]
  \n res[23:16] = val1[23:16] - val2[23:16]      \n res[31:24] = val1[31:24] - val2[31:24]
 */
uint32_t __QSUB8(uint32_t x, uint32_t y)
{
    int32_t r, s, t, u;

    r = __SSAT(((((int32_t)x << 24) >> 24) - (((int32_t)y << 24) >> 24)), 8) & (int32_t)0x000000FF;
    s = __SSAT(((((int32_t)x << 16) >> 24) - (((int32_t)y << 16) >> 24)), 8) & (int32_t)0x000000FF;
    t = __SSAT(((((int32_t)x << 8) >> 24) - (((int32_t)y << 8) >> 24)), 8) & (int32_t)0x000000FF;
    u = __SSAT(((((int32_t)x) >> 24) - (((int32_t)y) >> 24)), 8) & (int32_t)0x000000FF;

    return ((uint32_t)((u << 24) | (t << 16) | (s << 8) | (r)));
}

/**
  \brief   Dual 16-bit signed multiply with single 32-bit accumulator.
  \details This function enables you to perform two signed 16-bit multiplications,
           adding both results to a 32-bit accumulate operand.
  \param [in]    x   first 16-bit operands for each multiplication.
  \param [in]    y   second 16-bit operands for each multiplication.
  \param [in]  sum   accumulate value.
  \return        the product of each multiplication added to the accumulate value, as a 32-bit
  integer. \remark p1 = val1[15:0]  * val2[15:0]      \n p2 = val1[31:16] * val2[31:16]     \n
                 res[31:0] = p1 + p2 + val3[31:0]
 */

uint32_t __SMLAD(uint32_t x, uint32_t y, uint32_t sum)
{
    return ((uint32_t)(((((int32_t)x << 16) >> 16) * (((int32_t)y << 16) >> 16)) +
                       ((((int32_t)x) >> 16) * (((int32_t)y) >> 16)) + (((int32_t)sum))));
}
/**
  \brief   Dual 16-bit saturating addition.
  \details This function enables you to perform two 16-bit integer arithmetic additions in parallel,
           saturating the results to the 16-bit signed integer range -2^15 <= x <= 2^15 - 1.
  \param [in]    x   first two 16-bit summands.
  \param [in]    y   second two 16-bit summands.
  \return        the saturated addition of the low halfwords, in the low halfword of the return
  value.\n the saturated addition of the high halfwords, in the high halfword of the return value.\n
                 The returned results are saturated to the 16-bit signed integer range -2^15 <= x <=
  2^15 - 1. \remark res[15:0]  = val1[15:0]  + val2[15:0]        \n res[31:16] = val1[31:16] +
  val2[31:16]
 */
uint32_t __QADD16(uint32_t x, uint32_t y)
{
    int32_t r = 0, s = 0;

    r = __SSAT(((((int32_t)x << 16) >> 16) + (((int32_t)y << 16) >> 16)), 16) & (int32_t)0x0000FFFF;
    s = __SSAT(((((int32_t)x) >> 16) + (((int32_t)y) >> 16)), 16) & (int32_t)0x0000FFFF;

    return ((uint32_t)((s << 16) | (r)));
}

/**
 * @brief read and expand one Q7 word into two Q15 words
 */

void *read_and_pad(void *source, q31_t *out1, q31_t *out2)
{
    q31_t inA = *__SIMD32(source)++;
    q31_t inAbuf1 = __SXTB16(__ROR(inA, 8));
    q31_t inAbuf2 = __SXTB16(inA);

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

void *read_and_pad_reordered(void *source, q31_t *out1, q31_t *out2)
{
    q31_t inA = *__SIMD32(source)++;
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

q7_t *csi_nn_mat_mult_kernel_q7_q15_reordered(const q7_t *pA, const q15_t *pInBuffer,
                                              const uint16_t ch_im_out, const uint16_t numCol_A,
                                              const uint16_t bias_shift, const uint16_t out_shift,
                                              const q7_t *bias, q7_t *pOut);

q7_t *csi_nn_mat_mult_kernel_q7_q15(const q7_t *pA, const q15_t *pInBuffer,
                                    const uint16_t ch_im_out, const uint16_t numCol_A,
                                    const uint16_t bias_shift, const uint16_t out_shift,
                                    const q7_t *bias, q7_t *pOut);

/**
 * @brief A few utility functions used by pooling functions
 *
 */

void buffer_scale_back_q15_to_q7(q15_t *buffer, q7_t *target, uint16_t length, uint16_t scale);

void accumulate_q7_to_q15(q15_t *base, q7_t *target, const uint16_t length);

/**
 * @brief defition to adding rouding offset
 */
#ifndef CSKY_NN_TRUNCATE
#define NN_ROUND(out_shift) (0x1 << (out_shift - 1))
#else
#define NN_ROUND(out_shift) 0
#endif

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_INCLUDE_XT800_CSI_NNSUPPORTFUNCTIONS_H_
