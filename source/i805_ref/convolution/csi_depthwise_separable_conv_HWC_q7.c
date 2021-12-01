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
 * Title:        csi_depthwise_separable_conv_HWC_q7.c
 * Description:  Q7 depthwise separable convolution function
 *
 * -------------------------------------------------------------------- */

#include "csi_math.h"
#include "csi_nnfunctions.h"
#include "csi_nnsupportfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

/**
 * @brief Q7 depthwise separable convolution function
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in   input tensor dimention
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       wt          pointer to kernel weights
 * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel  filter kernel size
 * @param[in]       padding     padding sizes
 * @param[in]       stride      convolution stride
 * @param[in]       bias        pointer to bias
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in,out]   Im_out      pointer to output tensor
 * @param[in]       dim_im_out  output tensor dimension
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @return     The function returns either
 * <code>CSI_MATH_SIZE_MISMATCH</code> or <code>CSI_MATH_SUCCESS</code> based on the outcome of size checking.
 *
 * @details
 *
 * <b>Buffer size:</b>
 *
 * bufferA size: 2*ch_im_in*dim_kernel*dim_kernel
 *
 * <b>Input dimension constraints:</b>
 *
 * ch_im_in equals ch_im_out
 *
 * Implementation:
 * There are 3 nested loop here:
 * Inner loop: calculate each output value with MAC instruction over an accumulator
 * Mid   loop: loop over different output channel
 * Outer loop: loop over different output (x, y)
 */

void csi_depthwise_separable_conv_HWC_q7(const q7_t * Im_in,
                                               const uint16_t dim_im_in,
                                               const uint16_t ch_im_in,
                                               const q7_t * wt,
                                               const uint16_t ch_im_out,
                                               const uint16_t dim_kernel,
                                               const uint16_t padding,
                                               const uint16_t stride,
                                               const q7_t * bias,
                                               const uint16_t bias_shift,
                                               const uint16_t out_shift,
                                               q7_t * Im_out,
                                               const uint16_t dim_im_out,
                                               q15_t * bufferA)
{

#if defined (CSI_MATH_DSP)

    int16_t   i_out_y, i_out_x;
    int16_t   i_ker_y, i_ker_x;
    q7_t     *colBuffer = (q7_t *) bufferA;
    q7_t     *pBuffer = colBuffer;
    const q7_t *pBias = bias;
    q7_t     *pOut = Im_out;
    uint16_t  rowCnt;
    uint16_t  row_shift;

    /* do some checking here, basically ch_im_in == ch_im_out */
    if (ch_im_in != ch_im_out)
    {
        return;
    }

    for (i_out_y = 0; i_out_y < dim_im_out; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            /* we first do im2col here */
            for (i_ker_y = i_out_y * stride - padding;
                 i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding;
                     i_ker_x < i_out_x * stride - padding + dim_kernel;
                     i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in
                        || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        /* csi_fill_q7(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, ch_im_in);
                    } else
                    {
                        /* csi_copy_q7((q7_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer, ch_im_in); */
                        memcpy(pBuffer, (q7_t *) Im_in + (i_ker_y * dim_im_in
                                + i_ker_x) * ch_im_in, ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            /* we will do the computation here for each channel */
            rowCnt = ch_im_out >> 2;
            row_shift = 0;
            pBias = bias;

            while (rowCnt)
            {
                q31_t sum =  ((q31_t)(*pBias++) << bias_shift)
                    + NN_ROUND(out_shift);
                q31_t sum2 = ((q31_t)(*pBias++) << bias_shift)
                    + NN_ROUND(out_shift);
                q31_t sum3 = ((q31_t)(*pBias++) << bias_shift)
                    + NN_ROUND(out_shift);
                q31_t sum4 = ((q31_t)(*pBias++) << bias_shift)
                    + NN_ROUND(out_shift);

                uint16_t  colCnt = (dim_kernel * dim_kernel) >> 1;
                q7_t     *pB = colBuffer + row_shift;
                const q7_t *pA = wt + row_shift;
                row_shift += 4;

                while (colCnt)
                {
                    q31_t     inA1, inA2, inB1, inB2, opA, opB;

                    inB1 = *__SIMD32(pB);
                    pB += ch_im_in;
                    opB = *__SIMD32(pB);
                    pB += ch_im_in;
                    inB2 = __PKHTB(opB, inB1, 16);
                    inB1 = __PKHBT(inB1, opB, 16);
                    inA1 = *__SIMD32(pA);
                    pA += ch_im_in;
                    opB = *__SIMD32(pA);
                    pA += ch_im_in;
                    inA2 = __PKHTB(opB, inA1, 16);
                    inA1 = __PKHBT(inA1, opB, 16);
                    opA = __SXTB16(inA1);
                    opB = __SXTB16(inB1);
                    sum = __SMLAD(opA, opB, sum);
                    opA = __SXTB16(__ROR(inA1, 8));
                    opB = __SXTB16(__ROR(inB1, 8));
                    sum2 = __SMLAD(opA, opB, sum2);
                    opA = __SXTB16(inA2);
                    opB = __SXTB16(inB2);
                    sum3 = __SMLAD(opA, opB, sum3);
                    opA = __SXTB16(__ROR(inA2, 8));
                    opB = __SXTB16(__ROR(inB2, 8));
                    sum4 = __SMLAD(opA, opB, sum4);
                    colCnt--;
                }

                colCnt = (dim_kernel * dim_kernel) & 0x1;
                while (colCnt)
                {
                    union csi_nnword inA, inB;
                    inA.word = *__SIMD32(pA);
                    pA += ch_im_in;
                    inB.word = *__SIMD32(pB);
                    pB += ch_im_in;
                    sum += inA.bytes[0] * inB.bytes[0];
                    sum2 += inA.bytes[1] * inB.bytes[1];
                    sum3 += inA.bytes[2] * inB.bytes[2];
                    sum4 += inA.bytes[3] * inB.bytes[3];
                    colCnt--;
                }

                *pOut++ = (q7_t) __SSAT((sum >> out_shift), 8);
                *pOut++ = (q7_t) __SSAT((sum2 >> out_shift), 8);
                *pOut++ = (q7_t) __SSAT((sum3 >> out_shift), 8);
                *pOut++ = (q7_t) __SSAT((sum4 >> out_shift), 8);

                rowCnt--;
            }

            rowCnt = ch_im_out & 0x3;
            while (rowCnt)
            {
                q7_t *pB = colBuffer + row_shift;
                const q7_t *pA = wt + row_shift;
                q31_t sum = ((q31_t)(*pBias++) << bias_shift)
                    + NN_ROUND(out_shift);
                uint16_t colCnt = (dim_kernel * dim_kernel);

                row_shift += 1;

                while (colCnt)
                {
                    q7_t      A1 = *pA;
                    q7_t      B1 = *pB;
                    pA += ch_im_in;
                    pB += ch_im_in;
                    sum += A1 * B1;

                    colCnt--;
                }
                *pOut++ = (q7_t) __SSAT((sum >> out_shift), 8);
                rowCnt--;
            }

            /* clear counter and pointers */
            pBuffer = colBuffer;
        }
    }

#else
    int       i_out_y, i_out_x, i_ch_out, i_ker_x, i_ker_y;
    int       conv_out;

    /* do some checking here, basically ch_im_in == ch_im_out */
    if (ch_im_in != ch_im_out)
    {
        return;
    }

    for (i_out_y = 0; i_out_y < dim_im_out; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            for (i_ch_out = 0; i_ch_out < ch_im_out; i_ch_out++)
            {
                // for each output
                conv_out = ((q31_t)(bias[i_ch_out]) << bias_shift)
                    + NN_ROUND(out_shift);
                for (i_ker_y = 0; i_ker_y < dim_kernel; i_ker_y++)
                {
                    for (i_ker_x = 0; i_ker_x < dim_kernel; i_ker_x++)
                    {
                        int       in_row = stride * i_out_y + i_ker_y - padding;
                        int       in_col = stride * i_out_x + i_ker_x - padding;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in
                            && in_col < dim_im_in)
                        {
                            conv_out += Im_in[(in_row * dim_im_in + in_col)
                                * ch_im_in + i_ch_out]
                                * wt[(i_ker_y * dim_kernel + i_ker_x)
                                * ch_im_out + i_ch_out];
                        }
                    }
                }
                Im_out[(i_out_y * dim_im_out + i_out_x) * ch_im_out + i_ch_out]
                    = (q7_t) __SSAT((conv_out >> out_shift), 8);
            }
        }
    }

#endif                          /* CSI_MATH_DSP */

    /* Return to application */
    return;

}

/**
 * @} end of NNConv group
 */
