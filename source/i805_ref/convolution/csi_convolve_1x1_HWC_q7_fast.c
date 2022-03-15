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
 * Title:        csi_convolve_1x1_HWC_q7_fast_nonsquare.c
 * Description:  Fast Q7 version of 1x1 convolution (non-square shape)
 *
 * -------------------------------------------------------------------- */

#include "csi_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

/**
 * @brief Fast Q7 version of 1x1 convolution (non-sqaure shape)
 * @param[in]       Im_in        pointer to input tensor
 * @param[in]       dim_im_in_x  input tensor dimention x
 * @param[in]       dim_im_in_y  input tensor dimention y
 * @param[in]       ch_im_in     number of input tensor channels
 * @param[in]       wt           pointer to kernel weights
 * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
 * @param[in]       bias         pointer to bias
 * @param[in]       bias_shift   amount of left-shift for bias
 * @param[in]       out_shift    amount of right-shift for output
 * @param[in,out]   Im_out       pointer to output tensor
 * @param[in]       dim_im_out_x output tensor dimension x
 * @param[in]       dim_im_out_y output tensor dimension y
 * @param[in,out]   bufferA      pointer to buffer space for input
 * @return     The function returns either
 * <code>CSI_MATH_SIZE_MISMATCH</code> or <code>CSI_MATH_SUCCESS</code> based on the outcome of size checking.
 *
 * This function is optimized for convolution with 1x1 kernel size.
 * It can be used for the second half of MobileNets [1] after depthwise
 * separable convolution.
 *
 * This function is the version with full list of optimization tricks, but with
 * some contraints:
 *   ch_im_in is multiple of 4
 *   ch_im_out is multiple of 2
 *
 * [1] MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
 * https://arxiv.org/abs/1704.04861
 */

void csi_convolve_1x1_HWC_q7_fast(const q7_t * Im_in,
                                          const uint16_t dim_im_in_x,
                                          const uint16_t dim_im_in_y,
                                          const uint16_t ch_im_in,
                                          const q7_t * wt,
                                          const uint16_t ch_im_out,
                                          const q7_t * bias,
                                          const uint16_t bias_shift,
                                          const uint16_t out_shift,
                                          q7_t * Im_out,
                                          const uint16_t dim_im_out_x,
                                          const uint16_t dim_im_out_y,
                                          q15_t * bufferA)
{

#if defined (CSI_MATH_DSP)

    int16_t   i_out_y, i_out_x;
    int16_t   i_ch_out;

    /* -----------------------
     *  Here we use bufferA as q15_t internally as computation are done with q15_t level
     *  im2col are done to output in q15_t format from q7_t input
     */

    q15_t    *pBuffer = bufferA;
    q7_t     *pOut = Im_out;

    if (ch_im_in % 4 != 0 || ch_im_out % 2 != 0)
    {
        /* check if the input dimension meets the constraints */
        return;
    }

    for (i_out_y = 0; i_out_y < dim_im_out_y; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
        {
            /* This part implements the im2col function */
            csi_q7_to_q15_reordered_no_shift((q7_t *) Im_in +
                                              (i_out_y * dim_im_in_x + i_out_x)
                                              * ch_im_in, pBuffer,
                                             ch_im_in);
            pBuffer += ch_im_in;

            if (pBuffer == bufferA + 2 * ch_im_in)
            {
                pOut = csi_nn_mat_mult_kernel_q7_q15_reordered(wt, bufferA,
                                                                ch_im_out,
                                                                ch_im_in,
                                                                bias_shift,
                                                                out_shift,
                                                                bias, pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /* check if there is left-over for compute */
    if (pBuffer != bufferA)
    {
        const q7_t *pA = wt;
        for (i_ch_out = 0; i_ch_out < ch_im_out; i_ch_out++)
        {
            q31_t sum = ((q31_t)(bias[i_ch_out]) << bias_shift) +
                        NN_ROUND(out_shift);
            q15_t *pB = bufferA;
            /* basically each time it process 4 entries */
            uint16_t  colCnt = ch_im_in >> 2;

            while (colCnt)
            {

                q31_t     inA1, inA2;
                q31_t     inB1, inB2;

                pA = (const q7_t *)read_and_pad_reordered((void *)pA, &inA1,
                                                          &inA2);

                inB1 = *__SIMD32(pB)++;
                sum = __SMLAD(inA1, inB1, sum);
                inB2 = *__SIMD32(pB)++;
                sum = __SMLAD(inA2, inB2, sum);

                colCnt--;
            }
            colCnt = ch_im_in  & 0x3;
            while (colCnt)
            {
                q7_t      inA1 = *pA++;
                q15_t     inB1 = *pB++;
                sum += inA1 * inB1;
                colCnt--;
            }
            *pOut = (q7_t) __SSAT((sum >> out_shift), 8);
            pOut++;

        }

    }

#else

    int       i, j, k, l;
    int       conv_out;
    int       in_row, in_col;

     for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out_y; j++)
        {
            for (k = 0; k < dim_im_out_x; k++)
            {
                conv_out = ((q31_t)(bias[i]) << bias_shift)
                             + NN_ROUND(out_shift);
                // if-for implementation
                in_row = j;
                in_col = k;
                if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in_y
                    && in_col < dim_im_in_x)
                {
                    for (l = 0; l < ch_im_in; l++)
                    {
                        conv_out += Im_in[(in_row * dim_im_in_x
                                           + in_col) * ch_im_in + l] *
                            wt[i * ch_im_in + l];
                    }
                }
                Im_out[i + (j * dim_im_out_x + k) * ch_im_out] =
                    (q7_t) __SSAT((conv_out >> out_shift), 8);
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
