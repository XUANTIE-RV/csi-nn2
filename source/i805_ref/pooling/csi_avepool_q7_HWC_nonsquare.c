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

#include "csi_math.h"
#include "csi_nnfunctions.h"
#include "csi_nnsupportfunctions.h"

void csi_avepool_q7_HWC_nonsquare(
    q7_t *Im_in,                 // input image
    const uint16_t dim_im_in_x,  // input image dimension
    const uint16_t dim_im_in_y,  // input image dimension
    const uint16_t ch_im_in,     // number of input image channels
    const uint16_t dim_kernel_x, // window kernel size
    const uint16_t dim_kernel_y, // window kernel size
    const uint16_t padding_x,    // padding sizes
    const uint16_t padding_y,    // padding sizes
    const uint16_t stride_x,     // stride
    const uint16_t stride_y,     // stride
    const uint16_t dim_im_out_x, // output image dimension
    const uint16_t dim_im_out_y, // output image dimension
    q7_t *bufferA,               // a buffer for local storage
    q7_t *Im_out,                // output feature
    const uint16_t out_lshift)   // output left shift (scaling)
{
#if defined (CSI_MATH_DSP)

    q15_t    *buffer = (q15_t *) bufferA;
    int16_t   i_x, i_y, i;
    int16_t   count = 0;

    /* first does the pooling along x axis */
    for (i_y = 0; i_y < dim_im_in_y; i_y++)
    {

        for (i_x = 0; i_x < dim_im_out_x; i_x++)
        {
            /* for each output pixel */
            q7_t     *target = Im_in + (i_y * dim_im_in_x + i_x) * ch_im_in;
            q7_t     *win_start;
            q7_t     *win_stop;
            if (i_x * stride_x - padding_x < 0)
            {
                win_start = target;
            } else
            {
                win_start = Im_in + (i_y * dim_im_in_x + i_x * stride_x
                                     - padding_x) * ch_im_in;
            }

            if (i_x * stride_x - padding_x + dim_kernel_x >= dim_im_in_x)
            {
                win_stop = Im_in + (i_y * dim_im_in_x + dim_im_in_x) * ch_im_in;
            } else
            {
                win_stop = Im_in + (i_y * dim_im_in_x + i_x * stride_x - padding_x
                                    + dim_kernel_x) * ch_im_in;
            }

            /* first step is to copy over initial data */
            csi_q7_to_q15_no_shift(win_start, buffer, ch_im_in);
            count = 1;

            /* start the max operation from the second part */
            win_start += ch_im_in;
            for (; win_start < win_stop; win_start += ch_im_in)
            {
                accumulate_q7_to_q15(buffer, win_start, ch_im_in);
                count++;
            }
            buffer_scale_back_q15_to_q7(buffer, target, ch_im_in, count);
        }
    }

    /* then does the pooling along y axis */
    for (i_y = 0; i_y < dim_im_out_y; i_y++)
    {
        /* for each output row */
        q7_t     *target = Im_out + i_y * dim_im_out_x * ch_im_in;
        q7_t     *row_start;
        q7_t     *row_end;
        /* setting the starting row */
        if (i_y * stride_y - padding_y < 0)
        {
            row_start = Im_in;
        } else
        {
            row_start = Im_in + (i_y * stride_y - padding_y) * dim_im_in_x * ch_im_in;
        }
        /* setting the stopping row */
        if (i_y * stride_y - padding_y + dim_kernel_y >= dim_im_in_y)
        {
            row_end = Im_in + dim_im_in_x * dim_im_in_y * ch_im_in;
        } else
        {
            row_end = Im_in + (i_y * stride_y - padding_y + dim_kernel_y)
                * dim_im_in_x * ch_im_in;
        }

        /* copy over the first row */
        csi_q7_to_q15_no_shift(row_start, buffer, dim_im_out_x * ch_im_in);
        count = 1;

        /* move over to next row */
        row_start += ch_im_in * dim_im_in_x;

        for (; row_start < row_end; row_start += dim_im_in_x * ch_im_in)
        {
            accumulate_q7_to_q15(buffer, row_start, dim_im_out_x * ch_im_in);
            count++;
        }

        /* out left shift */
        for(i = 0; i < dim_im_out_x * ch_im_in; i++)
        {
            buffer[i] =  buffer[i] << out_lshift;
        }
        buffer_scale_back_q15_to_q7(buffer, target,
                                    dim_im_out_x * ch_im_in, count);
    }
#else

    int16_t i_ch_in, i_x, i_y;
    int16_t k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++) {
        for (i_y = 0; i_y < dim_im_out_y; i_y++) {
            for (i_x = 0; i_x < dim_im_out_x; i_x++) {
                int sum = 0;
                int count = 0;
                for (k_y = i_y * stride_y - padding_y;
                     k_y < i_y * stride_y - padding_y + dim_kernel_y; k_y++) {
                    for (k_x = i_x * stride_x - padding_x;
                         k_x < i_x * stride_x - padding_x + dim_kernel_x;
                         k_x++) {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_y &&
                            k_x < dim_im_in_x) {
                            sum += Im_in[i_ch_in +
                                         ch_im_in * (k_x + k_y * dim_im_in_x)];
                            count++;
                        }
                    }
                }
                    sum = __SSAT_8((sum << out_lshift) / count);
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = sum;
            }
        }
    }

#endif
}
