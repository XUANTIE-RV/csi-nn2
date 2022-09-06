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

#include "i805_ref_function.h"

void shl_avepool_q7_HWC_nonsquare(q7_t *Im_in,                  // input image
                                  const uint16_t dim_im_in_x,   // input image dimension
                                  const uint16_t dim_im_in_y,   // input image dimension
                                  const uint16_t ch_im_in,      // number of input image channels
                                  const uint16_t dim_kernel_x,  // window kernel size
                                  const uint16_t dim_kernel_y,  // window kernel size
                                  const uint16_t padding_x,     // padding sizes
                                  const uint16_t padding_y,     // padding sizes
                                  const uint16_t stride_x,      // stride
                                  const uint16_t stride_y,      // stride
                                  const uint16_t dim_im_out_x,  // output image dimension
                                  const uint16_t dim_im_out_y,  // output image dimension
                                  q7_t *bufferA,                // a buffer for local storage
                                  q7_t *Im_out,                 // output feature
                                  const uint16_t out_lshift)    // output left shift (scaling)
{
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
                         k_x < i_x * stride_x - padding_x + dim_kernel_x; k_x++) {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_y && k_x < dim_im_in_x) {
                            sum += Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)];
                            count++;
                        }
                    }
                }
                sum = __SSAT_8((sum << out_lshift) / count);
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = sum;
            }
        }
    }
}
