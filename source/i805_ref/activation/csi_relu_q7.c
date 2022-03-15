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
 * Title:        csi_relu_q7.c
 * Description:  Q7 version of ReLU
 *
 * -------------------------------------------------------------------- */

#include "csi_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup Acti
 * @{
 */

  /**
   * @brief Q7 RELU function
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @return none.
   * 
   * @details
   *
   * Optimized relu with QSUB instructions.
   *
   */

void csi_relu_q7(q7_t * data, uint16_t size)
{

#if defined (CSI_MATH_DSP)

    uint16_t  i = size >> 2;
    q7_t     *pIn = data;
    q7_t     *pOut = data;
    q31_t     in;
    q31_t     buf;
    q31_t     mask;

    while (i)
    {
        in = *__SIMD32(pIn)++;

        /* extract the first bit */
        buf = __ROR(in & 0x80808080, 7);

        /* if MSB=1, mask will be 0xFF, 0x0 otherwise */
        mask = __QSUB8(0x00000000, buf);

        *__SIMD32(pOut)++ = in & (~mask);
        i--;
    }

    i = size & 0x3;
    while (i)
    {
        if (*pIn < 0)
        {
            *pIn = 0;
        }
        pIn++;
        i--;
    }

#else

    uint16_t  i;

    for (i = 0; i < size; i++)
    {
        if (data[i] < 0)
            data[i] = 0;
    }

#endif                          /* CSI_MATH_DSP */

}

/**
 * @} end of Acti group
 */
