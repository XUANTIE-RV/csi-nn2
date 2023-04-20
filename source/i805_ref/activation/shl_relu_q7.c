/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
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
 * Title:        shl_relu_q7.c
 * Description:  Q7 version of ReLU
 *
 * -------------------------------------------------------------------- */

#include "i805_ref_function.h"

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

void shl_relu_q7(q7_t* data, uint16_t size)
{
    uint16_t i;

    for (i = 0; i < size; i++) {
        if (data[i] < 0) data[i] = 0;
    }
}
