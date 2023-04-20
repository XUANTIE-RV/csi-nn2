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

/* SHL version 2.1.x */

#include "shl_c908.h"

void gemm_int16_ncxhwx_12xpackn(int32_t *output, const int16_t *kernel, const int16_t *input, int k,
                                int n);

void shl_c908_ncxhwx_gemm_12xpackn_int16(int32_t *dst, const int16_t *sa, const int16_t *sb, int m,
                                         int k, int n)
{
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;

    int oc = 0;
    for (; oc + packn - 1 < m; oc += packn) {
        gemm_int16_ncxhwx_12xpackn(dst, sa, sb, k, n);
        sa += packn * k;
        dst += packn * n;
    }
}
