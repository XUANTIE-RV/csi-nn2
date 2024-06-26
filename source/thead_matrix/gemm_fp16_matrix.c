/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

#include "rvm/rvm.h"

void gemm_fp16_nhwc_matrix_rowxn(__fp16 *output, const __fp16 *kernel, const __fp16 *input,
                                 const __fp16 *bias, int row, int k, int n);
void gemm_fp16_nhwc_matrix_2rowxn(__fp16 *output, const __fp16 *kernel, const __fp16 *input,
                                  const __fp16 *bias, int row, int k, int n);
void gemm_fp16_nhwc_matrix_row_tailxn(__fp16 *output, const __fp16 *kernel, const __fp16 *input,
                                      const __fp16 *bias, int row, int k, int n);

void shl_rvm_nhwc_gemm_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, const __fp16 *bias,
                            int m, int k, int n)
{
    int mrows = csrr_xrlenb() / 4;
    int m2rows = mrows * 2;

    __fp16 *bias_shadow = NULL;
    if (bias == NULL) {
        bias_shadow = (__fp16 *)shl_mem_alloc(n * sizeof(__fp16));
        bias = bias_shadow;
    }
    int hw = 0;
    for (; hw + m2rows - 1 < m; hw += m2rows) {
        gemm_fp16_nhwc_matrix_2rowxn(dst, sa, sb, bias, mrows, k, n);
        sb += m2rows * k;
        dst += m2rows * n;
    }
    for (; hw + mrows - 1 < m; hw += mrows) {
        gemm_fp16_nhwc_matrix_rowxn(dst, sa, sb, bias, mrows, k, n);
        sb += mrows * k;
        dst += mrows * n;
    }
    if (hw < m) {
        gemm_fp16_nhwc_matrix_row_tailxn(dst, sa, sb, bias, m - hw, k, n);
    }
    if (bias_shadow) {
        shl_mem_free(bias_shadow);
        bias_shadow = NULL;
    }
}
