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

#include "rvm/rvm.h"

/*************************************************************
 * mrows = rlenb / 4
 * m2rows = mrows * 2
 * mcols = rlenb / sizeof(int8_t)
 * msize_n: m2rows, mrows, n_tail
 * msize_k: mcols, k_tail
 *
 * src: [n, k]
 * dst: [n/msize_n, k/msize_k, msize_n, msize_k]
 ************************************************************/
static void reorder_weight_m2rows_mcols_int8(int8_t *src, int8_t *dst, int n, int k)
{
    int mcols = csrr_xrlenb();
    int mrows = mcols / 4;
    int m2rows = mrows * 2;

    int r = 0;
    // m2rows
    mcfgm(mrows);
    for (; r + m2rows - 1 < n; r += m2rows) {
        int c = 0;
        while (c < k) {
            uint16_t msize_k = (k - c >= mcols) ? mcols : (k - c);
            mcfgk(msize_k * sizeof(int8_t));
            int8_t *s0_ptr = src + r * k + c;
            int8_t *s1_ptr = s0_ptr + mrows * k;
            mint8_t m0 = mld_i8(s0_ptr, k * sizeof(int8_t));
            mint8_t m1 = mld_i8(s1_ptr, k * sizeof(int8_t));
            msst_i8_mi8(dst, msize_k * sizeof(int8_t), m0);
            msst_i8_mi8(dst + mrows * msize_k, msize_k * sizeof(int8_t), m1);
            dst += m2rows * msize_k;
            c += msize_k;
        }
    }
    while (r < n) {
        int msize_n = (n - r >= mrows) ? mrows : (n - r);
        mcfgm(msize_n);
        int c = 0;
        while (c < k) {
            uint16_t msize_k = (k - c >= mcols) ? mcols : (k - c);
            mcfgk(msize_k * sizeof(int8_t));
            int8_t *s0_ptr = src + r * k + c;
            mint8_t m0 = mld_i8(s0_ptr, k * sizeof(int8_t));
            msst_i8_mi8(dst, msize_k * sizeof(int8_t), m0);
            dst += msize_n * msize_k;
            c += msize_k;
        }
        r += msize_n;
    }
}

void shl_rvm_fc_gemm_reorder_weight_int8(struct csinn_tensor *weights)
{
    int8_t *weight_data = (int8_t *)weights->data;

    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes

    int8_t *weight_reorder = (int8_t *)shl_mem_alloc(n * k * sizeof(int8_t));
    reorder_weight_m2rows_mcols_int8(weight_data, weight_reorder, n, k);
    memcpy(weight_data, weight_reorder, n * k * sizeof(int8_t));
    shl_mem_free(weight_reorder);
}

int shl_rvm_fullyconnected_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params)
{
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(input);
    }

    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *weights_data = (int8_t *)weights->data;
    int32_t *bias_data = (int32_t *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    int batches = 1;
    /* compute the outer size */
    for (int i = 0; i < output_dims_count - 1; i++) {
        batches *= output->dim[i];
    }
    const int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    const int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    int m = batches;
    int n = output_depth;
    int k = accum_depth;

    int32_t *multiplier = (int32_t *)shl_mem_alloc(n * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(n * sizeof(int32_t));
    if (weights->quant_channel > 1) {
        for (int c = 0; c < n; c++) {
            multiplier[c] = weights->qinfo[c].multiplier;
            shift[c] = -1 - weights->qinfo[c].shift;
        }
    } else if (weights->quant_channel == 1) {
        for (int c = 0; c < n; c++) {
            multiplier[c] = weights->qinfo[0].multiplier;
            shift[c] = -1 - weights->qinfo[0].shift;
        }
    }

#ifdef MATRIX_PW_I32
    shl_rvm_gemm_a0b1_int8_pw_i32(output_data, input_data, weights_data, bias_data, m, k, n,
                                  output->qinfo->zero_point, multiplier, shift);
#else
    shl_rvm_gemm_a0b1_int8_to_int32(output_data, input_data, weights_data, bias_data, m, k, n,
                                    output->qinfo->zero_point, multiplier, shift);
#endif  // MATRIX_PW_I32

    shl_mem_free(multiplier);
    shl_mem_free(shift);
    return CSINN_TRUE;
}
