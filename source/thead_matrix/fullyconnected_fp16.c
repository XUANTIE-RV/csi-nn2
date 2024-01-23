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
 * mcols = rlenb / sizeof(__fp16)
 * msize_n: m2rows, mrows, n_tail
 * msize_k: mcols, k_tail
 *
 * src: [n, k]
 * dst: [n/msize_n, k/msize_k, msize_n, msize_k]
 ************************************************************/
static void reorder_weight_m2rows_mcols_fp16(__fp16 *src, __fp16 *dst, int n, int k)
{
    int mrows = csrr_xrlenb() / 4;
    int m2rows = mrows * 2;
    int mcols = m2rows;

    int r = 0;
    // m2rows
    mcfgm(mrows);
    for (; r + m2rows - 1 < n; r += m2rows) {
        int c = 0;
        while (c < k) {
            uint16_t msize_k = (k - c >= mcols) ? mcols : (k - c);
            mcfgk(msize_k * sizeof(__fp16));
            __fp16 *s0_ptr = src + r * k + c;
            __fp16 *s1_ptr = s0_ptr + mrows * k;
            mfloat16_t m0 = mld_f16(s0_ptr, k * sizeof(__fp16));
            mfloat16_t m1 = mld_f16(s1_ptr, k * sizeof(__fp16));
            msst_f16_mf16(dst, msize_k * sizeof(__fp16), m0);
            msst_f16_mf16(dst + mrows * msize_k, msize_k * sizeof(__fp16), m1);
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
            mcfgk(msize_k * sizeof(__fp16));
            __fp16 *s0_ptr = src + r * k + c;
            mfloat16_t m0 = mld_f16(s0_ptr, k * sizeof(__fp16));
            msst_f16_mf16(dst, msize_k * sizeof(__fp16), m0);
            dst += msize_n * msize_k;
            c += msize_k;
        }
        r += msize_n;
    }
}

/*************************************************************
 * mrows = rlenb / 4
 * m2rows = mrows * 2
 * mcols = rlenb / sizeof(__fp16)
 * msize_n: m2rows, mrows, n_tail
 * msize_k: mcols, k_tail
 *
 * src: [n, k]
 * dst: [n/msize_n, k/msize_k, msize_n, msize_k]
 ************************************************************/
static void reorder_weight_m2rows_mcols_fp16_w_int8(int8_t *src, int8_t *dst, int n, int k)
{
    int mrows = csrr_xrlenb() / 4;
    int m2rows = mrows * 2;
    int mcols = m2rows;

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

void shl_rvm_fc_gemm_reorder_weight_fp16(struct csinn_tensor *weights)
{
    __fp16 *weight_data = (__fp16 *)weights->data;
    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes
    __fp16 *weight_reorder = (__fp16 *)shl_mem_alloc(n * k * sizeof(__fp16));
    reorder_weight_m2rows_mcols_fp16(weight_data, weight_reorder, n, k);
    memcpy(weight_data, weight_reorder, n * k * sizeof(__fp16));
    shl_mem_free(weight_reorder);
}

void shl_rvm_fc_gemm_reorder_weight_fp16_w_int8(struct csinn_tensor *weights)
{
    int8_t *weight_data = (int8_t *)weights->data;
    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes
    int8_t *weight_reorder = (int8_t *)shl_mem_alloc(n * k * sizeof(int8_t));
    reorder_weight_m2rows_mcols_fp16_w_int8(weight_data, weight_reorder, n, k);
    memcpy(weight_data, weight_reorder, n * k * sizeof(int8_t));
    shl_mem_free(weight_reorder);
}

int shl_rvm_fullyconnected_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params)
{
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(input);
    }

    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *weights_data = NULL;
    __fp16 *bias_data = (__fp16 *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    int batches = 1;
    /* compute the outer size */
    for (int i = 0; i < output_dims_count - 1; i++) {
        batches *= output->dim[i];
    }
    int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    int m = batches;
    int n = output_depth;
    int k = accum_depth;

    __fp16 *weights_fp16 = NULL;
    if (weights->is_const && weights->dtype == CSINN_DTYPE_INT8) {
        int size = csinn_tensor_size(weights);
        int8_t *weights_int8 = (int8_t *)weights->data;
        weights_fp16 = (__fp16 *)shl_mem_alloc(size * sizeof(__fp16));
        if (weights->quant_channel == 1) {
            int32_t zp = weights->qinfo->zero_point;
            float scale = weights->qinfo->scale;
            shl_rvv_dequantize_i8_to_f16(weights_int8, weights_fp16, size, zp, scale);
        } else if (weights->quant_channel == output_depth) {
            // support channel quantization
            for (int c = 0; c < output_depth; c++) {
                int32_t zp = weights->qinfo[c].zero_point;
                float scale = weights->qinfo[c].scale;
                shl_rvv_dequantize_i8_to_f16(weights_int8 + c * accum_depth,
                                             weights_fp16 + c * accum_depth, accum_depth, zp,
                                             scale);
            }
        }
        weights_data = weights_fp16;
    } else if (weights->dtype == CSINN_DTYPE_FLOAT16) {
        weights_data = (__fp16 *)weights->data;
    } else {
        shl_debug_error("weights unsupport dtype: %d\n", weights->dtype);
        return CSINN_FALSE;
    }

    shl_rvm_gemm_a0b1_fp16(output_data, input_data, weights_data, bias_data, m, k, n);

    if (weights->is_const && weights->dtype == CSINN_DTYPE_INT8) {
        shl_mem_free(weights_fp16);
        return CSINN_TRUE;
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, weights);
    return CSINN_TRUE;
}
