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

#include "rvv/rvv.h"

#ifdef SHL_USE_DOT_INT8
/*************************************************************
 * mf2 = vlenb / sizeof(int8_t) / 2
 * src: [n, k]
 * dst:
 *   k % 4 == 0: [n/mf2, k/4, mf2, 4]
 *   k_tail    : [n/mf2, k_tail, mf2]
 ************************************************************/
static void reorder_weight_nmf2z4_int8_dot(int8_t *src, int8_t *dst, int n, int k)
{
    int j = 0;
    while (j < n) {
        int vl = vsetvl_e8mf2(n - j);
        int8_t *s_ptr = src + j * k;
        int c = 0;
        for (; c + 3 < k; c += 4) {
            for (int i = 0; i < 4; i++) {
                vint8mf2_t _src = vlse8_v_i8mf2(s_ptr, k * sizeof(int8_t), vl);
                s_ptr += 1;
                vsse8_v_i8mf2(dst + i, 4 * sizeof(int8_t), _src, vl);
            }
            dst += 4 * vl;
        }
        // k_tail
        for (; c < k; c++) {
            vint8mf2_t _src = vlse8_v_i8mf2(s_ptr, k * sizeof(int8_t), vl);
            vse8_v_i8mf2(dst, _src, vl);
            s_ptr += 1;
            dst += vl;
        }
        j += vl;
    }
}
#else
/*************************************************************
 * packn = vlenb / sizeof(int8_t)
 * n_blk: packn/n_tail
 *
 * src: [n, k]
 * dst: [n/n_blk, k, n_blk]
 ************************************************************/
static void reorder_weight_npackn_int8(int8_t *b, int8_t *sb, int n, int k)
{
    int i = 0;
    while (i < n) {
        int vl = vsetvl_e8m1(n - i);
        for (int j = 0; j < k; j++) {
            int8_t *in_ptr = b + j;
            vint8m1_t _input = vlse8_v_i8m1(in_ptr, k * sizeof(int8_t), vl);
            vse8_v_i8m1(sb, _input, vl);
            sb += vl;
        }
        b += vl * k;
        i += vl;
    }
}
#endif  // SHL_USE_DOT_INT8

void shl_rvv_fc_gemm_reorder_weight_int8(struct csinn_tensor *weights)
{
    int8_t *weight_data = (int8_t *)weights->data;
    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes
    int8_t *kernel_reorder = (int8_t *)shl_mem_alloc(n * k * sizeof(int8_t));
#ifdef SHL_USE_DOT_INT8
    reorder_weight_nmf2z4_int8_dot(weight_data, kernel_reorder, n, k);
#else
    reorder_weight_npackn_int8(weight_data, kernel_reorder, n, k);
#endif  // SHL_USE_DOT_INT8
    memcpy(weight_data, kernel_reorder, n * k * sizeof(int8_t));
    shl_mem_free(kernel_reorder);
}

int shl_rvv_fullyconnected_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
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

    int8_t *input_reorder = (int8_t *)shl_mem_alloc(m * k * sizeof(int8_t));

#ifdef SHL_USE_DOT_INT8
    shl_rvv_matmul_reorder_mat0_n8z4_int8_dot(input_data, input_reorder, m, k, k);
    shl_rvv_gemm_a0b1_8xmf2_int8_dot(output_data, input_reorder, weights_data, bias_data, m, k, n,
                                     output->qinfo->zero_point, multiplier, shift);
#else
    shl_rvv_matmul_reorder_mat0_n4_int8(input_data, input_reorder, m, k, k);
    shl_rvv_gemm_a0b1_4xpackn_int8(output_data, input_reorder, weights_data, bias_data, m, k, n,
                                   output->qinfo->zero_point, multiplier, shift);
#endif  // SHL_USE_DOT_INT8

    shl_mem_free(multiplier);
    shl_mem_free(shift);
    shl_mem_free(input_reorder);
    return CSINN_TRUE;
}
