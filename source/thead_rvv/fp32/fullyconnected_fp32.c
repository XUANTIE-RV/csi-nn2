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

#include "rvv/rvv.h"

/*************************************************************
 * packn = vlenb / sizeof(float)
 * n_blk: pack2n/packn/n_tail
 *
 * src: [n, k]
 * dst: [n/n_blk, k, n_blk]
 ************************************************************/
static void reorder_weight_npack2n_fp32(const float *src, float *dst, int n, int k)
{
    const int packn = csrr_vlenb() / sizeof(float);
    const int pack2n = packn * 2;

    int i = 0;
    int vl = vsetvl_e32m2(pack2n);
    for (; i + pack2n - 1 < n; i += pack2n) {
        const float *s_ptr = src + i * k;
        for (int j = 0; j < k; j++) {
            vfloat32m2_t _src = vlse32_v_f32m2(s_ptr, k * sizeof(float), vl);
            vse32_v_f32m2(dst, _src, vl);
            s_ptr += 1;
            dst += vl;
        }
    }
    while (i < n) {
        int vl = vsetvl_e32m1(n - i);
        const float *s_ptr = src + i * k;
        for (int j = 0; j < k; j++) {
            vfloat32m1_t _src = vlse32_v_f32m1(s_ptr, k * sizeof(float), vl);
            vse32_v_f32m1(dst, _src, vl);
            s_ptr += 1;
            dst += vl;
        }
        i += vl;
    }
}

void shl_rvv_fc_gemm_reorder_weight_fp32(struct csinn_tensor *weights)
{
    float *weight_data = (float *)weights->data;
    int n = weights->dim[0];  // out_nodes
    int k = weights->dim[1];  // in_nodes
    float *pa_reorder = (float *)shl_mem_alloc(n * k * sizeof(float));
    reorder_weight_npack2n_fp32(weight_data, pa_reorder, n, k);
    memcpy(weight_data, pa_reorder, n * k * sizeof(float));
    shl_mem_free(pa_reorder);
}

int shl_rvv_fullyconnected_gemm_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params)
{
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(input);
    }

    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *weights_data = (float *)weights->data;
    float *bias_data = (float *)bias->data;
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

    float *input_reorder = (float *)shl_mem_alloc(m * k * sizeof(float));
    shl_rvv_reorder_a_block_12xk_fp32(input_data, input_reorder, m, k, m, k);
    shl_rvv_gemm_a0b1_12xpack2n_fp32(output_data, input_reorder, weights_data, bias_data, m, k, n);

    shl_mem_free(input_reorder);
    return CSINN_TRUE;
}
