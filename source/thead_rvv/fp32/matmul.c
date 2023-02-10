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

#include "shl_thead_rvv.h"

#define MATMUL_M_BLK 32
#define MATMUL_K_BLK 64
#define MATMUL_N_BLK 64

int shl_rvv_matmul_block_fp32(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                              struct csinn_tensor *output, struct csinn_matmul_params *params,
                              const int M_BLK, const int K_BLK, const int N_BLK)
{
    float *mat0_data = (float *)mat0->data;
    float *mat1_data = (float *)mat1->data;
    float *output_data = (float *)output->data;

    const int dims_count = mat0->dim_count;
    int batches_a = 1;
    int batches_b = 1;

    /* compute the outer size */
    for (int i = 0; i < dims_count - 2; i++) {
        batches_a *= mat0->dim[i];
        batches_b *= mat1->dim[i];
    }

    const int dim_m = mat0->dim[dims_count - (params->trans_a ? 1 : 2)];
    const int dim_k = mat0->dim[dims_count - (params->trans_a ? 2 : 1)];
    const int dim_n = mat1->dim[dims_count - (params->trans_b ? 2 : 1)];

    if (batches_a == batches_b) {
        if (!params->trans_a && !params->trans_b) {
            float *in0 = (float *)shl_mem_alloc(dim_m * dim_k * sizeof(float));
            float *in1;
            if (!(mat1->is_const)) {
                in1 = (float *)shl_mem_alloc(dim_k * dim_n * sizeof(float));
            }

            for (int b = 0; b < batches_a; b++) {
                shl_rvv_reorder_kernel_block_12xk_fp32(mat0_data, in0, dim_m, dim_k, M_BLK, K_BLK);
                if (!(mat1->is_const)) {
                    shl_rvv_reorder_input_block_pack2nxk_fp32(mat1_data, in1, dim_k, dim_n, K_BLK,
                                                              N_BLK);
                } else {
                    in1 = mat1_data;
                }

                shl_rvv_gemm_block_12xpack2n_fp32(output_data, in0, in1, NULL, dim_m, dim_k, dim_n,
                                                  M_BLK, K_BLK, N_BLK);

                mat0_data += dim_m * dim_k;
                mat1_data += dim_k * dim_n;
                output_data += dim_m * dim_n;
            }
            shl_mem_free(in0);
            if (!(mat1->is_const)) {
                shl_mem_free(in1);
            }
        } else {
            shl_ref_matmul_quant(mat0, mat1, output, params);
        }
    } else if (batches_a > 1 && batches_b == 1) {
        if (!params->trans_a && !params->trans_b) {
            float *in0 = (float *)shl_mem_alloc(dim_m * dim_k * sizeof(float));
            float *in1;
            if (!(mat1->is_const)) {
                in1 = (float *)shl_mem_alloc(dim_k * dim_n * sizeof(float));
                shl_rvv_reorder_input_block_pack2nxk_fp32(mat1_data, in1, dim_k, dim_n, K_BLK,
                                                          N_BLK);
            } else {
                in1 = mat1_data;
            }

            for (int b = 0; b < batches_a; b++) {
                shl_rvv_reorder_kernel_block_12xk_fp32(mat0_data, in0, dim_m, dim_k, M_BLK, K_BLK);

                shl_rvv_gemm_block_12xpack2n_fp32(output_data, in0, in1, NULL, dim_m, dim_k, dim_n,
                                                  M_BLK, K_BLK, N_BLK);

                mat0_data += dim_m * dim_k;
                output_data += dim_m * dim_n;
            }
            shl_mem_free(in0);
            if (!(mat1->is_const)) {
                shl_mem_free(in1);
            }
        } else {
            shl_debug_error("matmul unsupported this broadcast\n");
            return CSINN_FALSE;
        }
    } else {
        shl_debug_error("matmul unsupported this broadcast\n");
        return CSINN_FALSE;
    }

    return CSINN_TRUE;
}

/*************************************************************
 * packn = vlenb / sizeof(__fp16)
 * src: [k, n]
 * dst: [n/n_blk, k/k_blk, n_blk/pack2n, k_blk, pack2n]
 * n_blk: N_BLK, N_BLK/2, N_BLK/4, ..., pack2n
 * k_blk: K_BLK, K_tail
 ************************************************************/
void shl_rvv_matmul_reorder_weight_fp32(struct csinn_tensor *mat1, const int K_BLK, const int N_BLK)
{
    float *mat1_data = (float *)mat1->data;
    int dims_count = mat1->dim_count;
    int batch = 1;
    for (int i = 0; i < dims_count - 2; i++) {
        batch *= mat1->dim[i];
    }
    const int k = mat1->dim[dims_count - 2];
    const int n = mat1->dim[dims_count - 1];
    float *mat_reorder = (float *)shl_mem_alloc(k * n * sizeof(float));

    for (int b = 0; b < batch; b++) {
        float *init_mat = mat1_data + b * k * n;
        shl_rvv_reorder_input_block_pack2nxk_fp32(init_mat, mat_reorder, k, n, K_BLK, N_BLK);
        memcpy(init_mat, mat_reorder, k * n * sizeof(float));
    }

    shl_mem_free(mat_reorder);
}

int shl_rvv_matmul_fp32(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                        struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    return shl_rvv_matmul_block_fp32(mat0, mat1, output, params, MATMUL_M_BLK, MATMUL_K_BLK,
                                     MATMUL_N_BLK);
}

int shl_rvv_matmul_init_fp32(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                             struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    if (mat0->dtype == CSINN_DTYPE_FLOAT32) {
        if (mat1->dtype == CSINN_DTYPE_FLOAT32) {
            if (mat1->is_const) {
                shl_rvv_matmul_reorder_weight_fp32(mat1, MATMUL_K_BLK, MATMUL_N_BLK);
            }
            cb->exec = shl_rvv_matmul_fp32;
        } else {
            shl_debug_error("mat1 unsupported dtype: %d\n", mat1->dtype);
            return CSINN_FALSE;
        }
    } else {
        shl_debug_error("mat0 unsupported dtype: %d\n", mat0->dtype);
    }
    return CSINN_TRUE;
}
