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

#include "shl_c920.h"

/*************************************************************
  Matmul fp32 performance on C920@1.848GHz
  -------------------------
  |      mkn     | GFlops |
  |-----------------------|
  |   49,32,49   |  4.35  |
  |  64,192,576  |  10.32 |
  | 384,512,512  |  10.25 |
  | 196,1536,384 |  9.98  |
  -------------------------
 ************************************************************/

#define MATMUL_M_BLK 64
#define MATMUL_K_BLK 64
#define MATMUL_N_BLK 64

int shl_c920_matmul_fp32(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                         struct csinn_tensor *output, struct csinn_matmul_params *params)
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
    }

    // /* compute the outer size */
    for (int i = 0; i < mat1->dim_count - 2; i++) {
        batches_b *= mat1->dim[i];
    }

    const int dim_m = mat0->dim[dims_count - (params->trans_a ? 1 : 2)];
    const int dim_k = mat0->dim[dims_count - (params->trans_a ? 2 : 1)];
    const int dim_n = mat1->dim[mat1->dim_count - (params->trans_b ? 2 : 1)];

    if (batches_a == batches_b) {
        if (!params->trans_a && !params->trans_b) {
            float *in0 = (float *)shl_mem_alloc(dim_m * dim_k * sizeof(float));
            float *in1;
            if (!(mat1->is_const)) {
                in1 = (float *)shl_mem_alloc(dim_k * dim_n * sizeof(float));
            }

            for (int b = 0; b < batches_a; b++) {
                shl_c920_reorder_kernel_block_8xk_fp32(mat0_data, in0, dim_m, dim_k, MATMUL_M_BLK,
                                                       MATMUL_K_BLK);
                if (!(mat1->is_const)) {
                    shl_rvv_reorder_input_block_pack2nxk_fp32(mat1_data, in1, dim_k, dim_n,
                                                              MATMUL_K_BLK, MATMUL_N_BLK);
                } else {
                    in1 = mat1_data;
                }

                shl_c920_gemm_block_8xpack2n_fp32(output_data, in0, in1, NULL, dim_m, dim_k, dim_n,
                                                  MATMUL_M_BLK, MATMUL_K_BLK, MATMUL_N_BLK);

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
                shl_rvv_reorder_input_block_pack2nxk_fp32(mat1_data, in1, dim_k, dim_n,
                                                          MATMUL_K_BLK, MATMUL_N_BLK);
            } else {
                in1 = mat1_data;
            }

            for (int b = 0; b < batches_a; b++) {
                shl_c920_reorder_kernel_block_8xk_fp32(mat0_data, in0, dim_m, dim_k, MATMUL_M_BLK,
                                                       MATMUL_K_BLK);

                shl_c920_gemm_block_8xpack2n_fp32(output_data, in0, in1, NULL, dim_m, dim_k, dim_n,
                                                  MATMUL_M_BLK, MATMUL_K_BLK, MATMUL_N_BLK);

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

int shl_c920_matmul_init_fp32(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                              struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    if (mat0->dtype == CSINN_DTYPE_FLOAT32) {
        if (mat1->dtype == CSINN_DTYPE_FLOAT32) {
            if (mat1->is_const) {
                shl_rvv_matmul_reorder_weight_fp32(mat1, MATMUL_K_BLK, MATMUL_N_BLK);
            }
            cb->exec = shl_c920_matmul_fp32;
        } else {
            shl_debug_error("mat1 unsupported dtype: %d\n", mat1->dtype);
            return CSINN_FALSE;
        }
    } else {
        shl_debug_error("mat0 unsupported dtype: %d\n", mat0->dtype);
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}
