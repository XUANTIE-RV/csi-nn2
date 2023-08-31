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

#include "c920/c920.h"

/*************************************************************
  Matmul fp16 performance on C920@1.848GHz
  -------------------------
  |      mkn     | GFlops |
  |-----------------------|
  |   49,32,49   |  7.83  |
  |  64,192,576  |  21.57 |
  | 384,512,512  |  22.49 |
  | 196,1536,384 |  22.43 |
  -------------------------
 ************************************************************/

#define MATMUL_M_BLK 64
#define MATMUL_K_BLK 128
#define MATMUL_N_BLK 64

int shl_c920_matmul_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                         struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    if (mat0->layout >= CSINN_LAYOUT_NC1C0 && mat0->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(mat0);
    }
    if (mat1->layout >= CSINN_LAYOUT_NC1C0 && mat1->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(mat1);
    }

    __fp16 *mat0_data = (__fp16 *)mat0->data;
    __fp16 *mat1_data = (__fp16 *)mat1->data;
    __fp16 *output_data = (__fp16 *)output->data;

    const int dims_count = mat0->dim_count;
    int batches_a = 1;
    int batches_b = 1;

    /* compute the outer size */
    for (int i = 0; i < dims_count - 2; i++) {
        batches_a *= mat0->dim[i];
    }
    for (int i = 0; i < mat1->dim_count - 2; i++) {
        batches_b *= mat1->dim[i];
    }

    const int dim_m = mat0->dim[dims_count - (params->trans_a ? 1 : 2)];
    const int dim_k = mat0->dim[dims_count - (params->trans_a ? 2 : 1)];
    const int dim_n = mat1->dim[mat1->dim_count - (params->trans_b ? 2 : 1)];

    if (!params->trans_a && !params->trans_b) {
        if (batches_a == batches_b) {
            __fp16 *in0 = (__fp16 *)shl_mem_alloc(dim_m * dim_k * sizeof(__fp16));
            __fp16 *in1;
            if (!(mat1->is_const)) {
                in1 = (__fp16 *)shl_mem_alloc(dim_k * dim_n * sizeof(__fp16));
            }

            for (int b = 0; b < batches_a; b++) {
                shl_c920_reorder_kernel_block_8xk_fp16(mat0_data, in0, dim_m, dim_k, MATMUL_M_BLK,
                                                       MATMUL_K_BLK);
                if (!(mat1->is_const)) {
                    shl_rvv_reorder_input_block_pack2nxk_fp16(mat1_data, in1, dim_k, dim_n,
                                                              MATMUL_K_BLK, MATMUL_N_BLK);
                } else {
                    in1 = mat1_data;
                }

                shl_c920_gemm_block_8xpack2n_fp16(output_data, in0, in1, NULL, dim_m, dim_k, dim_n,
                                                  MATMUL_M_BLK, MATMUL_K_BLK, MATMUL_N_BLK);

                mat0_data += dim_m * dim_k;
                mat1_data += dim_k * dim_n;
                output_data += dim_m * dim_n;
            }
            shl_mem_free(in0);
            if (!(mat1->is_const)) {
                shl_mem_free(in1);
            }
            // requantize
            shl_rvv_sidcso_op_requantize_fp16(mat0, output, mat1);
        } else if (batches_a > 1 && batches_b == 1) {
            __fp16 *in0 = (__fp16 *)shl_mem_alloc(dim_m * dim_k * sizeof(__fp16));
            __fp16 *in1;
            if (!(mat1->is_const)) {
                in1 = (__fp16 *)shl_mem_alloc(dim_k * dim_n * sizeof(__fp16));
                shl_rvv_reorder_input_block_pack2nxk_fp16(mat1_data, in1, dim_k, dim_n,
                                                          MATMUL_K_BLK, MATMUL_N_BLK);
            } else {
                in1 = mat1_data;
            }

            for (int b = 0; b < batches_a; b++) {
                shl_c920_reorder_kernel_block_8xk_fp16(mat0_data, in0, dim_m, dim_k, MATMUL_M_BLK,
                                                       MATMUL_K_BLK);

                shl_c920_gemm_block_8xpack2n_fp16(output_data, in0, in1, NULL, dim_m, dim_k, dim_n,
                                                  MATMUL_M_BLK, MATMUL_K_BLK, MATMUL_N_BLK);

                mat0_data += dim_m * dim_k;
                output_data += dim_m * dim_n;
            }
            shl_mem_free(in0);
            if (!(mat1->is_const)) {
                shl_mem_free(in1);
            }
            // requantize
            shl_rvv_sidcso_op_requantize_fp16(mat0, output, mat1);
        } else {
            shl_debug_error("matmul unsupported this broadcast\n");
            return CSINN_FALSE;
        }
    } else {
        return shl_ref_matmul_quant(mat0, mat1, output, params);
    }

    return CSINN_TRUE;
}

int shl_c920_matmul_fp16_w_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                                struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    if (mat0->layout >= CSINN_LAYOUT_NC1C0 && mat0->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(mat0);
    }

    __fp16 *mat0_data = (__fp16 *)mat0->data;
    int8_t *mat1_data = (int8_t *)mat1->data;
    __fp16 *output_data = (__fp16 *)output->data;

    const int dims_count = mat0->dim_count;
    int batches_a = 1;
    int batches_b = 1;

    /* compute the outer size */
    for (int i = 0; i < dims_count - 2; i++) {
        batches_a *= mat0->dim[i];
    }
    for (int i = 0; i < mat1->dim_count - 2; i++) {
        batches_b *= mat1->dim[i];
    }

    const int dim_m = mat0->dim[dims_count - (params->trans_a ? 1 : 2)];
    const int dim_k = mat0->dim[dims_count - (params->trans_a ? 2 : 1)];
    const int dim_n = mat1->dim[mat1->dim_count - (params->trans_b ? 2 : 1)];

    int32_t zp = mat1->qinfo->zero_point;
    float scale = mat1->qinfo->scale;

    int api = params->base.api;
    int size1 = csinn_tensor_size(mat1);

    if (!params->trans_a && !params->trans_b) {
        if (batches_a == batches_b) {
            __fp16 *in0 = (__fp16 *)shl_mem_alloc(dim_m * dim_k * sizeof(__fp16));
            __fp16 *in1 = (__fp16 *)shl_mem_alloc(size1 * sizeof(__fp16));
            shl_rvv_dequantize_i8_to_f16(mat1_data, in1, size1, zp, scale);

            for (int b = 0; b < batches_a; b++) {
                shl_c920_reorder_kernel_block_8xk_fp16(mat0_data, in0, dim_m, dim_k, MATMUL_M_BLK,
                                                       MATMUL_K_BLK);

                shl_c920_gemm_block_8xpack2n_fp16(output_data, in0, in1 + b * dim_k * dim_n, NULL,
                                                  dim_m, dim_k, dim_n, MATMUL_M_BLK, MATMUL_K_BLK,
                                                  MATMUL_N_BLK);

                mat0_data += dim_m * dim_k;
                output_data += dim_m * dim_n;
            }
            shl_mem_free(in0);
            shl_mem_free(in1);
        } else if (batches_a > 1 && batches_b == 1) {
            __fp16 *in0 = (__fp16 *)shl_mem_alloc(dim_m * dim_k * sizeof(__fp16));
            __fp16 *in1 = (__fp16 *)shl_mem_alloc(size1 * sizeof(__fp16));
            shl_rvv_dequantize_i8_to_f16(mat1_data, in1, size1, zp, scale);

            for (int b = 0; b < batches_a; b++) {
                shl_c920_reorder_kernel_block_8xk_fp16(mat0_data, in0, dim_m, dim_k, MATMUL_M_BLK,
                                                       MATMUL_K_BLK);

                shl_c920_gemm_block_8xpack2n_fp16(output_data, in0, in1, NULL, dim_m, dim_k, dim_n,
                                                  MATMUL_M_BLK, MATMUL_K_BLK, MATMUL_N_BLK);

                mat0_data += dim_m * dim_k;
                output_data += dim_m * dim_n;
            }
            shl_mem_free(in0);
            shl_mem_free(in1);
        } else {
            shl_debug_error("matmul unsupported this broadcast\n");
            return CSINN_FALSE;
        }
    } else {
        return shl_ref_matmul_quant(mat0, mat1, output, params);
    }

    return CSINN_TRUE;
}

int shl_c920_matmul_init_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                              struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    if (!params->trans_a && !params->trans_b) {
        if (mat0->dtype == CSINN_DTYPE_FLOAT16) {
            if (mat1->is_const && mat1->dtype == CSINN_DTYPE_INT8) {
                shl_rvv_matmul_reorder_weight_fp16_w_int8(mat1, MATMUL_K_BLK, MATMUL_N_BLK);
                cb->exec = shl_c920_matmul_fp16_w_int8;
            } else if (mat1->dtype == CSINN_DTYPE_FLOAT16) {
                if (mat1->is_const) {
                    shl_rvv_matmul_reorder_weight_fp16(mat1, MATMUL_K_BLK, MATMUL_N_BLK);
                }
                cb->exec = shl_c920_matmul_fp16;
            }
        }
    }
    if (cb->exec == NULL) {
        shl_debug_warning(
            "matmul is not optimized to achieve under this condition, call reference func "
            "replaced.\n");
        cb->exec = shl_ref_matmul_quant;
    }
    return CSINN_TRUE;
}
