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

#define MATMUL_M_BLK 64
#define MATMUL_K_BLK 64
#define MATMUL_N_BLK 64

int shl_rvv_matmul_block_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                              struct csinn_tensor *output, struct csinn_matmul_params *params,
                              const int M_BLK, const int K_BLK, const int N_BLK)
{
    __fp16 *mat0_data = (__fp16 *)mat0->data;
    __fp16 *mat1_data = (__fp16 *)mat1->data;
    __fp16 *output_data = (__fp16 *)output->data;

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
            __fp16 *in0 = (__fp16 *)shl_mem_alloc(dim_m * dim_k * sizeof(__fp16));
            __fp16 *in1;
            if (!(mat1->is_const)) {
                in1 = (__fp16 *)shl_mem_alloc(dim_k * dim_n * sizeof(__fp16));
            }

            for (int b = 0; b < batches_a; b++) {
                shl_rvv_reorder_kernel_block_12xk_fp16(mat0_data, in0, dim_m, dim_k, M_BLK, K_BLK);
                if (!(mat1->is_const)) {
                    shl_rvv_reorder_input_block_pack2nxk_fp16(mat1_data, in1, dim_k, dim_n, K_BLK,
                                                              N_BLK);
                } else {
                    in1 = mat1_data;
                }

                shl_rvv_gemm_block_12xpack2n_fp16(output_data, in0, in1, NULL, dim_m, dim_k, dim_n,
                                                  M_BLK, K_BLK, N_BLK);

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
        } else {
            shl_ref_matmul_quant(mat0, mat1, output, params);
        }
    } else if (batches_a > 1 && batches_b == 1) {
        if (!params->trans_a && !params->trans_b) {
            __fp16 *in0 = (__fp16 *)shl_mem_alloc(dim_m * dim_k * sizeof(__fp16));
            __fp16 *in1;
            if (!(mat1->is_const)) {
                in1 = (__fp16 *)shl_mem_alloc(dim_k * dim_n * sizeof(__fp16));
                shl_rvv_reorder_input_block_pack2nxk_fp16(mat1_data, in1, dim_k, dim_n, K_BLK,
                                                          N_BLK);
            } else {
                in1 = mat1_data;
            }

            for (int b = 0; b < batches_a; b++) {
                shl_rvv_reorder_kernel_block_12xk_fp16(mat0_data, in0, dim_m, dim_k, M_BLK, K_BLK);

                shl_rvv_gemm_block_12xpack2n_fp16(output_data, in0, in1, NULL, dim_m, dim_k, dim_n,
                                                  M_BLK, K_BLK, N_BLK);

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
        shl_debug_error("matmul unsupported this broadcast\n");
        return CSINN_FALSE;
    }

    return CSINN_TRUE;
}

int shl_rvv_matmul_block_fp16_w_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                                     struct csinn_tensor *output,
                                     struct csinn_matmul_params *params, const int M_BLK,
                                     const int K_BLK, const int N_BLK)
{
    __fp16 *mat0_data = (__fp16 *)mat0->data;
    int8_t *mat1_data = (int8_t *)mat1->data;
    __fp16 *output_data = (__fp16 *)output->data;

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

    int32_t zp = mat1->qinfo->zero_point;
    float scale = mat1->qinfo->scale;

    int api = params->base.api;
    int size1 = csinn_tensor_size(mat1);

    if (batches_a == batches_b) {
        if (!params->trans_a && !params->trans_b) {
            __fp16 *in0 = (__fp16 *)shl_mem_alloc(dim_m * dim_k * sizeof(__fp16));
            __fp16 *in1 = (__fp16 *)shl_mem_alloc(size1 * sizeof(__fp16));
            shl_rvv_dequantize_i8_to_f16(mat1_data, in1, size1, zp, scale);

            for (int b = 0; b < batches_a; b++) {
                shl_rvv_reorder_kernel_block_12xk_fp16(mat0_data, in0, dim_m, dim_k, M_BLK, K_BLK);

                shl_rvv_gemm_block_12xpack2n_fp16(output_data, in0, in1 + b * dim_k * dim_n, NULL,
                                                  dim_m, dim_k, dim_n, M_BLK, K_BLK, N_BLK);

                mat0_data += dim_m * dim_k;
                output_data += dim_m * dim_n;
            }
            shl_mem_free(in0);
            shl_mem_free(in1);
        } else {
            shl_ref_matmul_quant(mat0, mat1, output, params);
        }
    } else if (batches_a > 1 && batches_b == 1) {
        if (!params->trans_a && !params->trans_b) {
            __fp16 *in0 = (__fp16 *)shl_mem_alloc(dim_m * dim_k * sizeof(__fp16));
            __fp16 *in1 = (__fp16 *)shl_mem_alloc(size1 * sizeof(__fp16));
            shl_rvv_dequantize_i8_to_f16(mat1_data, in1, size1, zp, scale);

            for (int b = 0; b < batches_a; b++) {
                shl_rvv_reorder_kernel_block_12xk_fp16(mat0_data, in0, dim_m, dim_k, M_BLK, K_BLK);

                shl_rvv_gemm_block_12xpack2n_fp16(output_data, in0, in1, NULL, dim_m, dim_k, dim_n,
                                                  M_BLK, K_BLK, N_BLK);

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
        shl_debug_error("matmul unsupported this broadcast\n");
        return CSINN_FALSE;
    }

    return CSINN_TRUE;
}

/*************************************************************
 * packn = vlenb / sizeof(__fp16)
 * src: [K_BLOCK, N_BLOCK]
 * dst: [N_BLOCK/n_blk, K_BLOCK, n_blk]
 * n_blk: pack2n/packn/n_tail
 ************************************************************/
static inline void reorder_matb_pack2nxk_fp16_w_int8(int8_t *src, int8_t *dst, int N_BLOCK,
                                                     int K_BLOCK, int ldb)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int pack2n = packn * 2;
    int vl = vsetvl_e16m2(pack2n);

    int j = 0;
    for (; j + pack2n - 1 < N_BLOCK; j += pack2n) {
        int8_t *s_ptr = src + j;
        for (int c = 0; c < K_BLOCK; c++) {
            vint8m1_t _src = vle8_v_i8m1(s_ptr, vl);
            vse8_v_i8m1(dst, _src, vl);
            s_ptr += ldb;
            dst += vl;
        }
    }
    while (j < N_BLOCK) {
        vl = vsetvl_e16m1(N_BLOCK - j);
        int8_t *s_ptr = src + j;
        for (int c = 0; c < K_BLOCK; c++) {
            vint8m1_t _src = vle8_v_i8m1(s_ptr, vl);
            vse8_v_i8m1(dst, _src, vl);
            s_ptr += ldb;
            dst += vl;
        }
        j += vl;
    }
}

/*************************************************************
 * packn = vlenb / sizeof(__fp16)
 * src: [k, n]
 * dst: [n/n_blk, k/k_blk, n_blk/pack2n, k_blk, pack2n]
 * n_blk: N_BLK, N_BLK/2, N_BLK/4, ..., pack2n
 * k_blk: K_BLK, K_tail
 ************************************************************/
void shl_rvv_matmul_reorder_weight_fp16_w_int8(struct csinn_tensor *mat1, const int K_BLK,
                                               const int N_BLK)
{
    int8_t *mat1_data = (int8_t *)mat1->data;
    int dims_count = mat1->dim_count;
    int batch = 1;
    for (int i = 0; i < dims_count - 2; i++) {
        batch *= mat1->dim[i];
    }
    const int k = mat1->dim[dims_count - 2];
    const int n = mat1->dim[dims_count - 1];
    int8_t *mat_reorder = (int8_t *)shl_mem_alloc(k * n * sizeof(int8_t));

    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int MIN_N_BLOCK = packn * 2;

    for (int b = 0; b < batch; b++) {
        int8_t *init_mat = mat1_data + b * k * n;
        int8_t *dst = mat_reorder;

        int k_block = K_BLK;
        int k_idx = 0;
        while (k_idx < k) {
            if (k - k_idx < k_block) {
                k_block = k - k_idx;
            }
            int n_block = N_BLK;
            int n_idx = 0;
            while (n_idx < n) {
                while (!(n_idx + n_block - 1 < n)) {
                    n_block /= 2;
                }
                if (n_block < MIN_N_BLOCK) {
                    n_block = n - n_idx;
                }
                int8_t *s_ptr = init_mat + k_idx * n + n_idx;
                int8_t *d_ptr = dst + n_idx * k + k_idx * n_block;
                reorder_matb_pack2nxk_fp16_w_int8(s_ptr, d_ptr, n_block, k_block, n);
                n_idx += n_block;
            }
            k_idx += k_block;
        }

        memcpy(init_mat, mat_reorder, k * n * sizeof(int8_t));
    }
    shl_mem_free(mat_reorder);
}

/*************************************************************
 * packn = vlenb / sizeof(__fp16)
 * src: [k, n]
 * dst: [n/n_blk, k/k_blk, n_blk/pack2n, k_blk, pack2n]
 * n_blk: N_BLK, N_BLK/2, N_BLK/4, ..., pack2n
 * k_blk: K_BLK, K_tail
 ************************************************************/
void shl_rvv_matmul_reorder_weight_fp16(struct csinn_tensor *mat1, const int K_BLK, const int N_BLK)
{
    __fp16 *mat1_data = (__fp16 *)mat1->data;
    int dims_count = mat1->dim_count;
    int batch = 1;
    for (int i = 0; i < dims_count - 2; i++) {
        batch *= mat1->dim[i];
    }
    const int k = mat1->dim[dims_count - 2];
    const int n = mat1->dim[dims_count - 1];
    __fp16 *mat_reorder = (__fp16 *)shl_mem_alloc(k * n * sizeof(__fp16));

    for (int b = 0; b < batch; b++) {
        __fp16 *init_mat = mat1_data + b * k * n;
        shl_rvv_reorder_input_block_pack2nxk_fp16(init_mat, mat_reorder, k, n, K_BLK, N_BLK);
        memcpy(init_mat, mat_reorder, k * n * sizeof(__fp16));
    }

    shl_mem_free(mat_reorder);
}

int shl_rvv_matmul_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                        struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    if (mat1->is_const && mat1->dtype == CSINN_DTYPE_INT8) {
        return shl_rvv_matmul_block_fp16_w_int8(mat0, mat1, output, params, MATMUL_M_BLK,
                                                MATMUL_K_BLK, MATMUL_N_BLK);
    } else if (mat1->dtype == CSINN_DTYPE_FLOAT16) {
        return shl_rvv_matmul_block_fp16(mat0, mat1, output, params, MATMUL_M_BLK, MATMUL_K_BLK,
                                         MATMUL_N_BLK);
    }
}

int shl_rvv_matmul_init_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                             struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    if (mat0->dtype == CSINN_DTYPE_FLOAT16) {
        if (mat1->is_const && mat1->dtype == CSINN_DTYPE_INT8) {
            shl_rvv_matmul_reorder_weight_fp16_w_int8(mat1, MATMUL_K_BLK, MATMUL_N_BLK);
        } else if (mat1->dtype == CSINN_DTYPE_FLOAT16) {
            if (mat1->is_const) {
                shl_rvv_matmul_reorder_weight_fp16(mat1, MATMUL_K_BLK, MATMUL_N_BLK);
            }
        } else {
            shl_debug_error("mat1 unsupported dtype: %d\n", mat1->dtype);
            return CSINN_FALSE;
        }
        cb->exec = shl_rvv_matmul_fp16;
    } else {
        shl_debug_error("mat0 unsupported dtype: %d\n", mat0->dtype);
    }
    return CSINN_TRUE;
}
