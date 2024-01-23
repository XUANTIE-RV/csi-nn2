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

int shl_c920_matmul_a0b0_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
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

    if (batches_a == batches_b) {
        __fp16 *in0 = (__fp16 *)shl_mem_alloc(dim_m * dim_k * sizeof(__fp16));
        __fp16 *in1;
        if (!(mat1->is_const)) {
            in1 = (__fp16 *)shl_mem_alloc(dim_k * dim_n * sizeof(__fp16));
        }

        for (int b = 0; b < batches_a; b++) {
            shl_c920_reorder_a_block_8xk_fp16(mat0_data, in0, dim_m, dim_k, MATMUL_M_BLK,
                                              MATMUL_K_BLK);
            if (!(mat1->is_const)) {
                shl_rvv_reorder_b_block_pack2nxk_fp16(mat1_data, in1, dim_k, dim_n, MATMUL_K_BLK,
                                                      MATMUL_N_BLK);
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
            shl_rvv_reorder_b_block_pack2nxk_fp16(mat1_data, in1, dim_k, dim_n, MATMUL_K_BLK,
                                                  MATMUL_N_BLK);
        } else {
            in1 = mat1_data;
        }

        for (int b = 0; b < batches_a; b++) {
            shl_c920_reorder_a_block_8xk_fp16(mat0_data, in0, dim_m, dim_k, MATMUL_M_BLK,
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

    return CSINN_TRUE;
}

int shl_c920_matmul_a0b0_fp16_w_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                                     struct csinn_tensor *output,
                                     struct csinn_matmul_params *params)
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
    int size1 = csinn_tensor_size(mat1);

    if (batches_a == batches_b) {
        __fp16 *in0 = (__fp16 *)shl_mem_alloc(dim_m * dim_k * sizeof(__fp16));
        __fp16 *in1 = (__fp16 *)shl_mem_alloc(size1 * sizeof(__fp16));
        shl_rvv_dequantize_i8_to_f16(mat1_data, in1, size1, zp, scale);

        for (int b = 0; b < batches_a; b++) {
            shl_c920_reorder_a_block_8xk_fp16(mat0_data, in0, dim_m, dim_k, MATMUL_M_BLK,
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
            shl_c920_reorder_a_block_8xk_fp16(mat0_data, in0, dim_m, dim_k, MATMUL_M_BLK,
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

    return CSINN_TRUE;
}

/*************************************************************
 * packn = vlenb / sizeof(__fp16)
 * n_blk: pack2n/packn/n_tail
 * src: [n, k]
 * dst: [n/n_blk, k, n_blk]
 ************************************************************/
static void reorder_mat1_npack2n_fp16(const __fp16 *src, __fp16 *dst, int n, int k)
{
    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int pack2n = packn * 2;

    int i = 0;
    int vl = vsetvl_e16m2(pack2n);
    for (; i + pack2n - 1 < n; i += pack2n) {
        const __fp16 *s_ptr = src + i * k;
        for (int j = 0; j < k; j++) {
            vfloat16m2_t _src = vlse16_v_f16m2(s_ptr, k * sizeof(__fp16), vl);
            vse16_v_f16m2(dst, _src, vl);
            s_ptr += 1;
            dst += vl;
        }
    }
    while (i < n) {
        int vl = vsetvl_e16m1(n - i);
        const __fp16 *s_ptr = src + i * k;
        for (int j = 0; j < k; j++) {
            vfloat16m1_t _src = vlse16_v_f16m1(s_ptr, k * sizeof(__fp16), vl);
            vse16_v_f16m1(dst, _src, vl);
            s_ptr += 1;
            dst += vl;
        }
        i += vl;
    }
}

int shl_c920_matmul_a0b1_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
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

    if (batches_a == batches_b) {
        __fp16 *in1 = (__fp16 *)shl_mem_alloc(dim_k * dim_n * sizeof(__fp16));

        for (int b = 0; b < batches_a; b++) {
            reorder_mat1_npack2n_fp16(mat1_data, in1, dim_n, dim_k);
            shl_c920_gemm_a0nb1r_8xpack2n_fp16(output_data, mat0_data, in1, NULL, dim_m, dim_k,
                                               dim_n);
            mat0_data += dim_m * dim_k;
            mat1_data += dim_k * dim_n;
            output_data += dim_m * dim_n;
        }

        shl_mem_free(in1);
    } else if (batches_a > 1 && batches_b == 1) {
        __fp16 *in1 = (__fp16 *)shl_mem_alloc(dim_k * dim_n * sizeof(__fp16));
        reorder_mat1_npack2n_fp16(mat1_data, in1, dim_n, dim_k);

        for (int b = 0; b < batches_a; b++) {
            shl_c920_gemm_a0nb1r_8xpack2n_fp16(output_data, mat0_data, in1, NULL, dim_m, dim_k,
                                               dim_n);
            mat0_data += dim_m * dim_k;
            output_data += dim_m * dim_n;
        }
        shl_mem_free(in1);
    } else {
        shl_debug_error("matmul unsupported this broadcast\n");
        return CSINN_FALSE;
    }

    return CSINN_TRUE;
}

int shl_c920_matmul_a0b1_fp16_block_quant(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                                          struct csinn_tensor *output,
                                          struct csinn_matmul_params *params)
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

    int size1 = csinn_tensor_size(mat1);
    __fp16 *scale_data;
    int weight_k = dim_k;
    void (*gemm_a0nb1n_dot_fp16)();
    if (mat1->mtype == CSINN_MEM_TYPE_BLOCK_Q8_0) {
        scale_data = (__fp16 *)(mat1_data + size1);
        gemm_a0nb1n_dot_fp16 = shl_c920_gemm_a0nb1n_dot_fp16_q8;
    } else if (mat1->mtype == CSINN_MEM_TYPE_BLOCK_Q8_0_REARRANGE) {
        scale_data = (__fp16 *)(mat1_data + size1);
        gemm_a0nb1n_dot_fp16 = shl_c920_gemm_a0nb1_dot_fp16_q8_rearrange;
    } else if (mat1->mtype == CSINN_MEM_TYPE_BLOCK_Q4_0) {
        // uint4 is only half of tensor size
        scale_data = (__fp16 *)(mat1_data + size1 / 2);
        weight_k = dim_k / 2;
        gemm_a0nb1n_dot_fp16 = shl_c920_gemm_a0nb1n_dot_fp16_q4;
    } else if (mat1->mtype == CSINN_MEM_TYPE_BLOCK_Q4_0_REARRANGE) {
        // uint4 is only half of tensor size
        scale_data = (__fp16 *)(mat1_data + size1 / 2);
        weight_k = dim_k / 2;
        gemm_a0nb1n_dot_fp16 = shl_c920_gemm_a0nb1_dot_fp16_q4_rearrange;
    } else {
        shl_debug_error("%s: unsupported mtype %d\n", __func__, mat1->mtype);
        return CSINN_FALSE;
    }

    if (batches_a == batches_b) {
        for (int b = 0; b < batches_a; b++) {
            gemm_a0nb1n_dot_fp16(output_data, mat0_data, mat1_data, NULL, dim_m, dim_k, dim_n,
                                 scale_data);
            mat0_data += dim_m * dim_k;
            mat1_data += dim_n * weight_k;
            scale_data += dim_n * dim_k / 32;
            output_data += dim_m * dim_n;
        }
    } else if (batches_a > 1 && batches_b == 1) {
        for (int b = 0; b < batches_a; b++) {
            gemm_a0nb1n_dot_fp16(output_data, mat0_data, mat1_data, NULL, dim_m, dim_k, dim_n,
                                 scale_data);
            mat0_data += dim_m * dim_k;
            output_data += dim_m * dim_n;
        }
    } else {
        shl_debug_error("matmul unsupported this broadcast\n");
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}

int shl_c920_matmul_init_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                              struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    struct csinn_session *sess = params->base.sess;
    bool binary_model_op_init = shl_c920_get_binary_model_op_init(sess);
    if (!params->trans_a && !params->trans_b) {
        if (mat0->dtype == CSINN_DTYPE_FLOAT16) {
            if (mat1->is_const && mat1->dtype == CSINN_DTYPE_INT8) {
                if (!binary_model_op_init) {
                    shl_rvv_matmul_reorder_weight_fp16_w_int8(mat1, MATMUL_K_BLK, MATMUL_N_BLK);
                }
                cb->exec = shl_c920_matmul_a0b0_fp16_w_int8;
            } else if (mat1->dtype == CSINN_DTYPE_FLOAT16) {
                if (mat1->is_const) {
                    if (!binary_model_op_init) {
                        shl_rvv_matmul_reorder_weight_fp16(mat1, MATMUL_K_BLK, MATMUL_N_BLK);
                    }
                }
                cb->exec = shl_c920_matmul_a0b0_fp16;
            }
        }
    }

    if (!params->trans_a && params->trans_b) {
        if (mat0->dtype == CSINN_DTYPE_FLOAT16 && mat1->dtype == CSINN_DTYPE_FLOAT16) {
            cb->exec = shl_c920_matmul_a0b1_fp16;
        } else if (mat0->dtype == CSINN_DTYPE_FLOAT16 &&
                   ((mat1->dtype == CSINN_DTYPE_INT8 && mat1->mtype == CSINN_MEM_TYPE_BLOCK_Q8_0) ||
                    (mat1->dtype == CSINN_DTYPE_INT4 && mat1->mtype == CSINN_MEM_TYPE_BLOCK_Q4_0) ||
                    (mat1->dtype == CSINN_DTYPE_INT8 &&
                     mat1->mtype == CSINN_MEM_TYPE_BLOCK_Q8_0_REARRANGE) ||
                    (mat1->dtype == CSINN_DTYPE_INT4 &&
                     mat1->mtype == CSINN_MEM_TYPE_BLOCK_Q4_0_REARRANGE))) {
            cb->exec = shl_c920_matmul_a0b1_fp16_block_quant;
        }
    }

    if (cb->exec == NULL) {
        shl_debug_warning(
            "matmul is not optimized to achieve under this condition on C920 FP16, call reference "
            "func replaced.\n");
        cb->exec = shl_ref_matmul_quant;
    }
    return CSINN_TRUE;
}
