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

int shl_c920_matmul_a0b0_fp32(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                              struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    if (mat0->layout >= CSINN_LAYOUT_NC1C0 && mat0->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(mat0);
    }
    if (mat1->layout >= CSINN_LAYOUT_NC1C0 && mat1->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(mat1);
    }

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
    for (int i = 0; i < mat1->dim_count - 2; i++) {
        batches_b *= mat1->dim[i];
    }

    const int dim_m = mat0->dim[dims_count - (params->trans_a ? 1 : 2)];
    const int dim_k = mat0->dim[dims_count - (params->trans_a ? 2 : 1)];
    const int dim_n = mat1->dim[mat1->dim_count - (params->trans_b ? 2 : 1)];

    if (batches_a == batches_b) {
        float *in0 = (float *)shl_mem_alloc(dim_m * dim_k * sizeof(float));
        float *in1;
        if (!(mat1->is_const)) {
            in1 = (float *)shl_mem_alloc(dim_k * dim_n * sizeof(float));
        }

        for (int b = 0; b < batches_a; b++) {
            shl_c920_reorder_a_block_8xk_fp32(mat0_data, in0, dim_m, dim_k, MATMUL_M_BLK,
                                              MATMUL_K_BLK);
            if (!(mat1->is_const)) {
                shl_rvv_reorder_b_block_pack2nxk_fp32(mat1_data, in1, dim_k, dim_n, MATMUL_K_BLK,
                                                      MATMUL_N_BLK);
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
    } else if (batches_a > 1 && batches_b == 1) {
        float *in0 = (float *)shl_mem_alloc(dim_m * dim_k * sizeof(float));
        float *in1;
        if (!(mat1->is_const)) {
            in1 = (float *)shl_mem_alloc(dim_k * dim_n * sizeof(float));
            shl_rvv_reorder_b_block_pack2nxk_fp32(mat1_data, in1, dim_k, dim_n, MATMUL_K_BLK,
                                                  MATMUL_N_BLK);
        } else {
            in1 = mat1_data;
        }

        for (int b = 0; b < batches_a; b++) {
            shl_c920_reorder_a_block_8xk_fp32(mat0_data, in0, dim_m, dim_k, MATMUL_M_BLK,
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

    return CSINN_TRUE;
}

/*************************************************************
 * packn = vlenb / sizeof(float)
 * n_blk: pack2n/packn/n_tail
 * src: [n, k]
 * dst: [n/n_blk, k, n_blk]
 ************************************************************/
static void reorder_mat1_npack2n_fp32(const float *src, float *dst, int n, int k)
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

int shl_c920_matmul_a0b1_fp32(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                              struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    if (mat0->layout >= CSINN_LAYOUT_NC1C0 && mat0->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(mat0);
    }
    if (mat1->layout >= CSINN_LAYOUT_NC1C0 && mat1->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(mat1);
    }

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
    for (int i = 0; i < mat1->dim_count - 2; i++) {
        batches_b *= mat1->dim[i];
    }

    const int dim_m = mat0->dim[dims_count - (params->trans_a ? 1 : 2)];
    const int dim_k = mat0->dim[dims_count - (params->trans_a ? 2 : 1)];
    const int dim_n = mat1->dim[mat1->dim_count - (params->trans_b ? 2 : 1)];

    if (batches_a == batches_b) {
        float *in1 = (float *)shl_mem_alloc(dim_k * dim_n * sizeof(float));

        for (int b = 0; b < batches_a; b++) {
            reorder_mat1_npack2n_fp32(mat1_data, in1, dim_n, dim_k);
            shl_c920_gemm_a0nb1r_8xpack2n_fp32(output_data, mat0_data, in1, NULL, dim_m, dim_k,
                                               dim_n);
            mat0_data += dim_m * dim_k;
            mat1_data += dim_k * dim_n;
            output_data += dim_m * dim_n;
        }

        shl_mem_free(in1);
    } else if (batches_a > 1 && batches_b == 1) {
        float *in1 = (float *)shl_mem_alloc(dim_k * dim_n * sizeof(float));
        reorder_mat1_npack2n_fp32(mat1_data, in1, dim_n, dim_k);

        for (int b = 0; b < batches_a; b++) {
            shl_c920_gemm_a0nb1r_8xpack2n_fp32(output_data, mat0_data, in1, NULL, dim_m, dim_k,
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

int shl_c920_matmul_a0b1_fp32_block_quant(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                                          struct csinn_tensor *output,
                                          struct csinn_matmul_params *params)
{
    if (mat0->layout >= CSINN_LAYOUT_NC1C0 && mat0->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(mat0);
    }

    float *mat0_data = (float *)mat0->data;
    int8_t *mat1_data = (int8_t *)mat1->data;
    float *output_data = (float *)output->data;

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
    void (*gemm_a0nb1n_dot_fp32)();
    if (mat1->mtype == CSINN_MEM_TYPE_BLOCK_Q8_0) {
        scale_data = (__fp16 *)(mat1_data + size1);
        gemm_a0nb1n_dot_fp32 = shl_c920_gemm_a0nb1n_dot_fp32_q8;
    } else if (mat1->mtype == CSINN_MEM_TYPE_BLOCK_Q4_0) {
        // uint4 is only half of tensor size
        scale_data = (__fp16 *)(mat1_data + size1 / 2);
        weight_k = dim_k / 2;
        gemm_a0nb1n_dot_fp32 = shl_c920_gemm_a0nb1n_dot_fp32_q4;
    } else {
        shl_debug_error("%s: unsupported mtype %d\n", __func__, mat1->mtype);
        return CSINN_FALSE;
    }

    if (batches_a == batches_b) {
        for (int b = 0; b < batches_a; b++) {
            gemm_a0nb1n_dot_fp32(output_data, mat0_data, mat1_data, NULL, dim_m, dim_k, dim_n,
                                 scale_data);
            mat0_data += dim_m * dim_k;
            mat1_data += dim_n * weight_k;
            scale_data += dim_n * dim_k / 32;
            output_data += dim_m * dim_n;
        }
    } else if (batches_a > 1 && batches_b == 1) {
        for (int b = 0; b < batches_a; b++) {
            gemm_a0nb1n_dot_fp32(output_data, mat0_data, mat1_data, NULL, dim_m, dim_k, dim_n,
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

int shl_c920_matmul_init_fp32(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                              struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    struct csinn_session *sess = params->base.sess;
    bool binary_model_op_init = shl_c920_get_binary_model_op_init(sess);
    if (!params->trans_a && !params->trans_b) {
        if (mat0->dtype == CSINN_DTYPE_FLOAT32 && mat1->dtype == CSINN_DTYPE_FLOAT32) {
            if (!binary_model_op_init) {
                if (mat1->is_const) {
                    shl_rvv_matmul_reorder_weight_fp32(mat1, MATMUL_K_BLK, MATMUL_N_BLK);
                }
            }
            cb->exec = shl_c920_matmul_a0b0_fp32;
        }
    }

    if (!params->trans_a && params->trans_b) {
        if (mat0->dtype == CSINN_DTYPE_FLOAT32 && mat1->dtype == CSINN_DTYPE_FLOAT32) {
            cb->exec = shl_c920_matmul_a0b1_fp32;
        } else if (mat0->dtype == CSINN_DTYPE_FLOAT32 &&
                   ((mat1->dtype == CSINN_DTYPE_INT8 && mat1->mtype == CSINN_MEM_TYPE_BLOCK_Q8_0) ||
                    (mat1->dtype == CSINN_DTYPE_INT4 &&
                     mat1->mtype == CSINN_MEM_TYPE_BLOCK_Q4_0))) {
            cb->exec = shl_c920_matmul_a0b1_fp32_block_quant;
        }
    }

    if (cb->exec == NULL) {
        shl_debug_warning(
            "matmul is not optimized to achieve under this condition on C920 FP32, call reference "
            "func replaced.\n");
        cb->exec = shl_ref_matmul_quant;
    }
    return CSINN_TRUE;
}
