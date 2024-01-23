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

int shl_rvv_matmul_common_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                               struct csinn_tensor *output, struct csinn_matmul_params *params,
                               void (*reorder_mat0)(int8_t *, int8_t *, int, int, int),
                               void (*reorder_mat1)(int8_t *, int8_t *, int, int, int),
                               void (*matmul)(int8_t *, const int8_t *, const int8_t *, int, int,
                                              int, int, int32_t, int32_t, int32_t, int32_t,
                                              int32_t))
{
    if (mat0->layout >= CSINN_LAYOUT_NC1C0 && mat0->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(mat0);
    }
    if (mat1->layout >= CSINN_LAYOUT_NC1C0 && mat1->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(mat1);
    }

    int8_t *mat0_data = (int8_t *)mat0->data;
    int8_t *mat1_data = (int8_t *)mat1->data;
    int8_t *output_data = (int8_t *)output->data;

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

    int32_t z1 = mat0->qinfo->zero_point;
    int32_t z2 = mat1->qinfo->zero_point;
    int32_t z3 = output->qinfo->zero_point;
    int32_t multiplier;
    int32_t shift;
    float real_scale = mat0->qinfo->scale * mat1->qinfo->scale / output->qinfo->scale;
    shl_quantize_multiplier(real_scale, &multiplier, &shift);

    if (!params->trans_a && !params->trans_b) {
        if (batches_a == batches_b) {
            int8_t *in0 = (int8_t *)shl_mem_alloc(dim_m * dim_k * sizeof(int8_t));
            int8_t *in1;
            if (!(mat1->is_const)) {
                in1 = (int8_t *)shl_mem_alloc(dim_k * dim_n * sizeof(int8_t));
            }

            for (int b = 0; b < batches_a; b++) {
                reorder_mat0(mat0_data, in0, dim_m, dim_k, dim_k);
                if (!(mat1->is_const)) {
                    reorder_mat1(mat1_data, in1, dim_k, dim_n, dim_n);
                } else {
                    in1 = mat1_data;
                }

                matmul(output_data, in0, in1, dim_m, dim_k, dim_n, dim_n, z1, z2, z3, multiplier,
                       shift);

                mat0_data += dim_m * dim_k;
                mat1_data += dim_k * dim_n;
                output_data += dim_m * dim_n;
            }
            shl_mem_free(in0);
            if (!(mat1->is_const)) {
                shl_mem_free(in1);
            }
        } else if (batches_a > 1 && batches_b == 1) {
            int8_t *in0 = (int8_t *)shl_mem_alloc(dim_m * dim_k * sizeof(int8_t));
            int8_t *in1;
            if (!(mat1->is_const)) {
                in1 = (int8_t *)shl_mem_alloc(dim_k * dim_n * sizeof(int8_t));
                reorder_mat1(mat1_data, in1, dim_k, dim_n, dim_n);
            } else {
                in1 = mat1_data;
            }

            for (int b = 0; b < batches_a; b++) {
                reorder_mat0(mat0_data, in0, dim_m, dim_k, dim_k);
                matmul(output_data, in0, in1, dim_m, dim_k, dim_n, dim_n, z1, z2, z3, multiplier,
                       shift);

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
        return shl_ref_matmul_quant(mat0, mat1, output, params);
    }

    return CSINN_TRUE;
}

void shl_rvv_matmul_reorder_weight_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1)
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

    for (int b = 0; b < batch; b++) {
        int8_t *init_mat = mat1_data + b * k * n;
#ifdef SHL_USE_DOT_INT8
        // XXX: use -z in dot has problems when z=-128
        if (mat0->qinfo->zero_point == INT8_MIN || mat1->qinfo->zero_point == INT8_MIN) {
            shl_rvv_matmul_reorder_mat1_zpackn_int8(init_mat, mat_reorder, k, n, n);
        } else {
            shl_rvv_matmul_reorder_mat1_zmf2n4_int8_dot(init_mat, mat_reorder, k, n, n);
        }
#else
        shl_rvv_matmul_reorder_mat1_zpackn_int8(init_mat, mat_reorder, k, n, n);
#endif  // SHL_USE_DOT_INT8
        memcpy(init_mat, mat_reorder, k * n * sizeof(int8_t));
    }

    shl_mem_free(mat_reorder);
}

int shl_rvv_matmul_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                        struct csinn_tensor *output, struct csinn_matmul_params *params)
{
#ifdef SHL_USE_DOT_INT8
    // XXX: use -z in dot has problems when z=-128
    if (mat0->qinfo->zero_point == INT8_MIN || mat1->qinfo->zero_point == INT8_MIN) {
        return shl_rvv_matmul_common_int8(
            mat0, mat1, output, params, shl_rvv_matmul_reorder_mat0_n4_int8,
            shl_rvv_matmul_reorder_mat1_zpackn_int8, shl_rvv_matmul_4xpackn_int8);
    } else {
        return shl_rvv_matmul_common_int8(
            mat0, mat1, output, params, shl_rvv_matmul_reorder_mat0_n8z4_int8_dot,
            shl_rvv_matmul_reorder_mat1_zmf2n4_int8_dot, shl_rvv_matmul_8xmf2_int8_dot);
    }
#else
    return shl_rvv_matmul_common_int8(
        mat0, mat1, output, params, shl_rvv_matmul_reorder_mat0_n4_int8,
        shl_rvv_matmul_reorder_mat1_zpackn_int8, shl_rvv_matmul_4xpackn_int8);
#endif  // SHL_USE_DOT_INT8
}

int shl_rvv_matmul_init_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                             struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    struct csinn_session *sess = params->base.sess;
    bool binary_model_op_init = shl_rvv_get_binary_model_op_init(sess);
    if (!params->trans_a && !params->trans_b) {
        if (mat0->dtype == CSINN_DTYPE_INT8 && mat1->dtype == CSINN_DTYPE_INT8) {
            if (!binary_model_op_init) {
                if (mat1->is_const) {
                    shl_rvv_matmul_reorder_weight_int8(mat0, mat1);
                }
            }
            cb->exec = shl_rvv_matmul_int8;
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
