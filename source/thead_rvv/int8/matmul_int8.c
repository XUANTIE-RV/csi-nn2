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

/* SHL version 2.1.x */

#include "shl_thead_rvv.h"

/************************************************************************************
 * [m, k] --> [k, m]
 ************************************************************************************/
static inline void transpose_mat(int8_t *mat, int m, int k)
{
    int8_t *trans = (int8_t *)shl_mem_alloc(k * m * sizeof(int8_t));
    for (int i = 0; i < k; i++) {
        int j = 0;
        while (j < m) {
            int vl = vsetvl_e8m1(m - j);
            vint8m1_t _src = vlse8_v_i8m1(mat + j * k + i, k, vl);
            vse8_v_i8m1(trans + i * m + j, _src, vl);
            j += vl;
        }
    }
    memcpy(mat, trans, m * k * sizeof(int8_t));
    shl_mem_free(trans);
}

/************************************************************************************
 * trans_a = 0
 * trans_b = 0
 * mat0:   [dim_i, dim_k]
 * mat1:   [dim_k, dim_j]
 * output: [dim_i, dim_j]
 ************************************************************************************/
static void matmul_int8_axb(int8_t *output, const int8_t *mat0, const int8_t *mat1, int dim_i,
                            int dim_k, int dim_j, int32_t z1, int32_t z2, int32_t z3, int32_t mult,
                            int32_t shift)
{
    for (int i = 0; i < dim_i; i++) {
        const int8_t *m1_ptr = mat1;
        int j = 0;
        while (j < dim_j) {
            int vl = vsetvl_e8m1(dim_j - j);
            const int8_t *m0_ptr = mat0;
            vint32m4_t _acc = vmv_v_x_i32m4(0, vl);

            for (int k = 0; k < dim_k; k++) {
                vint8m1_t _m1 = vle8_v_i8m1(m1_ptr, vl);
                vint16m2_t _m1_w = vwsub_vx_i16m2(_m1, z2, vl);
                int16_t m0_w = m0_ptr[0] - z1;
                vint32m4_t _mul = vwmul_vx_i32m4(_m1_w, m0_w, vl);
                _acc = vadd_vv_i32m4(_acc, _mul, vl);
                m0_ptr += 1;
                m1_ptr += vl;
            }

            vint32m4_t _mulh = vmulh_vx_i32m4(_acc, mult, vl);
            if (shift < 0) {
                _mulh = vssra_vx_i32m4(_mulh, -shift - 1, vl);
            } else {
                _mulh = vsll_vx_i32m4(_mulh, shift + 1, vl);
            }
            vint32m4_t _res0 = vadd_vx_i32m4(_mulh, z3, vl);
            vint16m2_t _res1 = vnclip_wx_i16m2(_res0, 0, vl);
            vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);
            vse8_v_i8m1(output, _res2, vl);
            output += vl;
            j += vl;
        }
        mat0 += dim_k;
    }
}

/************************************************************************************
 * trans_a = 0
 * trans_b = 1
 * mat0:   [dim_i, dim_k]
 * mat1:   [dim_j, dim_k]
 * output: [dim_i, dim_j]
 ************************************************************************************/
static void matmul_int8_axtb(int8_t *output, const int8_t *mat0, const int8_t *mat1, int dim_i,
                             int dim_k, int dim_j, int32_t z1, int32_t z2, int32_t z3, int32_t mult,
                             int32_t shift)
{
    for (int i = 0; i < dim_i; i++) {
        const int8_t *m1_ptr = mat1;
        int j = 0;
        while (j < dim_j) {
            int vl = vsetvl_e8m1(dim_j - j);
            const int8_t *m0_ptr = mat0;
            vint32m4_t _acc = vmv_v_x_i32m4(0, vl);

            for (int k = 0; k < dim_k; k++) {
                vint8m1_t _m1 = vlse8_v_i8m1(m1_ptr + j * dim_k + k, dim_k, vl);
                vint16m2_t _m1_w = vwsub_vx_i16m2(_m1, z2, vl);
                int16_t m0_w = m0_ptr[0] - z1;
                vint32m4_t _mul = vwmul_vx_i32m4(_m1_w, m0_w, vl);
                _acc = vadd_vv_i32m4(_acc, _mul, vl);
                m0_ptr += 1;
            }

            vint32m4_t _mulh = vmulh_vx_i32m4(_acc, mult, vl);
            if (shift < 0) {
                _mulh = vssra_vx_i32m4(_mulh, -shift - 1, vl);
            } else {
                _mulh = vsll_vx_i32m4(_mulh, shift + 1, vl);
            }
            vint32m4_t _res0 = vadd_vx_i32m4(_mulh, z3, vl);
            vint16m2_t _res1 = vnclip_wx_i16m2(_res0, 0, vl);
            vint8m1_t _res2 = vnclip_wx_i8m1(_res1, 0, vl);
            vse8_v_i8m1(output, _res2, vl);
            output += vl;
            j += vl;
        }
        mat0 += dim_k;
    }
}

int shl_rvv_matmul_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                        struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    int8_t *mat0_data = mat0->data;
    int8_t *mat1_data = mat1->data;
    int8_t *output_data = output->data;
    const int dims_count = mat0->dim_count;
    int batches_a = 1;
    int batches_b = 1;

    /* compute the outer size */
    for (int i = 0; i < dims_count - 2; i++) {
        batches_a *= mat0->dim[i];
        batches_b *= mat1->dim[i];
    }

    const int dim_i = mat0->dim[dims_count - (params->trans_a ? 1 : 2)];
    const int dim_k = mat0->dim[dims_count - (params->trans_a ? 2 : 1)];
    const int dim_j = mat1->dim[dims_count - (params->trans_b ? 2 : 1)];

    int32_t z1 = mat0->qinfo->zero_point;
    int32_t z2 = mat1->qinfo->zero_point;
    int32_t z3 = output->qinfo->zero_point;
    int32_t multiplier;
    int32_t shift;
    float real_scale = mat0->qinfo->scale * mat1->qinfo->scale / output->qinfo->scale;
    shl_quantize_multiplier(real_scale, &multiplier, &shift);

    if (batches_a == batches_b) {
        for (int b = 0; b < batches_a; b++) {
            if (!params->trans_a && !params->trans_b) {
                matmul_int8_axb(output_data, mat0_data, mat1_data, dim_i, dim_k, dim_j, z1, z2, z3,
                                multiplier, shift);
            } else if (!params->trans_a && params->trans_b) {
                matmul_int8_axtb(output_data, mat0_data, mat1_data, dim_i, dim_k, dim_j, z1, z2, z3,
                                 multiplier, shift);
            } else if (params->trans_a && !params->trans_b) {
                transpose_mat(mat0_data, dim_k, dim_i);
                matmul_int8_axb(output_data, mat0_data, mat1_data, dim_i, dim_k, dim_j, z1, z2, z3,
                                multiplier, shift);
            } else {
                matmul_int8_axb(output_data, mat1_data, mat0_data, dim_j, dim_k, dim_i, z2, z1, z3,
                                multiplier, shift);
                transpose_mat(output_data, dim_j, dim_i);
            }
            mat0_data += dim_i * dim_k;
            mat1_data += dim_k * dim_j;
            output_data += dim_i * dim_j;
        }
    } else if (batches_a > 1 && batches_b == 1) {
        for (int b = 0; b < batches_a; b++) {
            if (!params->trans_a && !params->trans_b) {
                matmul_int8_axb(output_data, mat0_data, mat1_data, dim_i, dim_k, dim_j, z1, z2, z3,
                                multiplier, shift);
            } else {
                shl_debug_error("matmul unsupport this broadcast\n");
                return CSINN_FALSE;
            }
            mat0_data += dim_i * dim_k;
            output_data += dim_i * dim_j;
        }
    } else {
        shl_debug_error("matmul unsupport this broadcast\n");
        return CSINN_FALSE;
    }

    return CSINN_TRUE;
}
