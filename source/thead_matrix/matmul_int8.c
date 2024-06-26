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

#include "rvm/rvm.h"

#ifdef MATRIX_PW_I32
/*************************************************************
 * mrows = rlenb / 4
 * mcols = rlenb / sizeof(int8_t)
 * msize_m: mrows, m_tail
 * msize_n: mrows, n_tail
 * msize_k: mcols, k_tail
 *
 * dst - output: [M, N]
 * sa - mat0:    [M, K]
 * sb - mat1:    [N/msize_n, K/msize_k, msize_n, msize_k]
 ************************************************************/
void shl_rvm_matmul_a0b0_int8_pw_i32(int8_t *dst, int8_t *sa, int8_t *sb, int M, int K, int N,
                                     int32_t z1, int32_t z2, int32_t z3, int32_t mult,
                                     int32_t shift)
{
    int mcols = csrr_xrlenb();
    int mrows = mcols / 4;

    int8_t z1_i8 = (int8_t)z1;
    int8_t z2_i8 = (int8_t)z2;

    int stride_a = K * sizeof(int8_t);
    int stride_c = N * sizeof(int8_t);

    int i = 0;
    // m = mrows/m_tail
    while (i < M) {
        uint8_t msize_m = (M - i >= mrows) ? mrows : (M - i);
        int8_t *sa_ptr = sa + i * K;

        int j = 0;
        // n = mrows/n_tail
        while (j < N) {
            uint8_t msize_n = (N - j >= mrows) ? mrows : (N - j);
            mcfgn(msize_n);
            int8_t *a0_ptr = sa_ptr;
            int8_t *k_ptr = sb + j * K;
            int8_t *c0_ptr = dst + i * N + j;

            mcfgk(msize_n * sizeof(int32_t));
            mcfgm(msize_m);
            mint32_t m4 = mzero_mi32();
            mint32_t m5 = mzero_mi32();
            mint32_t m6 = mzero_mi32();
            mint32_t m7 = mdup_i32(z1 * z2 * K);  // z1 * z2

            mcfgk(mcols * sizeof(int8_t));
            mint8_t m_z1 = mdup_i8(z1_i8);
            mint8_t m_z2 = mdup_i8(z2_i8);

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(int8_t));
                mint8_t m0 = mld_i8(a0_ptr, stride_a);  // a00
                a0_ptr += msize_k;

                mcfgm(msize_n);
                mint8_t m2 = msld_i8(k_ptr, msize_k * sizeof(int8_t));  // b00
                k_ptr += msize_n * msize_k;

                mcfgm(msize_m);
                m4 = mmaqa_mi8(m4, m2, m0);    // q1 * q2
                m5 = mmaqa_mi8(m5, m2, m_z1);  // z1 * q2
                m6 = mmaqa_mi8(m6, m_z2, m0);  // q1 * z2
                c += msize_k;
            }

            mcfgk(msize_n * sizeof(int32_t));
            m4 = msub_mi32(m4, m5);  // - z1 * q2
            m4 = msub_mi32(m4, m6);  // - q1 * z2
            m4 = madd_mi32(m4, m7);  // + z1 * z2

            // requantize
            m4 = mmulh_mi32_i32(m4, mult);
            m4 = msra_mi32_ui32(m4, -shift - 1);
            m4 = madd_mi32_i32(m4, z3);

            mint8_t res00 = mn4clip_mi32_ui32(m4, 0);

            mcfgk(msize_n * sizeof(int8_t));
            msst_i8_mi8(c0_ptr, stride_c, res00);
            j += msize_n;
        }
        i += msize_m;
    }
}
#else
static void sum_and_requantize_m4(int8_t *dst, int32_t *q1q2, int32_t *z1q2, int32_t *q1z2,
                                  int32_t z1z2, int row, int col, int32_t z3, int32_t mult,
                                  int32_t shift)
{
    for (int i = 0; i < row; i++) {
        int32_t *q1q2_ptr = q1q2 + i * col;
        int32_t *z1q2_ptr = z1q2 + i * col;
        int32_t *q1z2_ptr = q1z2 + i * col;
        int8_t *out_ptr = dst + i * col;
        int c = 0;
        while (c < col) {
            int vl = vsetvl_e32m4(col - c);
            vint32m4_t _q1q2 = vle32_v_i32m4(q1q2_ptr, vl);
            vint32m4_t _z1q2 = vle32_v_i32m4(z1q2_ptr, vl);
            vint32m4_t _q1z2 = vle32_v_i32m4(q1z2_ptr, vl);
            vint32m4_t _sum = vsub_vv_i32m4(_q1q2, _z1q2, vl);
            _sum = vsub_vv_i32m4(_sum, _q1z2, vl);
            _sum = vadd_vx_i32m4(_sum, z1z2, vl);
            vint32m4_t _mulh = vmulh_vx_i32m4(_sum, mult, vl);
            _mulh = vssra_vx_i32m4(_mulh, -shift - 1, vl);
            _mulh = vadd_vx_i32m4(_mulh, z3, vl);
            vint16m2_t _tmp1 = vnclip_wx_i16m2(_mulh, 0, vl);
            vint8m1_t _res = vnclip_wx_i8m1(_tmp1, 0, vl);
            vse8_v_i8m1(out_ptr, _res, vl);
            q1q2_ptr += vl;
            z1q2_ptr += vl;
            q1z2_ptr += vl;
            out_ptr += vl;
            c += vl;
        }
    }
}

/*************************************************************
 * mrows = rlenb / 4
 * mcols = rlenb / sizeof(int8_t)
 * msize_m: mrows, m_tail
 * msize_n: mrows, n_tail
 * msize_k: mcols, k_tail
 *
 * dst - output: [M, N]
 * sa - mat0:    [M, K]
 * sb - mat1:    [N/msize_n, K/msize_k, msize_n, msize_k]
 ************************************************************/
void shl_rvm_matmul_a0b0_int8_to_int32(int8_t *dst, int8_t *sa, int8_t *sb, int M, int K, int N,
                                       int32_t z1, int32_t z2, int32_t z3, int32_t mult,
                                       int32_t shift)
{
    int mcols = csrr_xrlenb();
    int mrows = mcols / 4;

    int8_t z1_i8 = (int8_t)z1;
    int8_t z2_i8 = (int8_t)z2;

    int stride_a = K * sizeof(int8_t);
    int stride_c = N * sizeof(int32_t);

    int32_t *q1q2 = (int32_t *)shl_mem_alloc(mrows * N * sizeof(int32_t));
    int32_t *z1q2 = (int32_t *)shl_mem_alloc(mrows * N * sizeof(int32_t));
    int32_t *q1z2 = (int32_t *)shl_mem_alloc(mrows * N * sizeof(int32_t));
    int32_t z1z2 = z1 * z2 * K;

    int i = 0;
    // m = mrows/m_tail
    while (i < M) {
        uint8_t msize_m = (M - i >= mrows) ? mrows : (M - i);
        int8_t *sa_ptr = sa + i * K;

        int j = 0;
        // n = mrows/n_tail
        while (j < N) {
            uint8_t msize_n = (N - j >= mrows) ? mrows : (N - j);
            mcfgn(msize_n);
            int8_t *a0_ptr = sa_ptr;
            int8_t *k_ptr = sb + j * K;
            int32_t *q1q2_ptr = q1q2 + j;
            int32_t *z1q2_ptr = z1q2 + j;
            int32_t *q1z2_ptr = q1z2 + j;

            mcfgk(msize_n * sizeof(int32_t));
            mcfgm(msize_m);
            mint32_t m4 = mzero_mi32();
            mint32_t m5 = mzero_mi32();
            mint32_t m6 = mzero_mi32();

            mcfgk(mcols * sizeof(int8_t));
            mint8_t m_z1 = mdup_i8(z1_i8);
            mint8_t m_z2 = mdup_i8(z2_i8);

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(int8_t));
                mint8_t m0 = mld_i8(a0_ptr, stride_a);  // a00
                a0_ptr += msize_k;

                mcfgm(msize_n);
                mint8_t m2 = msld_i8(k_ptr, msize_k * sizeof(int8_t));  // b00
                k_ptr += msize_n * msize_k;

                mcfgm(msize_m);
                m4 = mmaqa_mi8(m4, m2, m0);    // q1 * q2
                m5 = mmaqa_mi8(m5, m2, m_z1);  // z1 * q2
                m6 = mmaqa_mi8(m6, m_z2, m0);  // q1 * z2
                c += msize_k;
            }

            mcfgk(msize_n * sizeof(int32_t));
            msst_i32_mi32(q1q2_ptr, stride_c, m4);
            msst_i32_mi32(z1q2_ptr, stride_c, m5);
            msst_i32_mi32(q1z2_ptr, stride_c, m6);
            j += msize_n;
        }

        // sum(q1q2, z1q2, q1z2, z1z2) and requantize
        sum_and_requantize_m4(dst + i * N, q1q2, z1q2, q1z2, z1z2, msize_m, N, z3, mult, shift);
        i += msize_m;
    }
}
#endif  // MATRIX_PW_I32

/*************************************************************
 * mrows = rlenb / 4
 * mcols = rlenb / sizeof(int8_t)
 * msize_n: mrows, n_tail
 * msize_k: mcols, k_tail
 *
 * src: [k, n]
 * dst: [n/msize_n, k/msize_k, msize_n, msize_k]
 ************************************************************/
static void reorder_mat1_mrows_mcols_int8(int8_t *src, int8_t *dst, int k, int n)
{
    int mcols = csrr_xrlenb();
    int mrows = mcols / 4;

    int r = 0;
    while (r < n) {
        int msize_n = (n - r >= mrows) ? mrows : (n - r);
        int c = 0;
        while (c < k) {
            uint16_t msize_k = (k - c >= mcols) ? mcols : (k - c);
            for (int i = 0; i < msize_n; i++) {
                int8_t *s_ptr = src + c * n + r + i;
                int size = msize_k;
                while (size > 0) {
                    int vl = vsetvl_e8m4(size);
                    vint8m4_t _src = vlse8_v_i8m4(s_ptr, n * sizeof(int8_t), vl);
                    vse8_v_i8m4(dst, _src, vl);
                    s_ptr += vl * n;
                    dst += vl;
                    size -= vl;
                }
            }
            c += msize_k;
        }
        r += msize_n;
    }
}

void shl_rvm_matmul_reorder_weight_int8(struct csinn_tensor *mat1)
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
        reorder_mat1_mrows_mcols_int8(init_mat, mat_reorder, k, n);
        memcpy(init_mat, mat_reorder, k * n * sizeof(int8_t));
    }

    shl_mem_free(mat_reorder);
}

int shl_rvm_matmul_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                        struct csinn_tensor *output, struct csinn_matmul_params *params)
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

    void (*gemm_a0b0_int8)();
#ifdef MATRIX_PW_I32
    gemm_a0b0_int8 = shl_rvm_matmul_a0b0_int8_pw_i32;
#else
    gemm_a0b0_int8 = shl_rvm_matmul_a0b0_int8_to_int32;
#endif  // MATRIX_PW_I32

    if (!params->trans_a && !params->trans_b) {
        if (batches_a == batches_b) {
            int8_t *in1;
            if (!(mat1->is_const)) {
                in1 = (int8_t *)shl_mem_alloc(dim_k * dim_n * sizeof(int8_t));
            }

            for (int b = 0; b < batches_a; b++) {
                if (!(mat1->is_const)) {
                    reorder_mat1_mrows_mcols_int8(mat1_data, in1, dim_k, dim_n);
                } else {
                    in1 = mat1_data;
                }
                gemm_a0b0_int8(output_data, mat0_data, in1, dim_m, dim_k, dim_n, z1, z2, z3,
                               multiplier, shift);
                mat0_data += dim_m * dim_k;
                mat1_data += dim_k * dim_n;
                output_data += dim_m * dim_n;
            }

            if (!(mat1->is_const)) {
                shl_mem_free(in1);
            }
        } else if (batches_a > 1 && batches_b == 1) {
            int8_t *in1;
            if (!(mat1->is_const)) {
                in1 = (int8_t *)shl_mem_alloc(dim_k * dim_n * sizeof(int8_t));
                reorder_mat1_mrows_mcols_int8(mat1_data, in1, dim_k, dim_n);
            } else {
                in1 = mat1_data;
            }

            for (int b = 0; b < batches_a; b++) {
                gemm_a0b0_int8(output_data, mat0_data, in1, dim_m, dim_k, dim_n, z1, z2, z3,
                               multiplier, shift);

                mat0_data += dim_m * dim_k;
                output_data += dim_m * dim_n;
            }

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
