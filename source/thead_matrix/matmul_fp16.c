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

/*************************************************************
 * mrows = rlenb / 4
 * m2rows = mrows * 2
 * mcols = rlenb / sizeof(__fp16)
 * msize_m: m2rows, mrows, m_tail
 * msize_n: m2rows, mrows, n_tail
 * msize_k: mcols, k_tail
 *
 * dst - output: [M, N]
 * sa - mat0:    [M, K]
 * sb - mat1:    [N/msize_n, K/msize_k, msize_n, msize_k]
 ************************************************************/
void shl_rvm_matmul_a0b0_fp16(__fp16 *dst, __fp16 *sa, __fp16 *sb, int M, int K, int N)
{
    int mrows = mread_csr(RVM_XRLENB) / 4;
    int m2rows = mrows * 2;
    int mcols = m2rows;

    int stride_a = K * sizeof(__fp16);
    int stride_c = N * sizeof(__fp16);

    int i = 0;
    // m = m2rows
    uint8_t msize_m = mrows;
    for (; i + m2rows - 1 < M; i += m2rows) {
        __fp16 *sa_ptr = sa + i * K;

        int j = 0;
        // n = m2rows
        uint8_t msize_n = m2rows;
        mcfgn(msize_n);
        for (; j + m2rows - 1 < N; j += m2rows) {
            __fp16 *a0_ptr = sa_ptr;
            __fp16 *a1_ptr = a0_ptr + mrows * K;
            __fp16 *b_ptr = sb + j * K;
            __fp16 *c0_ptr = dst + i * N + j;
            __fp16 *c1_ptr = c0_ptr + mrows * N;

            mcfgk(msize_n * sizeof(__fp16));
            mcfgm(msize_m);
            // mfloat16_t m4 = mmov_mf16v(m0, 0);
            // mfloat16_t m5 = mmov_mf16v(m0, 0);
            mfloat16_t m4 = mzero_mf16();
            mfloat16_t m5 = mzero_mf16();

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(__fp16));
                mfloat16_t m0 = mld_f16(a0_ptr, stride_a);  // a00
                mfloat16_t m1 = mld_f16(a1_ptr, stride_a);  // a10
                a0_ptr += msize_k;
                a1_ptr += msize_k;

                // mcfgm(mrows);  // msizem == mrows
                mfloat16_t m2_0 = msld_f16(b_ptr, msize_k * sizeof(__fp16));  // b00
                mfloat16_t m2_1 =
                    msld_f16(b_ptr + mrows * msize_k, msize_k * sizeof(__fp16));  // b00 (half)
                mfloat16x2_t m2 = mmov_mf16x2(m2_0, m2_1);
                b_ptr += msize_n * msize_k;

                // mcfgm(msize_m);  // msizem == mrows
                m4 = fmmacc_mf16x2_mf16(m4, m2, m0);  // c00 += a00 * b00
                m5 = fmmacc_mf16x2_mf16(m5, m2, m1);  // c10 += a10 * b00
                c += msize_k;
            }

            mcfgk(msize_n * sizeof(__fp16));
            msst_f16_mf16(c0_ptr, stride_c, m4);
            msst_f16_mf16(c1_ptr, stride_c, m5);
        }

        // n = mrows/n_tail
        while (j < N) {
            uint8_t msize_n = (N - j >= mrows) ? mrows : (N - j);
            mcfgn(msize_n);
            __fp16 *a0_ptr = sa_ptr;
            __fp16 *a1_ptr = a0_ptr + mrows * K;
            __fp16 *b_ptr = sb + j * K;
            __fp16 *c0_ptr = dst + i * N + j;
            __fp16 *c1_ptr = c0_ptr + mrows * N;

            mcfgk(msize_n * sizeof(__fp16));
            mcfgm(msize_m);
            mfloat16_t m4 = mzero_mf16();
            mfloat16_t m5 = mzero_mf16();

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(__fp16));
                mfloat16_t m0 = mld_f16(a0_ptr, stride_a);  // a00
                mfloat16_t m1 = mld_f16(a1_ptr, stride_a);  // a10
                a0_ptr += msize_k;
                a1_ptr += msize_k;

                mcfgm(msize_n);
                mfloat16_t m2_0 = msld_f16(b_ptr, msize_k * sizeof(__fp16));  // b00
                mfloat16x2_t m2 = mmov_mf16x2(m2_0, mundefined_mf16());
                b_ptr += msize_n * msize_k;

                mcfgm(msize_m);
                m4 = fmmacc_mf16x2_mf16(m4, m2, m0);  // c00 += a00 * b00
                m5 = fmmacc_mf16x2_mf16(m5, m2, m1);  // c10 += a10 * b00
                c += msize_k;
            }

            mcfgk(msize_n * sizeof(__fp16));
            msst_f16_mf16(c0_ptr, stride_c, m4);
            msst_f16_mf16(c1_ptr, stride_c, m5);
            j += msize_n;
        }
    }

    // m = mrows/m_tail
    while (i < M) {
        uint8_t msize_m = (M - i >= mrows) ? mrows : (M - i);
        __fp16 *sa_ptr = sa + i * K;

        int j = 0;
        // n = m2rows
        uint8_t msize_n = m2rows;
        mcfgn(msize_n);
        for (; j + m2rows - 1 < N; j += m2rows) {
            __fp16 *a0_ptr = sa_ptr;
            __fp16 *b_ptr = sb + j * K;
            __fp16 *c0_ptr = dst + i * N + j;

            mcfgk(msize_n * sizeof(__fp16));
            mcfgm(msize_m);
            mfloat16_t m4 = mzero_mf16();

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(__fp16));
                mfloat16_t m0 = mld_f16(a0_ptr, stride_a);  // a00
                a0_ptr += msize_k;

                mcfgm(mrows);
                mfloat16_t m2_0 = msld_f16(b_ptr, msize_k * sizeof(__fp16));  // b00
                mfloat16_t m2_1 =
                    msld_f16(b_ptr + mrows * msize_k, msize_k * sizeof(__fp16));  // b00 (half)
                mfloat16x2_t m2 = mmov_mf16x2(m2_0, m2_1);
                b_ptr += msize_n * msize_k;

                mcfgm(msize_m);
                m4 = fmmacc_mf16x2_mf16(m4, m2, m0);  // c00 += a00 * b00
                c += msize_k;
            }

            mcfgk(msize_n * sizeof(__fp16));
            msst_f16_mf16(c0_ptr, stride_c, m4);
        }

        // n = mrows/n_tail
        while (j < N) {
            uint8_t msize_n = (N - j >= mrows) ? mrows : (N - j);
            mcfgn(msize_n);
            __fp16 *a0_ptr = sa_ptr;
            __fp16 *b_ptr = sb + j * K;
            __fp16 *c0_ptr = dst + i * N + j;

            mcfgk(msize_n * sizeof(__fp16));
            mcfgm(msize_m);
            mfloat16_t m4 = mzero_mf16();

            int c = 0;
            // k = mcols/k_tail
            while (c < K) {
                uint16_t msize_k = (K - c >= mcols) ? mcols : (K - c);
                mcfgk(msize_k * sizeof(__fp16));
                mfloat16_t m0 = mld_f16(a0_ptr, stride_a);  // a00
                a0_ptr += msize_k;

                mcfgm(msize_n);
                mfloat16_t m2_0 = msld_f16(b_ptr, msize_k * sizeof(__fp16));  // b00
                mfloat16x2_t m2 = mmov_mf16x2(m2_0, mundefined_mf16());
                b_ptr += msize_n * msize_k;

                mcfgm(msize_m);
                m4 = fmmacc_mf16x2_mf16(m4, m2, m0);  // c00 += a00 * b00
                c += msize_k;
            }

            mcfgk(msize_n * sizeof(__fp16));
            msst_f16_mf16(c0_ptr, stride_c, m4);
            j += msize_n;
        }
        i += msize_m;
    }
}

/*************************************************************
 * mrows = rlenb / 4
 * m2rows = mrows * 2
 * mcols = rlenb / sizeof(__fp16)
 * msize_n: m2rows, mrows, n_tail
 * msize_k: mcols, k_tail
 *
 * src: [k, n]
 * dst: [n/msize_n, k/msize_k, msize_n, msize_k]
 ************************************************************/
static void reorder_mat1_m2rows_mcols_fp16(__fp16 *src, __fp16 *dst, int k, int n)
{
    int mrows = csrr_xrlenb() / 4;
    int m2rows = mrows * 2;
    int mcols = m2rows;

    int r = 0;
    // m2rows
    for (; r + m2rows - 1 < n; r += m2rows) {
        int c = 0;
        while (c < k) {
            uint16_t msize_k = (k - c >= mcols) ? mcols : (k - c);
            for (int i = 0; i < m2rows; i++) {
                __fp16 *s_ptr = src + c * n + r + i;
                int size = msize_k;
                while (size > 0) {
                    int vl = vsetvl_e16m4(size);
                    vfloat16m4_t _src = vlse16_v_f16m4(s_ptr, n * sizeof(__fp16), vl);
                    vse16_v_f16m4(dst, _src, vl);
                    s_ptr += vl * n;
                    dst += vl;
                    size -= vl;
                }
            }
            c += msize_k;
        }
    }
    while (r < n) {
        int msize_n = (n - r >= mrows) ? mrows : (n - r);
        int c = 0;
        while (c < k) {
            uint16_t msize_k = (k - c >= mcols) ? mcols : (k - c);
            for (int i = 0; i < msize_n; i++) {
                __fp16 *s_ptr = src + c * n + r + i;
                int size = msize_k;
                while (size > 0) {
                    int vl = vsetvl_e16m4(size);
                    vfloat16m4_t _src = vlse16_v_f16m4(s_ptr, n * sizeof(__fp16), vl);
                    vse16_v_f16m4(dst, _src, vl);
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

/*************************************************************
 * mrows = rlenb / 4
 * m2rows = mrows * 2
 * mcols = rlenb / sizeof(__fp16)
 * msize_n: m2rows, mrows, n_tail
 * msize_k: mcols, k_tail
 *
 * src: [k, n]
 * dst: [n/msize_n, k/msize_k, msize_n, msize_k]
 ************************************************************/
static void reorder_mat1_m2rows_mcols_fp16_w_int8(int8_t *src, int8_t *dst, int k, int n)
{
    int mrows = csrr_xrlenb() / 4;
    int m2rows = mrows * 2;
    int mcols = m2rows;

    int r = 0;
    // m2rows
    for (; r + m2rows - 1 < n; r += m2rows) {
        int c = 0;
        while (c < k) {
            uint16_t msize_k = (k - c >= mcols) ? mcols : (k - c);
            for (int i = 0; i < m2rows; i++) {
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
    }
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

void shl_rvm_matmul_reorder_weight_fp16(struct csinn_tensor *mat1)
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
        reorder_mat1_m2rows_mcols_fp16(init_mat, mat_reorder, k, n);
        memcpy(init_mat, mat_reorder, k * n * sizeof(__fp16));
    }

    shl_mem_free(mat_reorder);
}

void shl_rvm_matmul_reorder_weight_fp16_w_int8(struct csinn_tensor *mat1)
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
        reorder_mat1_m2rows_mcols_fp16_w_int8(init_mat, mat_reorder, k, n);
        memcpy(init_mat, mat_reorder, k * n * sizeof(int8_t));
    }

    shl_mem_free(mat_reorder);
}

int shl_rvm_matmul_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
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
            __fp16 *in1;
            if (!(mat1->is_const)) {
                in1 = (__fp16 *)shl_mem_alloc(dim_k * dim_n * sizeof(__fp16));
            }

            for (int b = 0; b < batches_a; b++) {
                if (!(mat1->is_const)) {
                    reorder_mat1_m2rows_mcols_fp16(mat1_data, in1, dim_k, dim_n);
                } else {
                    in1 = mat1_data;
                }
                shl_rvm_matmul_a0b0_fp16(output_data, mat0_data, in1, dim_m, dim_k, dim_n);
                mat0_data += dim_m * dim_k;
                mat1_data += dim_k * dim_n;
                output_data += dim_m * dim_n;
            }

            if (!(mat1->is_const)) {
                shl_mem_free(in1);
            }
            // requantize
            shl_rvv_sidcso_op_requantize_fp16(mat0, output, mat1);
        } else if (batches_a > 1 && batches_b == 1) {
            __fp16 *in1;
            if (!(mat1->is_const)) {
                in1 = (__fp16 *)shl_mem_alloc(dim_k * dim_n * sizeof(__fp16));
                reorder_mat1_m2rows_mcols_fp16(mat1_data, in1, dim_k, dim_n);
            } else {
                in1 = mat1_data;
            }

            for (int b = 0; b < batches_a; b++) {
                shl_rvm_matmul_a0b0_fp16(output_data, mat0_data, in1, dim_m, dim_k, dim_n);
                mat0_data += dim_m * dim_k;
                output_data += dim_m * dim_n;
            }

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

int shl_rvm_matmul_fp16_w_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
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
    int size1 = csinn_tensor_size(mat1);

    if (!params->trans_a && !params->trans_b) {
        if (batches_a == batches_b) {
            __fp16 *in1 = (__fp16 *)shl_mem_alloc(size1 * sizeof(__fp16));
            shl_rvv_dequantize_i8_to_f16(mat1_data, in1, size1, zp, scale);

            for (int b = 0; b < batches_a; b++) {
                shl_rvm_matmul_a0b0_fp16(output_data, mat0_data, in1 + b * dim_k * dim_n, dim_m,
                                         dim_k, dim_n);
                mat0_data += dim_m * dim_k;
                output_data += dim_m * dim_n;
            }

            shl_mem_free(in1);
        } else if (batches_a > 1 && batches_b == 1) {
            __fp16 *in1 = (__fp16 *)shl_mem_alloc(size1 * sizeof(__fp16));
            shl_rvv_dequantize_i8_to_f16(mat1_data, in1, size1, zp, scale);

            for (int b = 0; b < batches_a; b++) {
                shl_rvm_matmul_a0b0_fp16(output_data, mat0_data, in1, dim_m, dim_k, dim_n);
                mat0_data += dim_m * dim_k;
                output_data += dim_m * dim_n;
            }

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
