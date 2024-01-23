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
#include "rvv_mathfun_fp16.h"

static inline void qk_t1_dot_4x4_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, int M, int K,
                                      int N, int lda, int ldb, int ldc);
static inline void trans_q_0132_fp16(__fp16 *src, __fp16 *dst, int sv, int head_dim);
static void q0k1_softmax_v1_fp16(__fp16 *q, __fp16 *k, __fp16 *v, __fp16 *o,
                                 struct csinn_scale_dot_attention_params *params, int32_t sq,
                                 int32_t sk, int32_t head_dim);

int shl_rvv_scaled_dot_product_attention_fp16(struct csinn_tensor *query, struct csinn_tensor *key,
                                              struct csinn_tensor *value,
                                              struct csinn_tensor *output_tensor,
                                              struct csinn_scale_dot_attention_params *params)
{
    __fp16 *query_data = query->data;
    __fp16 *key_data = key->data;
    __fp16 *value_data = value->data;
    __fp16 *output_data = output_tensor->data;
    // np: number of heads
    // sk: sequence number of kv
    // sq: sequence number of q
    int32_t batch = query->dim[0];  // batch = 1 only
    int32_t np = query->dim[1];
    int32_t sk = key->dim[2];
    int32_t sq = query->dim[2];
    int32_t head_dim = query->dim[3];

    if (shl_multithread_is_enable()) {
#pragma omp parallel for
        for (int i = 0; i < batch * np; i++) {
            __fp16 *q = query_data + i * sq * head_dim;
            __fp16 *k = key_data + i * sk * head_dim;
            __fp16 *v = value_data + i * sk * head_dim;
            __fp16 *o = output_data + i * sq * head_dim;
            if (params->transpose_v == 0) {
                __fp16 *value_transpose_tmp = malloc(sk * head_dim * sizeof(__fp16));
                trans_q_0132_fp16(v, value_transpose_tmp, sk, head_dim);
                q0k1_softmax_v1_fp16(q, k, value_transpose_tmp, o, params, sq, sk, head_dim);
                free(value_transpose_tmp);
            } else {
                q0k1_softmax_v1_fp16(q, k, v, o, params, sq, sk, head_dim);
            }
        }
    } else {
        for (int i = 0; i < batch * np; i++) {
            __fp16 *q = query_data + i * sq * head_dim;
            __fp16 *k = key_data + i * sk * head_dim;
            __fp16 *v = value_data + i * sk * head_dim;
            __fp16 *o = output_data + i * sq * head_dim;
            if (params->transpose_v == 0) {
                __fp16 *value_transpose_tmp = malloc(sk * head_dim * sizeof(__fp16));
                trans_q_0132_fp16(v, value_transpose_tmp, sk, head_dim);
                q0k1_softmax_v1_fp16(q, k, value_transpose_tmp, o, params, sq, sk, head_dim);
                free(value_transpose_tmp);
            } else {
                q0k1_softmax_v1_fp16(q, k, v, o, params, sq, sk, head_dim);
            }
        }
    }
    return CSINN_TRUE;
}

static inline void trans_q_0132_fp16(__fp16 *src, __fp16 *dst, int sv, int head_dim)
{
    for (int i = 0; i < sv; i++) {
        int size = head_dim;
        __fp16 *d_ptr = dst + i;
        while (size > 0) {
            int vl = vsetvl_e16m4(size);
            vfloat16m4_t _in = vle16_v_f16m4(src, vl);
            src += vl;
            vsse16_v_f16m4(d_ptr, sv * sizeof(__fp16), _in, vl);
            d_ptr += vl * sv;
            size -= vl;
        }
    }
    dst += head_dim * sv;
}

static inline void qk_t1_dot_4x4_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, int M, int K,
                                      int N, int lda, int ldb, int ldc)
{
    int i = 0;
    for (; i + 3 < M; i += 4) {
        const __fp16 *sa_ptr = sa + i * lda;
        int j = 0;
        for (; j + 3 < N; j += 4) {
            const __fp16 *a0_ptr = sa_ptr;
            const __fp16 *a1_ptr = sa_ptr + 1 * lda;
            const __fp16 *a2_ptr = sa_ptr + 2 * lda;
            const __fp16 *a3_ptr = sa_ptr + 3 * lda;
            const __fp16 *b0_ptr = sb + j * ldb;
            const __fp16 *b1_ptr = b0_ptr + 1 * ldb;
            const __fp16 *b2_ptr = b0_ptr + 2 * ldb;
            const __fp16 *b3_ptr = b0_ptr + 3 * ldb;

            int vlmax = vsetvl_e16m1(csrr_vlenb() / sizeof(__fp16));
            // dst[m, 0]
            vfloat16m1_t _acc00 = vfmv_v_f_f16m1(0.0f, vlmax);
            vfloat16m1_t _acc10 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc20 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc30 = vmv_v_v_f16m1(_acc00, vlmax);
            // dst[m, 1]
            vfloat16m1_t _acc01 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc11 = vmv_v_v_f16m1(_acc01, vlmax);
            vfloat16m1_t _acc21 = vmv_v_v_f16m1(_acc01, vlmax);
            vfloat16m1_t _acc31 = vmv_v_v_f16m1(_acc01, vlmax);
            // dst[m, 2]
            vfloat16m1_t _acc02 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc12 = vmv_v_v_f16m1(_acc02, vlmax);
            vfloat16m1_t _acc22 = vmv_v_v_f16m1(_acc02, vlmax);
            vfloat16m1_t _acc32 = vmv_v_v_f16m1(_acc02, vlmax);
            // dst[m, 3]
            vfloat16m1_t _acc03 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc13 = vmv_v_v_f16m1(_acc03, vlmax);
            vfloat16m1_t _acc23 = vmv_v_v_f16m1(_acc03, vlmax);
            vfloat16m1_t _acc33 = vmv_v_v_f16m1(_acc03, vlmax);

            int c = 0;
            while (c < K) {
                int vl = vsetvl_e16m1(K - c);
                vfloat16m1_t _a0 = vle16_v_f16m1(a0_ptr + c, vl);
                vfloat16m1_t _a1 = vle16_v_f16m1(a1_ptr + c, vl);
                vfloat16m1_t _a2 = vle16_v_f16m1(a2_ptr + c, vl);
                vfloat16m1_t _a3 = vle16_v_f16m1(a3_ptr + c, vl);
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr + c, vl);
                vfloat16m1_t _b1 = vle16_v_f16m1(b1_ptr + c, vl);
                vfloat16m1_t _b2 = vle16_v_f16m1(b2_ptr + c, vl);
                vfloat16m1_t _b3 = vle16_v_f16m1(b3_ptr + c, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vlmax);
                _acc10 = vfmacc_vv_f16m1(_acc10, _a1, _b0, vlmax);
                _acc20 = vfmacc_vv_f16m1(_acc20, _a2, _b0, vlmax);
                _acc30 = vfmacc_vv_f16m1(_acc30, _a3, _b0, vlmax);

                _acc01 = vfmacc_vv_f16m1(_acc01, _a0, _b1, vlmax);
                _acc11 = vfmacc_vv_f16m1(_acc11, _a1, _b1, vlmax);
                _acc21 = vfmacc_vv_f16m1(_acc21, _a2, _b1, vlmax);
                _acc31 = vfmacc_vv_f16m1(_acc31, _a3, _b1, vlmax);

                _acc02 = vfmacc_vv_f16m1(_acc02, _a0, _b2, vlmax);
                _acc12 = vfmacc_vv_f16m1(_acc12, _a1, _b2, vlmax);
                _acc22 = vfmacc_vv_f16m1(_acc22, _a2, _b2, vlmax);
                _acc32 = vfmacc_vv_f16m1(_acc32, _a3, _b2, vlmax);

                _acc03 = vfmacc_vv_f16m1(_acc03, _a0, _b3, vlmax);
                _acc13 = vfmacc_vv_f16m1(_acc13, _a1, _b3, vlmax);
                _acc23 = vfmacc_vv_f16m1(_acc23, _a2, _b3, vlmax);
                _acc33 = vfmacc_vv_f16m1(_acc33, _a3, _b3, vlmax);
                c += vl;
            }

            int idx00 = (i + 0) * ldc + (j + 0);
            int idx10 = (i + 1) * ldc + (j + 0);
            int idx20 = (i + 2) * ldc + (j + 0);
            int idx30 = (i + 3) * ldc + (j + 0);

            int idx01 = (i + 0) * ldc + (j + 1);
            int idx11 = (i + 1) * ldc + (j + 1);
            int idx21 = (i + 2) * ldc + (j + 1);
            int idx31 = (i + 3) * ldc + (j + 1);

            int idx02 = (i + 0) * ldc + (j + 2);
            int idx12 = (i + 1) * ldc + (j + 2);
            int idx22 = (i + 2) * ldc + (j + 2);
            int idx32 = (i + 3) * ldc + (j + 2);

            int idx03 = (i + 0) * ldc + (j + 3);
            int idx13 = (i + 1) * ldc + (j + 3);
            int idx23 = (i + 2) * ldc + (j + 3);
            int idx33 = (i + 3) * ldc + (j + 3);

            // dst[m, 0]
            vfloat16m1_t _sum00;
            vfloat16m1_t _sum10;
            vfloat16m1_t _sum20;
            vfloat16m1_t _sum30;
            // dst[m, 1]
            vfloat16m1_t _sum01;
            vfloat16m1_t _sum11;
            vfloat16m1_t _sum21;
            vfloat16m1_t _sum31;
            // dst[m, 2]
            vfloat16m1_t _sum02;
            vfloat16m1_t _sum12;
            vfloat16m1_t _sum22;
            vfloat16m1_t _sum32;
            // dst[m, 3]
            vfloat16m1_t _sum03;
            vfloat16m1_t _sum13;
            vfloat16m1_t _sum23;
            vfloat16m1_t _sum33;

            _sum00 = vfmv_v_f_f16m1(dst[idx00], 1);
            _sum10 = vfmv_v_f_f16m1(dst[idx10], 1);
            _sum20 = vfmv_v_f_f16m1(dst[idx20], 1);
            _sum30 = vfmv_v_f_f16m1(dst[idx30], 1);

            _sum01 = vfmv_v_f_f16m1(dst[idx01], 1);
            _sum11 = vfmv_v_f_f16m1(dst[idx11], 1);
            _sum21 = vfmv_v_f_f16m1(dst[idx21], 1);
            _sum31 = vfmv_v_f_f16m1(dst[idx31], 1);

            _sum02 = vfmv_v_f_f16m1(dst[idx02], 1);
            _sum12 = vfmv_v_f_f16m1(dst[idx12], 1);
            _sum22 = vfmv_v_f_f16m1(dst[idx22], 1);
            _sum32 = vfmv_v_f_f16m1(dst[idx32], 1);

            _sum03 = vfmv_v_f_f16m1(dst[idx03], 1);
            _sum13 = vfmv_v_f_f16m1(dst[idx13], 1);
            _sum23 = vfmv_v_f_f16m1(dst[idx23], 1);
            _sum33 = vfmv_v_f_f16m1(dst[idx33], 1);

            _sum00 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc00, _sum00, vlmax);
            _sum10 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc10, _sum10, vlmax);
            _sum20 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc20, _sum20, vlmax);
            _sum30 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc30, _sum30, vlmax);

            _sum01 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc01, _sum01, vlmax);
            _sum11 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc11, _sum11, vlmax);
            _sum21 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc21, _sum21, vlmax);
            _sum31 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc31, _sum31, vlmax);

            _sum02 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc02, _sum02, vlmax);
            _sum12 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc12, _sum12, vlmax);
            _sum22 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc22, _sum22, vlmax);
            _sum32 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc32, _sum32, vlmax);

            _sum03 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc03, _sum03, vlmax);
            _sum13 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc13, _sum13, vlmax);
            _sum23 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc23, _sum23, vlmax);
            _sum33 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc33, _sum33, vlmax);

            dst[idx00] = vfmv_f_s_f16m1_f16(_sum00);
            dst[idx10] = vfmv_f_s_f16m1_f16(_sum10);
            dst[idx20] = vfmv_f_s_f16m1_f16(_sum20);
            dst[idx30] = vfmv_f_s_f16m1_f16(_sum30);

            dst[idx01] = vfmv_f_s_f16m1_f16(_sum01);
            dst[idx11] = vfmv_f_s_f16m1_f16(_sum11);
            dst[idx21] = vfmv_f_s_f16m1_f16(_sum21);
            dst[idx31] = vfmv_f_s_f16m1_f16(_sum31);

            dst[idx02] = vfmv_f_s_f16m1_f16(_sum02);
            dst[idx12] = vfmv_f_s_f16m1_f16(_sum12);
            dst[idx22] = vfmv_f_s_f16m1_f16(_sum22);
            dst[idx32] = vfmv_f_s_f16m1_f16(_sum32);

            dst[idx03] = vfmv_f_s_f16m1_f16(_sum03);
            dst[idx13] = vfmv_f_s_f16m1_f16(_sum13);
            dst[idx23] = vfmv_f_s_f16m1_f16(_sum23);
            dst[idx33] = vfmv_f_s_f16m1_f16(_sum33);
        }
        for (; j < N; j++) {
            const __fp16 *a0_ptr = sa_ptr;
            const __fp16 *a1_ptr = sa_ptr + 1 * lda;
            const __fp16 *a2_ptr = sa_ptr + 2 * lda;
            const __fp16 *a3_ptr = sa_ptr + 3 * lda;
            const __fp16 *b0_ptr = sb + j * ldb;

            int vlmax = vsetvl_e16m1(csrr_vlenb() / sizeof(__fp16));
            // dst[m, 0]
            vfloat16m1_t _acc00 = vfmv_v_f_f16m1(0.0f, vlmax);
            vfloat16m1_t _acc10 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc20 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc30 = vmv_v_v_f16m1(_acc00, vlmax);

            int c = 0;
            while (c < K) {
                int vl = vsetvl_e16m1(K - c);
                vfloat16m1_t _a0 = vle16_v_f16m1(a0_ptr + c, vl);
                vfloat16m1_t _a1 = vle16_v_f16m1(a1_ptr + c, vl);
                vfloat16m1_t _a2 = vle16_v_f16m1(a2_ptr + c, vl);
                vfloat16m1_t _a3 = vle16_v_f16m1(a3_ptr + c, vl);
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr + c, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vlmax);
                _acc10 = vfmacc_vv_f16m1(_acc10, _a1, _b0, vlmax);
                _acc20 = vfmacc_vv_f16m1(_acc20, _a2, _b0, vlmax);
                _acc30 = vfmacc_vv_f16m1(_acc30, _a3, _b0, vlmax);
                c += vl;
            }

            int idx00 = (i + 0) * ldc + (j + 0);
            int idx10 = (i + 1) * ldc + (j + 0);
            int idx20 = (i + 2) * ldc + (j + 0);
            int idx30 = (i + 3) * ldc + (j + 0);

            // dst[m, 0]
            vfloat16m1_t _sum00;
            vfloat16m1_t _sum10;
            vfloat16m1_t _sum20;
            vfloat16m1_t _sum30;

            _sum00 = vfmv_v_f_f16m1(dst[idx00], 1);
            _sum10 = vfmv_v_f_f16m1(dst[idx10], 1);
            _sum20 = vfmv_v_f_f16m1(dst[idx20], 1);
            _sum30 = vfmv_v_f_f16m1(dst[idx30], 1);

            _sum00 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc00, _sum00, vlmax);
            _sum10 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc10, _sum10, vlmax);
            _sum20 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc20, _sum20, vlmax);
            _sum30 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc30, _sum30, vlmax);

            dst[idx00] = vfmv_f_s_f16m1_f16(_sum00);
            dst[idx10] = vfmv_f_s_f16m1_f16(_sum10);
            dst[idx20] = vfmv_f_s_f16m1_f16(_sum20);
            dst[idx30] = vfmv_f_s_f16m1_f16(_sum30);
        }
    }
    for (; i < M; i += 1) {
        const __fp16 *sa_ptr = sa + i * lda;
        int j = 0;
        for (; j + 3 < N; j += 4) {
            const __fp16 *a0_ptr = sa_ptr;
            const __fp16 *b0_ptr = sb + j * ldb;
            const __fp16 *b1_ptr = b0_ptr + 1 * ldb;
            const __fp16 *b2_ptr = b0_ptr + 2 * ldb;
            const __fp16 *b3_ptr = b0_ptr + 3 * ldb;

            int vlmax = vsetvl_e16m1(csrr_vlenb() / sizeof(__fp16));
            // dst[0, n]
            vfloat16m1_t _acc00 = vfmv_v_f_f16m1(0.0f, vlmax);
            vfloat16m1_t _acc01 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc02 = vmv_v_v_f16m1(_acc00, vlmax);
            vfloat16m1_t _acc03 = vmv_v_v_f16m1(_acc00, vlmax);

            int c = 0;
            while (c < K) {
                int vl = vsetvl_e16m1(K - c);
                vfloat16m1_t _a0 = vle16_v_f16m1(a0_ptr + c, vl);
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr + c, vl);
                vfloat16m1_t _b1 = vle16_v_f16m1(b1_ptr + c, vl);
                vfloat16m1_t _b2 = vle16_v_f16m1(b2_ptr + c, vl);
                vfloat16m1_t _b3 = vle16_v_f16m1(b3_ptr + c, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vlmax);
                _acc01 = vfmacc_vv_f16m1(_acc01, _a0, _b1, vlmax);
                _acc02 = vfmacc_vv_f16m1(_acc02, _a0, _b2, vlmax);
                _acc03 = vfmacc_vv_f16m1(_acc03, _a0, _b3, vlmax);
                c += vl;
            }

            int idx00 = (i + 0) * ldc + (j + 0);
            int idx01 = (i + 0) * ldc + (j + 1);
            int idx02 = (i + 0) * ldc + (j + 2);
            int idx03 = (i + 0) * ldc + (j + 3);

            // dst[0, n]
            vfloat16m1_t _sum00;
            vfloat16m1_t _sum01;
            vfloat16m1_t _sum02;
            vfloat16m1_t _sum03;

            // _sum00 = vfmv_v_f_f16m1(dst[idx00], 1);
            // _sum01 = vfmv_v_f_f16m1(dst[idx01], 1);
            // _sum02 = vfmv_v_f_f16m1(dst[idx02], 1);
            // _sum03 = vfmv_v_f_f16m1(dst[idx03], 1);
            _sum00 = vfmv_v_f_f16m1(0.0f, 1);
            _sum01 = vfmv_v_f_f16m1(0.0f, 1);
            _sum02 = vfmv_v_f_f16m1(0.0f, 1);
            _sum03 = vfmv_v_f_f16m1(0.0f, 1);

            _sum00 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc00, _sum00, vlmax);
            _sum01 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc01, _sum01, vlmax);
            _sum02 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc02, _sum02, vlmax);
            _sum03 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc03, _sum03, vlmax);

            dst[idx00] = vfmv_f_s_f16m1_f16(_sum00);
            dst[idx01] = vfmv_f_s_f16m1_f16(_sum01);
            dst[idx02] = vfmv_f_s_f16m1_f16(_sum02);
            dst[idx03] = vfmv_f_s_f16m1_f16(_sum03);
        }
        for (; j < N; j++) {
            const __fp16 *a0_ptr = sa_ptr;
            const __fp16 *b0_ptr = sb + j * ldb;

            int vlmax = vsetvl_e16m1(csrr_vlenb() / sizeof(__fp16));
            // dst[0, 0]
            vfloat16m1_t _acc00 = vfmv_v_f_f16m1(0.0f, vlmax);

            int c = 0;
            while (c < K) {
                int vl = vsetvl_e16m1(K - c);
                vfloat16m1_t _a0 = vle16_v_f16m1(a0_ptr + c, vl);
                vfloat16m1_t _b0 = vle16_v_f16m1(b0_ptr + c, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _a0, _b0, vlmax);
                c += vl;
            }

            int idx00 = (i + 0) * ldc + (j + 0);

            // dst[0, 0]
            vfloat16m1_t _sum00;

            _sum00 = vfmv_v_f_f16m1(dst[idx00], 1);

            _sum00 = vfredosum_vs_f16m1_f16m1(vundefined_f16m1(), _acc00, _sum00, vlmax);
            dst[idx00] = vfmv_f_s_f16m1_f16(_sum00);
        }
    }
}

/**
 * for llm
 * if prefill: q [batch,np,sq,dim_head]
 *             k [batch,np,sk,dim_head]
 *             v [batch,np,dim_head,sv]
 *             sq = sk =sv > 1
 * if decoder: q [batch,np,sq,dim_head]
 *             k [batch,np,sk,dim_head]
 *             v [batch,np,dim_head,sv]
 *             sq = 1, sk = sv > 1
 *
 */
static void q0k1_softmax_v1_fp16(__fp16 *q, __fp16 *k, __fp16 *v, __fp16 *o,
                                 struct csinn_scale_dot_attention_params *params, int32_t sq,
                                 int32_t sk, int32_t head_dim)
{
    __fp16 norm_factor = 1.0f / params->norm_factor;  // sqrt(128)
    size_t matmul_res_size = sq * sk * sizeof(__fp16);
    __fp16 *matmul_res_data = malloc(matmul_res_size);
    memset(matmul_res_data, 0, matmul_res_size);
    if (sq > 1) {
        const __fp16 *q_in = q;
        const __fp16 *k_in = k;
        const __fp16 *v_in = v;
        for (int i = 0; i < sq; i++) {
            __fp16 max = -65504;
            __fp16 acc_exp = 0.0f;
            int casual_cnt = sk;
            if (params->casual) casual_cnt = i + 1 + (sk - sq);
            const __fp16 *q_ptr = q_in + i * head_dim;
            int j = 0;
            const int stride = 4;
            int m1_vl = vsetvl_e16m1(csrr_vlenb() / sizeof(__fp16));
            ;
            vfloat16m1_t _max = vfmv_v_f_f16m1(max, m1_vl);
            for (; j + stride - 1 < casual_cnt; j += stride) {
                const __fp16 *k_ptr0 = k_in + j * head_dim;
                const __fp16 *k_ptr1 = k_in + (j + 1) * head_dim;
                const __fp16 *k_ptr2 = k_in + (j + 2) * head_dim;
                const __fp16 *k_ptr3 = k_in + (j + 3) * head_dim;
                int vl = vsetvl_e16m2(csrr_vlenb() / sizeof(__fp16) * 2);

                vfloat16m2_t _acc00 = vfmv_v_f_f16m2(0.0f, vl);
                vfloat16m2_t _acc01 = vfmv_v_f_f16m2(0.0f, vl);
                vfloat16m2_t _acc02 = vfmv_v_f_f16m2(0.0f, vl);
                vfloat16m2_t _acc03 = vfmv_v_f_f16m2(0.0f, vl);
                vfloat16m1_t _sum00 = vfmv_v_f_f16m1(matmul_res_data[i * sk + j], 1);
                vfloat16m1_t _sum01 = vfmv_v_f_f16m1(matmul_res_data[i * sk + j + 1], 1);
                vfloat16m1_t _sum02 = vfmv_v_f_f16m1(matmul_res_data[i * sk + j + 2], 1);
                vfloat16m1_t _sum03 = vfmv_v_f_f16m1(matmul_res_data[i * sk + j + 3], 1);

                int l = 0;
                while (l < head_dim) {
                    // vlen128 e16m2 = 16
                    int vl_ = vsetvl_e16m2(head_dim - l);
                    vfloat16m2_t _q0 = vle16_v_f16m2(q_ptr + l, vl_);
                    vfloat16m2_t _k0 = vle16_v_f16m2(k_ptr0 + l, vl_);
                    vfloat16m2_t _k1 = vle16_v_f16m2(k_ptr1 + l, vl_);
                    vfloat16m2_t _k2 = vle16_v_f16m2(k_ptr2 + l, vl_);
                    vfloat16m2_t _k3 = vle16_v_f16m2(k_ptr3 + l, vl_);

                    _acc00 = vfmacc_vv_f16m2(_acc00, _q0, _k0, vl);
                    _acc01 = vfmacc_vv_f16m2(_acc01, _q0, _k1, vl);
                    _acc02 = vfmacc_vv_f16m2(_acc02, _q0, _k2, vl);
                    _acc03 = vfmacc_vv_f16m2(_acc03, _q0, _k3, vl);
                    l += vl_;
                }
                __fp16 res[stride];
                _sum00 = vfredosum_vs_f16m2_f16m1(vundefined_f16m1(), _acc00, _sum00, vl);
                res[0] = vfmv_f_s_f16m1_f16(_sum00);
                _sum01 = vfredosum_vs_f16m2_f16m1(vundefined_f16m1(), _acc01, _sum01, vl);
                res[1] = vfmv_f_s_f16m1_f16(_sum01);
                _sum02 = vfredosum_vs_f16m2_f16m1(vundefined_f16m1(), _acc02, _sum02, vl);
                res[2] = vfmv_f_s_f16m1_f16(_sum02);
                _sum03 = vfredosum_vs_f16m2_f16m1(vundefined_f16m1(), _acc03, _sum03, vl);
                res[3] = vfmv_f_s_f16m1_f16(_sum03);
                int min_vl = vsetvl_e16m1(csrr_vlenb() / sizeof(__fp16) / 2);  // min_vl = 4
                vfloat16m1_t save = vle16_v_f16m1(res, min_vl);
                save = vfmul_vf_f16m1(save, norm_factor, min_vl);
                vse16_v_f16m1(matmul_res_data + i * sk + j, save, min_vl);
                _max = vfmax_vv_f16m1(save, _max, min_vl);
            }

            vfloat16m1_t _min_f = vfmv_v_f_f16m1(max, m1_vl);
            vfloat16m1_t _max0 = vfredmax_vs_f16m1_f16m1(vundefined_f16m1(), _max, _min_f, m1_vl);
            max = vfmv_f_s_f16m1_f16(_max0);
            for (; j < casual_cnt; j++) {
                const __fp16 *k_ptr = k_in + j * head_dim;
                int vl = vsetvl_e16m4(csrr_vlenb() / sizeof(__fp16) * 4);
                vfloat16m4_t _acc00 = vfmv_v_f_f16m4(0.0f, vl);
                vfloat16m1_t _sum00 = vfmv_v_f_f16m1(matmul_res_data[i * sk + j], 1);
                ;
                int l = 0;
                while (l < head_dim) {
                    // vlen128 e16m4 = 32
                    int vl_ = vsetvl_e16m4(head_dim - l);
                    vfloat16m4_t _q0 = vle16_v_f16m4(q_ptr + l, vl_);
                    vfloat16m4_t _k0 = vle16_v_f16m4(k_ptr + l, vl_);
                    _acc00 = vfmacc_vv_f16m4(_acc00, _q0, _k0, vl);
                    l += vl_;
                }
                _sum00 = vfredosum_vs_f16m4_f16m1(vundefined_f16m1(), _acc00, _sum00, vl);
                __fp16 res = vfmv_f_s_f16m1_f16(_sum00);
                res *= norm_factor;
                matmul_res_data[i * sk + j] = res;
                max = fmax(max, res);
            }

            vfloat16m1_t fred_sum1 = vfmv_v_f_f16m1(0.0f, 1);
            __fp16 *res_in = &matmul_res_data[i * sk];
            int vl_m4 = vsetvl_e16m4(csrr_vlenb() / sizeof(__fp16) * 4);
            int len = 0;
            vfloat16m4_t div_sum0 = vfmv_v_f_f16m4(0.0f, vl_m4);
            while (len < casual_cnt) {
                int vl_in = vsetvl_e16m4(casual_cnt - len);
                vfloat16m4_t _res = vle16_v_f16m4(res_in + len, vl_in);
                _res = vfadd_vf_f16m4(_res, -max, vl_in);
                _res = exp_ps_vfloat16m4(_res, vl_in);
                vse16_v_f16m4(res_in + len, _res, vl_in);
                div_sum0 = vfadd_vv_f16m4(div_sum0, _res, vl_m4);
                len += vl_in;
            }
            fred_sum1 = vfredosum_vs_f16m4_f16m1(vundefined_f16m1(), div_sum0, fred_sum1, vl_m4);
            acc_exp = vfmv_f_s_f16m1_f16(fred_sum1);
            len = 0;
            const __fp16 _mul_exp = 1.0f / acc_exp;
            while (len < casual_cnt) {
                int vl_in = vsetvl_e16m4(casual_cnt - len);
                vfloat16m4_t _mul_in = vle16_v_f16m4(res_in + len, vl_in);
                vfloat16m4_t _output_data = vfmul_vf_f16m4(_mul_in, _mul_exp, vl_in);
                vse16_v_f16m4(res_in + len, _output_data, vl_in);
                len += vl_in;
            }
        }
        __fp16 *o_out = o;
        const __fp16 *qk_ptr = &matmul_res_data[0];
        const __fp16 *v_ptr = v_in;
        int M = sq;
        int K = sk;  // not casual_cnt
        int N = head_dim;
        int lda = sk;
        int ldb = sk;
        int ldc = head_dim;
        qk_t1_dot_4x4_fp16(o_out, qk_ptr, v_ptr, M, K, N, lda, ldb, ldc);
    } else if (sq == 1) {
        const __fp16 *q_in = q;
        const __fp16 *k_in = k;
        const __fp16 *v_in = v;
        // for sq, but sq = 1
        __fp16 max = -65504;
        __fp16 acc_exp = 0.0f;
        int casual_cnt = sk;
        if (params->casual) casual_cnt = 1 + (sk - sq);
        const __fp16 *q_ptr = q_in;

        {
            const int stride = 4;
            int m1_vl = vsetvl_e16m1(csrr_vlenb() / sizeof(__fp16));
            vfloat16m1_t _max = vfmv_v_f_f16m1(max, m1_vl);
            int j = 0;
            for (; j + stride - 1 < casual_cnt; j += stride) {
                const __fp16 *k_ptr0 = k_in + j * head_dim;
                const __fp16 *k_ptr1 = k_in + (j + 1) * head_dim;
                const __fp16 *k_ptr2 = k_in + (j + 2) * head_dim;
                const __fp16 *k_ptr3 = k_in + (j + 3) * head_dim;
                int vl = vsetvl_e16m2(csrr_vlenb() / sizeof(__fp16) * 2);

                vfloat16m2_t _acc00 = vfmv_v_f_f16m2(0.0f, vl);
                vfloat16m2_t _acc01 = vfmv_v_f_f16m2(0.0f, vl);
                vfloat16m2_t _acc02 = vfmv_v_f_f16m2(0.0f, vl);
                vfloat16m2_t _acc03 = vfmv_v_f_f16m2(0.0f, vl);
                vfloat16m1_t _sum00 = vfmv_v_f_f16m1(matmul_res_data[j], 1);
                vfloat16m1_t _sum01 = vfmv_v_f_f16m1(matmul_res_data[j + 1], 1);
                vfloat16m1_t _sum02 = vfmv_v_f_f16m1(matmul_res_data[j + 2], 1);
                vfloat16m1_t _sum03 = vfmv_v_f_f16m1(matmul_res_data[j + 3], 1);

                int l = 0;
                while (l < head_dim) {
                    // vlen128 e16m2 = 16
                    int vl_ = vsetvl_e16m2(head_dim - l);
                    vfloat16m2_t _q0 = vle16_v_f16m2(q_ptr + l, vl_);
                    vfloat16m2_t _k0 = vle16_v_f16m2(k_ptr0 + l, vl_);
                    vfloat16m2_t _k1 = vle16_v_f16m2(k_ptr1 + l, vl_);
                    vfloat16m2_t _k2 = vle16_v_f16m2(k_ptr2 + l, vl_);
                    vfloat16m2_t _k3 = vle16_v_f16m2(k_ptr3 + l, vl_);

                    _acc00 = vfmacc_vv_f16m2(_acc00, _q0, _k0, vl);
                    _acc01 = vfmacc_vv_f16m2(_acc01, _q0, _k1, vl);
                    _acc02 = vfmacc_vv_f16m2(_acc02, _q0, _k2, vl);
                    _acc03 = vfmacc_vv_f16m2(_acc03, _q0, _k3, vl);
                    l += vl_;
                }
                __fp16 res[stride];
                _sum00 = vfredosum_vs_f16m2_f16m1(vundefined_f16m1(), _acc00, _sum00, vl);
                res[0] = vfmv_f_s_f16m1_f16(_sum00);
                _sum01 = vfredosum_vs_f16m2_f16m1(vundefined_f16m1(), _acc01, _sum01, vl);
                res[1] = vfmv_f_s_f16m1_f16(_sum01);
                _sum02 = vfredosum_vs_f16m2_f16m1(vundefined_f16m1(), _acc02, _sum02, vl);
                res[2] = vfmv_f_s_f16m1_f16(_sum02);
                _sum03 = vfredosum_vs_f16m2_f16m1(vundefined_f16m1(), _acc03, _sum03, vl);
                res[3] = vfmv_f_s_f16m1_f16(_sum03);
                int min_vl = vsetvl_e16m1(csrr_vlenb() / sizeof(__fp16) / 2);  // min_vl = 4
                vfloat16m1_t save = vle16_v_f16m1(res, min_vl);
                save = vfmul_vf_f16m1(save, norm_factor, min_vl);
                vse16_v_f16m1(matmul_res_data + j, save, min_vl);
                _max = vfmax_vv_f16m1(save, _max, min_vl);
            }

            vfloat16m1_t _min_f = vfmv_v_f_f16m1(max, m1_vl);
            vfloat16m1_t _max0 = vfredmax_vs_f16m1_f16m1(vundefined_f16m1(), _max, _min_f, m1_vl);
            max = vfmv_f_s_f16m1_f16(_max0);

            for (; j < casual_cnt; j++) {
                const __fp16 *k_ptr = k_in + j * head_dim;
                int vl = vsetvl_e16m4(csrr_vlenb() / sizeof(__fp16) * 4);
                vfloat16m4_t _acc00 = vfmv_v_f_f16m4(0.0f, vl);
                vfloat16m1_t _sum00 = vfmv_v_f_f16m1(matmul_res_data[j], 1);
                ;
                int l = 0;
                while (l < head_dim) {
                    // vlen128 e16m4=32
                    int vl_ = vsetvl_e16m4(head_dim - l);
                    vfloat16m4_t _q0 = vle16_v_f16m4(q_ptr + l, vl_);
                    vfloat16m4_t _k0 = vle16_v_f16m4(k_ptr + l, vl_);
                    _acc00 = vfmacc_vv_f16m4(_acc00, _q0, _k0, vl);
                    l += vl_;
                }

                _sum00 = vfredosum_vs_f16m4_f16m1(vundefined_f16m1(), _acc00, _sum00, vl);
                __fp16 res = vfmv_f_s_f16m1_f16(_sum00);
                res *= norm_factor;
                matmul_res_data[j] = res;
                max = fmax(max, res);
            }

            vfloat16m1_t fred_sum1 = vfmv_v_f_f16m1(0.0f, 1);
            int len = 0;
            __fp16 *res_in = &matmul_res_data[0];
            int vl_m4 = vsetvl_e16m4(32);
            vfloat16m4_t div_sum0 = vfmv_v_f_f16m4(0.0f, vl_m4);
            while (len < casual_cnt) {
                int vl_in = vsetvl_e16m4(casual_cnt - len);
                vfloat16m4_t _res = vle16_v_f16m4(res_in + len, vl_in);
                _res = vfadd_vf_f16m4(_res, -max, vl_in);
                _res = exp_ps_vfloat16m4(_res, vl_in);
                vse16_v_f16m4(res_in + len, _res, vl_in);
                div_sum0 = vfadd_vv_f16m4(div_sum0, _res, vl_m4);
                len += vl_in;
            }
            fred_sum1 = vfredosum_vs_f16m4_f16m1(vundefined_f16m1(), div_sum0, fred_sum1, vl_m4);
            acc_exp = vfmv_f_s_f16m1_f16(fred_sum1);
        }

        {
            __fp16 *o_out = o;  // for sq = 1
            const __fp16 *_in = matmul_res_data;
            const __fp16 _mul_exp = 1.0f / acc_exp;
            const int stride = 4;
            int m1_vl = vsetvl_e16m1(csrr_vlenb() / sizeof(__fp16));
            ;
            int dim = 0;
            for (; dim + stride - 1 < head_dim; dim += stride) {
                const __fp16 *v_ptr0 = v_in + dim * sk;
                const __fp16 *v_ptr1 = v_in + (dim + 1) * sk;
                const __fp16 *v_ptr2 = v_in + (dim + 2) * sk;
                const __fp16 *v_ptr3 = v_in + (dim + 3) * sk;
                int vl = vsetvl_e16m2(csrr_vlenb() / sizeof(__fp16) * 2);

                vfloat16m2_t _acc00 = vfmv_v_f_f16m2(0.0f, vl);
                vfloat16m2_t _acc01 = vfmv_v_f_f16m2(0.0f, vl);
                vfloat16m2_t _acc02 = vfmv_v_f_f16m2(0.0f, vl);
                vfloat16m2_t _acc03 = vfmv_v_f_f16m2(0.0f, vl);
                vfloat16m1_t _sum00 = vfmv_v_f_f16m1(0.0f, 1);
                vfloat16m1_t _sum01 = vfmv_v_f_f16m1(0.0f, 1);
                vfloat16m1_t _sum02 = vfmv_v_f_f16m1(0.0f, 1);
                vfloat16m1_t _sum03 = vfmv_v_f_f16m1(0.0f, 1);

                int j = 0;
                while (j < casual_cnt) {
                    // vlen128 e16m2 = 8
                    int vl_v = vsetvl_e16m2(casual_cnt - j);
                    vfloat16m2_t _in0 = vle16_v_f16m2(_in + j, vl_v);
                    vfloat16m2_t _v0 = vle16_v_f16m2(v_ptr0 + j, vl_v);
                    vfloat16m2_t _v1 = vle16_v_f16m2(v_ptr1 + j, vl_v);
                    vfloat16m2_t _v2 = vle16_v_f16m2(v_ptr2 + j, vl_v);
                    vfloat16m2_t _v3 = vle16_v_f16m2(v_ptr3 + j, vl_v);

                    _acc00 = vfmacc_vv_f16m2(_acc00, _in0, _v0, vl);
                    _acc01 = vfmacc_vv_f16m2(_acc01, _in0, _v1, vl);
                    _acc02 = vfmacc_vv_f16m2(_acc02, _in0, _v2, vl);
                    _acc03 = vfmacc_vv_f16m2(_acc03, _in0, _v3, vl);
                    j += vl_v;
                }
                __fp16 res[stride];
                _sum00 = vfredosum_vs_f16m2_f16m1(vundefined_f16m1(), _acc00, _sum00, vl);
                res[0] = vfmv_f_s_f16m1_f16(_sum00);
                _sum01 = vfredosum_vs_f16m2_f16m1(vundefined_f16m1(), _acc01, _sum01, vl);
                res[1] = vfmv_f_s_f16m1_f16(_sum01);
                _sum02 = vfredosum_vs_f16m2_f16m1(vundefined_f16m1(), _acc02, _sum02, vl);
                res[2] = vfmv_f_s_f16m1_f16(_sum02);
                _sum03 = vfredosum_vs_f16m2_f16m1(vundefined_f16m1(), _acc03, _sum03, vl);
                res[3] = vfmv_f_s_f16m1_f16(_sum03);
                vfloat16m1_t save = vle16_v_f16m1(res, m1_vl);
                save = vfmul_vf_f16m1(save, _mul_exp, m1_vl);
                vse16_v_f16m1(o_out + dim, save, m1_vl);
            }
            for (; dim < head_dim; dim++) {
                int j = 0;
                int vl_size = vsetvl_e16m4(csrr_vlenb() / sizeof(__fp16) * 4);
                vfloat16m4_t _acc_o0 = vfmv_v_f_f16m4(0.0f, vl_size);
                vfloat16m1_t _out;
                while (j < casual_cnt) {
                    const __fp16 *res_in = matmul_res_data + j;
                    int vl_v = vsetvl_e16m4(casual_cnt - j);
                    vfloat16m4_t _r0 = vle16_v_f16m4(res_in, vl_v);
                    const __fp16 *v_ptr = v_in + dim * sk + j;
                    vfloat16m4_t _v0 = vle16_v_f16m4(v_ptr, vl_v);
                    _acc_o0 = vfmacc_vv_f16m4(_acc_o0, _r0, _v0, vl_size);
                    j += vl_v;
                }
                _out = vfmv_v_f_f16m1(0.0f, 1);  //  colud not be blocked
                _out = vfredosum_vs_f16m4_f16m1(vundefined_f16m1(), _acc_o0, _out, vl_size);
                o_out[dim] = vfmv_f_s_f16m1_f16(_out);
                o_out[dim] = o_out[dim] / acc_exp;
            }
        }
    }
    free(matmul_res_data);
}