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
 * src: [M_BLOCK, K_BLOCK]
 * dst: [M_BLOCK/m_blk, K_BLOCK, m_blk]
 * m_blk: 8/4/2/1
 ************************************************************/
static inline void reorder_a_8xk_fp32(float *src, float *dst, int M_BLOCK, int K_BLOCK, int lda)
{
    int i = 0;
    for (; i + 7 < M_BLOCK; i += 8) {
        float *s_ptr = src + i * lda;
        float *d_ptr = dst + i * K_BLOCK;
        int stride = 8 * sizeof(float);
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e32m4(K_BLOCK - c);
            vfloat32m4_t _s0 = vle32_v_f32m4(s_ptr, vl);
            vfloat32m4_t _s1 = vle32_v_f32m4(s_ptr + lda, vl);
            vfloat32m4_t _s2 = vle32_v_f32m4(s_ptr + lda * 2, vl);
            vfloat32m4_t _s3 = vle32_v_f32m4(s_ptr + lda * 3, vl);
            vfloat32m4_t _s4 = vle32_v_f32m4(s_ptr + lda * 4, vl);
            vfloat32m4_t _s5 = vle32_v_f32m4(s_ptr + lda * 5, vl);
            vfloat32m4_t _s6 = vle32_v_f32m4(s_ptr + lda * 6, vl);
            vfloat32m4_t _s7 = vle32_v_f32m4(s_ptr + lda * 7, vl);
            vsse32_v_f32m4(d_ptr, stride, _s0, vl);
            vsse32_v_f32m4(d_ptr + 1, stride, _s1, vl);
            vsse32_v_f32m4(d_ptr + 2, stride, _s2, vl);
            vsse32_v_f32m4(d_ptr + 3, stride, _s3, vl);
            vsse32_v_f32m4(d_ptr + 4, stride, _s4, vl);
            vsse32_v_f32m4(d_ptr + 5, stride, _s5, vl);
            vsse32_v_f32m4(d_ptr + 6, stride, _s6, vl);
            vsse32_v_f32m4(d_ptr + 7, stride, _s7, vl);
            s_ptr += vl;
            d_ptr += vl * 8;
            c += vl;
        }
    }
    for (; i + 3 < M_BLOCK; i += 4) {
        float *s_ptr = src + i * lda;
        float *d_ptr = dst + i * K_BLOCK;
        int stride = 4 * sizeof(float);
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e32m4(K_BLOCK - c);
            vfloat32m4_t _s0 = vle32_v_f32m4(s_ptr, vl);
            vfloat32m4_t _s1 = vle32_v_f32m4(s_ptr + lda, vl);
            vfloat32m4_t _s2 = vle32_v_f32m4(s_ptr + lda * 2, vl);
            vfloat32m4_t _s3 = vle32_v_f32m4(s_ptr + lda * 3, vl);
            vsse32_v_f32m4(d_ptr, stride, _s0, vl);
            vsse32_v_f32m4(d_ptr + 1, stride, _s1, vl);
            vsse32_v_f32m4(d_ptr + 2, stride, _s2, vl);
            vsse32_v_f32m4(d_ptr + 3, stride, _s3, vl);
            s_ptr += vl;
            d_ptr += vl * 4;
            c += vl;
        }
    }
    for (; i + 1 < M_BLOCK; i += 2) {
        float *s_ptr = src + i * lda;
        float *d_ptr = dst + i * K_BLOCK;
        int stride = 2 * sizeof(float);
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e32m4(K_BLOCK - c);
            vfloat32m4_t _s0 = vle32_v_f32m4(s_ptr, vl);
            vfloat32m4_t _s1 = vle32_v_f32m4(s_ptr + lda, vl);
            vsse32_v_f32m4(d_ptr, stride, _s0, vl);
            vsse32_v_f32m4(d_ptr + 1, stride, _s1, vl);
            s_ptr += vl;
            d_ptr += vl * 2;
            c += vl;
        }
    }
    for (; i < M_BLOCK; i++) {
        float *s_ptr = src + i * lda;
        float *d_ptr = dst + i * K_BLOCK;
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e32m4(K_BLOCK - c);
            vfloat32m4_t _src = vle32_v_f32m4(s_ptr, vl);
            vse32_v_f32m4(d_ptr, _src, vl);
            s_ptr += vl;
            d_ptr += vl;
            c += vl;
        }
    }
}

/*************************************************************
 * src: [m, k]
 * dst: [m/m_blk, k/k_blk, m_blk/8, 8, k_blk]
 * m_blk: M_BLK, M_tail
 * k_blk: K_BLK, K_tail
 ************************************************************/
void shl_c920_reorder_a_block_8xk_fp32(float *src, float *dst, int m, int k, const int M_BLK,
                                       const int K_BLK)
{
    const int MIN_M_BLK = 8;

    int m_block = M_BLK;
    int m_idx = 0;
    while (m_idx < m) {
        if (m - m_idx < m_block) {
            m_block = m - m_idx;
        }

        int k_block = K_BLK;
        int k_idx = 0;
        while (k_idx < k) {
            if (k - k_idx < k_block) {
                k_block = k - k_idx;
            }
            float *s_ptr = src + m_idx * k + k_idx;
            float *d_ptr = dst + m_idx * k + k_idx * m_block;
            reorder_a_8xk_fp32(s_ptr, d_ptr, m_block, k_block, k);
            k_idx += k_block;
        }
        m_idx += m_block;
    }
}

/*************************************************************
 * src: [M_BLOCK, K_BLOCK]
 * dst: [M_BLOCK/m_blk, K_BLOCK, m_blk]
 * m_blk: 8/4/2/1
 ************************************************************/
static inline void reorder_a_8xk_fp16(__fp16 *src, __fp16 *dst, int M_BLOCK, int K_BLOCK, int lda)
{
    int i = 0;
    for (; i + 7 < M_BLOCK; i += 8) {
        __fp16 *s_ptr = src + i * lda;
        __fp16 *d_ptr = dst + i * K_BLOCK;
        int stride = 8 * sizeof(__fp16);
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e16m4(K_BLOCK - c);
            vfloat16m4_t _s0 = vle16_v_f16m4(s_ptr, vl);
            vfloat16m4_t _s1 = vle16_v_f16m4(s_ptr + lda, vl);
            vfloat16m4_t _s2 = vle16_v_f16m4(s_ptr + lda * 2, vl);
            vfloat16m4_t _s3 = vle16_v_f16m4(s_ptr + lda * 3, vl);
            vfloat16m4_t _s4 = vle16_v_f16m4(s_ptr + lda * 4, vl);
            vfloat16m4_t _s5 = vle16_v_f16m4(s_ptr + lda * 5, vl);
            vfloat16m4_t _s6 = vle16_v_f16m4(s_ptr + lda * 6, vl);
            vfloat16m4_t _s7 = vle16_v_f16m4(s_ptr + lda * 7, vl);
            vsse16_v_f16m4(d_ptr, stride, _s0, vl);
            vsse16_v_f16m4(d_ptr + 1, stride, _s1, vl);
            vsse16_v_f16m4(d_ptr + 2, stride, _s2, vl);
            vsse16_v_f16m4(d_ptr + 3, stride, _s3, vl);
            vsse16_v_f16m4(d_ptr + 4, stride, _s4, vl);
            vsse16_v_f16m4(d_ptr + 5, stride, _s5, vl);
            vsse16_v_f16m4(d_ptr + 6, stride, _s6, vl);
            vsse16_v_f16m4(d_ptr + 7, stride, _s7, vl);
            s_ptr += vl;
            d_ptr += vl * 8;
            c += vl;
        }
    }
    for (; i + 3 < M_BLOCK; i += 4) {
        __fp16 *s_ptr = src + i * lda;
        __fp16 *d_ptr = dst + i * K_BLOCK;
        int stride = 4 * sizeof(__fp16);
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e16m4(K_BLOCK - c);
            vfloat16m4_t _s0 = vle16_v_f16m4(s_ptr, vl);
            vfloat16m4_t _s1 = vle16_v_f16m4(s_ptr + lda, vl);
            vfloat16m4_t _s2 = vle16_v_f16m4(s_ptr + lda * 2, vl);
            vfloat16m4_t _s3 = vle16_v_f16m4(s_ptr + lda * 3, vl);
            vsse16_v_f16m4(d_ptr, stride, _s0, vl);
            vsse16_v_f16m4(d_ptr + 1, stride, _s1, vl);
            vsse16_v_f16m4(d_ptr + 2, stride, _s2, vl);
            vsse16_v_f16m4(d_ptr + 3, stride, _s3, vl);
            s_ptr += vl;
            d_ptr += vl * 4;
            c += vl;
        }
    }
    for (; i + 1 < M_BLOCK; i += 2) {
        __fp16 *s_ptr = src + i * lda;
        __fp16 *d_ptr = dst + i * K_BLOCK;
        int stride = 2 * sizeof(__fp16);
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e16m4(K_BLOCK - c);
            vfloat16m4_t _s0 = vle16_v_f16m4(s_ptr, vl);
            vfloat16m4_t _s1 = vle16_v_f16m4(s_ptr + lda, vl);
            vsse16_v_f16m4(d_ptr, stride, _s0, vl);
            vsse16_v_f16m4(d_ptr + 1, stride, _s1, vl);
            s_ptr += vl;
            d_ptr += vl * 2;
            c += vl;
        }
    }
    for (; i < M_BLOCK; i++) {
        __fp16 *s_ptr = src + i * lda;
        __fp16 *d_ptr = dst + i * K_BLOCK;
        int c = 0;
        while (c < K_BLOCK) {
            int vl = vsetvl_e16m4(K_BLOCK - c);
            vfloat16m4_t _src = vle16_v_f16m4(s_ptr, vl);
            vse16_v_f16m4(d_ptr, _src, vl);
            s_ptr += vl;
            d_ptr += vl;
            c += vl;
        }
    }
}

/*************************************************************
 * src: [m, k]
 * dst: [m/m_blk, k/k_blk, m_blk/8, 8, k_blk]
 * m_blk: M_BLK, M_tail
 * k_blk: K_BLK, K_tail
 ************************************************************/
void shl_c920_reorder_a_block_8xk_fp16(__fp16 *src, __fp16 *dst, int m, int k, const int M_BLK,
                                       const int K_BLK)
{
    const int MIN_M_BLK = 8;

    int m_block = M_BLK;
    int m_idx = 0;
    while (m_idx < m) {
        if (m - m_idx < m_block) {
            m_block = m - m_idx;
        }

        int k_block = K_BLK;
        int k_idx = 0;
        while (k_idx < k) {
            if (k - k_idx < k_block) {
                k_block = k - k_idx;
            }
            __fp16 *s_ptr = src + m_idx * k + k_idx;
            __fp16 *d_ptr = dst + m_idx * k + k_idx * m_block;
            reorder_a_8xk_fp16(s_ptr, d_ptr, m_block, k_block, k);
            k_idx += k_block;
        }
        m_idx += m_block;
    }
}
