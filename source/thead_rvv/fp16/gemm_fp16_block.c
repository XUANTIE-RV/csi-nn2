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

#include "rvv/rvv.h"

/*************************************************************
 * packn = vlenb / sizeof(__fp16)
 * m_blk: 12/8/4/2/1
 * n_blk: pack2n/packn/n_tail
 * k_blk: K_BLK
 ************************************************************/
static inline void gemm_12xpack2n_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb,
                                       __fp16 *bias, int M_BLOCK, int N_BLOCK, int K_BLOCK, int ldc,
                                       int k_idx)
{
    const __fp16 *kernel_data = sa;
    const __fp16 *input_data = sb;
    __fp16 *output_data = dst;
    __fp16 *bias_data = bias;

    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int pack2n = packn * 2;
    int vl = vsetvl_e16m1(packn);

    int i = 0;
    for (; i + 11 < M_BLOCK; i += 12) {
        const __fp16 *kernel_ptr = kernel_data + i * K_BLOCK;
        int j = 0;
        vl = vsetvl_e16m1(packn);
        for (; j + pack2n - 1 < N_BLOCK; j += pack2n) {
            const __fp16 *k_ptr = kernel_ptr;
            const __fp16 *in_ptr = input_data + j * K_BLOCK;
            __fp16 *out0_ptr = output_data + i * ldc + j;
            __fp16 *out1_ptr = out0_ptr + packn;

            // [n, 0]
            vfloat16m1_t _acc00;
            vfloat16m1_t _acc10;
            vfloat16m1_t _acc20;
            vfloat16m1_t _acc30;
            vfloat16m1_t _acc40;
            vfloat16m1_t _acc50;
            vfloat16m1_t _acc60;
            vfloat16m1_t _acc70;
            vfloat16m1_t _acc80;
            vfloat16m1_t _acc90;
            vfloat16m1_t _acca0;
            vfloat16m1_t _accb0;
            // [n, 1]
            vfloat16m1_t _acc01;
            vfloat16m1_t _acc11;
            vfloat16m1_t _acc21;
            vfloat16m1_t _acc31;
            vfloat16m1_t _acc41;
            vfloat16m1_t _acc51;
            vfloat16m1_t _acc61;
            vfloat16m1_t _acc71;
            vfloat16m1_t _acc81;
            vfloat16m1_t _acc91;
            vfloat16m1_t _acca1;
            vfloat16m1_t _accb1;
            if (k_idx == 0) {
                const __fp16 *b_ptr = bias_data + i;
                _acc00 = vfmv_v_f_f16m1(b_ptr[0], vl);
                _acc10 = vfmv_v_f_f16m1(b_ptr[1], vl);
                _acc20 = vfmv_v_f_f16m1(b_ptr[2], vl);
                _acc30 = vfmv_v_f_f16m1(b_ptr[3], vl);
                _acc40 = vfmv_v_f_f16m1(b_ptr[4], vl);
                _acc50 = vfmv_v_f_f16m1(b_ptr[5], vl);
                _acc60 = vfmv_v_f_f16m1(b_ptr[6], vl);
                _acc70 = vfmv_v_f_f16m1(b_ptr[7], vl);
                _acc80 = vfmv_v_f_f16m1(b_ptr[8], vl);
                _acc90 = vfmv_v_f_f16m1(b_ptr[9], vl);
                _acca0 = vfmv_v_f_f16m1(b_ptr[10], vl);
                _accb0 = vfmv_v_f_f16m1(b_ptr[11], vl);
                _acc01 = vmv_v_v_f16m1(_acc00, vl);
                _acc11 = vmv_v_v_f16m1(_acc10, vl);
                _acc21 = vmv_v_v_f16m1(_acc20, vl);
                _acc31 = vmv_v_v_f16m1(_acc30, vl);
                _acc41 = vmv_v_v_f16m1(_acc40, vl);
                _acc51 = vmv_v_v_f16m1(_acc50, vl);
                _acc61 = vmv_v_v_f16m1(_acc60, vl);
                _acc71 = vmv_v_v_f16m1(_acc70, vl);
                _acc81 = vmv_v_v_f16m1(_acc80, vl);
                _acc91 = vmv_v_v_f16m1(_acc90, vl);
                _acca1 = vmv_v_v_f16m1(_acca0, vl);
                _accb1 = vmv_v_v_f16m1(_accb0, vl);
            } else {
                _acc00 = vle16_v_f16m1(out0_ptr, vl);
                _acc10 = vle16_v_f16m1(out0_ptr + ldc, vl);
                _acc20 = vle16_v_f16m1(out0_ptr + ldc * 2, vl);
                _acc30 = vle16_v_f16m1(out0_ptr + ldc * 3, vl);
                _acc40 = vle16_v_f16m1(out0_ptr + ldc * 4, vl);
                _acc50 = vle16_v_f16m1(out0_ptr + ldc * 5, vl);
                _acc60 = vle16_v_f16m1(out0_ptr + ldc * 6, vl);
                _acc70 = vle16_v_f16m1(out0_ptr + ldc * 7, vl);
                _acc80 = vle16_v_f16m1(out0_ptr + ldc * 8, vl);
                _acc90 = vle16_v_f16m1(out0_ptr + ldc * 9, vl);
                _acca0 = vle16_v_f16m1(out0_ptr + ldc * 10, vl);
                _accb0 = vle16_v_f16m1(out0_ptr + ldc * 11, vl);
                _acc01 = vle16_v_f16m1(out1_ptr, vl);
                _acc11 = vle16_v_f16m1(out1_ptr + ldc, vl);
                _acc21 = vle16_v_f16m1(out1_ptr + ldc * 2, vl);
                _acc31 = vle16_v_f16m1(out1_ptr + ldc * 3, vl);
                _acc41 = vle16_v_f16m1(out1_ptr + ldc * 4, vl);
                _acc51 = vle16_v_f16m1(out1_ptr + ldc * 5, vl);
                _acc61 = vle16_v_f16m1(out1_ptr + ldc * 6, vl);
                _acc71 = vle16_v_f16m1(out1_ptr + ldc * 7, vl);
                _acc81 = vle16_v_f16m1(out1_ptr + ldc * 8, vl);
                _acc91 = vle16_v_f16m1(out1_ptr + ldc * 9, vl);
                _acca1 = vle16_v_f16m1(out1_ptr + ldc * 10, vl);
                _accb1 = vle16_v_f16m1(out1_ptr + ldc * 11, vl);
            }

            for (int c = 0; c < K_BLOCK; c++) {
                vfloat16m1_t _in0 = vle16_v_f16m1(in_ptr, vl);
                vfloat16m1_t _in1 = vle16_v_f16m1(in_ptr + packn, vl);
                in_ptr += pack2n;

                _acc00 = vfmacc_vf_f16m1(_acc00, k_ptr[0], _in0, vl);
                _acc10 = vfmacc_vf_f16m1(_acc10, k_ptr[1], _in0, vl);
                _acc20 = vfmacc_vf_f16m1(_acc20, k_ptr[2], _in0, vl);
                _acc30 = vfmacc_vf_f16m1(_acc30, k_ptr[3], _in0, vl);
                _acc40 = vfmacc_vf_f16m1(_acc40, k_ptr[4], _in0, vl);
                _acc50 = vfmacc_vf_f16m1(_acc50, k_ptr[5], _in0, vl);
                _acc60 = vfmacc_vf_f16m1(_acc60, k_ptr[6], _in0, vl);
                _acc70 = vfmacc_vf_f16m1(_acc70, k_ptr[7], _in0, vl);
                _acc80 = vfmacc_vf_f16m1(_acc80, k_ptr[8], _in0, vl);
                _acc90 = vfmacc_vf_f16m1(_acc90, k_ptr[9], _in0, vl);
                _acca0 = vfmacc_vf_f16m1(_acca0, k_ptr[10], _in0, vl);
                _accb0 = vfmacc_vf_f16m1(_accb0, k_ptr[11], _in0, vl);

                _acc01 = vfmacc_vf_f16m1(_acc01, k_ptr[0], _in1, vl);
                _acc11 = vfmacc_vf_f16m1(_acc11, k_ptr[1], _in1, vl);
                _acc21 = vfmacc_vf_f16m1(_acc21, k_ptr[2], _in1, vl);
                _acc31 = vfmacc_vf_f16m1(_acc31, k_ptr[3], _in1, vl);
                _acc41 = vfmacc_vf_f16m1(_acc41, k_ptr[4], _in1, vl);
                _acc51 = vfmacc_vf_f16m1(_acc51, k_ptr[5], _in1, vl);
                _acc61 = vfmacc_vf_f16m1(_acc61, k_ptr[6], _in1, vl);
                _acc71 = vfmacc_vf_f16m1(_acc71, k_ptr[7], _in1, vl);
                _acc81 = vfmacc_vf_f16m1(_acc81, k_ptr[8], _in1, vl);
                _acc91 = vfmacc_vf_f16m1(_acc91, k_ptr[9], _in1, vl);
                _acca1 = vfmacc_vf_f16m1(_acca1, k_ptr[10], _in1, vl);
                _accb1 = vfmacc_vf_f16m1(_accb1, k_ptr[11], _in1, vl);
                k_ptr += 12;
            }

            vse16_v_f16m1(out0_ptr, _acc00, vl);
            vse16_v_f16m1(out0_ptr + ldc, _acc10, vl);
            vse16_v_f16m1(out0_ptr + ldc * 2, _acc20, vl);
            vse16_v_f16m1(out0_ptr + ldc * 3, _acc30, vl);
            vse16_v_f16m1(out0_ptr + ldc * 4, _acc40, vl);
            vse16_v_f16m1(out0_ptr + ldc * 5, _acc50, vl);
            vse16_v_f16m1(out0_ptr + ldc * 6, _acc60, vl);
            vse16_v_f16m1(out0_ptr + ldc * 7, _acc70, vl);
            vse16_v_f16m1(out0_ptr + ldc * 8, _acc80, vl);
            vse16_v_f16m1(out0_ptr + ldc * 9, _acc90, vl);
            vse16_v_f16m1(out0_ptr + ldc * 10, _acca0, vl);
            vse16_v_f16m1(out0_ptr + ldc * 11, _accb0, vl);
            vse16_v_f16m1(out1_ptr, _acc01, vl);
            vse16_v_f16m1(out1_ptr + ldc, _acc11, vl);
            vse16_v_f16m1(out1_ptr + ldc * 2, _acc21, vl);
            vse16_v_f16m1(out1_ptr + ldc * 3, _acc31, vl);
            vse16_v_f16m1(out1_ptr + ldc * 4, _acc41, vl);
            vse16_v_f16m1(out1_ptr + ldc * 5, _acc51, vl);
            vse16_v_f16m1(out1_ptr + ldc * 6, _acc61, vl);
            vse16_v_f16m1(out1_ptr + ldc * 7, _acc71, vl);
            vse16_v_f16m1(out1_ptr + ldc * 8, _acc81, vl);
            vse16_v_f16m1(out1_ptr + ldc * 9, _acc91, vl);
            vse16_v_f16m1(out1_ptr + ldc * 10, _acca1, vl);
            vse16_v_f16m1(out1_ptr + ldc * 11, _accb1, vl);
        }
        while (j < N_BLOCK) {
            vl = vsetvl_e16m1(N_BLOCK - j);
            const __fp16 *k_ptr = kernel_ptr;
            const __fp16 *in_ptr = input_data + j * K_BLOCK;
            __fp16 *out0_ptr = output_data + i * ldc + j;

            vfloat16m1_t _acc00;
            vfloat16m1_t _acc10;
            vfloat16m1_t _acc20;
            vfloat16m1_t _acc30;
            vfloat16m1_t _acc40;
            vfloat16m1_t _acc50;
            vfloat16m1_t _acc60;
            vfloat16m1_t _acc70;
            vfloat16m1_t _acc80;
            vfloat16m1_t _acc90;
            vfloat16m1_t _acca0;
            vfloat16m1_t _accb0;
            if (k_idx == 0) {
                const __fp16 *b_ptr = bias_data + i;
                _acc00 = vfmv_v_f_f16m1(b_ptr[0], vl);
                _acc10 = vfmv_v_f_f16m1(b_ptr[1], vl);
                _acc20 = vfmv_v_f_f16m1(b_ptr[2], vl);
                _acc30 = vfmv_v_f_f16m1(b_ptr[3], vl);
                _acc40 = vfmv_v_f_f16m1(b_ptr[4], vl);
                _acc50 = vfmv_v_f_f16m1(b_ptr[5], vl);
                _acc60 = vfmv_v_f_f16m1(b_ptr[6], vl);
                _acc70 = vfmv_v_f_f16m1(b_ptr[7], vl);
                _acc80 = vfmv_v_f_f16m1(b_ptr[8], vl);
                _acc90 = vfmv_v_f_f16m1(b_ptr[9], vl);
                _acca0 = vfmv_v_f_f16m1(b_ptr[10], vl);
                _accb0 = vfmv_v_f_f16m1(b_ptr[11], vl);
            } else {
                _acc00 = vle16_v_f16m1(out0_ptr, vl);
                _acc10 = vle16_v_f16m1(out0_ptr + ldc, vl);
                _acc20 = vle16_v_f16m1(out0_ptr + ldc * 2, vl);
                _acc30 = vle16_v_f16m1(out0_ptr + ldc * 3, vl);
                _acc40 = vle16_v_f16m1(out0_ptr + ldc * 4, vl);
                _acc50 = vle16_v_f16m1(out0_ptr + ldc * 5, vl);
                _acc60 = vle16_v_f16m1(out0_ptr + ldc * 6, vl);
                _acc70 = vle16_v_f16m1(out0_ptr + ldc * 7, vl);
                _acc80 = vle16_v_f16m1(out0_ptr + ldc * 8, vl);
                _acc90 = vle16_v_f16m1(out0_ptr + ldc * 9, vl);
                _acca0 = vle16_v_f16m1(out0_ptr + ldc * 10, vl);
                _accb0 = vle16_v_f16m1(out0_ptr + ldc * 11, vl);
            }

            for (int c = 0; c < K_BLOCK; c++) {
                vfloat16m1_t _in0 = vle16_v_f16m1(in_ptr, vl);
                in_ptr += vl;

                _acc00 = vfmacc_vf_f16m1(_acc00, k_ptr[0], _in0, vl);
                _acc10 = vfmacc_vf_f16m1(_acc10, k_ptr[1], _in0, vl);
                _acc20 = vfmacc_vf_f16m1(_acc20, k_ptr[2], _in0, vl);
                _acc30 = vfmacc_vf_f16m1(_acc30, k_ptr[3], _in0, vl);
                _acc40 = vfmacc_vf_f16m1(_acc40, k_ptr[4], _in0, vl);
                _acc50 = vfmacc_vf_f16m1(_acc50, k_ptr[5], _in0, vl);
                _acc60 = vfmacc_vf_f16m1(_acc60, k_ptr[6], _in0, vl);
                _acc70 = vfmacc_vf_f16m1(_acc70, k_ptr[7], _in0, vl);
                _acc80 = vfmacc_vf_f16m1(_acc80, k_ptr[8], _in0, vl);
                _acc90 = vfmacc_vf_f16m1(_acc90, k_ptr[9], _in0, vl);
                _acca0 = vfmacc_vf_f16m1(_acca0, k_ptr[10], _in0, vl);
                _accb0 = vfmacc_vf_f16m1(_accb0, k_ptr[11], _in0, vl);
                k_ptr += 12;
            }

            vse16_v_f16m1(out0_ptr, _acc00, vl);
            vse16_v_f16m1(out0_ptr + ldc, _acc10, vl);
            vse16_v_f16m1(out0_ptr + ldc * 2, _acc20, vl);
            vse16_v_f16m1(out0_ptr + ldc * 3, _acc30, vl);
            vse16_v_f16m1(out0_ptr + ldc * 4, _acc40, vl);
            vse16_v_f16m1(out0_ptr + ldc * 5, _acc50, vl);
            vse16_v_f16m1(out0_ptr + ldc * 6, _acc60, vl);
            vse16_v_f16m1(out0_ptr + ldc * 7, _acc70, vl);
            vse16_v_f16m1(out0_ptr + ldc * 8, _acc80, vl);
            vse16_v_f16m1(out0_ptr + ldc * 9, _acc90, vl);
            vse16_v_f16m1(out0_ptr + ldc * 10, _acca0, vl);
            vse16_v_f16m1(out0_ptr + ldc * 11, _accb0, vl);
            j += vl;
        }
    }
    for (; i + 7 < M_BLOCK; i += 8) {
        const __fp16 *kernel_ptr = kernel_data + i * K_BLOCK;
        int j = 0;
        vl = vsetvl_e16m1(packn);
        for (; j + pack2n - 1 < N_BLOCK; j += pack2n) {
            const __fp16 *k_ptr = kernel_ptr;
            const __fp16 *in_ptr = input_data + j * K_BLOCK;
            __fp16 *out0_ptr = output_data + i * ldc + j;
            __fp16 *out1_ptr = out0_ptr + packn;

            // [n, 0]
            vfloat16m1_t _acc00;
            vfloat16m1_t _acc10;
            vfloat16m1_t _acc20;
            vfloat16m1_t _acc30;
            vfloat16m1_t _acc40;
            vfloat16m1_t _acc50;
            vfloat16m1_t _acc60;
            vfloat16m1_t _acc70;
            // [n, 1]
            vfloat16m1_t _acc01;
            vfloat16m1_t _acc11;
            vfloat16m1_t _acc21;
            vfloat16m1_t _acc31;
            vfloat16m1_t _acc41;
            vfloat16m1_t _acc51;
            vfloat16m1_t _acc61;
            vfloat16m1_t _acc71;
            if (k_idx == 0) {
                const __fp16 *b_ptr = bias_data + i;
                _acc00 = vfmv_v_f_f16m1(b_ptr[0], vl);
                _acc10 = vfmv_v_f_f16m1(b_ptr[1], vl);
                _acc20 = vfmv_v_f_f16m1(b_ptr[2], vl);
                _acc30 = vfmv_v_f_f16m1(b_ptr[3], vl);
                _acc40 = vfmv_v_f_f16m1(b_ptr[4], vl);
                _acc50 = vfmv_v_f_f16m1(b_ptr[5], vl);
                _acc60 = vfmv_v_f_f16m1(b_ptr[6], vl);
                _acc70 = vfmv_v_f_f16m1(b_ptr[7], vl);
                _acc01 = vmv_v_v_f16m1(_acc00, vl);
                _acc11 = vmv_v_v_f16m1(_acc10, vl);
                _acc21 = vmv_v_v_f16m1(_acc20, vl);
                _acc31 = vmv_v_v_f16m1(_acc30, vl);
                _acc41 = vmv_v_v_f16m1(_acc40, vl);
                _acc51 = vmv_v_v_f16m1(_acc50, vl);
                _acc61 = vmv_v_v_f16m1(_acc60, vl);
                _acc71 = vmv_v_v_f16m1(_acc70, vl);
            } else {
                _acc00 = vle16_v_f16m1(out0_ptr, vl);
                _acc10 = vle16_v_f16m1(out0_ptr + ldc, vl);
                _acc20 = vle16_v_f16m1(out0_ptr + ldc * 2, vl);
                _acc30 = vle16_v_f16m1(out0_ptr + ldc * 3, vl);
                _acc40 = vle16_v_f16m1(out0_ptr + ldc * 4, vl);
                _acc50 = vle16_v_f16m1(out0_ptr + ldc * 5, vl);
                _acc60 = vle16_v_f16m1(out0_ptr + ldc * 6, vl);
                _acc70 = vle16_v_f16m1(out0_ptr + ldc * 7, vl);
                _acc01 = vle16_v_f16m1(out1_ptr, vl);
                _acc11 = vle16_v_f16m1(out1_ptr + ldc, vl);
                _acc21 = vle16_v_f16m1(out1_ptr + ldc * 2, vl);
                _acc31 = vle16_v_f16m1(out1_ptr + ldc * 3, vl);
                _acc41 = vle16_v_f16m1(out1_ptr + ldc * 4, vl);
                _acc51 = vle16_v_f16m1(out1_ptr + ldc * 5, vl);
                _acc61 = vle16_v_f16m1(out1_ptr + ldc * 6, vl);
                _acc71 = vle16_v_f16m1(out1_ptr + ldc * 7, vl);
            }

            for (int c = 0; c < K_BLOCK; c++) {
                vfloat16m1_t _in0 = vle16_v_f16m1(in_ptr, vl);
                vfloat16m1_t _in1 = vle16_v_f16m1(in_ptr + packn, vl);
                in_ptr += pack2n;

                _acc00 = vfmacc_vf_f16m1(_acc00, k_ptr[0], _in0, vl);
                _acc10 = vfmacc_vf_f16m1(_acc10, k_ptr[1], _in0, vl);
                _acc20 = vfmacc_vf_f16m1(_acc20, k_ptr[2], _in0, vl);
                _acc30 = vfmacc_vf_f16m1(_acc30, k_ptr[3], _in0, vl);
                _acc40 = vfmacc_vf_f16m1(_acc40, k_ptr[4], _in0, vl);
                _acc50 = vfmacc_vf_f16m1(_acc50, k_ptr[5], _in0, vl);
                _acc60 = vfmacc_vf_f16m1(_acc60, k_ptr[6], _in0, vl);
                _acc70 = vfmacc_vf_f16m1(_acc70, k_ptr[7], _in0, vl);
                _acc01 = vfmacc_vf_f16m1(_acc01, k_ptr[0], _in1, vl);
                _acc11 = vfmacc_vf_f16m1(_acc11, k_ptr[1], _in1, vl);
                _acc21 = vfmacc_vf_f16m1(_acc21, k_ptr[2], _in1, vl);
                _acc31 = vfmacc_vf_f16m1(_acc31, k_ptr[3], _in1, vl);
                _acc41 = vfmacc_vf_f16m1(_acc41, k_ptr[4], _in1, vl);
                _acc51 = vfmacc_vf_f16m1(_acc51, k_ptr[5], _in1, vl);
                _acc61 = vfmacc_vf_f16m1(_acc61, k_ptr[6], _in1, vl);
                _acc71 = vfmacc_vf_f16m1(_acc71, k_ptr[7], _in1, vl);
                k_ptr += 8;
            }

            vse16_v_f16m1(out0_ptr, _acc00, vl);
            vse16_v_f16m1(out0_ptr + ldc, _acc10, vl);
            vse16_v_f16m1(out0_ptr + ldc * 2, _acc20, vl);
            vse16_v_f16m1(out0_ptr + ldc * 3, _acc30, vl);
            vse16_v_f16m1(out0_ptr + ldc * 4, _acc40, vl);
            vse16_v_f16m1(out0_ptr + ldc * 5, _acc50, vl);
            vse16_v_f16m1(out0_ptr + ldc * 6, _acc60, vl);
            vse16_v_f16m1(out0_ptr + ldc * 7, _acc70, vl);
            vse16_v_f16m1(out1_ptr, _acc01, vl);
            vse16_v_f16m1(out1_ptr + ldc, _acc11, vl);
            vse16_v_f16m1(out1_ptr + ldc * 2, _acc21, vl);
            vse16_v_f16m1(out1_ptr + ldc * 3, _acc31, vl);
            vse16_v_f16m1(out1_ptr + ldc * 4, _acc41, vl);
            vse16_v_f16m1(out1_ptr + ldc * 5, _acc51, vl);
            vse16_v_f16m1(out1_ptr + ldc * 6, _acc61, vl);
            vse16_v_f16m1(out1_ptr + ldc * 7, _acc71, vl);
        }
        while (j < N_BLOCK) {
            vl = vsetvl_e16m1(N_BLOCK - j);
            const __fp16 *k_ptr = kernel_ptr;
            const __fp16 *in_ptr = input_data + j * K_BLOCK;
            __fp16 *out0_ptr = output_data + i * ldc + j;

            vfloat16m1_t _acc00;
            vfloat16m1_t _acc10;
            vfloat16m1_t _acc20;
            vfloat16m1_t _acc30;
            vfloat16m1_t _acc40;
            vfloat16m1_t _acc50;
            vfloat16m1_t _acc60;
            vfloat16m1_t _acc70;
            if (k_idx == 0) {
                const __fp16 *b_ptr = bias_data + i;
                _acc00 = vfmv_v_f_f16m1(b_ptr[0], vl);
                _acc10 = vfmv_v_f_f16m1(b_ptr[1], vl);
                _acc20 = vfmv_v_f_f16m1(b_ptr[2], vl);
                _acc30 = vfmv_v_f_f16m1(b_ptr[3], vl);
                _acc40 = vfmv_v_f_f16m1(b_ptr[4], vl);
                _acc50 = vfmv_v_f_f16m1(b_ptr[5], vl);
                _acc60 = vfmv_v_f_f16m1(b_ptr[6], vl);
                _acc70 = vfmv_v_f_f16m1(b_ptr[7], vl);
            } else {
                _acc00 = vle16_v_f16m1(out0_ptr, vl);
                _acc10 = vle16_v_f16m1(out0_ptr + ldc, vl);
                _acc20 = vle16_v_f16m1(out0_ptr + ldc * 2, vl);
                _acc30 = vle16_v_f16m1(out0_ptr + ldc * 3, vl);
                _acc40 = vle16_v_f16m1(out0_ptr + ldc * 4, vl);
                _acc50 = vle16_v_f16m1(out0_ptr + ldc * 5, vl);
                _acc60 = vle16_v_f16m1(out0_ptr + ldc * 6, vl);
                _acc70 = vle16_v_f16m1(out0_ptr + ldc * 7, vl);
            }

            for (int c = 0; c < K_BLOCK; c++) {
                vfloat16m1_t _in0 = vle16_v_f16m1(in_ptr, vl);
                in_ptr += vl;

                _acc00 = vfmacc_vf_f16m1(_acc00, k_ptr[0], _in0, vl);
                _acc10 = vfmacc_vf_f16m1(_acc10, k_ptr[1], _in0, vl);
                _acc20 = vfmacc_vf_f16m1(_acc20, k_ptr[2], _in0, vl);
                _acc30 = vfmacc_vf_f16m1(_acc30, k_ptr[3], _in0, vl);
                _acc40 = vfmacc_vf_f16m1(_acc40, k_ptr[4], _in0, vl);
                _acc50 = vfmacc_vf_f16m1(_acc50, k_ptr[5], _in0, vl);
                _acc60 = vfmacc_vf_f16m1(_acc60, k_ptr[6], _in0, vl);
                _acc70 = vfmacc_vf_f16m1(_acc70, k_ptr[7], _in0, vl);
                k_ptr += 8;
            }

            vse16_v_f16m1(out0_ptr, _acc00, vl);
            vse16_v_f16m1(out0_ptr + ldc, _acc10, vl);
            vse16_v_f16m1(out0_ptr + ldc * 2, _acc20, vl);
            vse16_v_f16m1(out0_ptr + ldc * 3, _acc30, vl);
            vse16_v_f16m1(out0_ptr + ldc * 4, _acc40, vl);
            vse16_v_f16m1(out0_ptr + ldc * 5, _acc50, vl);
            vse16_v_f16m1(out0_ptr + ldc * 6, _acc60, vl);
            vse16_v_f16m1(out0_ptr + ldc * 7, _acc70, vl);
            j += vl;
        }
    }
    for (; i + 3 < M_BLOCK; i += 4) {
        const __fp16 *kernel_ptr = kernel_data + i * K_BLOCK;
        int j = 0;
        vl = vsetvl_e16m1(packn);
        for (; j + pack2n - 1 < N_BLOCK; j += pack2n) {
            const __fp16 *k_ptr = kernel_ptr;
            const __fp16 *in_ptr = input_data + j * K_BLOCK;
            __fp16 *out0_ptr = output_data + i * ldc + j;
            __fp16 *out1_ptr = out0_ptr + packn;

            vfloat16m1_t _acc00;
            vfloat16m1_t _acc10;
            vfloat16m1_t _acc20;
            vfloat16m1_t _acc30;
            vfloat16m1_t _acc01;
            vfloat16m1_t _acc11;
            vfloat16m1_t _acc21;
            vfloat16m1_t _acc31;
            if (k_idx == 0) {
                const __fp16 *b_ptr = bias_data + i;
                _acc00 = vfmv_v_f_f16m1(b_ptr[0], vl);
                _acc10 = vfmv_v_f_f16m1(b_ptr[1], vl);
                _acc20 = vfmv_v_f_f16m1(b_ptr[2], vl);
                _acc30 = vfmv_v_f_f16m1(b_ptr[3], vl);
                _acc01 = vmv_v_v_f16m1(_acc00, vl);
                _acc11 = vmv_v_v_f16m1(_acc10, vl);
                _acc21 = vmv_v_v_f16m1(_acc20, vl);
                _acc31 = vmv_v_v_f16m1(_acc30, vl);
            } else {
                _acc00 = vle16_v_f16m1(out0_ptr, vl);
                _acc10 = vle16_v_f16m1(out0_ptr + ldc, vl);
                _acc20 = vle16_v_f16m1(out0_ptr + ldc * 2, vl);
                _acc30 = vle16_v_f16m1(out0_ptr + ldc * 3, vl);
                _acc01 = vle16_v_f16m1(out1_ptr, vl);
                _acc11 = vle16_v_f16m1(out1_ptr + ldc, vl);
                _acc21 = vle16_v_f16m1(out1_ptr + ldc * 2, vl);
                _acc31 = vle16_v_f16m1(out1_ptr + ldc * 3, vl);
            }

            for (int c = 0; c < K_BLOCK; c++) {
                vfloat16m1_t _in0 = vle16_v_f16m1(in_ptr, vl);
                vfloat16m1_t _in1 = vle16_v_f16m1(in_ptr + packn, vl);
                in_ptr += pack2n;

                _acc00 = vfmacc_vf_f16m1(_acc00, k_ptr[0], _in0, vl);
                _acc10 = vfmacc_vf_f16m1(_acc10, k_ptr[1], _in0, vl);
                _acc20 = vfmacc_vf_f16m1(_acc20, k_ptr[2], _in0, vl);
                _acc30 = vfmacc_vf_f16m1(_acc30, k_ptr[3], _in0, vl);
                _acc01 = vfmacc_vf_f16m1(_acc01, k_ptr[0], _in1, vl);
                _acc11 = vfmacc_vf_f16m1(_acc11, k_ptr[1], _in1, vl);
                _acc21 = vfmacc_vf_f16m1(_acc21, k_ptr[2], _in1, vl);
                _acc31 = vfmacc_vf_f16m1(_acc31, k_ptr[3], _in1, vl);
                k_ptr += 4;
            }

            vse16_v_f16m1(out0_ptr, _acc00, vl);
            vse16_v_f16m1(out0_ptr + ldc, _acc10, vl);
            vse16_v_f16m1(out0_ptr + ldc * 2, _acc20, vl);
            vse16_v_f16m1(out0_ptr + ldc * 3, _acc30, vl);
            vse16_v_f16m1(out1_ptr, _acc01, vl);
            vse16_v_f16m1(out1_ptr + ldc, _acc11, vl);
            vse16_v_f16m1(out1_ptr + ldc * 2, _acc21, vl);
            vse16_v_f16m1(out1_ptr + ldc * 3, _acc31, vl);
        }
        while (j < N_BLOCK) {
            vl = vsetvl_e16m1(N_BLOCK - j);
            const __fp16 *k_ptr = kernel_ptr;
            const __fp16 *in_ptr = input_data + j * K_BLOCK;
            __fp16 *out0_ptr = output_data + i * ldc + j;

            vfloat16m1_t _acc00;
            vfloat16m1_t _acc10;
            vfloat16m1_t _acc20;
            vfloat16m1_t _acc30;
            if (k_idx == 0) {
                const __fp16 *b_ptr = bias_data + i;
                _acc00 = vfmv_v_f_f16m1(b_ptr[0], vl);
                _acc10 = vfmv_v_f_f16m1(b_ptr[1], vl);
                _acc20 = vfmv_v_f_f16m1(b_ptr[2], vl);
                _acc30 = vfmv_v_f_f16m1(b_ptr[3], vl);
            } else {
                _acc00 = vle16_v_f16m1(out0_ptr, vl);
                _acc10 = vle16_v_f16m1(out0_ptr + ldc, vl);
                _acc20 = vle16_v_f16m1(out0_ptr + ldc * 2, vl);
                _acc30 = vle16_v_f16m1(out0_ptr + ldc * 3, vl);
            }

            for (int c = 0; c < K_BLOCK; c++) {
                vfloat16m1_t _in0 = vle16_v_f16m1(in_ptr, vl);
                in_ptr += vl;

                _acc00 = vfmacc_vf_f16m1(_acc00, k_ptr[0], _in0, vl);
                _acc10 = vfmacc_vf_f16m1(_acc10, k_ptr[1], _in0, vl);
                _acc20 = vfmacc_vf_f16m1(_acc20, k_ptr[2], _in0, vl);
                _acc30 = vfmacc_vf_f16m1(_acc30, k_ptr[3], _in0, vl);
                k_ptr += 4;
            }

            vse16_v_f16m1(out0_ptr, _acc00, vl);
            vse16_v_f16m1(out0_ptr + ldc, _acc10, vl);
            vse16_v_f16m1(out0_ptr + ldc * 2, _acc20, vl);
            vse16_v_f16m1(out0_ptr + ldc * 3, _acc30, vl);
            j += vl;
        }
    }
    for (; i + 1 < M_BLOCK; i += 2) {
        const __fp16 *kernel_ptr = kernel_data + i * K_BLOCK;
        int j = 0;
        vl = vsetvl_e16m1(packn);
        for (; j + pack2n - 1 < N_BLOCK; j += pack2n) {
            const __fp16 *k_ptr = kernel_ptr;
            const __fp16 *in_ptr = input_data + j * K_BLOCK;
            __fp16 *out0_ptr = output_data + i * ldc + j;
            __fp16 *out1_ptr = out0_ptr + packn;

            vfloat16m1_t _acc00;
            vfloat16m1_t _acc10;
            vfloat16m1_t _acc01;
            vfloat16m1_t _acc11;
            if (k_idx == 0) {
                const __fp16 *b_ptr = bias_data + i;
                _acc00 = vfmv_v_f_f16m1(b_ptr[0], vl);
                _acc10 = vfmv_v_f_f16m1(b_ptr[1], vl);
                _acc01 = vmv_v_v_f16m1(_acc00, vl);
                _acc11 = vmv_v_v_f16m1(_acc10, vl);
            } else {
                _acc00 = vle16_v_f16m1(out0_ptr, vl);
                _acc10 = vle16_v_f16m1(out0_ptr + ldc, vl);
                _acc01 = vle16_v_f16m1(out1_ptr, vl);
                _acc11 = vle16_v_f16m1(out1_ptr + ldc, vl);
            }

            for (int c = 0; c < K_BLOCK; c++) {
                vfloat16m1_t _in0 = vle16_v_f16m1(in_ptr, vl);
                vfloat16m1_t _in1 = vle16_v_f16m1(in_ptr + packn, vl);
                in_ptr += pack2n;

                _acc00 = vfmacc_vf_f16m1(_acc00, k_ptr[0], _in0, vl);
                _acc10 = vfmacc_vf_f16m1(_acc10, k_ptr[1], _in0, vl);
                _acc01 = vfmacc_vf_f16m1(_acc01, k_ptr[0], _in1, vl);
                _acc11 = vfmacc_vf_f16m1(_acc11, k_ptr[1], _in1, vl);
                k_ptr += 2;
            }

            vse16_v_f16m1(out0_ptr, _acc00, vl);
            vse16_v_f16m1(out0_ptr + ldc, _acc10, vl);
            vse16_v_f16m1(out1_ptr, _acc01, vl);
            vse16_v_f16m1(out1_ptr + ldc, _acc11, vl);
        }
        while (j < N_BLOCK) {
            vl = vsetvl_e16m1(N_BLOCK - j);
            const __fp16 *k_ptr = kernel_ptr;
            const __fp16 *in_ptr = input_data + j * K_BLOCK;
            __fp16 *out0_ptr = output_data + i * ldc + j;

            vfloat16m1_t _acc00;
            vfloat16m1_t _acc10;
            if (k_idx == 0) {
                const __fp16 *b_ptr = bias_data + i;
                _acc00 = vfmv_v_f_f16m1(b_ptr[0], vl);
                _acc10 = vfmv_v_f_f16m1(b_ptr[1], vl);
            } else {
                _acc00 = vle16_v_f16m1(out0_ptr, vl);
                _acc10 = vle16_v_f16m1(out0_ptr + ldc, vl);
            }

            for (int c = 0; c < K_BLOCK; c++) {
                vfloat16m1_t _in0 = vle16_v_f16m1(in_ptr, vl);
                in_ptr += vl;

                _acc00 = vfmacc_vf_f16m1(_acc00, k_ptr[0], _in0, vl);
                _acc10 = vfmacc_vf_f16m1(_acc10, k_ptr[1], _in0, vl);
                k_ptr += 2;
            }

            vse16_v_f16m1(out0_ptr, _acc00, vl);
            vse16_v_f16m1(out0_ptr + ldc, _acc10, vl);
            j += vl;
        }
    }
    for (; i < M_BLOCK; i++) {
        const __fp16 *kernel_ptr = kernel_data + i * K_BLOCK;
        int j = 0;
        vl = vsetvl_e16m1(packn);
        for (; j + pack2n - 1 < N_BLOCK; j += pack2n) {
            const __fp16 *k_ptr = kernel_ptr;
            const __fp16 *in_ptr = input_data + j * K_BLOCK;
            __fp16 *out0_ptr = output_data + i * ldc + j;
            __fp16 *out1_ptr = out0_ptr + packn;

            vfloat16m1_t _acc00;
            vfloat16m1_t _acc01;
            if (k_idx == 0) {
                const __fp16 *b_ptr = bias_data + i;
                _acc00 = vfmv_v_f_f16m1(b_ptr[0], vl);
                _acc01 = vmv_v_v_f16m1(_acc00, vl);
            } else {
                _acc00 = vle16_v_f16m1(out0_ptr, vl);
                _acc01 = vle16_v_f16m1(out1_ptr, vl);
            }

            for (int c = 0; c < K_BLOCK; c++) {
                vfloat16m1_t _in0 = vle16_v_f16m1(in_ptr, vl);
                vfloat16m1_t _in1 = vle16_v_f16m1(in_ptr + packn, vl);
                in_ptr += pack2n;

                _acc00 = vfmacc_vf_f16m1(_acc00, k_ptr[0], _in0, vl);
                _acc01 = vfmacc_vf_f16m1(_acc01, k_ptr[0], _in1, vl);
                k_ptr += 1;
            }

            vse16_v_f16m1(out0_ptr, _acc00, vl);
            vse16_v_f16m1(out1_ptr, _acc01, vl);
        }
        while (j < N_BLOCK) {
            vl = vsetvl_e16m1(N_BLOCK - j);
            const __fp16 *k_ptr = kernel_ptr;
            const __fp16 *in_ptr = input_data + j * K_BLOCK;
            __fp16 *out0_ptr = output_data + i * ldc + j;

            vfloat16m1_t _acc00;
            if (k_idx == 0) {
                const __fp16 *b_ptr = bias_data + i;
                _acc00 = vfmv_v_f_f16m1(b_ptr[0], vl);
            } else {
                _acc00 = vle16_v_f16m1(out0_ptr, vl);
            }

            for (int c = 0; c < K_BLOCK; c++) {
                vfloat16m1_t _in0 = vle16_v_f16m1(in_ptr, vl);
                in_ptr += vl;

                _acc00 = vfmacc_vf_f16m1(_acc00, k_ptr[0], _in0, vl);
                k_ptr += 1;
            }

            vse16_v_f16m1(out0_ptr, _acc00, vl);
            j += vl;
        }
    }
}

/*************************************************************
 * packn = vlenb / sizeof(__fp16)
 * m_blk: M_BLK, M_tail
 * n_blk: N_BLK, N_tail
 * k_blk: K_BLK, K_tail
 *
 * dst - output: [m, n]
 * sa - kernel:  [m/m_blk, k/k_blk, m_blk/12, k_blk, 12]
 * sb - input:   [n/n_blk, k/k_blk, n_blk/pack2n, k_blk, pack2n]
 * bias:         [m]
 ************************************************************/
void shl_rvv_gemm_block_12xpack2n_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb,
                                       __fp16 *bias, int m, int k, int n, const int M_BLK,
                                       const int K_BLK, const int N_BLK)
{
    const __fp16 *kernel_data = sa;
    const __fp16 *input_data = sb;
    __fp16 *output_data = dst;

    int flag_bias = 1;  // default: conv2d layer include bias
    if (bias == NULL) {
        flag_bias = 0;
        bias = (__fp16 *)shl_mem_alloc(m * sizeof(__fp16));
    }

    int m_block = M_BLK;
    int m_idx = 0;
    while (m_idx < m) {
        if (m - m_idx < m_block) {
            m_block = m - m_idx;
        }

        int n_block = N_BLK;
        int n_idx = 0;
        while (n_idx < n) {
            if (n - n_idx < n_block) {
                n_block = n - n_idx;
            }

            int k_block = K_BLK;
            int k_idx = 0;
            while (k_idx < k) {
                if (k - k_idx < k_block) {
                    k_block = k - k_idx;
                }
                __fp16 *out = output_data + m_idx * n + n_idx;
                const __fp16 *ker = kernel_data + m_idx * k + k_idx * m_block;
                const __fp16 *in = input_data + n_idx * k + k_idx * n_block;
                gemm_12xpack2n_fp16(out, ker, in, bias + m_idx, m_block, n_block, k_block, n,
                                    k_idx);
                k_idx += k_block;
            }

            n_idx += n_block;
        }

        m_idx += m_block;
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
