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

/************************************************************************
 * input matrix and kernel matrix have been reordered
 ***********************************************************************/

// vlen=128
void shl_rvv_gemm_8x16_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, __fp16 *bias, int m,
                            int k, int n, int ldc)
{
    __fp16 *kernel_data = (__fp16 *)sa;
    __fp16 *input_data = (__fp16 *)sb;
    __fp16 *output_data = dst;

    int flag_bias = 1;  // default: conv2d layer include bias
    if (bias == NULL) {
        flag_bias = 0;
        bias = (__fp16 *)shl_mem_alloc(m * sizeof(__fp16));
    }
    __fp16 *bias_ptr = bias;

    int vl;

    int i = 0;
    // m8 loop
    for (; i + 7 < m; i += 8) {
        vl = vsetvl_e16m2(16);

        __fp16 *in_ptr = input_data;

        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = out_ptr0 + ldc;
        __fp16 *out_ptr2 = out_ptr1 + ldc;
        __fp16 *out_ptr3 = out_ptr2 + ldc;
        __fp16 *out_ptr4 = out_ptr3 + ldc;
        __fp16 *out_ptr5 = out_ptr4 + ldc;
        __fp16 *out_ptr6 = out_ptr5 + ldc;
        __fp16 *out_ptr7 = out_ptr6 + ldc;

        int j = 0;
        // m8n16 loop
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            vfloat16m2_t _acc0 = vfmv_v_f_f16m2(bias_ptr[0], vl);
            vfloat16m2_t _acc1 = vfmv_v_f_f16m2(bias_ptr[1], vl);
            vfloat16m2_t _acc2 = vfmv_v_f_f16m2(bias_ptr[2], vl);
            vfloat16m2_t _acc3 = vfmv_v_f_f16m2(bias_ptr[3], vl);
            vfloat16m2_t _acc4 = vfmv_v_f_f16m2(bias_ptr[4], vl);
            vfloat16m2_t _acc5 = vfmv_v_f_f16m2(bias_ptr[5], vl);
            vfloat16m2_t _acc6 = vfmv_v_f_f16m2(bias_ptr[6], vl);
            vfloat16m2_t _acc7 = vfmv_v_f_f16m2(bias_ptr[7], vl);  // init acc with bias_data

            for (int c = 0; c < k; c++) {
                vfloat16m2_t _input = vle16_v_f16m2(in_ptr, vl);

                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                __fp16 k4 = kernel_ptr[4];
                __fp16 k5 = kernel_ptr[5];
                __fp16 k6 = kernel_ptr[6];
                __fp16 k7 = kernel_ptr[7];

                _acc0 = vfmacc_vf_f16m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m2(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f16m2(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f16m2(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f16m2(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f16m2(_acc7, k7, _input, vl);

                kernel_ptr += 8;
                in_ptr += 16;
            }
            vse16_v_f16m2(out_ptr0, _acc0, vl);
            vse16_v_f16m2(out_ptr1, _acc1, vl);
            vse16_v_f16m2(out_ptr2, _acc2, vl);
            vse16_v_f16m2(out_ptr3, _acc3, vl);
            vse16_v_f16m2(out_ptr4, _acc4, vl);
            vse16_v_f16m2(out_ptr5, _acc5, vl);
            vse16_v_f16m2(out_ptr6, _acc6, vl);
            vse16_v_f16m2(out_ptr7, _acc7, vl);
            out_ptr0 += 16;
            out_ptr1 += 16;
            out_ptr2 += 16;
            out_ptr3 += 16;
            out_ptr4 += 16;
            out_ptr5 += 16;
            out_ptr6 += 16;
            out_ptr7 += 16;
        }

        vl = vsetvl_e16m1(8);

        // m8n8
        for (; j + 7 < n; j += 8) {
            __fp16 *kernel_ptr = kernel_data;
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);
            vfloat16m1_t _acc4 = vfmv_v_f_f16m1(bias_ptr[4], vl);
            vfloat16m1_t _acc5 = vfmv_v_f_f16m1(bias_ptr[5], vl);
            vfloat16m1_t _acc6 = vfmv_v_f_f16m1(bias_ptr[6], vl);
            vfloat16m1_t _acc7 = vfmv_v_f_f16m1(bias_ptr[7], vl);  // init acc with bias_data

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);

                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                __fp16 k4 = kernel_ptr[4];
                __fp16 k5 = kernel_ptr[5];
                __fp16 k6 = kernel_ptr[6];
                __fp16 k7 = kernel_ptr[7];

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f16m1(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f16m1(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f16m1(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f16m1(_acc7, k7, _input, vl);

                kernel_ptr += 8;
                in_ptr += 8;
            }
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);
            vse16_v_f16m1(out_ptr4, _acc4, vl);
            vse16_v_f16m1(out_ptr5, _acc5, vl);
            vse16_v_f16m1(out_ptr6, _acc6, vl);
            vse16_v_f16m1(out_ptr7, _acc7, vl);
            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
            out_ptr4 += 8;
            out_ptr5 += 8;
            out_ptr6 += 8;
            out_ptr7 += 8;
        }

        // m8n4
        for (; j + 3 < n; j += 4) {
            vfloat16m1_t _acc0 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc1 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc2 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc3 = vle16_v_f16m1(bias_ptr, vl);  // init acc with bias_data

            __fp16 *kernel_ptr = kernel_data;

            __fp16 *in_ptr0 = in_ptr;
            __fp16 *in_ptr1 = in_ptr0 + k;
            __fp16 *in_ptr2 = in_ptr1 + k;
            __fp16 *in_ptr3 = in_ptr2 + k;

            out_ptr1 = out_ptr0 + 1;
            out_ptr2 = out_ptr0 + 2;
            out_ptr3 = out_ptr0 + 3;

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _kernel = vle16_v_f16m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, in_ptr1[c], _kernel, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, in_ptr2[c], _kernel, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, in_ptr3[c], _kernel, vl);
                kernel_ptr += 8;
            }
            vsse16_v_f16m1(out_ptr0, ldc * sizeof(__fp16), _acc0, vl);
            vsse16_v_f16m1(out_ptr1, ldc * sizeof(__fp16), _acc1, vl);
            vsse16_v_f16m1(out_ptr2, ldc * sizeof(__fp16), _acc2, vl);
            vsse16_v_f16m1(out_ptr3, ldc * sizeof(__fp16), _acc3, vl);
            out_ptr0 += 4;
            in_ptr += 4 * k;
        }

        // m8n2
        for (; j + 1 < n; j += 2) {
            vfloat16m1_t _acc0 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc1 = vle16_v_f16m1(bias_ptr, vl);

            __fp16 *kernel_ptr = kernel_data;

            __fp16 *in_ptr0 = in_ptr;
            __fp16 *in_ptr1 = in_ptr0 + k;

            out_ptr1 = out_ptr0 + 1;

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _kernel = vle16_v_f16m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, in_ptr1[c], _kernel, vl);
                kernel_ptr += 8;
            }
            vsse16_v_f16m1(out_ptr0, ldc * sizeof(__fp16), _acc0, vl);
            vsse16_v_f16m1(out_ptr1, ldc * sizeof(__fp16), _acc1, vl);
            out_ptr0 += 2;
            in_ptr += 2 * k;
        }

        // m8n1
        for (; j < n; j++) {
            vfloat16m1_t _acc0 = vle16_v_f16m1(bias_ptr, vl);
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr0 = in_ptr;

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _kernel = vle16_v_f16m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, in_ptr0[c], _kernel, vl);
                kernel_ptr += 8;
            }
            vsse16_v_f16m1(out_ptr0, ldc * sizeof(__fp16), _acc0, vl);
        }
        kernel_data += 8 * k;
        output_data += 8 * ldc;
        bias_ptr += 8;
    }

    // m4
    for (; i + 3 < m; i += 4) {
        vl = vsetvl_e16m2(16);

        __fp16 *in_ptr = input_data;

        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = out_ptr0 + ldc;
        __fp16 *out_ptr2 = out_ptr1 + ldc;
        __fp16 *out_ptr3 = out_ptr2 + ldc;

        int j = 0;
        // m4n16 loop
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            vfloat16m2_t _acc0 = vfmv_v_f_f16m2(bias_ptr[0], vl);
            vfloat16m2_t _acc1 = vfmv_v_f_f16m2(bias_ptr[1], vl);
            vfloat16m2_t _acc2 = vfmv_v_f_f16m2(bias_ptr[2], vl);
            vfloat16m2_t _acc3 = vfmv_v_f_f16m2(bias_ptr[3], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m2_t _input = vle16_v_f16m2(in_ptr, vl);

                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                _acc0 = vfmacc_vf_f16m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m2(_acc3, k3, _input, vl);

                kernel_ptr += 4;
                in_ptr += 16;
            }
            vse16_v_f16m2(out_ptr0, _acc0, vl);
            vse16_v_f16m2(out_ptr1, _acc1, vl);
            vse16_v_f16m2(out_ptr2, _acc2, vl);
            vse16_v_f16m2(out_ptr3, _acc3, vl);
            out_ptr0 += 16;
            out_ptr1 += 16;
            out_ptr2 += 16;
            out_ptr3 += 16;
        }

        // m4n8
        for (; j + 7 < n; j += 8) {
            vl = vsetvl_e16m1(8);

            __fp16 *kernel_ptr = kernel_data;
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);
            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);

                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, k3, _input, vl);

                kernel_ptr += 4;
                in_ptr += 8;
            }
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);

            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
        }

        // TODO: rvv opt
        for (; j < n; j++) {
            __fp16 acc0 = bias_ptr[0];
            __fp16 acc1 = bias_ptr[1];
            __fp16 acc2 = bias_ptr[2];
            __fp16 acc3 = bias_ptr[3];
            for (int c = 0; c < k; c++) {
                acc0 += kernel_data[4 * c] * in_ptr[c];
                acc1 += kernel_data[4 * c + 1] * in_ptr[c];
                acc2 += kernel_data[4 * c + 2] * in_ptr[c];
                acc3 += kernel_data[4 * c + 3] * in_ptr[c];
            }
            *out_ptr0++ = acc0;
            *out_ptr1++ = acc1;
            *out_ptr2++ = acc2;
            *out_ptr3++ = acc3;
            in_ptr += k;
        }
        kernel_data += 4 * k;
        output_data += 4 * ldc;
        bias_ptr += 4;
    }

    // m2
    for (; i + 1 < m; i += 2) {
        vl = vsetvl_e16m2(16);

        __fp16 *in_ptr = input_data;
        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = out_ptr0 + ldc;

        int j = 0;
        // m2n16 loop
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            vfloat16m2_t _acc0 = vfmv_v_f_f16m2(bias_ptr[0], vl);
            vfloat16m2_t _acc1 = vfmv_v_f_f16m2(bias_ptr[1], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m2_t _input = vle16_v_f16m2(in_ptr, vl);

                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                _acc0 = vfmacc_vf_f16m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m2(_acc1, k1, _input, vl);
                kernel_ptr += 2;
                in_ptr += 16;
            }
            vse16_v_f16m2(out_ptr0, _acc0, vl);
            vse16_v_f16m2(out_ptr1, _acc1, vl);
            out_ptr0 += 16;
            out_ptr1 += 16;
        }

        vl = vsetvl_e16m1(8);
        // m2n8
        for (; j + 7 < n; j += 8) {
            __fp16 *kernel_ptr = kernel_data;
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);

                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);

                kernel_ptr += 2;
                in_ptr += 8;
            }
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            out_ptr0 += 8;
            out_ptr1 += 8;
        }

        // TODO: rvv opt
        for (; j < n; j++) {
            __fp16 acc0 = bias_ptr[0];
            __fp16 acc1 = bias_ptr[1];
            for (int c = 0; c < k; c++) {
                acc0 += kernel_data[2 * c] * in_ptr[c];
                acc1 += kernel_data[2 * c + 1] * in_ptr[c];
            }
            *out_ptr0++ = acc0;
            *out_ptr1++ = acc1;
            in_ptr += k;
        }
        kernel_data += 2 * k;
        output_data += 2 * ldc;
        bias_ptr += 2;
    }

    // m1
    for (; i < m; i++) {
        vl = vsetvl_e16m2(16);

        __fp16 *in_ptr = input_data;
        __fp16 *out_ptr0 = output_data;

        int j = 0;
        // m1n16 loop
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            vfloat16m2_t _acc0 = vfmv_v_f_f16m2(bias_ptr[0], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m2_t _input = vle16_v_f16m2(in_ptr, vl);
                __fp16 k0 = kernel_ptr[0];
                _acc0 = vfmacc_vf_f16m2(_acc0, k0, _input, vl);
                kernel_ptr += 1;
                in_ptr += 16;
            }
            vse16_v_f16m2(out_ptr0, _acc0, vl);
            out_ptr0 += 16;
        }

        vl = vsetvl_e16m1(8);
        // m1n8
        for (; j + 7 < n; j += 8) {
            __fp16 *kernel_ptr = kernel_data;
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                __fp16 k0 = kernel_ptr[0];
                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                kernel_ptr += 1;
                in_ptr += 8;
            }
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            out_ptr0 += 8;
        }

        // TODO: rvv opt
        for (; j < n; j++) {
            __fp16 acc0 = bias_ptr[0];
            for (int c = 0; c < k; c++) {
                acc0 += kernel_data[c] * in_ptr[c];
            }
            *out_ptr0++ = acc0;
            in_ptr += k;
        }
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}

// vlen=256
void shl_rvv256_gemm_16x16_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, __fp16 *bias,
                                int m, int k, int n, int ldc)
{
    __fp16 *kernel_data = (__fp16 *)sa;
    __fp16 *input_data = (__fp16 *)sb;
    __fp16 *output_data = dst;

    int flag_bias = 1;  // default: conv2d layer include bias
    if (bias == NULL) {
        flag_bias = 0;
        bias = (__fp16 *)shl_mem_alloc(m * 2);
    }
    __fp16 *bias_ptr = bias;

    int vl;

    int i = 0;
    // m16 loop
    for (; i + 15 < m; i += 16) {
        vl = vsetvl_e16m1(16);

        __fp16 *in_ptr = input_data;

        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = out_ptr0 + ldc;
        __fp16 *out_ptr2 = out_ptr1 + ldc;
        __fp16 *out_ptr3 = out_ptr2 + ldc;
        __fp16 *out_ptr4 = out_ptr3 + ldc;
        __fp16 *out_ptr5 = out_ptr4 + ldc;
        __fp16 *out_ptr6 = out_ptr5 + ldc;
        __fp16 *out_ptr7 = out_ptr6 + ldc;
        __fp16 *out_ptr8 = out_ptr7 + ldc;
        __fp16 *out_ptr9 = out_ptr8 + ldc;
        __fp16 *out_ptr10 = out_ptr9 + ldc;
        __fp16 *out_ptr11 = out_ptr10 + ldc;
        __fp16 *out_ptr12 = out_ptr11 + ldc;
        __fp16 *out_ptr13 = out_ptr12 + ldc;
        __fp16 *out_ptr14 = out_ptr13 + ldc;
        __fp16 *out_ptr15 = out_ptr14 + ldc;

        int j = 0;
        // m16n16 loop
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);
            vfloat16m1_t _acc4 = vfmv_v_f_f16m1(bias_ptr[4], vl);
            vfloat16m1_t _acc5 = vfmv_v_f_f16m1(bias_ptr[5], vl);
            vfloat16m1_t _acc6 = vfmv_v_f_f16m1(bias_ptr[6], vl);
            vfloat16m1_t _acc7 = vfmv_v_f_f16m1(bias_ptr[7], vl);
            vfloat16m1_t _acc8 = vfmv_v_f_f16m1(bias_ptr[8], vl);
            vfloat16m1_t _acc9 = vfmv_v_f_f16m1(bias_ptr[9], vl);
            vfloat16m1_t _acc10 = vfmv_v_f_f16m1(bias_ptr[10], vl);
            vfloat16m1_t _acc11 = vfmv_v_f_f16m1(bias_ptr[11], vl);
            vfloat16m1_t _acc12 = vfmv_v_f_f16m1(bias_ptr[12], vl);
            vfloat16m1_t _acc13 = vfmv_v_f_f16m1(bias_ptr[13], vl);
            vfloat16m1_t _acc14 = vfmv_v_f_f16m1(bias_ptr[14], vl);
            vfloat16m1_t _acc15 = vfmv_v_f_f16m1(bias_ptr[15], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);

                _acc0 = vfmacc_vf_f16m1(_acc0, kernel_ptr[0], _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, kernel_ptr[1], _input, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, kernel_ptr[2], _input, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, kernel_ptr[3], _input, vl);
                _acc4 = vfmacc_vf_f16m1(_acc4, kernel_ptr[4], _input, vl);
                _acc5 = vfmacc_vf_f16m1(_acc5, kernel_ptr[5], _input, vl);
                _acc6 = vfmacc_vf_f16m1(_acc6, kernel_ptr[6], _input, vl);
                _acc7 = vfmacc_vf_f16m1(_acc7, kernel_ptr[7], _input, vl);
                _acc8 = vfmacc_vf_f16m1(_acc8, kernel_ptr[8], _input, vl);
                _acc9 = vfmacc_vf_f16m1(_acc9, kernel_ptr[9], _input, vl);
                _acc10 = vfmacc_vf_f16m1(_acc10, kernel_ptr[10], _input, vl);
                _acc11 = vfmacc_vf_f16m1(_acc11, kernel_ptr[11], _input, vl);
                _acc12 = vfmacc_vf_f16m1(_acc12, kernel_ptr[12], _input, vl);
                _acc13 = vfmacc_vf_f16m1(_acc13, kernel_ptr[13], _input, vl);
                _acc14 = vfmacc_vf_f16m1(_acc14, kernel_ptr[14], _input, vl);
                _acc15 = vfmacc_vf_f16m1(_acc15, kernel_ptr[15], _input, vl);

                kernel_ptr += 16;
                in_ptr += 16;
            }
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);
            vse16_v_f16m1(out_ptr4, _acc4, vl);
            vse16_v_f16m1(out_ptr5, _acc5, vl);
            vse16_v_f16m1(out_ptr6, _acc6, vl);
            vse16_v_f16m1(out_ptr7, _acc7, vl);
            vse16_v_f16m1(out_ptr8, _acc8, vl);
            vse16_v_f16m1(out_ptr9, _acc9, vl);
            vse16_v_f16m1(out_ptr10, _acc10, vl);
            vse16_v_f16m1(out_ptr11, _acc11, vl);
            vse16_v_f16m1(out_ptr12, _acc12, vl);
            vse16_v_f16m1(out_ptr13, _acc13, vl);
            vse16_v_f16m1(out_ptr14, _acc14, vl);
            vse16_v_f16m1(out_ptr15, _acc15, vl);
            out_ptr0 += 16;
            out_ptr1 += 16;
            out_ptr2 += 16;
            out_ptr3 += 16;
            out_ptr4 += 16;
            out_ptr5 += 16;
            out_ptr6 += 16;
            out_ptr7 += 16;
            out_ptr8 += 16;
            out_ptr9 += 16;
            out_ptr10 += 16;
            out_ptr11 += 16;
            out_ptr12 += 16;
            out_ptr13 += 16;
            out_ptr14 += 16;
            out_ptr15 += 16;
        }
        // m16n8
        for (; j + 7 < n; j += 8) {
            vfloat16m1_t _acc0 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc1 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc2 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc3 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc4 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc5 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc6 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc7 = vle16_v_f16m1(bias_ptr, vl);

            __fp16 *kernel_ptr = kernel_data;

            __fp16 *in_ptr0 = in_ptr;
            __fp16 *in_ptr1 = in_ptr0 + k;
            __fp16 *in_ptr2 = in_ptr1 + k;
            __fp16 *in_ptr3 = in_ptr2 + k;
            __fp16 *in_ptr4 = in_ptr3 + k;
            __fp16 *in_ptr5 = in_ptr4 + k;
            __fp16 *in_ptr6 = in_ptr5 + k;
            __fp16 *in_ptr7 = in_ptr6 + k;

            out_ptr1 = out_ptr0 + 1;
            out_ptr2 = out_ptr0 + 2;
            out_ptr3 = out_ptr0 + 3;
            out_ptr4 = out_ptr0 + 4;
            out_ptr5 = out_ptr0 + 5;
            out_ptr6 = out_ptr0 + 6;
            out_ptr7 = out_ptr0 + 7;

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _kernel = vle16_v_f16m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, in_ptr1[c], _kernel, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, in_ptr2[c], _kernel, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, in_ptr3[c], _kernel, vl);
                _acc4 = vfmacc_vf_f16m1(_acc4, in_ptr4[c], _kernel, vl);
                _acc5 = vfmacc_vf_f16m1(_acc5, in_ptr5[c], _kernel, vl);
                _acc6 = vfmacc_vf_f16m1(_acc6, in_ptr6[c], _kernel, vl);
                _acc7 = vfmacc_vf_f16m1(_acc7, in_ptr7[c], _kernel, vl);
                kernel_ptr += 16;
            }
            vsse16_v_f16m1(out_ptr0, ldc * sizeof(__fp16), _acc0, vl);
            vsse16_v_f16m1(out_ptr1, ldc * sizeof(__fp16), _acc1, vl);
            vsse16_v_f16m1(out_ptr2, ldc * sizeof(__fp16), _acc2, vl);
            vsse16_v_f16m1(out_ptr3, ldc * sizeof(__fp16), _acc3, vl);
            vsse16_v_f16m1(out_ptr4, ldc * sizeof(__fp16), _acc4, vl);
            vsse16_v_f16m1(out_ptr5, ldc * sizeof(__fp16), _acc5, vl);
            vsse16_v_f16m1(out_ptr6, ldc * sizeof(__fp16), _acc6, vl);
            vsse16_v_f16m1(out_ptr7, ldc * sizeof(__fp16), _acc7, vl);
            out_ptr0 += 8;
            in_ptr += 8 * k;
        }
        // m16n4
        for (; j + 3 < n; j += 4) {
            vfloat16m1_t _acc0 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc1 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc2 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc3 = vle16_v_f16m1(bias_ptr, vl);

            __fp16 *kernel_ptr = kernel_data;

            __fp16 *in_ptr0 = in_ptr;
            __fp16 *in_ptr1 = in_ptr0 + k;
            __fp16 *in_ptr2 = in_ptr1 + k;
            __fp16 *in_ptr3 = in_ptr2 + k;

            out_ptr1 = out_ptr0 + 1;
            out_ptr2 = out_ptr0 + 2;
            out_ptr3 = out_ptr0 + 3;

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _kernel = vle16_v_f16m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, in_ptr1[c], _kernel, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, in_ptr2[c], _kernel, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, in_ptr3[c], _kernel, vl);
                kernel_ptr += 16;
            }
            vsse16_v_f16m1(out_ptr0, ldc * sizeof(__fp16), _acc0, vl);
            vsse16_v_f16m1(out_ptr1, ldc * sizeof(__fp16), _acc1, vl);
            vsse16_v_f16m1(out_ptr2, ldc * sizeof(__fp16), _acc2, vl);
            vsse16_v_f16m1(out_ptr3, ldc * sizeof(__fp16), _acc3, vl);
            out_ptr0 += 4;
            in_ptr += 4 * k;
        }
        // m16n2
        for (; j + 1 < n; j += 2) {
            vfloat16m1_t _acc0 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc1 = vle16_v_f16m1(bias_ptr, vl);

            __fp16 *kernel_ptr = kernel_data;

            __fp16 *in_ptr0 = in_ptr;
            __fp16 *in_ptr1 = in_ptr0 + k;

            out_ptr1 = out_ptr0 + 1;

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _kernel = vle16_v_f16m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, in_ptr1[c], _kernel, vl);
                kernel_ptr += 16;
            }
            vsse16_v_f16m1(out_ptr0, ldc * sizeof(__fp16), _acc0, vl);
            vsse16_v_f16m1(out_ptr1, ldc * sizeof(__fp16), _acc1, vl);
            out_ptr0 += 2;
            in_ptr += 2 * k;
        }
        // m16n1
        for (; j < n; j++) {
            vfloat16m1_t _acc0 = vle16_v_f16m1(bias_ptr, vl);
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr0 = in_ptr;

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _kernel = vle16_v_f16m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, in_ptr0[c], _kernel, vl);
                kernel_ptr += 16;
            }
            vsse16_v_f16m1(out_ptr0, ldc * sizeof(__fp16), _acc0, vl);
            out_ptr0 += 1;
            in_ptr += 1 * k;
        }
        kernel_data += 16 * k;
        output_data += 16 * ldc;
        bias_ptr += 16;
    }

    // m8
    for (; i + 7 < m; i += 8) {
        vl = vsetvl_e16m1(16);

        __fp16 *in_ptr = input_data;

        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = out_ptr0 + ldc;
        __fp16 *out_ptr2 = out_ptr1 + ldc;
        __fp16 *out_ptr3 = out_ptr2 + ldc;
        __fp16 *out_ptr4 = out_ptr3 + ldc;
        __fp16 *out_ptr5 = out_ptr4 + ldc;
        __fp16 *out_ptr6 = out_ptr5 + ldc;
        __fp16 *out_ptr7 = out_ptr6 + ldc;

        int j = 0;
        // m8n16 loop
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);
            vfloat16m1_t _acc4 = vfmv_v_f_f16m1(bias_ptr[4], vl);
            vfloat16m1_t _acc5 = vfmv_v_f_f16m1(bias_ptr[5], vl);
            vfloat16m1_t _acc6 = vfmv_v_f_f16m1(bias_ptr[6], vl);
            vfloat16m1_t _acc7 = vfmv_v_f_f16m1(bias_ptr[7], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);

                _acc0 = vfmacc_vf_f16m1(_acc0, kernel_ptr[0], _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, kernel_ptr[1], _input, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, kernel_ptr[2], _input, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, kernel_ptr[3], _input, vl);
                _acc4 = vfmacc_vf_f16m1(_acc4, kernel_ptr[4], _input, vl);
                _acc5 = vfmacc_vf_f16m1(_acc5, kernel_ptr[5], _input, vl);
                _acc6 = vfmacc_vf_f16m1(_acc6, kernel_ptr[6], _input, vl);
                _acc7 = vfmacc_vf_f16m1(_acc7, kernel_ptr[7], _input, vl);

                kernel_ptr += 8;
                in_ptr += 16;
            }
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);
            vse16_v_f16m1(out_ptr4, _acc4, vl);
            vse16_v_f16m1(out_ptr5, _acc5, vl);
            vse16_v_f16m1(out_ptr6, _acc6, vl);
            vse16_v_f16m1(out_ptr7, _acc7, vl);
            out_ptr0 += 16;
            out_ptr1 += 16;
            out_ptr2 += 16;
            out_ptr3 += 16;
            out_ptr4 += 16;
            out_ptr5 += 16;
            out_ptr6 += 16;
            out_ptr7 += 16;
        }
        // m8n8
        // TODO: rvv opt
        for (; j < n; j++) {
            float acc0 = bias_ptr[0];
            float acc1 = bias_ptr[1];
            float acc2 = bias_ptr[2];
            float acc3 = bias_ptr[3];
            float acc4 = bias_ptr[4];
            float acc5 = bias_ptr[5];
            float acc6 = bias_ptr[6];
            float acc7 = bias_ptr[7];
            for (int c = 0; c < k; c++) {
                acc0 += kernel_data[8 * c] * in_ptr[c];
                acc1 += kernel_data[8 * c + 1] * in_ptr[c];
                acc2 += kernel_data[8 * c + 2] * in_ptr[c];
                acc3 += kernel_data[8 * c + 3] * in_ptr[c];
                acc4 += kernel_data[8 * c + 4] * in_ptr[c];
                acc5 += kernel_data[8 * c + 5] * in_ptr[c];
                acc6 += kernel_data[8 * c + 6] * in_ptr[c];
                acc7 += kernel_data[8 * c + 7] * in_ptr[c];
            }
            *out_ptr0++ = acc0;
            *out_ptr1++ = acc1;
            *out_ptr2++ = acc2;
            *out_ptr3++ = acc3;
            *out_ptr4++ = acc4;
            *out_ptr5++ = acc5;
            *out_ptr6++ = acc6;
            *out_ptr7++ = acc7;
            in_ptr += k;
        }
        kernel_data += 8 * k;
        output_data += 8 * ldc;
        bias_ptr += 8;
    }

    // m4
    for (; i + 3 < m; m += 4) {
        vl = vsetvl_e16m1(16);

        __fp16 *in_ptr = input_data;

        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = out_ptr0 + ldc;
        __fp16 *out_ptr2 = out_ptr1 + ldc;
        __fp16 *out_ptr3 = out_ptr2 + ldc;

        int j = 0;
        // m4n16 loop
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);

                _acc0 = vfmacc_vf_f16m1(_acc0, kernel_ptr[0], _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, kernel_ptr[1], _input, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, kernel_ptr[2], _input, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, kernel_ptr[3], _input, vl);

                kernel_ptr += 4;
                in_ptr += 16;
            }
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);
            out_ptr0 += 16;
            out_ptr1 += 16;
            out_ptr2 += 16;
            out_ptr3 += 16;
        }
        // m4n8
        // TODO: rvv opt
        for (; j < n; j++) {
            float acc0 = bias_ptr[0];
            float acc1 = bias_ptr[1];
            float acc2 = bias_ptr[2];
            float acc3 = bias_ptr[3];
            for (int c = 0; c < k; c++) {
                acc0 += kernel_data[4 * c] * in_ptr[c];
                acc1 += kernel_data[4 * c + 1] * in_ptr[c];
                acc2 += kernel_data[4 * c + 2] * in_ptr[c];
                acc3 += kernel_data[4 * c + 3] * in_ptr[c];
            }
            *out_ptr0++ = acc0;
            *out_ptr1++ = acc1;
            *out_ptr2++ = acc2;
            *out_ptr3++ = acc3;
            in_ptr += k;
        }
        kernel_data += 4 * k;
        output_data += 4 * ldc;
        bias_ptr += 4;
    }

    // m2
    for (; i + 1 < m; m += 2) {
        vl = vsetvl_e16m1(16);

        __fp16 *in_ptr = input_data;

        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = out_ptr0 + ldc;

        int j = 0;
        // m2n16 loop
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);

                _acc0 = vfmacc_vf_f16m1(_acc0, kernel_ptr[0], _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, kernel_ptr[1], _input, vl);

                kernel_ptr += 2;
                in_ptr += 16;
            }
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            out_ptr0 += 16;
            out_ptr1 += 16;
        }
        // m2n8
        for (; j < n; j++) {
            float acc0 = bias_ptr[0];
            float acc1 = bias_ptr[1];
            for (int c = 0; c < k; c++) {
                acc0 += kernel_data[2 * c] * in_ptr[c];
                acc1 += kernel_data[2 * c + 1] * in_ptr[c];
            }
            *out_ptr0++ = acc0;
            *out_ptr1++ = acc1;
            in_ptr += k;
        }
        kernel_data += 2 * k;
        output_data += 2 * ldc;
        bias_ptr += 2;
    }

    // m1
    for (; i < m; i++) {
        vl = vsetvl_e16m1(16);

        __fp16 *in_ptr = input_data;
        __fp16 *out_ptr0 = output_data;

        int j = 0;
        // m1n16 loop
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, kernel_ptr[0], _input, vl);
                kernel_ptr += 1;
                in_ptr += 16;
            }
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            out_ptr0 += 16;
        }
        // m1n8
        for (; j < n; j++) {
            float acc0 = bias_ptr[0];
            for (int c = 0; c < k; c++) {
                acc0 += kernel_data[c] * in_ptr[c];
            }
            *out_ptr0++ = acc0;
            in_ptr += k;
        }
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
