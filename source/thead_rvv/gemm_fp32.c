/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

/************************************************************************
 * input matrix and kernel matrix have been reordered
 ***********************************************************************/

/*
    dst - output:[m, n]
    sa - kernel: [m, k]
    sb - input:  [k, n]
*/
// vlen=128
void shl_rvv_gemm_8x8_fp32(float *dst, const float *sa, const float *sb, float *bias, int m, int k,
                           int n, int ldc)
{
    float *kernel_data = (float *)sa;
    float *input_data = (float *)sb;
    float *output_data = dst;

    int flag_bias = 1;  // default: conv2d layer include bias
    if (bias == NULL) {
        flag_bias = 0;
        bias = (float *)shl_mem_alloc(m * sizeof(float));
    }
    float *bias_ptr = bias;

    int vl;

    int i = 0;
    // m8 loop
    vl = vsetvl_e32m2(8);
    for (; i + 7 < m; i += 8) {
        float *in_ptr = input_data;

        float *out_ptr0 = output_data;
        float *out_ptr1 = out_ptr0 + ldc;
        float *out_ptr2 = out_ptr1 + ldc;
        float *out_ptr3 = out_ptr2 + ldc;
        float *out_ptr4 = out_ptr3 + ldc;
        float *out_ptr5 = out_ptr4 + ldc;
        float *out_ptr6 = out_ptr5 + ldc;
        float *out_ptr7 = out_ptr6 + ldc;

        int j = 0;
        // m8n8 loop
        for (; j + 7 < n; j += 8) {
            float *kernel_ptr = kernel_data;
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            vfloat32m2_t _acc1 = vfmv_v_f_f32m2(bias_ptr[1], vl);
            vfloat32m2_t _acc2 = vfmv_v_f_f32m2(bias_ptr[2], vl);
            vfloat32m2_t _acc3 = vfmv_v_f_f32m2(bias_ptr[3], vl);
            vfloat32m2_t _acc4 = vfmv_v_f_f32m2(bias_ptr[4], vl);
            vfloat32m2_t _acc5 = vfmv_v_f_f32m2(bias_ptr[5], vl);
            vfloat32m2_t _acc6 = vfmv_v_f_f32m2(bias_ptr[6], vl);
            vfloat32m2_t _acc7 = vfmv_v_f_f32m2(bias_ptr[7], vl);  // init acc with bias_data

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);

                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                float k2 = kernel_ptr[2];
                float k3 = kernel_ptr[3];
                float k4 = kernel_ptr[4];
                float k5 = kernel_ptr[5];
                float k6 = kernel_ptr[6];
                float k7 = kernel_ptr[7];

                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f32m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f32m2(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f32m2(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f32m2(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f32m2(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f32m2(_acc7, k7, _input, vl);

                kernel_ptr += 8;
                in_ptr += 8;
            }
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            vse32_v_f32m2(out_ptr1, _acc1, vl);
            vse32_v_f32m2(out_ptr2, _acc2, vl);
            vse32_v_f32m2(out_ptr3, _acc3, vl);
            vse32_v_f32m2(out_ptr4, _acc4, vl);
            vse32_v_f32m2(out_ptr5, _acc5, vl);
            vse32_v_f32m2(out_ptr6, _acc6, vl);
            vse32_v_f32m2(out_ptr7, _acc7, vl);
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
            vfloat32m2_t _acc0 = vle32_v_f32m2(bias_ptr, vl);
            vfloat32m2_t _acc1 = vle32_v_f32m2(bias_ptr, vl);
            vfloat32m2_t _acc2 = vle32_v_f32m2(bias_ptr, vl);
            vfloat32m2_t _acc3 = vle32_v_f32m2(bias_ptr, vl);  // init acc with bias_data

            float *kernel_ptr = kernel_data;

            float *in_ptr0 = in_ptr;
            float *in_ptr1 = in_ptr0 + k;
            float *in_ptr2 = in_ptr1 + k;
            float *in_ptr3 = in_ptr2 + k;

            out_ptr1 = out_ptr0 + 1;
            out_ptr2 = out_ptr0 + 2;
            out_ptr3 = out_ptr0 + 3;

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _kernel = vle32_v_f32m2(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, in_ptr1[c], _kernel, vl);
                _acc2 = vfmacc_vf_f32m2(_acc2, in_ptr2[c], _kernel, vl);
                _acc3 = vfmacc_vf_f32m2(_acc3, in_ptr3[c], _kernel, vl);
                kernel_ptr += 8;
            }
            vsse32_v_f32m2(out_ptr0, ldc * sizeof(float), _acc0, vl);
            vsse32_v_f32m2(out_ptr1, ldc * sizeof(float), _acc1, vl);
            vsse32_v_f32m2(out_ptr2, ldc * sizeof(float), _acc2, vl);
            vsse32_v_f32m2(out_ptr3, ldc * sizeof(float), _acc3, vl);
            out_ptr0 += 4;
            in_ptr += 4 * k;
        }
        // m8n2
        for (; j + 1 < n; j += 2) {
            vfloat32m2_t _acc0 = vle32_v_f32m2(bias_ptr, vl);
            vfloat32m2_t _acc1 = vle32_v_f32m2(bias_ptr, vl);

            float *kernel_ptr = kernel_data;

            float *in_ptr0 = in_ptr;
            float *in_ptr1 = in_ptr0 + k;

            out_ptr1 = out_ptr0 + 1;

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _kernel = vle32_v_f32m2(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, in_ptr1[c], _kernel, vl);
                kernel_ptr += 8;
            }
            vsse32_v_f32m2(out_ptr0, ldc * sizeof(float), _acc0, vl);
            vsse32_v_f32m2(out_ptr1, ldc * sizeof(float), _acc1, vl);
            out_ptr0 += 2;
            in_ptr += 2 * k;
        }
        // m8n1
        for (; j < n; j++) {
            vfloat32m2_t _acc0 = vle32_v_f32m2(bias_ptr, vl);
            float *kernel_ptr = kernel_data;
            float *in_ptr0 = in_ptr;

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _kernel = vle32_v_f32m2(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, in_ptr0[c], _kernel, vl);
                kernel_ptr += 8;
            }
            vsse32_v_f32m2(out_ptr0, ldc * sizeof(float), _acc0, vl);
        }
        kernel_data += 8 * k;
        output_data += 8 * ldc;
        bias_ptr += 8;
    }

    // m4
    for (; i + 3 < m; i += 4) {
        vl = vsetvl_e32m2(8);
        float *in_ptr = input_data;

        float *out_ptr0 = output_data;
        float *out_ptr1 = out_ptr0 + ldc;
        float *out_ptr2 = out_ptr1 + ldc;
        float *out_ptr3 = out_ptr2 + ldc;

        int j = 0;
        // m4n8 loop
        for (; j + 7 < n; j += 8) {
            float *kernel_ptr = kernel_data;
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            vfloat32m2_t _acc1 = vfmv_v_f_f32m2(bias_ptr[1], vl);
            vfloat32m2_t _acc2 = vfmv_v_f_f32m2(bias_ptr[2], vl);
            vfloat32m2_t _acc3 = vfmv_v_f_f32m2(bias_ptr[3], vl);

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);
                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                float k2 = kernel_ptr[2];
                float k3 = kernel_ptr[3];

                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f32m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f32m2(_acc3, k3, _input, vl);

                kernel_ptr += 4;
                in_ptr += 8;
            }
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            vse32_v_f32m2(out_ptr1, _acc1, vl);
            vse32_v_f32m2(out_ptr2, _acc2, vl);
            vse32_v_f32m2(out_ptr3, _acc3, vl);
            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
        }
        vl = vsetvl_e32m1(4);
        // m4n4
        for (; j + 3 < n; j += 4) {
            vfloat32m1_t _acc0 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc1 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc2 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc3 = vle32_v_f32m1(bias_ptr, vl);  // init acc with bias_data

            float *kernel_ptr = kernel_data;

            float *in_ptr0 = in_ptr;
            float *in_ptr1 = in_ptr0 + k;
            float *in_ptr2 = in_ptr1 + k;
            float *in_ptr3 = in_ptr2 + k;

            out_ptr1 = out_ptr0 + 1;
            out_ptr2 = out_ptr0 + 2;
            out_ptr3 = out_ptr0 + 3;

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel = vle32_v_f32m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, in_ptr1[c], _kernel, vl);
                _acc2 = vfmacc_vf_f32m1(_acc2, in_ptr2[c], _kernel, vl);
                _acc3 = vfmacc_vf_f32m1(_acc3, in_ptr3[c], _kernel, vl);
                kernel_ptr += 4;
            }
            vsse32_v_f32m1(out_ptr0, ldc * sizeof(float), _acc0, vl);
            vsse32_v_f32m1(out_ptr1, ldc * sizeof(float), _acc1, vl);
            vsse32_v_f32m1(out_ptr2, ldc * sizeof(float), _acc2, vl);
            vsse32_v_f32m1(out_ptr3, ldc * sizeof(float), _acc3, vl);
            out_ptr0 += 4;
            in_ptr += 4 * k;
        }
        // m4n2
        for (; j + 1 < n; j += 2) {
            vfloat32m1_t _acc0 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc1 = vle32_v_f32m1(bias_ptr, vl);  // init acc with bias_data

            float *kernel_ptr = kernel_data;
            float *in_ptr0 = in_ptr;
            float *in_ptr1 = in_ptr0 + k;
            out_ptr1 = out_ptr0 + 1;

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel = vle32_v_f32m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, in_ptr1[c], _kernel, vl);
                kernel_ptr += 4;
            }
            vsse32_v_f32m1(out_ptr0, ldc * sizeof(float), _acc0, vl);
            vsse32_v_f32m1(out_ptr1, ldc * sizeof(float), _acc1, vl);
            out_ptr0 += 2;
            in_ptr += 2 * k;
        }
        // m4n1
        for (; j < n; j++) {
            vfloat32m1_t _acc0 = vle32_v_f32m1(bias_ptr, vl);  // init acc with bias_data
            float *kernel_ptr = kernel_data;
            float *in_ptr0 = in_ptr;

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel = vle32_v_f32m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, in_ptr0[c], _kernel, vl);
                kernel_ptr += 4;
            }
            vsse32_v_f32m1(out_ptr0, ldc * sizeof(float), _acc0, vl);
        }
        kernel_data += 4 * k;
        output_data += 4 * ldc;
        bias_ptr += 4;
    }

    // m2
    for (; i + 1 < m; i += 2) {
        vl = vsetvl_e32m2(8);
        float *in_ptr = input_data;
        float *out_ptr0 = output_data;
        float *out_ptr1 = out_ptr0 + ldc;

        int j = 0;
        // m2n8 loop
        for (; j + 7 < n; j += 8) {
            float *kernel_ptr = kernel_data;
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            vfloat32m2_t _acc1 = vfmv_v_f_f32m2(bias_ptr[1], vl);

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);
                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k1, _input, vl);
                kernel_ptr += 2;
                in_ptr += 8;
            }
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            vse32_v_f32m2(out_ptr1, _acc1, vl);
            out_ptr0 += 8;
            out_ptr1 += 8;
        }

        // TODO: rvv opt
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
        vl = vsetvl_e32m2(8);
        float *in_ptr = input_data;
        float *out_ptr0 = output_data;

        int j = 0;
        // m1n8 loop
        for (; j + 7 < n; j += 8) {
            float *kernel_ptr = kernel_data;
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            for (int c = 0; c < k; c++) {
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);
                float k0 = kernel_ptr[0];
                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                kernel_ptr += 1;
                in_ptr += 8;
            }
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            out_ptr0 += 8;
        }

        // TODO: rvv opt
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

// vlen=256
void shl_rvv256_gemm_8x16_fp32(float *dst, const float *sa, const float *sb, float *bias, int m,
                               int k, int n, int ldc)
{
    float *kernel_data = (float *)sa;
    float *input_data = (float *)sb;
    float *output_data = dst;

    int flag_bias = 1;  // default: conv2d layer include bias
    if (bias == NULL) {
        flag_bias = 0;
        bias = (float *)shl_mem_alloc(m * sizeof(float));
    }
    float *bias_ptr = bias;

    int vl;

    int i = 0;
    // m8 loop
    for (; i + 7 < m; i += 8) {
        vl = vsetvl_e32m2(16);

        float *in_ptr = input_data;

        float *out_ptr0 = output_data;
        float *out_ptr1 = out_ptr0 + ldc;
        float *out_ptr2 = out_ptr1 + ldc;
        float *out_ptr3 = out_ptr2 + ldc;
        float *out_ptr4 = out_ptr3 + ldc;
        float *out_ptr5 = out_ptr4 + ldc;
        float *out_ptr6 = out_ptr5 + ldc;
        float *out_ptr7 = out_ptr6 + ldc;

        int j = 0;
        // m8n16 loop
        for (; j + 15 < n; j += 16) {
            float *kernel_ptr = kernel_data;
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            vfloat32m2_t _acc1 = vfmv_v_f_f32m2(bias_ptr[1], vl);
            vfloat32m2_t _acc2 = vfmv_v_f_f32m2(bias_ptr[2], vl);
            vfloat32m2_t _acc3 = vfmv_v_f_f32m2(bias_ptr[3], vl);
            vfloat32m2_t _acc4 = vfmv_v_f_f32m2(bias_ptr[4], vl);
            vfloat32m2_t _acc5 = vfmv_v_f_f32m2(bias_ptr[5], vl);
            vfloat32m2_t _acc6 = vfmv_v_f_f32m2(bias_ptr[6], vl);
            vfloat32m2_t _acc7 = vfmv_v_f_f32m2(bias_ptr[7], vl);

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);

                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                float k2 = kernel_ptr[2];
                float k3 = kernel_ptr[3];
                float k4 = kernel_ptr[4];
                float k5 = kernel_ptr[5];
                float k6 = kernel_ptr[6];
                float k7 = kernel_ptr[7];

                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f32m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f32m2(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f32m2(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f32m2(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f32m2(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f32m2(_acc7, k7, _input, vl);

                kernel_ptr += 8;
                in_ptr += 16;
            }
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            vse32_v_f32m2(out_ptr1, _acc1, vl);
            vse32_v_f32m2(out_ptr2, _acc2, vl);
            vse32_v_f32m2(out_ptr3, _acc3, vl);
            vse32_v_f32m2(out_ptr4, _acc4, vl);
            vse32_v_f32m2(out_ptr5, _acc5, vl);
            vse32_v_f32m2(out_ptr6, _acc6, vl);
            vse32_v_f32m2(out_ptr7, _acc7, vl);
            out_ptr0 += 16;
            out_ptr1 += 16;
            out_ptr2 += 16;
            out_ptr3 += 16;
            out_ptr4 += 16;
            out_ptr5 += 16;
            out_ptr6 += 16;
            out_ptr7 += 16;
        }

        vl = vsetvl_e32m1(8);

        // m8n8
        for (; j + 7 < n; j += 8) {
            float *kernel_ptr = kernel_data;
            vfloat32m1_t _acc0 = vfmv_v_f_f32m1(bias_ptr[0], vl);
            vfloat32m1_t _acc1 = vfmv_v_f_f32m1(bias_ptr[1], vl);
            vfloat32m1_t _acc2 = vfmv_v_f_f32m1(bias_ptr[2], vl);
            vfloat32m1_t _acc3 = vfmv_v_f_f32m1(bias_ptr[3], vl);
            vfloat32m1_t _acc4 = vfmv_v_f_f32m1(bias_ptr[4], vl);
            vfloat32m1_t _acc5 = vfmv_v_f_f32m1(bias_ptr[5], vl);
            vfloat32m1_t _acc6 = vfmv_v_f_f32m1(bias_ptr[6], vl);
            vfloat32m1_t _acc7 = vfmv_v_f_f32m1(bias_ptr[7], vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _input = vle32_v_f32m1(in_ptr, vl);

                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                float k2 = kernel_ptr[2];
                float k3 = kernel_ptr[3];
                float k4 = kernel_ptr[4];
                float k5 = kernel_ptr[5];
                float k6 = kernel_ptr[6];
                float k7 = kernel_ptr[7];

                _acc0 = vfmacc_vf_f32m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f32m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f32m1(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f32m1(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f32m1(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f32m1(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f32m1(_acc7, k7, _input, vl);

                kernel_ptr += 8;
                in_ptr += 8;
            }
            vse32_v_f32m1(out_ptr0, _acc0, vl);
            vse32_v_f32m1(out_ptr1, _acc1, vl);
            vse32_v_f32m1(out_ptr2, _acc2, vl);
            vse32_v_f32m1(out_ptr3, _acc3, vl);
            vse32_v_f32m1(out_ptr4, _acc4, vl);
            vse32_v_f32m1(out_ptr5, _acc5, vl);
            vse32_v_f32m1(out_ptr6, _acc6, vl);
            vse32_v_f32m1(out_ptr7, _acc7, vl);
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
            vfloat32m1_t _acc0 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc1 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc2 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc3 = vle32_v_f32m1(bias_ptr, vl);

            float *kernel_ptr = kernel_data;

            float *in_ptr0 = in_ptr;
            float *in_ptr1 = in_ptr0 + k;
            float *in_ptr2 = in_ptr1 + k;
            float *in_ptr3 = in_ptr2 + k;

            out_ptr1 = out_ptr0 + 1;
            out_ptr2 = out_ptr0 + 2;
            out_ptr3 = out_ptr0 + 3;

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel = vle32_v_f32m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, in_ptr1[c], _kernel, vl);
                _acc2 = vfmacc_vf_f32m1(_acc2, in_ptr2[c], _kernel, vl);
                _acc3 = vfmacc_vf_f32m1(_acc3, in_ptr3[c], _kernel, vl);
                kernel_ptr += 8;
            }
            vsse32_v_f32m1(out_ptr0, ldc * sizeof(float), _acc0, vl);
            vsse32_v_f32m1(out_ptr1, ldc * sizeof(float), _acc1, vl);
            vsse32_v_f32m1(out_ptr2, ldc * sizeof(float), _acc2, vl);
            vsse32_v_f32m1(out_ptr3, ldc * sizeof(float), _acc3, vl);
            out_ptr0 += 4;
            in_ptr += 4 * k;
        }
        // m8n2
        for (; j + 1 < n; j += 2) {
            vfloat32m1_t _acc0 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc1 = vle32_v_f32m1(bias_ptr, vl);

            float *kernel_ptr = kernel_data;

            float *in_ptr0 = in_ptr;
            float *in_ptr1 = in_ptr0 + k;

            out_ptr1 = out_ptr0 + 1;

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel = vle32_v_f32m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, in_ptr1[c], _kernel, vl);
                kernel_ptr += 8;
            }
            vsse32_v_f32m1(out_ptr0, ldc * sizeof(float), _acc0, vl);
            vsse32_v_f32m1(out_ptr1, ldc * sizeof(float), _acc1, vl);
            out_ptr0 += 2;
            in_ptr += 2 * k;
        }
        // m8n1
        for (; j < n; j++) {
            vfloat32m1_t _acc0 = vle32_v_f32m1(bias_ptr, vl);
            float *kernel_ptr = kernel_data;
            float *in_ptr0 = in_ptr;

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel = vle32_v_f32m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, in_ptr0[c], _kernel, vl);
                kernel_ptr += 8;
            }
            vsse32_v_f32m1(out_ptr0, ldc * sizeof(__fp16), _acc0, vl);
        }
        kernel_data += 8 * k;
        output_data += 8 * ldc;
        bias_ptr += 8;
    }

    // m4
    for (; i + 3 < m; i += 4) {
        vl = vsetvl_e32m2(16);

        float *in_ptr = input_data;

        float *out_ptr0 = output_data;
        float *out_ptr1 = out_ptr0 + ldc;
        float *out_ptr2 = out_ptr1 + ldc;
        float *out_ptr3 = out_ptr2 + ldc;

        int j = 0;
        // m4n16 loop
        for (; j + 15 < n; j += 16) {
            float *kernel_ptr = kernel_data;
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            vfloat32m2_t _acc1 = vfmv_v_f_f32m2(bias_ptr[1], vl);
            vfloat32m2_t _acc2 = vfmv_v_f_f32m2(bias_ptr[2], vl);
            vfloat32m2_t _acc3 = vfmv_v_f_f32m2(bias_ptr[3], vl);

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);

                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                float k2 = kernel_ptr[2];
                float k3 = kernel_ptr[3];
                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f32m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f32m2(_acc3, k3, _input, vl);

                kernel_ptr += 4;
                in_ptr += 16;
            }
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            vse32_v_f32m2(out_ptr1, _acc1, vl);
            vse32_v_f32m2(out_ptr2, _acc2, vl);
            vse32_v_f32m2(out_ptr3, _acc3, vl);
            out_ptr0 += 16;
            out_ptr1 += 16;
            out_ptr2 += 16;
            out_ptr3 += 16;
        }

        // m4n8
        for (; j + 7 < n; j += 8) {
            vl = vsetvl_e16m1(8);

            float *kernel_ptr = kernel_data;
            vfloat32m1_t _acc0 = vfmv_v_f_f32m1(bias_ptr[0], vl);
            vfloat32m1_t _acc1 = vfmv_v_f_f32m1(bias_ptr[1], vl);
            vfloat32m1_t _acc2 = vfmv_v_f_f32m1(bias_ptr[2], vl);
            vfloat32m1_t _acc3 = vfmv_v_f_f32m1(bias_ptr[3], vl);
            for (int c = 0; c < k; c++) {
                vfloat32m1_t _input = vle32_v_f32m1(in_ptr, vl);

                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                float k2 = kernel_ptr[2];
                float k3 = kernel_ptr[3];

                _acc0 = vfmacc_vf_f32m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f32m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f32m1(_acc3, k3, _input, vl);

                kernel_ptr += 4;
                in_ptr += 8;
            }
            vse32_v_f32m1(out_ptr0, _acc0, vl);
            vse32_v_f32m1(out_ptr1, _acc1, vl);
            vse32_v_f32m1(out_ptr2, _acc2, vl);
            vse32_v_f32m1(out_ptr3, _acc3, vl);

            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
        }

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
    for (; i + 1 < m; i += 2) {
        vl = vsetvl_e16m2(16);

        float *in_ptr = input_data;
        float *out_ptr0 = output_data;
        float *out_ptr1 = out_ptr0 + ldc;

        int j = 0;
        // m2n16 loop
        for (; j + 15 < n; j += 16) {
            float *kernel_ptr = kernel_data;
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            vfloat32m2_t _acc1 = vfmv_v_f_f32m2(bias_ptr[1], vl);

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);

                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k1, _input, vl);
                kernel_ptr += 2;
                in_ptr += 16;
            }
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            vse32_v_f32m2(out_ptr1, _acc1, vl);
            out_ptr0 += 16;
            out_ptr1 += 16;
        }

        vl = vsetvl_e32m1(8);
        // m2n8
        for (; j + 7 < n; j += 8) {
            float *kernel_ptr = kernel_data;
            vfloat32m1_t _acc0 = vfmv_v_f_f32m1(bias_ptr[0], vl);
            vfloat32m1_t _acc1 = vfmv_v_f_f32m1(bias_ptr[1], vl);
            for (int c = 0; c < k; c++) {
                vfloat32m1_t _input = vle32_v_f32m1(in_ptr, vl);

                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                _acc0 = vfmacc_vf_f32m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k1, _input, vl);

                kernel_ptr += 2;
                in_ptr += 8;
            }
            vse32_v_f32m1(out_ptr0, _acc0, vl);
            vse32_v_f32m1(out_ptr1, _acc1, vl);
            out_ptr0 += 8;
            out_ptr1 += 8;
        }

        // TODO: rvv opt
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
        vl = vsetvl_e32m2(16);

        float *in_ptr = input_data;
        float *out_ptr0 = output_data;

        int j = 0;
        // m1n16 loop
        for (; j + 15 < n; j += 16) {
            float *kernel_ptr = kernel_data;
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);
                float k0 = kernel_ptr[0];
                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                kernel_ptr += 1;
                in_ptr += 16;
            }
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            out_ptr0 += 16;
        }

        vl = vsetvl_e32m1(8);
        // m1n8
        for (; j + 7 < n; j += 8) {
            float *kernel_ptr = kernel_data;
            vfloat32m1_t _acc0 = vfmv_v_f_f32m1(bias_ptr[0], vl);
            for (int c = 0; c < k; c++) {
                vfloat32m1_t _input = vle32_v_f32m1(in_ptr, vl);
                float k0 = kernel_ptr[0];
                _acc0 = vfmacc_vf_f32m1(_acc0, k0, _input, vl);
                kernel_ptr += 1;
                in_ptr += 8;
            }
            vse32_v_f32m1(out_ptr0, _acc0, vl);
            out_ptr0 += 8;
        }
        // TODO: rvv opt
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
