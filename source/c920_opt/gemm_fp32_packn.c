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

#include "shl_c920.h"

/**************************************************************
 * dst - output: [m/packn, n, packn]
 * sa - kernel:  [m/pack2n, k, pack2n]  [m/packn, k, packn]
 * sb - input:   [n/8, k, 8]
 **************************************************************/
void shl_c920_ncxhwx_gemm_8xpack2n_fp32(float *dst, const float *sa, const float *sb, float *bias,
                                        int m, int k, int n, int ldc)
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

    const int packn = csrr_vlenb() / sizeof(float);
    const int pack2n = packn * 2;
    int vl = vsetvl_e32m1(packn);

    int oc = 0;
    for (; oc + pack2n - 1 < m; oc += pack2n) {
        float *output0 = output_data + oc * n;  // 8 channel dot output
        float *output1 = output0 + packn * n;
        const float *img0 = input_data;
        const float *b0 = bias_ptr + oc;
        int t = 0;
        for (; t + 7 < n; t += 8) {
            const float *k0 = kernel_data + oc * k;  // 8 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);
            vfloat32m1_t _acc01 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc02 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc03 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc04 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc05 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc06 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc07 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc10 = vle32_v_f32m1(b0 + packn, vl);
            vfloat32m1_t _acc11 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc12 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc13 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc14 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc15 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc16 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc17 = vmv_v_v_f32m1(_acc10, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                vfloat32m1_t _kernel1 = vle32_v_f32m1(k0 + packn, vl);
                k0 += pack2n;

                vfloat32m1_t _img0 = vfmv_v_f_f32m1(img0[0], vl);
                vfloat32m1_t _img1 = vfmv_v_f_f32m1(img0[1], vl);
                vfloat32m1_t _img2 = vfmv_v_f_f32m1(img0[2], vl);
                vfloat32m1_t _img3 = vfmv_v_f_f32m1(img0[3], vl);
                vfloat32m1_t _img4 = vfmv_v_f_f32m1(img0[4], vl);
                vfloat32m1_t _img5 = vfmv_v_f_f32m1(img0[5], vl);
                vfloat32m1_t _img6 = vfmv_v_f_f32m1(img0[6], vl);
                vfloat32m1_t _img7 = vfmv_v_f_f32m1(img0[7], vl);
                img0 += 8;

                _acc00 = vfmacc_vv_f32m1(_acc00, _img0, _kernel0, vl);
                _acc01 = vfmacc_vv_f32m1(_acc01, _img1, _kernel0, vl);
                _acc02 = vfmacc_vv_f32m1(_acc02, _img2, _kernel0, vl);
                _acc03 = vfmacc_vv_f32m1(_acc03, _img3, _kernel0, vl);
                _acc04 = vfmacc_vv_f32m1(_acc04, _img4, _kernel0, vl);
                _acc05 = vfmacc_vv_f32m1(_acc05, _img5, _kernel0, vl);
                _acc06 = vfmacc_vv_f32m1(_acc06, _img6, _kernel0, vl);
                _acc07 = vfmacc_vv_f32m1(_acc07, _img7, _kernel0, vl);

                _acc10 = vfmacc_vv_f32m1(_acc10, _img0, _kernel1, vl);
                _acc11 = vfmacc_vv_f32m1(_acc11, _img1, _kernel1, vl);
                _acc12 = vfmacc_vv_f32m1(_acc12, _img2, _kernel1, vl);
                _acc13 = vfmacc_vv_f32m1(_acc13, _img3, _kernel1, vl);
                _acc14 = vfmacc_vv_f32m1(_acc14, _img4, _kernel1, vl);
                _acc15 = vfmacc_vv_f32m1(_acc15, _img5, _kernel1, vl);
                _acc16 = vfmacc_vv_f32m1(_acc16, _img6, _kernel1, vl);
                _acc17 = vfmacc_vv_f32m1(_acc17, _img7, _kernel1, vl);
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + packn * 1, _acc01, vl);
            vse32_v_f32m1(output0 + packn * 2, _acc02, vl);
            vse32_v_f32m1(output0 + packn * 3, _acc03, vl);
            vse32_v_f32m1(output0 + packn * 4, _acc04, vl);
            vse32_v_f32m1(output0 + packn * 5, _acc05, vl);
            vse32_v_f32m1(output0 + packn * 6, _acc06, vl);
            vse32_v_f32m1(output0 + packn * 7, _acc07, vl);
            output0 += packn * 8;

            vse32_v_f32m1(output1, _acc10, vl);
            vse32_v_f32m1(output1 + packn * 1, _acc11, vl);
            vse32_v_f32m1(output1 + packn * 2, _acc12, vl);
            vse32_v_f32m1(output1 + packn * 3, _acc13, vl);
            vse32_v_f32m1(output1 + packn * 4, _acc14, vl);
            vse32_v_f32m1(output1 + packn * 5, _acc15, vl);
            vse32_v_f32m1(output1 + packn * 6, _acc16, vl);
            vse32_v_f32m1(output1 + packn * 7, _acc17, vl);
            output1 += packn * 8;
        }
        for (; t + 3 < n; t += 4) {
            const float *k0 = kernel_data + oc * k;  // 8 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);
            vfloat32m1_t _acc01 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc02 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc03 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc10 = vle32_v_f32m1(b0 + packn, vl);
            vfloat32m1_t _acc11 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc12 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc13 = vmv_v_v_f32m1(_acc10, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                vfloat32m1_t _kernel1 = vle32_v_f32m1(k0 + packn, vl);
                k0 += pack2n;

                vfloat32m1_t _img0 = vfmv_v_f_f32m1(img0[0], vl);
                vfloat32m1_t _img1 = vfmv_v_f_f32m1(img0[1], vl);
                vfloat32m1_t _img2 = vfmv_v_f_f32m1(img0[2], vl);
                vfloat32m1_t _img3 = vfmv_v_f_f32m1(img0[3], vl);
                img0 += 4;

                _acc00 = vfmacc_vv_f32m1(_acc00, _img0, _kernel0, vl);
                _acc01 = vfmacc_vv_f32m1(_acc01, _img1, _kernel0, vl);
                _acc02 = vfmacc_vv_f32m1(_acc02, _img2, _kernel0, vl);
                _acc03 = vfmacc_vv_f32m1(_acc03, _img3, _kernel0, vl);

                _acc10 = vfmacc_vv_f32m1(_acc10, _img0, _kernel1, vl);
                _acc11 = vfmacc_vv_f32m1(_acc11, _img1, _kernel1, vl);
                _acc12 = vfmacc_vv_f32m1(_acc12, _img2, _kernel1, vl);
                _acc13 = vfmacc_vv_f32m1(_acc13, _img3, _kernel1, vl);
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + packn * 1, _acc01, vl);
            vse32_v_f32m1(output0 + packn * 2, _acc02, vl);
            vse32_v_f32m1(output0 + packn * 3, _acc03, vl);
            output0 += packn * 4;

            vse32_v_f32m1(output1, _acc10, vl);
            vse32_v_f32m1(output1 + packn * 1, _acc11, vl);
            vse32_v_f32m1(output1 + packn * 2, _acc12, vl);
            vse32_v_f32m1(output1 + packn * 3, _acc13, vl);
            output1 += packn * 4;
        }
        for (; t + 1 < n; t += 2) {
            const float *k0 = kernel_data + oc * k;  // 8 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);
            vfloat32m1_t _acc01 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc10 = vle32_v_f32m1(b0 + packn, vl);
            vfloat32m1_t _acc11 = vmv_v_v_f32m1(_acc10, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                vfloat32m1_t _kernel1 = vle32_v_f32m1(k0 + packn, vl);
                k0 += pack2n;

                vfloat32m1_t _img0 = vfmv_v_f_f32m1(img0[0], vl);
                vfloat32m1_t _img1 = vfmv_v_f_f32m1(img0[1], vl);
                img0 += 2;

                _acc00 = vfmacc_vv_f32m1(_acc00, _img0, _kernel0, vl);
                _acc01 = vfmacc_vv_f32m1(_acc01, _img1, _kernel0, vl);

                _acc10 = vfmacc_vv_f32m1(_acc10, _img0, _kernel1, vl);
                _acc11 = vfmacc_vv_f32m1(_acc11, _img1, _kernel1, vl);
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + packn * 1, _acc01, vl);
            output0 += packn * 2;

            vse32_v_f32m1(output1, _acc10, vl);
            vse32_v_f32m1(output1 + packn * 1, _acc11, vl);
            output1 += packn * 2;
        }
        for (; t < n; t++) {
            const float *k0 = kernel_data + oc * k;  // 8 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);
            vfloat32m1_t _acc10 = vle32_v_f32m1(b0 + packn, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                vfloat32m1_t _kernel1 = vle32_v_f32m1(k0 + packn, vl);
                k0 += pack2n;
                vfloat32m1_t _img0 = vfmv_v_f_f32m1(img0[0], vl);
                img0 += 1;

                _acc00 = vfmacc_vv_f32m1(_acc00, _img0, _kernel0, vl);
                _acc10 = vfmacc_vv_f32m1(_acc10, _img0, _kernel1, vl);
            }
            vse32_v_f32m1(output0, _acc00, vl);
            output0 += packn * 1;

            vse32_v_f32m1(output1, _acc10, vl);
            output1 += packn * 1;
        }
    }

    for (; oc + packn - 1 < m; oc += packn) {
        float *output0 = output_data + oc * n;  // 4 channel dot output
        const float *img0 = input_data;
        const float *b0 = bias_ptr + oc;
        int t = 0;
        for (; t + 7 < n; t += 8) {
            const float *k0 = kernel_data + oc * k;  // 4 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);
            vfloat32m1_t _acc01 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc02 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc03 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc04 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc05 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc06 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc07 = vmv_v_v_f32m1(_acc00, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                k0 += packn;

                vfloat32m1_t _img0 = vfmv_v_f_f32m1(img0[0], vl);
                vfloat32m1_t _img1 = vfmv_v_f_f32m1(img0[1], vl);
                vfloat32m1_t _img2 = vfmv_v_f_f32m1(img0[2], vl);
                vfloat32m1_t _img3 = vfmv_v_f_f32m1(img0[3], vl);
                vfloat32m1_t _img4 = vfmv_v_f_f32m1(img0[4], vl);
                vfloat32m1_t _img5 = vfmv_v_f_f32m1(img0[5], vl);
                vfloat32m1_t _img6 = vfmv_v_f_f32m1(img0[6], vl);
                vfloat32m1_t _img7 = vfmv_v_f_f32m1(img0[7], vl);
                img0 += 8;

                _acc00 = vfmacc_vv_f32m1(_acc00, _img0, _kernel0, vl);
                _acc01 = vfmacc_vv_f32m1(_acc01, _img1, _kernel0, vl);
                _acc02 = vfmacc_vv_f32m1(_acc02, _img2, _kernel0, vl);
                _acc03 = vfmacc_vv_f32m1(_acc03, _img3, _kernel0, vl);
                _acc04 = vfmacc_vv_f32m1(_acc04, _img4, _kernel0, vl);
                _acc05 = vfmacc_vv_f32m1(_acc05, _img5, _kernel0, vl);
                _acc06 = vfmacc_vv_f32m1(_acc06, _img6, _kernel0, vl);
                _acc07 = vfmacc_vv_f32m1(_acc07, _img7, _kernel0, vl);
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + packn * 1, _acc01, vl);
            vse32_v_f32m1(output0 + packn * 2, _acc02, vl);
            vse32_v_f32m1(output0 + packn * 3, _acc03, vl);
            vse32_v_f32m1(output0 + packn * 4, _acc04, vl);
            vse32_v_f32m1(output0 + packn * 5, _acc05, vl);
            vse32_v_f32m1(output0 + packn * 6, _acc06, vl);
            vse32_v_f32m1(output0 + packn * 7, _acc07, vl);
            output0 += packn * 8;
        }
        for (; t + 3 < n; t += 4) {
            const float *k0 = kernel_data + oc * k;  // 4 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);
            vfloat32m1_t _acc01 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc02 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc03 = vmv_v_v_f32m1(_acc00, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                k0 += packn;

                vfloat32m1_t _img0 = vfmv_v_f_f32m1(img0[0], vl);
                vfloat32m1_t _img1 = vfmv_v_f_f32m1(img0[1], vl);
                vfloat32m1_t _img2 = vfmv_v_f_f32m1(img0[2], vl);
                vfloat32m1_t _img3 = vfmv_v_f_f32m1(img0[3], vl);
                img0 += 4;

                _acc00 = vfmacc_vv_f32m1(_acc00, _img0, _kernel0, vl);
                _acc01 = vfmacc_vv_f32m1(_acc01, _img1, _kernel0, vl);
                _acc02 = vfmacc_vv_f32m1(_acc02, _img2, _kernel0, vl);
                _acc03 = vfmacc_vv_f32m1(_acc03, _img3, _kernel0, vl);
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + packn * 1, _acc01, vl);
            vse32_v_f32m1(output0 + packn * 2, _acc02, vl);
            vse32_v_f32m1(output0 + packn * 3, _acc03, vl);
            output0 += packn * 4;
        }
        for (; t + 1 < n; t += 2) {
            const float *k0 = kernel_data + oc * k;  // 4 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);
            vfloat32m1_t _acc01 = vmv_v_v_f32m1(_acc00, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                k0 += packn;

                vfloat32m1_t _img0 = vfmv_v_f_f32m1(img0[0], vl);
                vfloat32m1_t _img1 = vfmv_v_f_f32m1(img0[1], vl);
                img0 += 2;

                _acc00 = vfmacc_vv_f32m1(_acc00, _img0, _kernel0, vl);
                _acc01 = vfmacc_vv_f32m1(_acc01, _img1, _kernel0, vl);
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + packn * 1, _acc01, vl);
            output0 += packn * 2;
        }
        for (; t < n; t++) {
            const float *k0 = kernel_data + oc * k;  // 4 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                k0 += packn;
                vfloat32m1_t _img0 = vfmv_v_f_f32m1(img0[0], vl);
                img0 += 1;
                _acc00 = vfmacc_vv_f32m1(_acc00, _img0, _kernel0, vl);
            }
            vse32_v_f32m1(output0, _acc00, vl);
            output0 += packn * 1;
        }
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
