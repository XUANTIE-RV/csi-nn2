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

/*************************************************************
 * note: VLEN = 128/256 ... flexible vlen
 * input matrix and kernel matrix have been reordered
 * PS: 这里实现了两种寄存器分块，以vlen128 fp32 类型为例，分别是 8*8 和 8*12，
 * 两份代码可以合成一份，用宏或者条件来控制
 *************************************************************/

/**************************************************************
 * dst - output: [m/packn, n, packn]
 * sa - kernel:  [m/pack2n, k, pack2n]  [m/packn, k, packn]
 * sb - input:   [n/8, k, 8]
 **************************************************************/
// XXX: unsupported fuse relu
void shl_rvv_ncxhwx_gemm_8xpack2n_fp32(float *dst, const float *sa, const float *sb, float *bias,
                                       int m, int k, int n, bool fuse_relu)
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                _acc04 = vfmacc_vf_f32m1(_acc04, img0[4], _kernel0, vl);
                _acc05 = vfmacc_vf_f32m1(_acc05, img0[5], _kernel0, vl);
                _acc06 = vfmacc_vf_f32m1(_acc06, img0[6], _kernel0, vl);
                _acc07 = vfmacc_vf_f32m1(_acc07, img0[7], _kernel0, vl);

                _acc10 = vfmacc_vf_f32m1(_acc10, img0[0], _kernel1, vl);
                _acc11 = vfmacc_vf_f32m1(_acc11, img0[1], _kernel1, vl);
                _acc12 = vfmacc_vf_f32m1(_acc12, img0[2], _kernel1, vl);
                _acc13 = vfmacc_vf_f32m1(_acc13, img0[3], _kernel1, vl);
                _acc14 = vfmacc_vf_f32m1(_acc14, img0[4], _kernel1, vl);
                _acc15 = vfmacc_vf_f32m1(_acc15, img0[5], _kernel1, vl);
                _acc16 = vfmacc_vf_f32m1(_acc16, img0[6], _kernel1, vl);
                _acc17 = vfmacc_vf_f32m1(_acc17, img0[7], _kernel1, vl);
                img0 += 8;
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);

                _acc10 = vfmacc_vf_f32m1(_acc10, img0[0], _kernel1, vl);
                _acc11 = vfmacc_vf_f32m1(_acc11, img0[1], _kernel1, vl);
                _acc12 = vfmacc_vf_f32m1(_acc12, img0[2], _kernel1, vl);
                _acc13 = vfmacc_vf_f32m1(_acc13, img0[3], _kernel1, vl);
                img0 += 4;
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);

                _acc10 = vfmacc_vf_f32m1(_acc10, img0[0], _kernel1, vl);
                _acc11 = vfmacc_vf_f32m1(_acc11, img0[1], _kernel1, vl);
                img0 += 2;
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc10 = vfmacc_vf_f32m1(_acc10, img0[0], _kernel1, vl);
                img0 += 1;
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                _acc04 = vfmacc_vf_f32m1(_acc04, img0[4], _kernel0, vl);
                _acc05 = vfmacc_vf_f32m1(_acc05, img0[5], _kernel0, vl);
                _acc06 = vfmacc_vf_f32m1(_acc06, img0[6], _kernel0, vl);
                _acc07 = vfmacc_vf_f32m1(_acc07, img0[7], _kernel0, vl);
                img0 += 8;
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                img0 += 4;
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                img0 += 2;
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                img0 += 1;
            }
            vse32_v_f32m1(output0, _acc00, vl);
            output0 += packn * 1;
        }
    }

    /* tail output_channel */
    if (oc < m) {
        vl = vsetvl_e32m1(m - oc);
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
                k0 += vl;
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                _acc04 = vfmacc_vf_f32m1(_acc04, img0[4], _kernel0, vl);
                _acc05 = vfmacc_vf_f32m1(_acc05, img0[5], _kernel0, vl);
                _acc06 = vfmacc_vf_f32m1(_acc06, img0[6], _kernel0, vl);
                _acc07 = vfmacc_vf_f32m1(_acc07, img0[7], _kernel0, vl);
                img0 += 8;
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + vl * 1, _acc01, vl);
            vse32_v_f32m1(output0 + vl * 2, _acc02, vl);
            vse32_v_f32m1(output0 + vl * 3, _acc03, vl);
            vse32_v_f32m1(output0 + vl * 4, _acc04, vl);
            vse32_v_f32m1(output0 + vl * 5, _acc05, vl);
            vse32_v_f32m1(output0 + vl * 6, _acc06, vl);
            vse32_v_f32m1(output0 + vl * 7, _acc07, vl);
            output0 += vl * 8;
        }
        for (; t + 3 < n; t += 4) {
            const float *k0 = kernel_data + oc * k;  // 4 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);
            vfloat32m1_t _acc01 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc02 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc03 = vmv_v_v_f32m1(_acc00, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                k0 += vl;
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                img0 += 4;
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + vl * 1, _acc01, vl);
            vse32_v_f32m1(output0 + vl * 2, _acc02, vl);
            vse32_v_f32m1(output0 + vl * 3, _acc03, vl);
            output0 += vl * 4;
        }
        for (; t + 1 < n; t += 2) {
            const float *k0 = kernel_data + oc * k;  // 4 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);
            vfloat32m1_t _acc01 = vmv_v_v_f32m1(_acc00, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                k0 += vl;
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                img0 += 2;
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + vl * 1, _acc01, vl);
            output0 += vl * 2;
        }
        for (; t < n; t++) {
            const float *k0 = kernel_data + oc * k;  // 4 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                k0 += vl;
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                img0 += 1;
            }
            vse32_v_f32m1(output0, _acc00, vl);
            output0 += vl * 1;
        }
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}

/**************************************************************
 * dst - output: [m/packn, n, packn]
 * sa - kernel:  [m/pack2n, k, pack2n]  [m/packn, k, packn]
 * sb - input:   [n/12, k, 12]
 **************************************************************/
// XXX: unsupported fuse relu
void shl_rvv_ncxhwx_gemm_12xpack2n_fp32(float *dst, const float *sa, const float *sb, float *bias,
                                        int m, int k, int n, bool fuse_relu)
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
        for (; t + 11 < n; t += 12) {
            const float *k0 = kernel_data + oc * k;  // 8 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);
            vfloat32m1_t _acc01 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc02 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc03 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc04 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc05 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc06 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc07 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc08 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc09 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc0a = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc0b = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc10 = vle32_v_f32m1(b0 + packn, vl);
            vfloat32m1_t _acc11 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc12 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc13 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc14 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc15 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc16 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc17 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc18 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc19 = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc1a = vmv_v_v_f32m1(_acc10, vl);
            vfloat32m1_t _acc1b = vmv_v_v_f32m1(_acc10, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                vfloat32m1_t _kernel1 = vle32_v_f32m1(k0 + packn, vl);
                k0 += pack2n;
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                _acc04 = vfmacc_vf_f32m1(_acc04, img0[4], _kernel0, vl);
                _acc05 = vfmacc_vf_f32m1(_acc05, img0[5], _kernel0, vl);
                _acc06 = vfmacc_vf_f32m1(_acc06, img0[6], _kernel0, vl);
                _acc07 = vfmacc_vf_f32m1(_acc07, img0[7], _kernel0, vl);
                _acc08 = vfmacc_vf_f32m1(_acc08, img0[8], _kernel0, vl);
                _acc09 = vfmacc_vf_f32m1(_acc09, img0[9], _kernel0, vl);
                _acc0a = vfmacc_vf_f32m1(_acc0a, img0[10], _kernel0, vl);
                _acc0b = vfmacc_vf_f32m1(_acc0b, img0[11], _kernel0, vl);

                _acc10 = vfmacc_vf_f32m1(_acc10, img0[0], _kernel1, vl);
                _acc11 = vfmacc_vf_f32m1(_acc11, img0[1], _kernel1, vl);
                _acc12 = vfmacc_vf_f32m1(_acc12, img0[2], _kernel1, vl);
                _acc13 = vfmacc_vf_f32m1(_acc13, img0[3], _kernel1, vl);
                _acc14 = vfmacc_vf_f32m1(_acc14, img0[4], _kernel1, vl);
                _acc15 = vfmacc_vf_f32m1(_acc15, img0[5], _kernel1, vl);
                _acc16 = vfmacc_vf_f32m1(_acc16, img0[6], _kernel1, vl);
                _acc17 = vfmacc_vf_f32m1(_acc17, img0[7], _kernel1, vl);
                _acc18 = vfmacc_vf_f32m1(_acc18, img0[8], _kernel1, vl);
                _acc19 = vfmacc_vf_f32m1(_acc19, img0[9], _kernel1, vl);
                _acc1a = vfmacc_vf_f32m1(_acc1a, img0[10], _kernel1, vl);
                _acc1b = vfmacc_vf_f32m1(_acc1b, img0[11], _kernel1, vl);
                img0 += 12;
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + packn * 1, _acc01, vl);
            vse32_v_f32m1(output0 + packn * 2, _acc02, vl);
            vse32_v_f32m1(output0 + packn * 3, _acc03, vl);
            vse32_v_f32m1(output0 + packn * 4, _acc04, vl);
            vse32_v_f32m1(output0 + packn * 5, _acc05, vl);
            vse32_v_f32m1(output0 + packn * 6, _acc06, vl);
            vse32_v_f32m1(output0 + packn * 7, _acc07, vl);
            vse32_v_f32m1(output0 + packn * 8, _acc08, vl);
            vse32_v_f32m1(output0 + packn * 9, _acc09, vl);
            vse32_v_f32m1(output0 + packn * 10, _acc0a, vl);
            vse32_v_f32m1(output0 + packn * 11, _acc0b, vl);
            output0 += packn * 12;

            vse32_v_f32m1(output1, _acc10, vl);
            vse32_v_f32m1(output1 + packn * 1, _acc11, vl);
            vse32_v_f32m1(output1 + packn * 2, _acc12, vl);
            vse32_v_f32m1(output1 + packn * 3, _acc13, vl);
            vse32_v_f32m1(output1 + packn * 4, _acc14, vl);
            vse32_v_f32m1(output1 + packn * 5, _acc15, vl);
            vse32_v_f32m1(output1 + packn * 6, _acc16, vl);
            vse32_v_f32m1(output1 + packn * 7, _acc17, vl);
            vse32_v_f32m1(output1 + packn * 8, _acc18, vl);
            vse32_v_f32m1(output1 + packn * 9, _acc19, vl);
            vse32_v_f32m1(output1 + packn * 10, _acc1a, vl);
            vse32_v_f32m1(output1 + packn * 11, _acc1b, vl);
            output1 += packn * 12;
        }
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                _acc04 = vfmacc_vf_f32m1(_acc04, img0[4], _kernel0, vl);
                _acc05 = vfmacc_vf_f32m1(_acc05, img0[5], _kernel0, vl);
                _acc06 = vfmacc_vf_f32m1(_acc06, img0[6], _kernel0, vl);
                _acc07 = vfmacc_vf_f32m1(_acc07, img0[7], _kernel0, vl);

                _acc10 = vfmacc_vf_f32m1(_acc10, img0[0], _kernel1, vl);
                _acc11 = vfmacc_vf_f32m1(_acc11, img0[1], _kernel1, vl);
                _acc12 = vfmacc_vf_f32m1(_acc12, img0[2], _kernel1, vl);
                _acc13 = vfmacc_vf_f32m1(_acc13, img0[3], _kernel1, vl);
                _acc14 = vfmacc_vf_f32m1(_acc14, img0[4], _kernel1, vl);
                _acc15 = vfmacc_vf_f32m1(_acc15, img0[5], _kernel1, vl);
                _acc16 = vfmacc_vf_f32m1(_acc16, img0[6], _kernel1, vl);
                _acc17 = vfmacc_vf_f32m1(_acc17, img0[7], _kernel1, vl);
                img0 += 8;
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);

                _acc10 = vfmacc_vf_f32m1(_acc10, img0[0], _kernel1, vl);
                _acc11 = vfmacc_vf_f32m1(_acc11, img0[1], _kernel1, vl);
                _acc12 = vfmacc_vf_f32m1(_acc12, img0[2], _kernel1, vl);
                _acc13 = vfmacc_vf_f32m1(_acc13, img0[3], _kernel1, vl);
                img0 += 4;
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);

                _acc10 = vfmacc_vf_f32m1(_acc10, img0[0], _kernel1, vl);
                _acc11 = vfmacc_vf_f32m1(_acc11, img0[1], _kernel1, vl);
                img0 += 2;
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc10 = vfmacc_vf_f32m1(_acc10, img0[0], _kernel1, vl);
                img0 += 1;
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
        for (; t + 11 < n; t += 12) {
            const float *k0 = kernel_data + oc * k;  // 4 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);
            vfloat32m1_t _acc01 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc02 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc03 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc04 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc05 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc06 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc07 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc08 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc09 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc0a = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc0b = vmv_v_v_f32m1(_acc00, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                k0 += packn;
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                _acc04 = vfmacc_vf_f32m1(_acc04, img0[4], _kernel0, vl);
                _acc05 = vfmacc_vf_f32m1(_acc05, img0[5], _kernel0, vl);
                _acc06 = vfmacc_vf_f32m1(_acc06, img0[6], _kernel0, vl);
                _acc07 = vfmacc_vf_f32m1(_acc07, img0[7], _kernel0, vl);
                _acc08 = vfmacc_vf_f32m1(_acc08, img0[8], _kernel0, vl);
                _acc09 = vfmacc_vf_f32m1(_acc09, img0[9], _kernel0, vl);
                _acc0a = vfmacc_vf_f32m1(_acc0a, img0[10], _kernel0, vl);
                _acc0b = vfmacc_vf_f32m1(_acc0b, img0[11], _kernel0, vl);

                img0 += 12;
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + packn * 1, _acc01, vl);
            vse32_v_f32m1(output0 + packn * 2, _acc02, vl);
            vse32_v_f32m1(output0 + packn * 3, _acc03, vl);
            vse32_v_f32m1(output0 + packn * 4, _acc04, vl);
            vse32_v_f32m1(output0 + packn * 5, _acc05, vl);
            vse32_v_f32m1(output0 + packn * 6, _acc06, vl);
            vse32_v_f32m1(output0 + packn * 7, _acc07, vl);
            vse32_v_f32m1(output0 + packn * 8, _acc08, vl);
            vse32_v_f32m1(output0 + packn * 9, _acc09, vl);
            vse32_v_f32m1(output0 + packn * 10, _acc0a, vl);
            vse32_v_f32m1(output0 + packn * 11, _acc0b, vl);
            output0 += packn * 12;
        }
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                _acc04 = vfmacc_vf_f32m1(_acc04, img0[4], _kernel0, vl);
                _acc05 = vfmacc_vf_f32m1(_acc05, img0[5], _kernel0, vl);
                _acc06 = vfmacc_vf_f32m1(_acc06, img0[6], _kernel0, vl);
                _acc07 = vfmacc_vf_f32m1(_acc07, img0[7], _kernel0, vl);
                img0 += 8;
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                img0 += 4;
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                img0 += 2;
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
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                img0 += 1;
            }
            vse32_v_f32m1(output0, _acc00, vl);
            output0 += packn * 1;
        }
    }

    /* tail output_channel */
    if (oc < m) {
        vl = vsetvl_e32m1(m - oc);
        float *output0 = output_data + oc * n;  // tial channel dot output
        const float *img0 = input_data;
        const float *b0 = bias_ptr + oc;
        int t = 0;
        for (; t + 11 < n; t += 12) {
            const float *k0 = kernel_data + oc * k;  // tail channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);
            vfloat32m1_t _acc01 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc02 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc03 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc04 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc05 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc06 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc07 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc08 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc09 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc0a = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc0b = vmv_v_v_f32m1(_acc00, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                k0 += vl;
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                _acc04 = vfmacc_vf_f32m1(_acc04, img0[4], _kernel0, vl);
                _acc05 = vfmacc_vf_f32m1(_acc05, img0[5], _kernel0, vl);
                _acc06 = vfmacc_vf_f32m1(_acc06, img0[6], _kernel0, vl);
                _acc07 = vfmacc_vf_f32m1(_acc07, img0[7], _kernel0, vl);
                _acc08 = vfmacc_vf_f32m1(_acc08, img0[8], _kernel0, vl);
                _acc09 = vfmacc_vf_f32m1(_acc09, img0[9], _kernel0, vl);
                _acc0a = vfmacc_vf_f32m1(_acc0a, img0[10], _kernel0, vl);
                _acc0b = vfmacc_vf_f32m1(_acc0b, img0[11], _kernel0, vl);

                img0 += 12;
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + vl * 1, _acc01, vl);
            vse32_v_f32m1(output0 + vl * 2, _acc02, vl);
            vse32_v_f32m1(output0 + vl * 3, _acc03, vl);
            vse32_v_f32m1(output0 + vl * 4, _acc04, vl);
            vse32_v_f32m1(output0 + vl * 5, _acc05, vl);
            vse32_v_f32m1(output0 + vl * 6, _acc06, vl);
            vse32_v_f32m1(output0 + vl * 7, _acc07, vl);
            vse32_v_f32m1(output0 + vl * 8, _acc08, vl);
            vse32_v_f32m1(output0 + vl * 9, _acc09, vl);
            vse32_v_f32m1(output0 + vl * 10, _acc0a, vl);
            vse32_v_f32m1(output0 + vl * 11, _acc0b, vl);
            output0 += vl * 12;
        }
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
                k0 += vl;
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                _acc04 = vfmacc_vf_f32m1(_acc04, img0[4], _kernel0, vl);
                _acc05 = vfmacc_vf_f32m1(_acc05, img0[5], _kernel0, vl);
                _acc06 = vfmacc_vf_f32m1(_acc06, img0[6], _kernel0, vl);
                _acc07 = vfmacc_vf_f32m1(_acc07, img0[7], _kernel0, vl);
                img0 += 8;
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + vl * 1, _acc01, vl);
            vse32_v_f32m1(output0 + vl * 2, _acc02, vl);
            vse32_v_f32m1(output0 + vl * 3, _acc03, vl);
            vse32_v_f32m1(output0 + vl * 4, _acc04, vl);
            vse32_v_f32m1(output0 + vl * 5, _acc05, vl);
            vse32_v_f32m1(output0 + vl * 6, _acc06, vl);
            vse32_v_f32m1(output0 + vl * 7, _acc07, vl);
            output0 += vl * 8;
        }
        for (; t + 3 < n; t += 4) {
            const float *k0 = kernel_data + oc * k;  // 4 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);
            vfloat32m1_t _acc01 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc02 = vmv_v_v_f32m1(_acc00, vl);
            vfloat32m1_t _acc03 = vmv_v_v_f32m1(_acc00, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                k0 += vl;
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                _acc02 = vfmacc_vf_f32m1(_acc02, img0[2], _kernel0, vl);
                _acc03 = vfmacc_vf_f32m1(_acc03, img0[3], _kernel0, vl);
                img0 += 4;
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + vl * 1, _acc01, vl);
            vse32_v_f32m1(output0 + vl * 2, _acc02, vl);
            vse32_v_f32m1(output0 + vl * 3, _acc03, vl);
            output0 += vl * 4;
        }
        for (; t + 1 < n; t += 2) {
            const float *k0 = kernel_data + oc * k;  // 4 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);
            vfloat32m1_t _acc01 = vmv_v_v_f32m1(_acc00, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                k0 += vl;
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                _acc01 = vfmacc_vf_f32m1(_acc01, img0[1], _kernel0, vl);
                img0 += 2;
            }
            vse32_v_f32m1(output0, _acc00, vl);
            vse32_v_f32m1(output0 + vl * 1, _acc01, vl);
            output0 += vl * 2;
        }
        for (; t < n; t++) {
            const float *k0 = kernel_data + oc * k;  // 4 channel kernel
            vfloat32m1_t _acc00 = vle32_v_f32m1(b0, vl);

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel0 = vle32_v_f32m1(k0, vl);
                k0 += vl;
                _acc00 = vfmacc_vf_f32m1(_acc00, img0[0], _kernel0, vl);
                img0 += 1;
            }
            vse32_v_f32m1(output0, _acc00, vl);
            output0 += vl * 1;
        }
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
