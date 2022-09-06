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

/* CSI-NN2 version 2.0.x */

#include "shl_thead_rvv.h"

/*************************************************************
    note: VLEN = 128/256 ... flexible vlen
*************************************************************/
int shl_rvv_dwconv3x3s1_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];  // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int vl = vsetvl_e16m1(packn);

    __fp16 *input_padd_buf =
        (__fp16 *)shl_mem_alloc(in_c * (in_h + params->pad_top + params->pad_down) *
                                (in_w + params->pad_left + params->pad_right) * sizeof(float));

    shl_rvv_pad_input_packn_fp16(
        input_data, input_padd_buf, in_c, in_h, in_w, in_h + params->pad_top + params->pad_down,
        in_w + params->pad_left + params->pad_right, params->pad_top, params->pad_left);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

#pragma omp parallel for num_threads(1)
    for (int c = 0; c + packn - 1 < in_c; c += packn) {
        __fp16 *out0 = output_data + c * out_h * out_w;
        __fp16 *out1 = out0 + out_w * packn;

        const __fp16 *r0 = input_padd_buf + c * in_h * in_w;
        const __fp16 *r1 = r0 + in_w * packn;
        const __fp16 *r2 = r1 + in_w * packn;
        const __fp16 *r3 = r2 + in_w * packn;

        const __fp16 *kernel0 = kernel_data + c * 9;

        vfloat16m1_t _k00 = vle16_v_f16m1(kernel0, vl);
        vfloat16m1_t _k01 = vle16_v_f16m1(kernel0 + 1 * packn, vl);
        vfloat16m1_t _k02 = vle16_v_f16m1(kernel0 + 2 * packn, vl);
        vfloat16m1_t _k10 = vle16_v_f16m1(kernel0 + 3 * packn, vl);
        vfloat16m1_t _k11 = vle16_v_f16m1(kernel0 + 4 * packn, vl);
        vfloat16m1_t _k12 = vle16_v_f16m1(kernel0 + 5 * packn, vl);
        vfloat16m1_t _k20 = vle16_v_f16m1(kernel0 + 6 * packn, vl);
        vfloat16m1_t _k21 = vle16_v_f16m1(kernel0 + 7 * packn, vl);
        vfloat16m1_t _k22 = vle16_v_f16m1(kernel0 + 8 * packn, vl);

        vfloat16m1_t _bias0;
        _bias0 = bias_data ? vle16_v_f16m1(bias_data + c, vl) : vfmv_v_f_f16m1(0.0f, vl);

        int h = 0;
        // h2 loop
        for (; h + 1 < out_h; h += 2) {
            int w = 0;
            // h2w4 loop
            for (; w + 3 < out_w; w += 4) {
                vfloat16m1_t _acc00 = _bias0;
                vfloat16m1_t _acc01 = _bias0;
                vfloat16m1_t _acc02 = _bias0;
                vfloat16m1_t _acc03 = _bias0;
                vfloat16m1_t _acc10 = _bias0;
                vfloat16m1_t _acc11 = _bias0;
                vfloat16m1_t _acc12 = _bias0;
                vfloat16m1_t _acc13 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + 1 * packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + 2 * packn, vl);
                vfloat16m1_t _r03 = vle16_v_f16m1(r0 + 3 * packn, vl);
                vfloat16m1_t _r04 = vle16_v_f16m1(r0 + 4 * packn, vl);
                vfloat16m1_t _r05 = vle16_v_f16m1(r0 + 5 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k00, _r00, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k01, _r01, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k02, _r02, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k00, _r01, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k01, _r02, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k02, _r03, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k00, _r02, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k01, _r03, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k02, _r04, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k00, _r03, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k01, _r04, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k02, _r05, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + 1 * packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + 2 * packn, vl);
                vfloat16m1_t _r13 = vle16_v_f16m1(r1 + 3 * packn, vl);
                vfloat16m1_t _r14 = vle16_v_f16m1(r1 + 4 * packn, vl);
                vfloat16m1_t _r15 = vle16_v_f16m1(r1 + 5 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k10, _r10, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k11, _r11, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k12, _r12, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k10, _r11, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k11, _r12, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k12, _r13, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k10, _r12, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k11, _r13, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k12, _r14, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k10, _r13, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k11, _r14, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k12, _r15, vl);  //
                _acc10 = vfmacc_vv_f16m1(_acc10, _k00, _r10, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k01, _r11, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k02, _r12, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k00, _r11, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k01, _r12, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k02, _r13, vl);
                _acc12 = vfmacc_vv_f16m1(_acc12, _k00, _r12, vl);
                _acc12 = vfmacc_vv_f16m1(_acc12, _k01, _r13, vl);
                _acc12 = vfmacc_vv_f16m1(_acc12, _k02, _r14, vl);
                _acc13 = vfmacc_vv_f16m1(_acc13, _k00, _r13, vl);
                _acc13 = vfmacc_vv_f16m1(_acc13, _k01, _r14, vl);
                _acc13 = vfmacc_vv_f16m1(_acc13, _k02, _r15, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + 1 * packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + 2 * packn, vl);
                vfloat16m1_t _r23 = vle16_v_f16m1(r2 + 3 * packn, vl);
                vfloat16m1_t _r24 = vle16_v_f16m1(r2 + 4 * packn, vl);
                vfloat16m1_t _r25 = vle16_v_f16m1(r2 + 5 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k20, _r20, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k21, _r21, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k22, _r22, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k20, _r21, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k21, _r22, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k22, _r23, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k20, _r22, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k21, _r23, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k22, _r24, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k20, _r23, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k21, _r24, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k22, _r25, vl);  //
                _acc10 = vfmacc_vv_f16m1(_acc10, _k10, _r20, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k11, _r21, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k12, _r22, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k10, _r21, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k11, _r22, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k12, _r23, vl);
                _acc12 = vfmacc_vv_f16m1(_acc12, _k10, _r22, vl);
                _acc12 = vfmacc_vv_f16m1(_acc12, _k11, _r23, vl);
                _acc12 = vfmacc_vv_f16m1(_acc12, _k12, _r24, vl);
                _acc13 = vfmacc_vv_f16m1(_acc13, _k10, _r23, vl);
                _acc13 = vfmacc_vv_f16m1(_acc13, _k11, _r24, vl);
                _acc13 = vfmacc_vv_f16m1(_acc13, _k12, _r25, vl);

                vfloat16m1_t _r30 = vle16_v_f16m1(r3, vl);
                vfloat16m1_t _r31 = vle16_v_f16m1(r3 + 1 * packn, vl);
                vfloat16m1_t _r32 = vle16_v_f16m1(r3 + 2 * packn, vl);
                vfloat16m1_t _r33 = vle16_v_f16m1(r3 + 3 * packn, vl);
                vfloat16m1_t _r34 = vle16_v_f16m1(r3 + 4 * packn, vl);
                vfloat16m1_t _r35 = vle16_v_f16m1(r3 + 5 * packn, vl);

                _acc10 = vfmacc_vv_f16m1(_acc10, _k20, _r30, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k21, _r31, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k22, _r32, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k20, _r31, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k21, _r32, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k22, _r33, vl);
                _acc12 = vfmacc_vv_f16m1(_acc12, _k20, _r32, vl);
                _acc12 = vfmacc_vv_f16m1(_acc12, _k21, _r33, vl);
                _acc12 = vfmacc_vv_f16m1(_acc12, _k22, _r34, vl);
                _acc13 = vfmacc_vv_f16m1(_acc13, _k20, _r33, vl);
                _acc13 = vfmacc_vv_f16m1(_acc13, _k21, _r34, vl);
                _acc13 = vfmacc_vv_f16m1(_acc13, _k22, _r35, vl);

                vse16_v_f16m1(out0, _acc00, vl);
                vse16_v_f16m1(out0 + 1 * packn, _acc01, vl);
                vse16_v_f16m1(out0 + 2 * packn, _acc02, vl);
                vse16_v_f16m1(out0 + 3 * packn, _acc03, vl);
                vse16_v_f16m1(out1, _acc10, vl);
                vse16_v_f16m1(out1 + 1 * packn, _acc11, vl);
                vse16_v_f16m1(out1 + 2 * packn, _acc12, vl);
                vse16_v_f16m1(out1 + 3 * packn, _acc13, vl);

                out0 += packn * 4;
                out1 += packn * 4;

                r0 += packn * 4;
                r1 += packn * 4;
                r2 += packn * 4;
                r3 += packn * 4;
            }
            // h2w2
            for (; w + 1 < out_w; w += 2) {
                vfloat16m1_t _acc00 = _bias0;
                vfloat16m1_t _acc01 = _bias0;
                vfloat16m1_t _acc10 = _bias0;
                vfloat16m1_t _acc11 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + 1 * packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + 2 * packn, vl);
                vfloat16m1_t _r03 = vle16_v_f16m1(r0 + 3 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k00, _r00, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k01, _r01, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k02, _r02, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k00, _r01, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k01, _r02, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k02, _r03, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + 1 * packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + 2 * packn, vl);
                vfloat16m1_t _r13 = vle16_v_f16m1(r1 + 3 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k10, _r10, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k11, _r11, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k12, _r12, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k10, _r11, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k11, _r12, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k12, _r13, vl);  // 0
                _acc10 = vfmacc_vv_f16m1(_acc10, _k00, _r10, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k01, _r11, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k02, _r12, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k00, _r11, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k01, _r12, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k02, _r13, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + 1 * packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + 2 * packn, vl);
                vfloat16m1_t _r23 = vle16_v_f16m1(r2 + 3 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k20, _r20, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k21, _r21, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k22, _r22, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k20, _r21, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k21, _r22, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k22, _r23, vl);  //
                _acc10 = vfmacc_vv_f16m1(_acc10, _k10, _r20, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k11, _r21, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k12, _r22, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k10, _r21, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k11, _r22, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k12, _r23, vl);

                vfloat16m1_t _r30 = vle16_v_f16m1(r3, vl);
                vfloat16m1_t _r31 = vle16_v_f16m1(r3 + 1 * packn, vl);
                vfloat16m1_t _r32 = vle16_v_f16m1(r3 + 2 * packn, vl);
                vfloat16m1_t _r33 = vle16_v_f16m1(r3 + 3 * packn, vl);

                _acc10 = vfmacc_vv_f16m1(_acc10, _k20, _r30, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k21, _r31, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k22, _r32, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k20, _r31, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k21, _r32, vl);
                _acc11 = vfmacc_vv_f16m1(_acc11, _k22, _r33, vl);

                vse16_v_f16m1(out0, _acc00, vl);
                vse16_v_f16m1(out0 + 1 * packn, _acc01, vl);
                vse16_v_f16m1(out1, _acc10, vl);
                vse16_v_f16m1(out1 + 1 * packn, _acc11, vl);

                out0 += packn * 2;
                out1 += packn * 2;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
                r3 += packn * 2;
            }
            // h2w1
            for (; w < out_w; w++) {
                vfloat16m1_t _acc00 = _bias0;
                vfloat16m1_t _acc10 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + 1 * packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + 2 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k00, _r00, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k01, _r01, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k02, _r02, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + 1 * packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + 2 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k10, _r10, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k11, _r11, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k12, _r12, vl);  // 0
                _acc10 = vfmacc_vv_f16m1(_acc10, _k00, _r10, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k01, _r11, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k02, _r12, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + 1 * packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + 2 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k20, _r20, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k21, _r21, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k22, _r22, vl);  //
                _acc10 = vfmacc_vv_f16m1(_acc10, _k10, _r20, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k11, _r21, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k12, _r22, vl);

                vfloat16m1_t _r30 = vle16_v_f16m1(r3, vl);
                vfloat16m1_t _r31 = vle16_v_f16m1(r3 + 1 * packn, vl);
                vfloat16m1_t _r32 = vle16_v_f16m1(r3 + 2 * packn, vl);

                _acc10 = vfmacc_vv_f16m1(_acc10, _k20, _r30, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k21, _r31, vl);
                _acc10 = vfmacc_vv_f16m1(_acc10, _k22, _r32, vl);

                vse16_v_f16m1(out0, _acc00, vl);
                vse16_v_f16m1(out1, _acc10, vl);

                out0 += packn * 1;
                out1 += packn * 1;

                r0 += packn * 1;
                r1 += packn * 1;
                r2 += packn * 1;
                r3 += packn * 1;
            }
            r0 += (2 + in_w) * packn;
            r1 += (2 + in_w) * packn;
            r2 += (2 + in_w) * packn;
            r3 += (2 + in_w) * packn;

            out0 += out_w * packn;
            out1 += out_w * packn;
        }

        // h1
        for (; h < out_h; h++) {
            int w = 0;
            // h1w4 loop
            for (; w + 3 < out_w; w += 4) {
                vfloat16m1_t _acc00 = _bias0;
                vfloat16m1_t _acc01 = _bias0;
                vfloat16m1_t _acc02 = _bias0;
                vfloat16m1_t _acc03 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + 1 * packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + 2 * packn, vl);
                vfloat16m1_t _r03 = vle16_v_f16m1(r0 + 3 * packn, vl);
                vfloat16m1_t _r04 = vle16_v_f16m1(r0 + 4 * packn, vl);
                vfloat16m1_t _r05 = vle16_v_f16m1(r0 + 5 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k00, _r00, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k01, _r01, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k02, _r02, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k00, _r01, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k01, _r02, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k02, _r03, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k00, _r02, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k01, _r03, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k02, _r04, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k00, _r03, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k01, _r04, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k02, _r05, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + 1 * packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + 2 * packn, vl);
                vfloat16m1_t _r13 = vle16_v_f16m1(r1 + 3 * packn, vl);
                vfloat16m1_t _r14 = vle16_v_f16m1(r1 + 4 * packn, vl);
                vfloat16m1_t _r15 = vle16_v_f16m1(r1 + 5 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k10, _r10, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k11, _r11, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k12, _r12, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k10, _r11, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k11, _r12, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k12, _r13, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k10, _r12, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k11, _r13, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k12, _r14, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k10, _r13, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k11, _r14, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k12, _r15, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + 1 * packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + 2 * packn, vl);
                vfloat16m1_t _r23 = vle16_v_f16m1(r2 + 3 * packn, vl);
                vfloat16m1_t _r24 = vle16_v_f16m1(r2 + 4 * packn, vl);
                vfloat16m1_t _r25 = vle16_v_f16m1(r2 + 5 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k20, _r20, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k21, _r21, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k22, _r22, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k20, _r21, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k21, _r22, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k22, _r23, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k20, _r22, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k21, _r23, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k22, _r24, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k20, _r23, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k21, _r24, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k22, _r25, vl);

                vse16_v_f16m1(out0, _acc00, vl);
                vse16_v_f16m1(out0 + 1 * packn, _acc01, vl);
                vse16_v_f16m1(out0 + 2 * packn, _acc02, vl);
                vse16_v_f16m1(out0 + 3 * packn, _acc03, vl);

                out0 += packn * 4;

                r0 += packn * 4;
                r1 += packn * 4;
                r2 += packn * 4;
            }
            // h1w2
            for (; w + 1 < out_w; w += 2) {
                vfloat16m1_t _acc00 = _bias0;
                vfloat16m1_t _acc01 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + 1 * packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + 2 * packn, vl);
                vfloat16m1_t _r03 = vle16_v_f16m1(r0 + 3 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k00, _r00, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k01, _r01, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k02, _r02, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k00, _r01, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k01, _r02, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k02, _r03, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + 1 * packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + 2 * packn, vl);
                vfloat16m1_t _r13 = vle16_v_f16m1(r1 + 3 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k10, _r10, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k11, _r11, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k12, _r12, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k10, _r11, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k11, _r12, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k12, _r13, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + 1 * packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + 2 * packn, vl);
                vfloat16m1_t _r23 = vle16_v_f16m1(r2 + 3 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k20, _r20, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k21, _r21, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k22, _r22, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k20, _r21, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k21, _r22, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k22, _r23, vl);

                vse16_v_f16m1(out0, _acc00, vl);
                vse16_v_f16m1(out0 + 1 * packn, _acc01, vl);

                out0 += packn * 2;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
            }
            // h1w1
            for (; w < out_w; w++) {
                vfloat16m1_t _acc00 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + 1 * packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + 2 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k00, _r00, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k01, _r01, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k02, _r02, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + 1 * packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + 2 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k10, _r10, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k11, _r11, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k12, _r12, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + 1 * packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + 2 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k20, _r20, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k21, _r21, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k22, _r22, vl);

                vse16_v_f16m1(out0, _acc00, vl);

                out0 += packn * 1;

                r0 += packn * 1;
                r1 += packn * 1;
                r2 += packn * 1;
            }
        }
    }
    shl_mem_free(input_padd_buf);
    return CSINN_TRUE;
}

int shl_rvv_dwconv3x3s2_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];  // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int vl = vsetvl_e16m1(packn);

    __fp16 *input_padd_buf =
        (__fp16 *)shl_mem_alloc(in_c * (in_h + params->pad_top + params->pad_down) *
                                (in_w + params->pad_left + params->pad_right) * sizeof(float));

    shl_rvv_pad_input_packn_fp16(
        input_data, input_padd_buf, in_c, in_h, in_w, in_h + params->pad_top + params->pad_down,
        in_w + params->pad_left + params->pad_right, params->pad_top, params->pad_left);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

    int tailstep = (in_w - 2 * out_w + in_w) * packn;

#pragma omp parallel for num_threads(1)
    for (int c = 0; c + packn - 1 < in_c; c += packn) {
        __fp16 *out0 = output_data + c * out_h * out_w;

        const __fp16 *r0 = input_padd_buf + c * in_h * in_w;
        const __fp16 *r1 = r0 + in_w * packn;
        const __fp16 *r2 = r1 + in_w * packn;

        const __fp16 *kernel0 = kernel_data + c * 9;

        vfloat16m1_t _k00 = vle16_v_f16m1(kernel0, vl);
        vfloat16m1_t _k01 = vle16_v_f16m1(kernel0 + 1 * packn, vl);
        vfloat16m1_t _k02 = vle16_v_f16m1(kernel0 + 2 * packn, vl);
        vfloat16m1_t _k10 = vle16_v_f16m1(kernel0 + 3 * packn, vl);
        vfloat16m1_t _k11 = vle16_v_f16m1(kernel0 + 4 * packn, vl);
        vfloat16m1_t _k12 = vle16_v_f16m1(kernel0 + 5 * packn, vl);
        vfloat16m1_t _k20 = vle16_v_f16m1(kernel0 + 6 * packn, vl);
        vfloat16m1_t _k21 = vle16_v_f16m1(kernel0 + 7 * packn, vl);
        vfloat16m1_t _k22 = vle16_v_f16m1(kernel0 + 8 * packn, vl);

        vfloat16m1_t _bias0;
        _bias0 = bias_data ? vle16_v_f16m1(bias_data + c, vl) : vfmv_v_f_f16m1(0.0f, vl);

        for (int h = 0; h < out_h; h++) {
            int w = 0;
            // h1w4 loop
            for (; w + 3 < out_w; w += 4) {
                vfloat16m1_t _acc00 = _bias0;
                vfloat16m1_t _acc01 = _bias0;
                vfloat16m1_t _acc02 = _bias0;
                vfloat16m1_t _acc03 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + 1 * packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + 2 * packn, vl);
                vfloat16m1_t _r03 = vle16_v_f16m1(r0 + 3 * packn, vl);
                vfloat16m1_t _r04 = vle16_v_f16m1(r0 + 4 * packn, vl);
                vfloat16m1_t _r05 = vle16_v_f16m1(r0 + 5 * packn, vl);
                vfloat16m1_t _r06 = vle16_v_f16m1(r0 + 6 * packn, vl);
                vfloat16m1_t _r07 = vle16_v_f16m1(r0 + 7 * packn, vl);
                vfloat16m1_t _r08 = vle16_v_f16m1(r0 + 8 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k00, _r00, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k01, _r01, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k02, _r02, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k00, _r02, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k01, _r03, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k02, _r04, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k00, _r04, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k01, _r05, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k02, _r06, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k00, _r06, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k01, _r07, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k02, _r08, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + 1 * packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + 2 * packn, vl);
                vfloat16m1_t _r13 = vle16_v_f16m1(r1 + 3 * packn, vl);
                vfloat16m1_t _r14 = vle16_v_f16m1(r1 + 4 * packn, vl);
                vfloat16m1_t _r15 = vle16_v_f16m1(r1 + 5 * packn, vl);
                vfloat16m1_t _r16 = vle16_v_f16m1(r1 + 6 * packn, vl);
                vfloat16m1_t _r17 = vle16_v_f16m1(r1 + 7 * packn, vl);
                vfloat16m1_t _r18 = vle16_v_f16m1(r1 + 8 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k10, _r10, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k11, _r11, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k12, _r12, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k10, _r12, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k11, _r13, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k12, _r14, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k10, _r14, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k11, _r15, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k12, _r16, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k10, _r16, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k11, _r17, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k12, _r18, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + 1 * packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + 2 * packn, vl);
                vfloat16m1_t _r23 = vle16_v_f16m1(r2 + 3 * packn, vl);
                vfloat16m1_t _r24 = vle16_v_f16m1(r2 + 4 * packn, vl);
                vfloat16m1_t _r25 = vle16_v_f16m1(r2 + 5 * packn, vl);
                vfloat16m1_t _r26 = vle16_v_f16m1(r2 + 6 * packn, vl);
                vfloat16m1_t _r27 = vle16_v_f16m1(r2 + 7 * packn, vl);
                vfloat16m1_t _r28 = vle16_v_f16m1(r2 + 8 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k20, _r20, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k21, _r21, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k22, _r22, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k20, _r22, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k21, _r23, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k22, _r24, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k20, _r24, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k21, _r25, vl);
                _acc02 = vfmacc_vv_f16m1(_acc02, _k22, _r26, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k20, _r26, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k21, _r27, vl);
                _acc03 = vfmacc_vv_f16m1(_acc03, _k22, _r28, vl);

                vse16_v_f16m1(out0, _acc00, vl);
                vse16_v_f16m1(out0 + 1 * packn, _acc01, vl);
                vse16_v_f16m1(out0 + 2 * packn, _acc02, vl);
                vse16_v_f16m1(out0 + 3 * packn, _acc03, vl);

                out0 += packn * 4;

                r0 += packn * 8;
                r1 += packn * 8;
                r2 += packn * 8;
            }
            for (; w + 1 < out_w; w += 2) {
                vfloat16m1_t _acc00 = _bias0;
                vfloat16m1_t _acc01 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + 1 * packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + 2 * packn, vl);
                vfloat16m1_t _r03 = vle16_v_f16m1(r0 + 3 * packn, vl);
                vfloat16m1_t _r04 = vle16_v_f16m1(r0 + 4 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k00, _r00, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k01, _r01, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k02, _r02, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k00, _r02, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k01, _r03, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k02, _r04, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + 1 * packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + 2 * packn, vl);
                vfloat16m1_t _r13 = vle16_v_f16m1(r1 + 3 * packn, vl);
                vfloat16m1_t _r14 = vle16_v_f16m1(r1 + 4 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k10, _r10, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k11, _r11, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k12, _r12, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k10, _r12, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k11, _r13, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k12, _r14, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + 1 * packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + 2 * packn, vl);
                vfloat16m1_t _r23 = vle16_v_f16m1(r2 + 3 * packn, vl);
                vfloat16m1_t _r24 = vle16_v_f16m1(r2 + 4 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k20, _r20, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k21, _r21, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k22, _r22, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k20, _r22, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k21, _r23, vl);
                _acc01 = vfmacc_vv_f16m1(_acc01, _k22, _r24, vl);

                vse16_v_f16m1(out0, _acc00, vl);
                vse16_v_f16m1(out0 + 1 * packn, _acc01, vl);

                out0 += packn * 2;

                r0 += packn * 4;
                r1 += packn * 4;
                r2 += packn * 4;
            }
            for (; w < out_w; w++) {
                vfloat16m1_t _acc00 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + 1 * packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + 2 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k00, _r00, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k01, _r01, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k02, _r02, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + 1 * packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + 2 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k10, _r10, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k11, _r11, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k12, _r12, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + 1 * packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + 2 * packn, vl);

                _acc00 = vfmacc_vv_f16m1(_acc00, _k20, _r20, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k21, _r21, vl);
                _acc00 = vfmacc_vv_f16m1(_acc00, _k22, _r22, vl);

                vse16_v_f16m1(out0, _acc00, vl);
                out0 += packn * 1;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
            }
            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
    shl_mem_free(input_padd_buf);
    return CSINN_TRUE;
}

void shl_rvv_dwconv_reorder_kernel_packn_fp16(struct csinn_tensor *kernel,
                                              struct csinn_conv2d_params *params)
{
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    const int out_ch = kernel->dim[0];
    const int maxk = kernel->dim[2] * kernel->dim[3];
    __fp16 *kernel_trans = (__fp16 *)shl_mem_alloc(out_ch * maxk * sizeof(__fp16));

    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int vl = vsetvl_e16m1(packn);

    for (int oc = 0; oc + packn - 1 < out_ch; oc += packn) {
        __fp16 *ksrc = kernel_data + oc * maxk;
        __fp16 *kdst = kernel_trans + oc * maxk;
        for (int ic = 0; ic < maxk; ic++) {
            vfloat16m1_t _tmp = vlse16_v_f16m1(ksrc + ic, maxk * sizeof(__fp16), vl);
            vse16_v_f16m1(kdst, _tmp, vl);
            kdst += vl;
        }
    }
    memcpy(kernel_data, kernel_trans, out_ch * maxk * sizeof(__fp16));
    shl_mem_free(kernel_trans);
}
