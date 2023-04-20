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

#include "shl_thead_rvv.h"

/*************************************************************
    note: VLEN = 128/256
*************************************************************/
int shl_rvv_dwconv3x3s1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
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

    __fp16 *input_padd_buf =
        (__fp16 *)shl_mem_alloc(in_c * (in_h + params->pad_top + params->pad_down) *
                                (in_w + params->pad_left + params->pad_right) * sizeof(float));

    shl_rvv_pad_input_fp16(
        input_data, input_padd_buf, in_c, in_h, in_w, in_h + params->pad_top + params->pad_down,
        in_w + params->pad_left + params->pad_right, params->pad_top, params->pad_left);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

#pragma omp parallel for num_threads(1)
    for (int c = 0; c < in_c; c++) {
        __fp16 *out = output_data + c * out_h * out_w;
        __fp16 *outptr0 = out;
        __fp16 *outptr1 = outptr0 + out_w;

        const __fp16 bias0 = bias_data ? bias_data[c] : 0.0f;

        __fp16 *img0 = input_padd_buf + c * in_h * in_w;
        __fp16 *r0 = img0;
        __fp16 *r1 = r0 + in_w;
        __fp16 *r2 = r1 + in_w;
        __fp16 *r3 = r2 + in_w;

        const __fp16 *kernel0 = kernel_data + c * 9;

        __fp16 k00 = kernel0[0];
        __fp16 k01 = kernel0[1];
        __fp16 k02 = kernel0[2];
        __fp16 k10 = kernel0[3];
        __fp16 k11 = kernel0[4];
        __fp16 k12 = kernel0[5];
        __fp16 k20 = kernel0[6];
        __fp16 k21 = kernel0[7];
        __fp16 k22 = kernel0[8];

        int vl;
        int w_loop = csrr_vlenb() / sizeof(__fp16);  // VLEN128=8  VLEN256=16

        int h = 0;
        // h2 loop
        for (; h + 1 < out_h; h += 2) {
            vl = vsetvl_e16m1(w_loop);

            int w = 0;
            // h2w8 loop
            for (; w + w_loop - 1 < out_w; w += w_loop) {
                vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias0, vl);
                vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias0, vl);

                vfloat16m1_t _r0_0_7 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r0_1_8 = vle16_v_f16m1(r0 + 1, vl);
                vfloat16m1_t _r0_2_9 = vle16_v_f16m1(r0 + 2, vl);

                vfloat16m1_t _r1_0_7 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r1_1_8 = vle16_v_f16m1(r1 + 1, vl);
                vfloat16m1_t _r1_2_9 = vle16_v_f16m1(r1 + 2, vl);

                vfloat16m1_t _r2_0_7 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r2_1_8 = vle16_v_f16m1(r2 + 1, vl);
                vfloat16m1_t _r2_2_9 = vle16_v_f16m1(r2 + 2, vl);

                vfloat16m1_t _r3_0_7 = vle16_v_f16m1(r3, vl);
                vfloat16m1_t _r3_1_8 = vle16_v_f16m1(r3 + 1, vl);
                vfloat16m1_t _r3_2_9 = vle16_v_f16m1(r3 + 2, vl);

                _acc0 = vfmacc_vf_f16m1(_acc0, k00, _r0_0_7, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k01, _r0_1_8, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k02, _r0_2_9, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k10, _r1_0_7, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k11, _r1_1_8, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k12, _r1_2_9, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k20, _r2_0_7, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k21, _r2_1_8, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k22, _r2_2_9, vl);

                _acc1 = vfmacc_vf_f16m1(_acc1, k00, _r1_0_7, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k01, _r1_1_8, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k02, _r1_2_9, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k10, _r2_0_7, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k11, _r2_1_8, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k12, _r2_2_9, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k20, _r3_0_7, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k21, _r3_1_8, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k22, _r3_2_9, vl);

                vse16_v_f16m1(outptr0, _acc0, vl);
                vse16_v_f16m1(outptr1, _acc1, vl);

                r0 += vl;
                r1 += vl;
                r2 += vl;
                r3 += vl;
                outptr0 += vl;
                outptr1 += vl;
            }

            // h2w4
            for (; w + w_loop / 2 - 1 < out_w; w += w_loop / 2) {
                vl = vsetvl_e16m1(w_loop / 2);

                vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias0, vl);
                vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias0, vl);

                vfloat16m1_t _r0_0_3 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r0_1_4 = vle16_v_f16m1(r0 + 1, vl);
                vfloat16m1_t _r0_2_5 = vle16_v_f16m1(r0 + 2, vl);

                vfloat16m1_t _r1_0_3 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r1_1_4 = vle16_v_f16m1(r1 + 1, vl);
                vfloat16m1_t _r1_2_5 = vle16_v_f16m1(r1 + 2, vl);

                vfloat16m1_t _r2_0_3 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r2_1_4 = vle16_v_f16m1(r2 + 1, vl);
                vfloat16m1_t _r2_2_5 = vle16_v_f16m1(r2 + 2, vl);

                vfloat16m1_t _r3_0_3 = vle16_v_f16m1(r3, vl);
                vfloat16m1_t _r3_1_4 = vle16_v_f16m1(r3 + 1, vl);
                vfloat16m1_t _r3_2_5 = vle16_v_f16m1(r3 + 2, vl);

                _acc0 = vfmacc_vf_f16m1(_acc0, k00, _r0_0_3, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k01, _r0_1_4, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k02, _r0_2_5, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k10, _r1_0_3, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k11, _r1_1_4, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k12, _r1_2_5, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k20, _r2_0_3, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k21, _r2_1_4, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k22, _r2_2_5, vl);

                _acc1 = vfmacc_vf_f16m1(_acc1, k00, _r1_0_3, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k01, _r1_1_4, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k02, _r1_2_5, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k10, _r2_0_3, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k11, _r2_1_4, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k12, _r2_2_5, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k20, _r3_0_3, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k21, _r3_1_4, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k22, _r3_2_5, vl);

                vse16_v_f16m1(outptr0, _acc0, vl);
                vse16_v_f16m1(outptr1, _acc1, vl);

                r0 += vl;
                r1 += vl;
                r2 += vl;
                r3 += vl;
                outptr0 += vl;
                outptr1 += vl;
            }

            vl = vsetvl_e16m1(3);

            vfloat16m1_t _k0 = vle16_v_f16m1(kernel0, vl);
            vfloat16m1_t _k1 = vle16_v_f16m1(kernel0 + 3, vl);
            vfloat16m1_t _k2 = vle16_v_f16m1(kernel0 + 6, vl);

            vfloat16m1_t _tmp = vfmv_v_f_f16m1(bias0, vl);

            // h2w_tail
            for (; w < out_w; w++) {
                vfloat16m1_t _r0 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r1 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r2 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r3 = vle16_v_f16m1(r3, vl);

                vfloat16m1_t _acc0 = vfmul_vv_f16m1(_k0, _r0, vl);
                _acc0 = vfmacc_vv_f16m1(_acc0, _k1, _r1, vl);
                _acc0 = vfmacc_vv_f16m1(_acc0, _k2, _r2, vl);
                vfloat16m1_t _acc0_tmp =
                    vfredusum_vs_f16m1_f16m1(vundefined_f16m1(), _acc0, _tmp, vl);
                __fp16 res0 = vfmv_f_s_f16m1_f16(_acc0_tmp);

                vfloat16m1_t _acc1 = vfmul_vv_f16m1(_k0, _r1, vl);
                _acc1 = vfmacc_vv_f16m1(_acc1, _k1, _r2, vl);
                _acc1 = vfmacc_vv_f16m1(_acc1, _k2, _r3, vl);
                vfloat16m1_t _acc1_tmp =
                    vfredusum_vs_f16m1_f16m1(vundefined_f16m1(), _acc1, _tmp, vl);
                __fp16 res1 = vfmv_f_s_f16m1_f16(_acc1_tmp);

                r0++;
                r1++;
                r2++;
                r3++;
                *outptr0++ = res0;
                *outptr1++ = res1;
            }
            r0 += 2 + in_w;
            r1 += 2 + in_w;
            r2 += 2 + in_w;
            r3 += 2 + in_w;

            outptr0 += out_w;
            outptr1 += out_w;
        }

        // h1
        for (; h < out_h; h++) {
            vl = vsetvl_e16m1(w_loop);
            int w = 0;
            // h1w8 loop 使用了 v 寄存器一半位宽资源
            for (; w + w_loop - 1 < out_w; w += w_loop) {
                vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias0, vl);

                vfloat16m1_t _r0_0_7 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r0_1_8 = vle16_v_f16m1(r0 + 1, vl);
                vfloat16m1_t _r0_2_9 = vle16_v_f16m1(r0 + 2, vl);

                vfloat16m1_t _r1_0_7 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r1_1_8 = vle16_v_f16m1(r1 + 1, vl);
                vfloat16m1_t _r1_2_9 = vle16_v_f16m1(r1 + 2, vl);

                vfloat16m1_t _r2_0_7 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r2_1_8 = vle16_v_f16m1(r2 + 1, vl);
                vfloat16m1_t _r2_2_9 = vle16_v_f16m1(r2 + 2, vl);

                _acc0 = vfmacc_vf_f16m1(_acc0, k00, _r0_0_7, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k01, _r0_1_8, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k02, _r0_2_9, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k10, _r1_0_7, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k11, _r1_1_8, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k12, _r1_2_9, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k20, _r2_0_7, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k21, _r2_1_8, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k22, _r2_2_9, vl);

                vse16_v_f16m1(outptr0, _acc0, vl);

                r0 += vl;
                r1 += vl;
                r2 += vl;
                outptr0 += vl;
            }

            // h1w4
            for (; w + w_loop / 2 - 1 < out_w; w += w_loop / 2) {
                vl = vsetvl_e16m1(w_loop / 2);

                vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias0, vl);

                vfloat16m1_t _r0_0_3 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r0_1_4 = vle16_v_f16m1(r0 + 1, vl);
                vfloat16m1_t _r0_2_5 = vle16_v_f16m1(r0 + 2, vl);

                vfloat16m1_t _r1_0_3 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r1_1_4 = vle16_v_f16m1(r1 + 1, vl);
                vfloat16m1_t _r1_2_5 = vle16_v_f16m1(r1 + 2, vl);

                vfloat16m1_t _r2_0_3 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r2_1_4 = vle16_v_f16m1(r2 + 1, vl);
                vfloat16m1_t _r2_2_5 = vle16_v_f16m1(r2 + 2, vl);

                _acc0 = vfmacc_vf_f16m1(_acc0, k00, _r0_0_3, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k01, _r0_1_4, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k02, _r0_2_5, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k10, _r1_0_3, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k11, _r1_1_4, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k12, _r1_2_5, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k20, _r2_0_3, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k21, _r2_1_4, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, k22, _r2_2_5, vl);

                vse16_v_f16m1(outptr0, _acc0, vl);

                r0 += vl;
                r1 += vl;
                r2 += vl;
                outptr0 += vl;
            }
            vl = vsetvl_e16m1(3);

            vfloat16m1_t _k0 = vle16_v_f16m1(kernel0, vl);
            vfloat16m1_t _k1 = vle16_v_f16m1(kernel0 + 3, vl);
            vfloat16m1_t _k2 = vle16_v_f16m1(kernel0 + 6, vl);

            vfloat16m1_t _tmp = vfmv_v_f_f16m1(bias0, vl);
            // h1w_tail
            for (; w < out_w; w++) {
                vfloat16m1_t _r0 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r1 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r2 = vle16_v_f16m1(r2, vl);

                vfloat16m1_t _acc0 = vfmul_vv_f16m1(_k0, _r0, vl);
                _acc0 = vfmacc_vv_f16m1(_acc0, _k1, _r1, vl);
                _acc0 = vfmacc_vv_f16m1(_acc0, _k2, _r2, vl);
                vfloat16m1_t _acc0_tmp =
                    vfredusum_vs_f16m1_f16m1(vundefined_f16m1(), _acc0, _tmp, vl);
                float res0 = vfmv_f_s_f16m1_f16(_acc0_tmp);

                r0++;
                r1++;
                r2++;
                *outptr0++ = res0;
            }
        }
    }
    shl_mem_free(input_padd_buf);
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}

int shl_rvv_dwconv3x3s2_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
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

    __fp16 *input_padd_buf =
        (__fp16 *)shl_mem_alloc(in_c * (in_h + params->pad_top + params->pad_down) *
                                (in_w + params->pad_left + params->pad_right) * sizeof(float));

    shl_rvv_pad_input_fp16(
        input_data, input_padd_buf, in_c, in_h, in_w, in_h + params->pad_top + params->pad_down,
        in_w + params->pad_left + params->pad_right, params->pad_top, params->pad_left);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

    int tailstep = in_w - 2 * out_w + in_w;

#pragma omp parallel for num_threads(1)
    for (int c = 0; c < in_c; c++) {
        __fp16 *out = output_data + c * out_h * out_w;
        __fp16 *outptr0 = out;

        const __fp16 bias0 = bias_data ? bias_data[c] : 0.0f;

        __fp16 *img0 = input_padd_buf + c * in_h * in_w;
        __fp16 *r0 = img0;
        __fp16 *r1 = r0 + in_w;
        __fp16 *r2 = r1 + in_w;

        const __fp16 *kernel0 = kernel_data + c * 9;

        __fp16 k00 = kernel0[0];
        __fp16 k01 = kernel0[1];
        __fp16 k02 = kernel0[2];
        __fp16 k10 = kernel0[3];
        __fp16 k11 = kernel0[4];
        __fp16 k12 = kernel0[5];
        __fp16 k20 = kernel0[6];
        __fp16 k21 = kernel0[7];
        __fp16 k22 = kernel0[8];
        int vl;
        int w_loop = csrr_vlenb() / sizeof(__fp16);  // VLEN128=8  VLEN256=16

        for (int h = 0; h < out_h; h++) {
            vl = vsetvl_e16m1(w_loop);
            int w = 0;
            // h1w8 loop
            for (; w + w_loop - 1 < out_w; w += w_loop) {
                vfloat16m1_t _acc = vfmv_v_f_f16m1(bias0, vl);

                vfloat16m1_t _r0_0_6, _r0_1_7;
                vfloat16m1_t _r1_0_6, _r1_1_7;
                vfloat16m1_t _r2_0_6, _r2_1_7;

                vlseg2e16_v_f16m1(&_r0_0_6, &_r0_1_7, r0, vl);
                r0 += 2;
                vfloat16m1_t _r0_2_8 = vlse16_v_f16m1(r0, 2 * 2, vl);
                r0 += (w_loop - 1) * 2;

                vlseg2e16_v_f16m1(&_r1_0_6, &_r1_1_7, r1, vl);
                r1 += 2;
                vfloat16m1_t _r1_2_8 = vlse16_v_f16m1(r1, 2 * 2, vl);
                r1 += (w_loop - 1) * 2;

                vlseg2e16_v_f16m1(&_r2_0_6, &_r2_1_7, r2, vl);
                r2 += 2;
                vfloat16m1_t _r2_2_8 = vlse16_v_f16m1(r2, 2 * 2, vl);
                r2 += (w_loop - 1) * 2;

                _acc = vfmacc_vf_f16m1(_acc, k00, _r0_0_6, vl);
                _acc = vfmacc_vf_f16m1(_acc, k01, _r0_1_7, vl);
                _acc = vfmacc_vf_f16m1(_acc, k02, _r0_2_8, vl);
                _acc = vfmacc_vf_f16m1(_acc, k10, _r1_0_6, vl);
                _acc = vfmacc_vf_f16m1(_acc, k11, _r1_1_7, vl);
                _acc = vfmacc_vf_f16m1(_acc, k12, _r1_2_8, vl);
                _acc = vfmacc_vf_f16m1(_acc, k20, _r2_0_6, vl);
                _acc = vfmacc_vf_f16m1(_acc, k21, _r2_1_7, vl);
                _acc = vfmacc_vf_f16m1(_acc, k22, _r2_2_8, vl);

                vse16_v_f16m1(outptr0, _acc, vl);
                outptr0 += vl;
            }

            // h1w4
            for (; w + w_loop / 2 - 1 < out_w; w += w_loop / 2) {
                vl = vsetvl_e16m1(w_loop / 2);
                vfloat16m1_t _acc = vfmv_v_f_f16m1(bias0, vl);

                vfloat16m1_t _r0_0_3, _r0_1_4;
                vfloat16m1_t _r1_0_3, _r1_1_4;
                vfloat16m1_t _r2_0_3, _r2_1_4;

                vlseg2e16_v_f16m1(&_r0_0_3, &_r0_1_4, r0, vl);
                r0 += 2;
                vfloat16m1_t _r0_2_5 = vlse16_v_f16m1(r0, 2 * 2, vl);
                r0 += w_loop - 2;

                vlseg2e16_v_f16m1(&_r1_0_3, &_r1_1_4, r1, vl);
                r1 += 2;
                vfloat16m1_t _r1_2_5 = vlse16_v_f16m1(r1, 2 * 2, vl);
                r1 += w_loop - 2;

                vlseg2e16_v_f16m1(&_r2_0_3, &_r2_1_4, r2, vl);
                r2 += 2;
                vfloat16m1_t _r2_2_5 = vlse16_v_f16m1(r2, 2 * 2, vl);
                r2 += w_loop - 2;

                _acc = vfmacc_vf_f16m1(_acc, k00, _r0_0_3, vl);
                _acc = vfmacc_vf_f16m1(_acc, k01, _r0_1_4, vl);
                _acc = vfmacc_vf_f16m1(_acc, k02, _r0_2_5, vl);
                _acc = vfmacc_vf_f16m1(_acc, k10, _r1_0_3, vl);
                _acc = vfmacc_vf_f16m1(_acc, k11, _r1_1_4, vl);
                _acc = vfmacc_vf_f16m1(_acc, k12, _r1_2_5, vl);
                _acc = vfmacc_vf_f16m1(_acc, k20, _r2_0_3, vl);
                _acc = vfmacc_vf_f16m1(_acc, k21, _r2_1_4, vl);
                _acc = vfmacc_vf_f16m1(_acc, k22, _r2_2_5, vl);

                vse16_v_f16m1(outptr0, _acc, vl);
                outptr0 += vl;
            }
            vl = vsetvl_e16m1(3);

            vfloat16m1_t _k0 = vle16_v_f16m1(kernel0, vl);
            vfloat16m1_t _k1 = vle16_v_f16m1(kernel0 + 3, vl);
            vfloat16m1_t _k2 = vle16_v_f16m1(kernel0 + 6, vl);

            vfloat16m1_t _tmp = vfmv_v_f_f16m1(bias0, vl);
            // h1w_tail
            for (; w < out_w; w++) {
                vfloat16m1_t _r0 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r1 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r2 = vle16_v_f16m1(r2, vl);

                vfloat16m1_t _acc0 = vfmul_vv_f16m1(_k0, _r0, vl);
                _acc0 = vfmacc_vv_f16m1(_acc0, _k1, _r1, vl);
                _acc0 = vfmacc_vv_f16m1(_acc0, _k2, _r2, vl);
                vfloat16m1_t _acc0_tmp =
                    vfredusum_vs_f16m1_f16m1(vundefined_f16m1(), _acc0, _tmp, vl);
                __fp16 res0 = vfmv_f_s_f16m1_f16(_acc0_tmp);

                r0 += 2;
                r1 += 2;
                r2 += 2;
                *outptr0++ = res0;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }

    shl_mem_free(input_padd_buf);
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}
