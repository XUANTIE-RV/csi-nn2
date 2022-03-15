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

/* CSI-NN2 version 1.12.x */

#include "csi_thead_rvv.h"

/*************************************************************
    note: VLEN = 128/256
*************************************************************/
int csi_nn_rvv_dwconv3x3s1_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *kernel, struct csi_tensor *bias,
                                struct conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];  // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    float *input_padd_buf =
        (float *)csi_mem_alloc(in_c * (in_h + params->pad_top + params->pad_down) *
                               (in_w + params->pad_left + params->pad_right) * sizeof(float));

    csi_nn_rvv_pad_input_fp32(
        input_data, input_padd_buf, in_c, in_h, in_w, in_h + params->pad_top + params->pad_down,
        in_w + params->pad_left + params->pad_right, params->pad_top, params->pad_left);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

#pragma omp parallel for num_threads(1)
    for (int c = 0; c < in_c; c++) {
        float *out = output_data + c * out_h * out_w;
        float *outptr0 = out;
        float *outptr1 = outptr0 + out_w;

        const float bias0 = bias_data ? bias_data[c] : 0.0f;

        float *img0 = input_padd_buf + c * in_h * in_w;
        float *r0 = img0;
        float *r1 = r0 + in_w;
        float *r2 = r1 + in_w;
        float *r3 = r2 + in_w;

        const float *kernel0 = kernel_data + c * 9;

        float k00 = kernel0[0];
        float k01 = kernel0[1];
        float k02 = kernel0[2];
        float k10 = kernel0[3];
        float k11 = kernel0[4];
        float k12 = kernel0[5];
        float k20 = kernel0[6];
        float k21 = kernel0[7];
        float k22 = kernel0[8];

        int vl;
        int w_loop = csrr_vlenb() / sizeof(float);  // VLEN128=4  VLEN256=8
        int w2_loop = w_loop * 2;

        // TODO: 优化指令序列，调整 intrinsic ，达到和汇编类似的指令序列
        int h = 0;
        // h2 loop
        for (; h + 1 < out_h; h += 2) {
            vl = vsetvl_e32m2(w2_loop);
            int w = 0;
            // h2w8 loop
            for (; w + w2_loop - 1 < out_w; w += w2_loop) {
                vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias0, vl);
                vfloat32m2_t _acc1 = vfmv_v_f_f32m2(bias0, vl);

                vfloat32m2_t _r0_0_7 = vle32_v_f32m2(r0, vl);
                vfloat32m2_t _r0_1_8 = vle32_v_f32m2(r0 + 1, vl);
                vfloat32m2_t _r0_2_9 = vle32_v_f32m2(r0 + 2, vl);

                vfloat32m2_t _r1_0_7 = vle32_v_f32m2(r1, vl);
                vfloat32m2_t _r1_1_8 = vle32_v_f32m2(r1 + 1, vl);
                vfloat32m2_t _r1_2_9 = vle32_v_f32m2(r1 + 2, vl);

                vfloat32m2_t _r2_0_7 = vle32_v_f32m2(r2, vl);
                vfloat32m2_t _r2_1_8 = vle32_v_f32m2(r2 + 1, vl);
                vfloat32m2_t _r2_2_9 = vle32_v_f32m2(r2 + 2, vl);

                vfloat32m2_t _r3_0_7 = vle32_v_f32m2(r3, vl);
                vfloat32m2_t _r3_1_8 = vle32_v_f32m2(r3 + 1, vl);
                vfloat32m2_t _r3_2_9 = vle32_v_f32m2(r3 + 2, vl);

                _acc0 = vfmacc_vf_f32m2(_acc0, k00, _r0_0_7, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k01, _r0_1_8, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k02, _r0_2_9, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k10, _r1_0_7, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k11, _r1_1_8, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k12, _r1_2_9, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k20, _r2_0_7, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k21, _r2_1_8, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k22, _r2_2_9, vl);

                _acc1 = vfmacc_vf_f32m2(_acc1, k00, _r1_0_7, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k01, _r1_1_8, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k02, _r1_2_9, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k10, _r2_0_7, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k11, _r2_1_8, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k12, _r2_2_9, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k20, _r3_0_7, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k21, _r3_1_8, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k22, _r3_2_9, vl);

                vse32_v_f32m2(outptr0, _acc0, vl);
                vse32_v_f32m2(outptr1, _acc1, vl);

                r0 += vl;
                r1 += vl;
                r2 += vl;
                r3 += vl;
                outptr0 += vl;
                outptr1 += vl;
            }

            // h2w4
            for (; w + w_loop - 1 < out_w; w += w_loop) {
                vl = vsetvl_e32m1(w_loop);

                vfloat32m1_t _acc0 = vfmv_v_f_f32m1(bias0, vl);
                vfloat32m1_t _acc1 = vfmv_v_f_f32m1(bias0, vl);

                vfloat32m1_t _r0_0_3 = vle32_v_f32m1(r0, vl);
                vfloat32m1_t _r0_1_4 = vle32_v_f32m1(r0 + 1, vl);
                vfloat32m1_t _r0_2_5 = vle32_v_f32m1(r0 + 2, vl);

                vfloat32m1_t _r1_0_3 = vle32_v_f32m1(r1, vl);
                vfloat32m1_t _r1_1_4 = vle32_v_f32m1(r1 + 1, vl);
                vfloat32m1_t _r1_2_5 = vle32_v_f32m1(r1 + 2, vl);

                vfloat32m1_t _r2_0_3 = vle32_v_f32m1(r2, vl);
                vfloat32m1_t _r2_1_4 = vle32_v_f32m1(r2 + 1, vl);
                vfloat32m1_t _r2_2_5 = vle32_v_f32m1(r2 + 2, vl);

                vfloat32m1_t _r3_0_3 = vle32_v_f32m1(r3, vl);
                vfloat32m1_t _r3_1_4 = vle32_v_f32m1(r3 + 1, vl);
                vfloat32m1_t _r3_2_5 = vle32_v_f32m1(r3 + 2, vl);

                _acc0 = vfmacc_vf_f32m1(_acc0, k00, _r0_0_3, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k01, _r0_1_4, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k02, _r0_2_5, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k10, _r1_0_3, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k11, _r1_1_4, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k12, _r1_2_5, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k20, _r2_0_3, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k21, _r2_1_4, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k22, _r2_2_5, vl);

                _acc1 = vfmacc_vf_f32m1(_acc1, k00, _r1_0_3, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k01, _r1_1_4, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k02, _r1_2_5, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k10, _r2_0_3, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k11, _r2_1_4, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k12, _r2_2_5, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k20, _r3_0_3, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k21, _r3_1_4, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k22, _r3_2_5, vl);

                vse32_v_f32m1(outptr0, _acc0, vl);
                vse32_v_f32m1(outptr1, _acc1, vl);

                r0 += vl;
                r1 += vl;
                r2 += vl;
                r3 += vl;
                outptr0 += vl;
                outptr1 += vl;
            }

            vl = vsetvl_e32m1(3);

            vfloat32m1_t _k0 = vle32_v_f32m1(kernel0, vl);
            vfloat32m1_t _k1 = vle32_v_f32m1(kernel0 + 3, vl);
            vfloat32m1_t _k2 = vle32_v_f32m1(kernel0 + 6, vl);

            vfloat32m1_t _tmp = vfmv_v_f_f32m1(bias0, vl);

            // h2w_tail
            for (; w < out_w; w++) {
                vfloat32m1_t _r0 = vle32_v_f32m1(r0, vl);
                vfloat32m1_t _r1 = vle32_v_f32m1(r1, vl);
                vfloat32m1_t _r2 = vle32_v_f32m1(r2, vl);
                vfloat32m1_t _r3 = vle32_v_f32m1(r3, vl);

                vfloat32m1_t _acc0 = vfmul_vv_f32m1(_k0, _r0, vl);
                _acc0 = vfmacc_vv_f32m1(_acc0, _k1, _r1, vl);
                _acc0 = vfmacc_vv_f32m1(_acc0, _k2, _r2, vl);
                vfloat32m1_t _acc0_tmp =
                    vfredusum_vs_f32m1_f32m1(vundefined_f32m1(), _acc0, _tmp, vl);
                float res0 = vfmv_f_s_f32m1_f32(_acc0_tmp);

                vfloat32m1_t _acc1 = vfmul_vv_f32m1(_k0, _r1, vl);
                _acc1 = vfmacc_vv_f32m1(_acc1, _k1, _r2, vl);
                _acc1 = vfmacc_vv_f32m1(_acc1, _k2, _r3, vl);
                vfloat32m1_t _acc1_tmp =
                    vfredusum_vs_f32m1_f32m1(vundefined_f32m1(), _acc1, _tmp, vl);
                float res1 = vfmv_f_s_f32m1_f32(_acc1_tmp);

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
            vl = vsetvl_e32m2(w2_loop);
            int w = 0;
            // h1w8 loop
            for (; w + w2_loop - 1 < out_w; w += w2_loop) {
                vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias0, vl);

                vfloat32m2_t _r0_0_7 = vle32_v_f32m2(r0, vl);
                vfloat32m2_t _r0_1_8 = vle32_v_f32m2(r0 + 1, vl);
                vfloat32m2_t _r0_2_9 = vle32_v_f32m2(r0 + 2, vl);

                vfloat32m2_t _r1_0_7 = vle32_v_f32m2(r1, vl);
                vfloat32m2_t _r1_1_8 = vle32_v_f32m2(r1 + 1, vl);
                vfloat32m2_t _r1_2_9 = vle32_v_f32m2(r1 + 2, vl);

                vfloat32m2_t _r2_0_7 = vle32_v_f32m2(r2, vl);
                vfloat32m2_t _r2_1_8 = vle32_v_f32m2(r2 + 1, vl);
                vfloat32m2_t _r2_2_9 = vle32_v_f32m2(r2 + 2, vl);

                _acc0 = vfmacc_vf_f32m2(_acc0, k00, _r0_0_7, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k01, _r0_1_8, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k02, _r0_2_9, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k10, _r1_0_7, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k11, _r1_1_8, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k12, _r1_2_9, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k20, _r2_0_7, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k21, _r2_1_8, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, k22, _r2_2_9, vl);

                vse32_v_f32m2(outptr0, _acc0, vl);

                r0 += vl;
                r1 += vl;
                r2 += vl;
                outptr0 += vl;
            }

            // h1w4
            for (; w + w_loop - 1 < out_w; w += w_loop) {
                vl = vsetvl_e32m1(w_loop);

                vfloat32m1_t _acc0 = vfmv_v_f_f32m1(bias0, vl);

                vfloat32m1_t _r0_0_3 = vle32_v_f32m1(r0, vl);
                vfloat32m1_t _r0_1_4 = vle32_v_f32m1(r0 + 1, vl);
                vfloat32m1_t _r0_2_5 = vle32_v_f32m1(r0 + 2, vl);

                vfloat32m1_t _r1_0_3 = vle32_v_f32m1(r1, vl);
                vfloat32m1_t _r1_1_4 = vle32_v_f32m1(r1 + 1, vl);
                vfloat32m1_t _r1_2_5 = vle32_v_f32m1(r1 + 2, vl);

                vfloat32m1_t _r2_0_3 = vle32_v_f32m1(r2, vl);
                vfloat32m1_t _r2_1_4 = vle32_v_f32m1(r2 + 1, vl);
                vfloat32m1_t _r2_2_5 = vle32_v_f32m1(r2 + 2, vl);

                _acc0 = vfmacc_vf_f32m1(_acc0, k00, _r0_0_3, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k01, _r0_1_4, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k02, _r0_2_5, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k10, _r1_0_3, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k11, _r1_1_4, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k12, _r1_2_5, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k20, _r2_0_3, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k21, _r2_1_4, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, k22, _r2_2_5, vl);

                vse32_v_f32m1(outptr0, _acc0, vl);

                r0 += vl;
                r1 += vl;
                r2 += vl;
                outptr0 += vl;
            }

            vl = vsetvl_e32m1(3);

            vfloat32m1_t _k0 = vle32_v_f32m1(kernel0, vl);
            vfloat32m1_t _k1 = vle32_v_f32m1(kernel0 + 3, vl);
            vfloat32m1_t _k2 = vle32_v_f32m1(kernel0 + 6, vl);

            vfloat32m1_t _tmp = vfmv_v_f_f32m1(bias0, vl);
            // h1w_tail
            for (; w < out_w; w++) {
                vfloat32m1_t _r0 = vle32_v_f32m1(r0, vl);
                vfloat32m1_t _r1 = vle32_v_f32m1(r1, vl);
                vfloat32m1_t _r2 = vle32_v_f32m1(r2, vl);

                vfloat32m1_t _acc0 = vfmul_vv_f32m1(_k0, _r0, vl);
                _acc0 = vfmacc_vv_f32m1(_acc0, _k1, _r1, vl);
                _acc0 = vfmacc_vv_f32m1(_acc0, _k2, _r2, vl);
                vfloat32m1_t _acc0_tmp =
                    vfredusum_vs_f32m1_f32m1(vundefined_f32m1(), _acc0, _tmp, vl);
                float res0 = vfmv_f_s_f32m1_f32(_acc0_tmp);

                r0++;
                r1++;
                r2++;
                *outptr0++ = res0;
            }
        }
    }

    csi_mem_free(input_padd_buf);
    return CSINN_TRUE;
}

int csi_nn_rvv_dwconv3x3s2_fp32(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *kernel, struct csi_tensor *bias,
                                struct conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];  // group = in_channel
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];

    int32_t out_c = output->dim[1];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    float *input_padd_buf =
        (float *)csi_mem_alloc(in_c * (in_h + params->pad_top + params->pad_down) *
                               (in_w + params->pad_left + params->pad_right) * sizeof(float));

    csi_nn_rvv_pad_input_fp32(
        input_data, input_padd_buf, in_c, in_h, in_w, in_h + params->pad_top + params->pad_down,
        in_w + params->pad_left + params->pad_right, params->pad_top, params->pad_left);

    in_h = in_h + params->pad_top + params->pad_down;
    in_w = in_w + params->pad_left + params->pad_right;

    int tailstep = in_w - 2 * out_w + in_w;

#pragma omp parallel for num_threads(1)
    for (int c = 0; c < in_c; c++) {
        float *out = output_data + c * out_h * out_w;
        float *outptr0 = out;

        const float bias0 = bias_data ? bias_data[c] : 0.0f;

        float *img0 = input_padd_buf + c * in_h * in_w;
        float *r0 = img0;
        float *r1 = r0 + in_w;
        float *r2 = r1 + in_w;

        const float *kernel0 = kernel_data + c * 9;

        float k00 = kernel0[0];
        float k01 = kernel0[1];
        float k02 = kernel0[2];
        float k10 = kernel0[3];
        float k11 = kernel0[4];
        float k12 = kernel0[5];
        float k20 = kernel0[6];
        float k21 = kernel0[7];
        float k22 = kernel0[8];
        int vl;
        int w_loop = csrr_vlenb() / sizeof(float);  // VLEN128=4  VLEN256=8

        for (int h = 0; h < out_h; h++) {
            vl = vsetvl_e32m1(w_loop);
            int w = 0;
            // h1w4 loop
            for (; w + w_loop - 1 < out_w; w += w_loop) {
                vfloat32m1_t _acc = vfmv_v_f_f32m1(bias0, vl);

                vfloat32m1_t _r0_0_6, _r0_1_7;
                vfloat32m1_t _r1_0_6, _r1_1_7;
                vfloat32m1_t _r2_0_6, _r2_1_7;

                vlseg2e32_v_f32m1(&_r0_0_6, &_r0_1_7, r0, vl);
                r0 += 2;
                vfloat32m1_t _r0_2_8 = vlse32_v_f32m1(r0, 2 * sizeof(float), vl);
                r0 += (w_loop - 1) * 2;

                vlseg2e32_v_f32m1(&_r1_0_6, &_r1_1_7, r1, vl);
                r1 += 2;
                vfloat32m1_t _r1_2_8 = vlse32_v_f32m1(r1, 2 * sizeof(float), vl);
                r1 += (w_loop - 1) * 2;

                vlseg2e32_v_f32m1(&_r2_0_6, &_r2_1_7, r2, vl);
                r2 += 2;
                vfloat32m1_t _r2_2_8 = vlse32_v_f32m1(r2, 2 * sizeof(float), vl);
                r2 += (w_loop - 1) * 2;

                _acc = vfmacc_vf_f32m1(_acc, k00, _r0_0_6, vl);
                _acc = vfmacc_vf_f32m1(_acc, k01, _r0_1_7, vl);
                _acc = vfmacc_vf_f32m1(_acc, k02, _r0_2_8, vl);
                _acc = vfmacc_vf_f32m1(_acc, k10, _r1_0_6, vl);
                _acc = vfmacc_vf_f32m1(_acc, k11, _r1_1_7, vl);
                _acc = vfmacc_vf_f32m1(_acc, k12, _r1_2_8, vl);
                _acc = vfmacc_vf_f32m1(_acc, k20, _r2_0_6, vl);
                _acc = vfmacc_vf_f32m1(_acc, k21, _r2_1_7, vl);
                _acc = vfmacc_vf_f32m1(_acc, k22, _r2_2_8, vl);

                vse32_v_f32m1(outptr0, _acc, vl);
                outptr0 += vl;
            }

            vl = vsetvl_e32m1(3);

            vfloat32m1_t _k0 = vle32_v_f32m1(kernel0, vl);
            vfloat32m1_t _k1 = vle32_v_f32m1(kernel0 + 3, vl);
            vfloat32m1_t _k2 = vle32_v_f32m1(kernel0 + 6, vl);

            vfloat32m1_t _tmp = vfmv_v_f_f32m1(bias0, vl);
            // h1w_tail
            for (; w < out_w; w++) {
                vfloat32m1_t _r0 = vle32_v_f32m1(r0, vl);
                vfloat32m1_t _r1 = vle32_v_f32m1(r1, vl);
                vfloat32m1_t _r2 = vle32_v_f32m1(r2, vl);

                vfloat32m1_t _acc0 = vfmul_vv_f32m1(_k0, _r0, vl);
                _acc0 = vfmacc_vv_f32m1(_acc0, _k1, _r1, vl);
                _acc0 = vfmacc_vv_f32m1(_acc0, _k2, _r2, vl);
                vfloat32m1_t _acc0_tmp =
                    vfredusum_vs_f32m1_f32m1(vundefined_f32m1(), _acc0, _tmp, vl);
                float res0 = vfmv_f_s_f32m1_f32(_acc0_tmp);

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

    csi_mem_free(input_padd_buf);
    return CSINN_TRUE;
}
