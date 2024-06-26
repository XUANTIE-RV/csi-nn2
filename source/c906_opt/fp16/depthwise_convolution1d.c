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

#include "c906/c906.h"

int shl_c906_dwconv8s1_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                            struct csinn_conv1d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = NULL;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];  // group = in_channel
    int32_t in_w = input->dim[2];

    int32_t out_c = output->dim[1];
    int32_t out_w = output->dim[2];

    __fp16 *kernel_fp16 = NULL;
    if (kernel->dtype == CSINN_DTYPE_INT8) {
        int size = csinn_tensor_size(kernel);
        int8_t *kernel_int8 = (int8_t *)kernel->data;
        kernel_fp16 = (__fp16 *)shl_mem_alloc(size * sizeof(__fp16));
        if (kernel->quant_channel > 1) {
            const int maxk = kernel->dim[2];
            for (int c = 0; c < in_c; c++) {
                int32_t zp = kernel->qinfo[c].zero_point;
                float scale = kernel->qinfo[c].scale;
                shl_rvv_dequantize_i8_to_f16(kernel_int8 + c * maxk, kernel_fp16 + c * maxk, maxk,
                                             zp, scale);
            }
        } else {
            int32_t zp = kernel->qinfo->zero_point;
            float scale = kernel->qinfo->scale;
            shl_rvv_dequantize_i8_to_f16(kernel_int8, kernel_fp16, size, zp, scale);
        }
        kernel_data = kernel_fp16;
    } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
        kernel_data = (__fp16 *)kernel->data;
    } else {
        shl_debug_error("kernel unsupport dtype: %d\n", kernel->dtype);
        return CSINN_FALSE;
    }

    if (params->pad_left == 0 && params->pad_right == 0) {
        for (int c = 0; c < in_c; c++) {
            __fp16 *out0 = output_data + c * out_w;
            __fp16 *img0 = input_data + c * in_w;
            const __fp16 *ker0 = kernel_data + c * 8;
            const __fp16 bias0 = bias_data ? bias_data[c] : 0.0f;

            int w_loop = csrr_vlenb() / sizeof(__fp16);  // VLEN128=8  VLEN256=16
            int vl = vsetvl_e16m1(w_loop);
            int w = 0;
            for (; w + w_loop - 1 < out_w; w += w_loop) {
                vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias0, vl);

                vfloat16m1_t _r0_0_7 = vle16_v_f16m1(img0, vl);
                vfloat16m1_t _r0_1_8 = vle16_v_f16m1(img0 + 1, vl);
                vfloat16m1_t _r0_2_9 = vle16_v_f16m1(img0 + 2, vl);
                vfloat16m1_t _r0_3_a = vle16_v_f16m1(img0 + 3, vl);
                vfloat16m1_t _r0_4_b = vle16_v_f16m1(img0 + 4, vl);
                vfloat16m1_t _r0_5_c = vle16_v_f16m1(img0 + 5, vl);
                vfloat16m1_t _r0_6_d = vle16_v_f16m1(img0 + 6, vl);
                vfloat16m1_t _r0_7_e = vle16_v_f16m1(img0 + 7, vl);

                _acc0 = vfmacc_vf_f16m1(_acc0, ker0[0], _r0_0_7, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, ker0[1], _r0_1_8, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, ker0[2], _r0_2_9, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, ker0[3], _r0_3_a, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, ker0[4], _r0_4_b, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, ker0[5], _r0_5_c, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, ker0[6], _r0_6_d, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, ker0[7], _r0_7_e, vl);

                vse16_v_f16m1(out0, _acc0, vl);
                img0 += vl;
                out0 += vl;
            }
            vl = vsetvl_e16m1(8);
            vfloat16m1_t _k0_0_7 = vle16_v_f16m1(ker0, vl);
            vfloat16m1_t _tmp = vfmv_v_f_f16m1(bias0, vl);

            for (; w < out_w; w++) {
                vfloat16m1_t _r0_0_7 = vle16_v_f16m1(img0, vl);
                vfloat16m1_t _acc0 = vfmul_vv_f16m1(_k0_0_7, _r0_0_7, vl);

                vfloat16m1_t _acc0_tmp =
                    vfredusum_vs_f16m1_f16m1(vundefined_f16m1(), _acc0, _tmp, vl);
                __fp16 res0 = vfmv_f_s_f16m1_f16(_acc0_tmp);
                img0 += 1;
                *out0++ = res0;
            }
        }
    } else {
        shl_debug_error("Dwconv1d unsupported padding params [%d %d]\n", params->pad_left,
                        params->pad_right);
        return CSINN_FALSE;
    }
    if (kernel->dtype == CSINN_DTYPE_INT8) {
        shl_mem_free(kernel_fp16);
    }
    return CSINN_TRUE;
}

int shl_c906_depthwise_conv1d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv1d_params *params)
{
    int32_t batch = input->dim[0];
    int32_t in_ch = input->dim[1];
    int32_t in_w = input->dim[2];

    int32_t out_ch = output->dim[1];
    int32_t out_w = output->dim[3];

    int32_t kernel_w = kernel->dim[2];
    int32_t stride_w = params->stride_width;
    struct csinn_callback *cb = params->base.cb;

    if (kernel_w == 8 && stride_w == 1) {
        cb->exec = shl_c906_dwconv8s1_fp16;
    } else {
        cb->exec = shl_ref_conv1d_quant;
    }
    return CSINN_TRUE;
}
