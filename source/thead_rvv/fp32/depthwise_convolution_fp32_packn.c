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
 *************************************************************/
int shl_rvv_dwconv_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params)
{
    if (input->layout == CSINN_LAYOUT_NCHW) {
        shl_rvv_tensor_ndarray_to_nc1xc0_replace_fp32(input);
    }
    if (output->layout == CSINN_LAYOUT_NCHW) {
        output->dim[1] /= input->dim[4];
        output->dim[4] = input->dim[4];
        output->dim_count = 5;
        output->layout = CSINN_LAYOUT_NC1HWC0;
    }
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    int batch = input->dim[0];
    int in_c = input->dim[1] * input->dim[4];  // group = in_channel
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int out_c = in_c;
    int out_h = output->dim[2];
    int out_w = output->dim[3];

    int kernel_h = kernel->dim[2];
    int kernel_w = kernel->dim[3];
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;

    int padded_in_h = in_h + params->pad_top + params->pad_down;
    int padded_in_w = in_w + params->pad_left + params->pad_right;

    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);

    float *input_padd_buf =
        (float *)shl_mem_alloc(in_c * padded_in_h * padded_in_w * sizeof(float));

#pragma omp parallel for num_threads(1)
    for (int b = 0; b < batch; b++) {
        shl_rvv_pad_input_packn_fp32(input_data, input_padd_buf, in_c, in_h, in_w, padded_in_h,
                                     padded_in_w, params->pad_top, params->pad_left);
        for (int c = 0; c + packn <= in_c; c += packn) {
            const float *src = input_padd_buf + c * padded_in_h * padded_in_w;
            float *dst = output_data + c * out_h * out_w;
            const float *kernel0 = kernel_data + c * kernel_h * kernel_w;
            vfloat32m1_t _bias0 =
                bias_data ? vle32_v_f32m1(bias_data + c, vl) : vfmv_v_f_f32m1(0.0f, vl);

            for (int oh = 0; oh < out_h; oh++) {
                int i_h_start = oh * stride_h;
                int i_h_end = i_h_start + kernel_h;
                for (int ow = 0; ow < out_w; ow++) {
                    int i_w_start = ow * stride_w;
                    int i_w_end = i_w_start + kernel_w;

                    vfloat32m1_t _acc = _bias0;
                    for (int ih = i_h_start, i_kh = 0; ih < i_h_end; ih++, i_kh++) {
                        for (int iw = i_w_start, i_kw = 0; iw < i_w_end; iw++, i_kw++) {
                            const float *in_ptr = src + (ih * padded_in_w + iw) * packn;
                            const float *k_ptr = kernel0 + (i_kh * kernel_w + i_kw) * packn;
                            vfloat32m1_t _rxx = vle32_v_f32m1(in_ptr, vl);
                            vfloat32m1_t _kxx = vle32_v_f32m1(k_ptr, vl);
                            _acc = vfmacc_vv_f32m1(_acc, _rxx, _kxx, vl);
                        }
                    }
                    float *out_ptr = dst + (oh * out_w + ow) * packn;
                    vse32_v_f32m1(out_ptr, _acc, vl);
                }
            }
        }
        input_data += in_c * in_h * in_w;
        output_data += out_c * out_h * out_w;
    }
    shl_mem_free(input_padd_buf);
    return CSINN_TRUE;
}
