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

#include "shl_c908.h"

void shl_c908_conv_im2col_gemm_reorder_kernel_packnto1_int8(struct csinn_tensor *kernel,
                                                            struct csinn_conv2d_params *params)
{
    shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_int8(kernel, params);
}

int shl_c908_conv_im2col_gemm_packnto1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)params->conv_extra.kernel_tm->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];
    int32_t out_c = kernel->dim[0];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];
    int32_t ksize_h = kernel->dim[2];
    int32_t ksize_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;

    int32_t m = out_c / group;
    int32_t in_cp = in_c / group;
    int32_t maxk = ksize_h * ksize_w;
    int32_t n = out_h * out_w;

    int8_t *output_ncxhwx = (int8_t *)shl_mem_alloc(m * n * sizeof(int8_t));

    int32_t *multiplier = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));

    for (int i = 0; i < batch; i++) {
        for (int g = 0, j = 0; g < group; g++) {
            // paddding
            int padded_in_hw = (in_h + params->pad_top + params->pad_down) *
                               (in_w + params->pad_left + params->pad_right);
            int8_t *input_pad_buf = (int8_t *)shl_mem_alloc(in_cp * padded_in_hw * sizeof(int8_t));
            shl_rvv_pad_input_packn_int8(input_data, input_pad_buf, in_cp, in_h, in_w,
                                         (in_h + params->pad_top + params->pad_down),
                                         (in_w + params->pad_left + params->pad_right),
                                         params->pad_top, params->pad_left,
                                         input->qinfo->zero_point);

            // im2col
            const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
            const int vl = vsetvl_e8mf2(packn);

            // [in_c/packn, maxk, out_h, out_w, packn]
            int8_t *im2col_buf = (int8_t *)shl_mem_alloc(in_cp / packn * maxk * out_h * out_w *
                                                         packn * sizeof(int8_t));
            const int tailstep =
                ((in_w + params->pad_left + params->pad_right) * stride_h - out_w * stride_w) *
                packn;

            for (int c = 0; c + packn - 1 < in_cp; c += packn) {
                const int8_t *img0 = input_pad_buf + c * padded_in_hw;
                int8_t *dst_ptr = im2col_buf + c * maxk * out_h * out_w;

                for (int a = 0; a < ksize_h; a++) {
                    for (int b = 0; b < ksize_w; b++) {
                        const int8_t *img1 =
                            img0 + a * (in_w + params->pad_left + params->pad_right) * packn +
                            b * packn;

                        for (int p = 0; p < out_h; p++) {
                            for (int q = 0; q < out_w; q++) {
                                vint8mf2_t _tmp = vle8_v_i8mf2(img1, vl);
                                img1 += stride_w * packn;
                                vse8_v_i8mf2(dst_ptr, _tmp, vl);
                                dst_ptr += packn;
                            }
                            img1 += tailstep;
                        }
                    }
                }
            }
            shl_mem_free(input_pad_buf);

            if (kernel->quant_channel > 1) {
                for (int c = 0; c < m; c++, j++) {
                    multiplier[c] = kernel->qinfo[j].multiplier;
                    shift[c] = kernel->qinfo[j].shift;
                }
            } else if (kernel->quant_channel == 1) {
                for (int c = 0; c < m; c++) {
                    multiplier[c] = kernel->qinfo[0].multiplier;
                    shift[c] = kernel->qinfo[0].shift;
                }
            }

            int8_t *ker_ptr = kernel_data + g * m * maxk * in_cp;
            int32_t *bias_ptr = bias_data + g * m;  // bias_data != NULL with fusing zp to bias

            int8_t *reorder_buf = (int8_t *)shl_mem_alloc(in_cp * maxk * n * sizeof(int8_t));

#ifdef SHL_USE_DOT_INT8
            shl_rvv_reorder_input_z12_packn_int8_dot(im2col_buf, reorder_buf, in_cp * maxk, n, n);
            shl_mem_free(im2col_buf);
            shl_c908_ncxhwx_gemm_12xpackn_int8_dot(output_ncxhwx, ker_ptr, reorder_buf, bias_ptr, m,
                                                   in_cp * maxk, n, output->qinfo->zero_point,
                                                   multiplier, shift);
#else
            shl_rvv_reorder_input_z4_packn_int8(im2col_buf, reorder_buf, in_cp * maxk, n, n);
            shl_mem_free(im2col_buf);
            shl_c908_ncxhwx_gemm_4xpack2n_int8(output_ncxhwx, ker_ptr, reorder_buf, bias_ptr, m,
                                               in_cp * maxk, n, output->qinfo->zero_point,
                                               multiplier, shift);
#endif  // SHL_USE_DOT_INT8

            shl_rvv_reorder_input_packnto1_int8(output_ncxhwx, output_data, m, out_h, out_w);
            shl_mem_free(reorder_buf);

            input_data += in_cp * in_h * in_w;
            output_data += m * n;
        }
    }
    shl_mem_free(multiplier);
    shl_mem_free(shift);
    shl_mem_free(output_ncxhwx);
    return CSINN_TRUE;
}
