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

#include "shl_c908.h"

void shl_c908_conv_im2col_gemm_reorder_kernel_pack1ton_int8(struct csinn_tensor *kernel,
                                                            struct csinn_conv2d_params *params)
{
    shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_int8(kernel, params);
}

int shl_c908_conv_im2col_gemm_pack1ton_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params)
{
    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(input);
    }
    if (output->layout == CSINN_LAYOUT_NCHW) {
        const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
        output->dim[1] /= packn;
        output->dim[4] = packn;
        output->dim_count = 5;
        output->layout = CSINN_LAYOUT_NC1HWC0;
    }
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
    int32_t dilation_h = params->dilation_height;
    int32_t dilation_w = params->dilation_width;

    int32_t m = out_c / group;
    int32_t in_cp = in_c / group;
    int32_t maxk = ksize_h * ksize_w;
    int32_t n = out_h * out_w;

    int32_t *multiplier = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));

    for (int i = 0; i < batch; i++) {
        for (int g = 0, j = 0; g < group; g++) {
            // padding
            int padded_in_h = in_h + params->pad_top + params->pad_down;
            int padded_in_w = in_w + params->pad_left + params->pad_right;
            int padded_in_hw = padded_in_h * padded_in_w;
            int8_t *input_pad_buf = (int8_t *)shl_mem_alloc(in_cp * padded_in_hw * sizeof(int8_t));
            shl_rvv_pad_input_pack1ton_int8(input_data, input_pad_buf, in_cp, in_h, in_w,
                                            padded_in_h, padded_in_w, params->pad_top,
                                            params->pad_left, input->qinfo->zero_point);

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
            int32_t *bias_ptr = bias_data + g * m;

            const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
            int vl = vsetvl_e8mf2(packn);
#ifdef SHL_USE_DOT_INT8
            // im2col
            int in_cp4 = ((in_cp - 1) & -4) + 4;
            // [in_cp4/packn, maxk, out_h, out_w, packn] + [maxk, out_h, out_w, in_cp4%packn]
            int8_t *im2col_buf = (int8_t *)shl_mem_alloc(in_cp4 * maxk * n * sizeof(int8_t));
            const int tailstep = (padded_in_w * stride_h - out_w * stride_w);

            const int8_t *img0 = input_pad_buf;
            int8_t *dst_ptr = im2col_buf;

            int loop_c = in_cp;
            while (loop_c > 0) {
                vl = vsetvl_e8mf2(loop_c);
                int vl4 = ((vl - 1) & -4) + 4;
                for (int a = 0; a < ksize_h; a++) {
                    for (int b = 0; b < ksize_w; b++) {
                        const int8_t *img1 =
                            img0 + a * dilation_h * padded_in_w * vl + b * dilation_w * vl;

                        for (int p = 0; p < out_h; p++) {
                            for (int q = 0; q < out_w; q++) {
                                vint8mf2_t _tmp = vle8_v_i8mf2(img1, vl);
                                img1 += stride_w * vl;
                                vse8_v_i8mf2(dst_ptr, _tmp, vl);
                                dst_ptr += vl4;  // XXX: dst align 4
                            }
                            img1 += tailstep * vl;
                        }
                    }
                }
                img0 += padded_in_hw * vl;
                // dst_ptr += maxk * out_h * out_w * vl;
                loop_c -= vl;
            }
            shl_mem_free(input_pad_buf);
            // reorder(pack)
            int8_t *reorder_buf = (int8_t *)shl_mem_alloc(in_cp4 * maxk * n * sizeof(int8_t));
            shl_rvv_reorder_input_z12_pack1ton_int8_dot(im2col_buf, reorder_buf, in_cp4, maxk, n,
                                                        n);
            shl_mem_free(im2col_buf);
            int8_t *ker_ptr = kernel_data + g * m * maxk * in_cp4;
            // gemm
            shl_c908_ncxhwx_gemm_12xpackn_int8_dot(output_data, ker_ptr, reorder_buf, bias_ptr, m,
                                                   in_cp4 * maxk, n, output->qinfo->zero_point,
                                                   multiplier, shift);
#else
            // im2col
            // [in_c/packn, maxk, out_h, out_w, packn] + [maxk, out_h, out_w, in_c%packn]
            int8_t *im2col_buf = (int8_t *)shl_mem_alloc(in_cp * maxk * n * sizeof(int8_t));
            const int tailstep = (padded_in_w * stride_h - out_w * stride_w);

            const int8_t *img0 = input_pad_buf;
            int8_t *dst_ptr = im2col_buf;

            int loop_c = in_cp;
            while (loop_c > 0) {
                vl = vsetvl_e8mf2(loop_c);
                for (int a = 0; a < ksize_h; a++) {
                    for (int b = 0; b < ksize_w; b++) {
                        const int8_t *img1 =
                            img0 + a * dilation_h * padded_in_w * vl + b * dilation_w * vl;

                        for (int p = 0; p < out_h; p++) {
                            for (int q = 0; q < out_w; q++) {
                                vint8mf2_t _tmp = vle8_v_i8mf2(img1, vl);
                                img1 += stride_w * vl;
                                vse8_v_i8mf2(dst_ptr, _tmp, vl);
                                dst_ptr += vl;
                            }
                            img1 += tailstep * vl;
                        }
                    }
                }
                img0 += padded_in_hw * vl;
                loop_c -= vl;
            }
            shl_mem_free(input_pad_buf);

            // reorder(pack)
            int8_t *reorder_buf = (int8_t *)shl_mem_alloc(in_cp * maxk * n * sizeof(int8_t));
            shl_rvv_reorder_input_z4_pack1ton_int8(im2col_buf, reorder_buf, in_cp, maxk, n, n);
            shl_mem_free(im2col_buf);
            int8_t *ker_ptr = kernel_data + g * m * maxk * in_cp;
            // gemm
            shl_c908_ncxhwx_gemm_4xpack2n_int8(output_data, ker_ptr, reorder_buf, bias_ptr, m,
                                               in_cp * maxk, n, output->qinfo->zero_point,
                                               multiplier, shift);
#endif  // SHL_USE_DOT_INT8

            shl_mem_free(reorder_buf);
            input_data += in_cp * in_h * in_w;
            output_data += m * n;
        }
    }

    shl_mem_free(multiplier);
    shl_mem_free(shift);
    return CSINN_TRUE;
}
