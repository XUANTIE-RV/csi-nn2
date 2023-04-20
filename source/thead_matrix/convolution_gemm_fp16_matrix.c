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

#include "shl_thead_rvm.h"

static void im2col_gemm_reorder_kernel_nhwc_per_group_fp16(__fp16 *src, __fp16 *dst, int out_c,
                                                           int in_c, int maxk)
{
    int m2rows = csrr_xrlenb() / 2;
    int cols = m2rows;
    int K = maxk * in_c;
    int oc = 0;
    for (; oc + m2rows <= out_c; oc += m2rows) {
        __fp16 *src_m = src + oc * K;
        int j = 0;
        for (; j + cols - 1 < K; j += cols) {
            __fp16 *src_n = src_m + j;
            for (int i = 0; i < m2rows; i++) {
                __fp16 *src_i = src_n + i * K;
                memcpy(dst, src_i, cols * sizeof(__fp16));
                dst += cols;
            }
        }
        // k_tail
        if (j < K) {
            __fp16 *src_n = src_m + j;
            for (int i = 0; i < m2rows; i++) {
                __fp16 *src_i = src_n + i * K;
                memcpy(dst, src_i, (K - j) * sizeof(__fp16));
                dst += cols;
            }
        }
    }
    // oc_tail
    if (oc < out_c) {
        __fp16 *src_m = src + oc * K;
        int j = 0;
        for (; j + cols - 1 < K; j += cols) {
            __fp16 *src_n = src_m + j;
            for (int i = 0; i < (out_c - oc); i++) {
                __fp16 *src_i = src_n + i * K;
                memcpy(dst, src_i, cols * sizeof(__fp16));
                dst += cols;
            }
            dst += (oc + m2rows - out_c) * cols;  // padding
        }
        // k_tail
        if (j < K) {
            __fp16 *src_n = src_m + j;
            for (int i = 0; i < (out_c - oc); i++) {
                __fp16 *src_i = src_n + i * K;
                memcpy(dst, src_i, (K - j) * sizeof(__fp16));
                dst += cols;
            }
        }
    }
}

void shl_rvm_conv_im2col_gemm_reorder_kernel_fp16(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params)
{
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    int group = params->group;

    int out_c = kernel->dim[0];
    int out_cp = out_c / group;  // per-group out channel
    int in_c = kernel->dim[3];
    int maxk = kernel->dim[1] * kernel->dim[2];

    int oc_per_group_align = ((out_cp - 1) & -(csrr_xrlenb() / 2)) + csrr_xrlenb() / 2;
    int k_align = ((in_c * maxk - 1) & -(csrr_xrlenb() / 2)) + csrr_xrlenb() / 2;

    params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
    params->conv_extra.kernel_tm->data =
        (__fp16 *)shl_mem_alloc(group * oc_per_group_align * k_align * sizeof(__fp16));

    __fp16 *pa_reorder = (__fp16 *)params->conv_extra.kernel_tm->data;

    for (int g = 0; g < group; g++) {
        __fp16 *ker_ptr = kernel_data + g * out_cp * in_c * maxk;
        __fp16 *ker_tm_ptr = pa_reorder + g * oc_per_group_align * k_align;
        im2col_gemm_reorder_kernel_nhwc_per_group_fp16(ker_ptr, ker_tm_ptr, out_cp, in_c, maxk);
    }
    // memcpy(kernel_data, kernel_reorder, out_c * in_c * maxk * sizeof(__fp16));
    // shl_mem_free(kernel_reorder);
}

int shl_rvm_conv_im2col_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)params->conv_extra.kernel_tm->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_h = input->dim[1];
    int32_t in_w = input->dim[2];
    int32_t in_c = input->dim[3];
    int32_t out_h = output->dim[1];
    int32_t out_w = output->dim[2];
    int32_t out_c = kernel->dim[0];
    int32_t ksize_h = kernel->dim[1];
    int32_t ksize_w = kernel->dim[2];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t dilation_h = params->dilation_height;
    int32_t dilation_w = params->dilation_width;

    int32_t m = out_h * out_w;
    int32_t maxk = ksize_h * ksize_w;
    int32_t n = out_c;
    int32_t k = in_c * maxk;
    int32_t k_align = ((k - 1) & -(csrr_xrlenb() / 2)) + csrr_xrlenb() / 2;

    // padding
    int32_t padded_in_h = in_h + params->pad_top + params->pad_down;
    int32_t padded_in_w = in_w + params->pad_left + params->pad_right;
    int32_t padded_in_hw = padded_in_w * padded_in_h;
    int32_t flag_pad =
        params->pad_top + params->pad_down + params->pad_left + params->pad_right > 0 ? 1 : 0;
    __fp16 *input_pad_buf = input_data;
    if (flag_pad) {
        input_pad_buf = (__fp16 *)shl_mem_alloc(padded_in_hw * in_c * sizeof(__fp16));
    }

    // im2col [out_h, out_w, maxk, in_c]
    __fp16 *im2col_buf = (__fp16 *)shl_mem_alloc(k_align * out_h * out_w * sizeof(__fp16));

    for (int b = 0; b < batch; b++) {
        if (flag_pad) {
            shl_rvv_pad_input_nhwc_fp16(input_data, input_pad_buf, in_h, in_w, in_c, padded_in_h,
                                        padded_in_w, params->pad_top, params->pad_left);
        }
        // im2col
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                const __fp16 *img0 =
                    input_pad_buf + (oh * stride_h * padded_in_w + ow * stride_w) * in_c;
                __fp16 *dst0 = im2col_buf + (oh * out_w + ow) * k_align;
                for (int kh = 0; kh < ksize_h; kh++) {
                    for (int kw = 0; kw < ksize_w; kw++) {
                        const __fp16 *img1 =
                            img0 + (kh * dilation_h * padded_in_w + kw * dilation_w) * in_c;
                        __fp16 *dst1 = dst0 + (kh * ksize_w + kw) * in_c;
                        int ic = 0;
                        while (ic < in_c) {
                            int vl = vsetvl_e16m1(in_c - ic);
                            vfloat16m1_t _tmp = vle16_v_f16m1(img1 + ic, vl);
                            vse16_v_f16m1(dst1 + ic, _tmp, vl);
                            ic += vl;
                        }
                    }
                }
            }
        }

        // gemm
        __fp16 *ker_ptr = kernel_data;
        __fp16 *bias_ptr = bias_data ? bias_data : NULL;
        shl_rvm_nhwc_gemm_fp16(output_data, ker_ptr, im2col_buf, bias_ptr, m, k_align, n);

        input_data += in_h * in_w * in_c;
        output_data += m * n;
    }
    shl_mem_free(im2col_buf);
    if (flag_pad) {
        shl_mem_free(input_pad_buf);
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}

// Split the group conv2d into multiple common conv2ds on HHB
int shl_rvm_group_conv_im2col_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params)
{
    return CSINN_FALSE;
}
