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

static void im2col_gemm_reorder_kernel_per_group_int8_matrix(int8_t *src, int8_t *dst, int out_c,
                                                             int in_c, int maxk)
{
    const int col = csrr_xrlenb();
    const int row = col / 4;
    int oc = 0;
    for (; oc + 2 * row <= out_c; oc += 2 * row) {
        int8_t *src_m = src + oc * in_c * maxk;
        int k = 0;
        for (; k + col <= maxk * in_c; k += col) {
            int8_t *src_n = src_m + k;
            for (int i = 0; i < 2 * row; i++) {
                int8_t *src_i = src_n + i * in_c * maxk;
                memcpy(dst, src_i, col * sizeof(int8_t));
                dst += col;
            }
        }
        // k_tail
        if (k < maxk * in_c) {
            int8_t *src_n = src_m + k;
            for (int i = 0; i < 2 * row; i++) {
                int8_t *src_i = src_n + i * in_c * maxk;
                memcpy(dst, src_i, (maxk * in_c - k) * sizeof(int8_t));
                dst += col;
            }
        }
    }
    for (; oc + row <= out_c; oc += row) {
        int8_t *src_m = src + oc * in_c * maxk;
        int k = 0;
        for (; k + col <= maxk * in_c; k += col) {
            int8_t *src_n = src_m + k;
            for (int i = 0; i < row; i++) {
                int8_t *src_i = src_n + i * in_c * maxk;
                memcpy(dst, src_i, col * sizeof(int8_t));
                dst += col;
            }
        }
        if (k < maxk * in_c) {
            int8_t *src_n = src_m + k;
            for (int i = 0; i < row; i++) {
                int8_t *src_i = src_n + i * in_c * maxk;
                memcpy(dst, src_i, (maxk * in_c - k) * sizeof(int8_t));
                dst += col;
            }
        }
    }
    // oc_tail
    if (oc < out_c) {
        int8_t *src_m = src + oc * in_c * maxk;
        int k = 0;
        for (; k + col <= maxk * in_c; k += col) {
            int8_t *src_n = src_m + k;
            for (int i = 0; i < (out_c - oc); i++) {
                int8_t *src_i = src_n + i * in_c * maxk;
                memcpy(dst, src_i, col * sizeof(int8_t));
                dst += col;
            }
            dst += (oc + row - out_c) * col;  // padding
        }
        if (k < maxk * in_c) {
            int8_t *src_n = src_m + k;
            for (int i = 0; i < (out_c - oc); i++) {
                int8_t *src_i = src_n + i * in_c * maxk;
                memcpy(dst, src_i, (maxk * in_c - k) * sizeof(int8_t));
                dst += col;
            }
        }
    }
}

void shl_rvm_conv_im2col_gemm_reorder_kernel_int8(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params)
{
    int8_t *kernel_data = (int8_t *)kernel->data;
    int group = params->group;

    int out_c = kernel->dim[0];
    int out_cp = out_c / group;  // per-group out channel
    int in_c = kernel->dim[3];
    int maxk = kernel->dim[1] * kernel->dim[2];

    int oc_per_group_align = ((out_cp - 1) & -(csrr_xrlenb() / 4)) + csrr_xrlenb() / 4;
    int k_align = ((in_c * maxk - 1) & -csrr_xrlenb()) + csrr_xrlenb();

    params->conv_extra.kernel_tm = csinn_alloc_tensor(NULL);
    params->conv_extra.kernel_tm->data =
        (int8_t *)shl_mem_alloc(group * oc_per_group_align * k_align * sizeof(int8_t));
    // int8_t *pa_reorder = (int8_t *)shl_mem_alloc(out_c * in_c * maxk * sizeof(int8_t));
    int8_t *pa_reorder = (int8_t *)params->conv_extra.kernel_tm->data;

    for (int g = 0; g < group; g++) {
        int8_t *ker_ptr = kernel_data + g * out_cp * in_c * maxk;
        int8_t *ker_tm_ptr = pa_reorder + g * oc_per_group_align * k_align;
        im2col_gemm_reorder_kernel_per_group_int8_matrix(ker_ptr, ker_tm_ptr, out_cp, in_c, maxk);
    }
    // memcpy(kernel_data, pa_reorder, out_c * in_c * maxk * sizeof(int8_t));
    // shl_mem_free(pa_reorder);
}

int shl_rvm_conv_im2col_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)params->conv_extra.kernel_tm->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[3];
    int32_t in_h = input->dim[1];
    int32_t in_w = input->dim[2];
    int32_t out_c = kernel->dim[0];
    int32_t out_h = output->dim[1];
    int32_t out_w = output->dim[2];
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
    int32_t k_align = ((k - 1) & -csrr_xrlenb()) + csrr_xrlenb();

    int32_t *multiplier = (int32_t *)shl_mem_alloc(n * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(n * sizeof(int32_t));

    // paddding
    int32_t padded_in_h = in_h + params->pad_top + params->pad_down;
    int32_t padded_in_w = in_w + params->pad_left + params->pad_right;
    int32_t padded_in_hw = padded_in_w * padded_in_h;
    int32_t flag_pad =
        params->pad_top + params->pad_down + params->pad_left + params->pad_right > 0 ? 1 : 0;
    int8_t *input_pad_buf = input_data;
    if (flag_pad) {
        input_pad_buf = (int8_t *)shl_mem_alloc(in_c * padded_in_hw * sizeof(int8_t));
    }

    // im2col [out_h, out_w, maxk, in_c]
    int8_t *im2col_buf = (int8_t *)shl_mem_alloc(k_align * out_h * out_w * sizeof(int8_t));

    for (int i = 0; i < batch; i++) {
        if (flag_pad) {
            shl_rvv_pad_input_nhwc_int8(input_data, input_pad_buf, in_h, in_w, in_c, padded_in_h,
                                        padded_in_w, params->pad_top, params->pad_left,
                                        input->qinfo->zero_point);
        }
        // im2col
        int vl = vsetvl_e8m1(csrr_vlenb() / sizeof(int8_t));
        int8_t *im2col_buf_shadow = im2col_buf;
        for (int p = 0; p < out_h; p++) {
            for (int q = 0; q < out_w; q++) {
                const int8_t *img0 =
                    input_pad_buf + (p * stride_h * padded_in_w + q * stride_w) * in_c;
                for (int a = 0; a < ksize_h; a++) {
                    for (int b = 0; b < ksize_w; b++) {
                        const int8_t *img1 =
                            img0 + (a * dilation_h * padded_in_w + b * dilation_w) * in_c;
                        int size = in_c;
                        while (size > 0) {
                            vl = vsetvl_e8m1(size);
                            vint8m1_t _input = vle8_v_i8m1(img1, vl);
                            img1 += vl;
                            vse8_v_i8m1(im2col_buf_shadow, _input, vl);
                            im2col_buf_shadow += vl;
                            size -= vl;
                        }
                    }
                }
                im2col_buf_shadow += k_align - k;  // align for mlenb
            }
        }

        if (kernel->quant_channel > 1) {
            for (int c = 0; c < n; c++) {
                multiplier[c] = kernel->qinfo[c].multiplier;
                shift[c] = -1 - kernel->qinfo[c].shift;
            }
        } else if (kernel->quant_channel == 1) {
            for (int c = 0; c < n; c++) {
                multiplier[c] = kernel->qinfo[0].multiplier;
                shift[c] = -1 - kernel->qinfo[0].shift;
            }
        }

        // gemm
        int8_t *ker_ptr = kernel_data;
        int32_t *bias_ptr = bias_data;  // bias_data != NULL with fusing zp to bias
        shl_rvm_nhwc_gemm_int8(output_data, ker_ptr, im2col_buf, bias_ptr, m, k_align, n,
                               output->qinfo->zero_point, multiplier, shift);
        input_data += in_c * in_h * in_w;
        output_data += m * n;
    }
    shl_mem_free(multiplier);
    shl_mem_free(shift);
    shl_mem_free(im2col_buf);
    if (flag_pad) {
        shl_mem_free(input_pad_buf);
    }
    return CSINN_TRUE;
}

// Split the group conv2d into multiple common conv2ds on HHB
int shl_rvm_group_conv_im2col_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params)
{
    return CSINN_FALSE;
}
