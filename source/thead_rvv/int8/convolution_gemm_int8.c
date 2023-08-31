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

void shl_rvv_conv_im2col_gemm_reorder_kernel_int8(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params)
{
    int8_t *kernel_data = (int8_t *)kernel->data;
    int group = params->group;

    int m = kernel->dim[0] / group;  // m = out_ch / group
    int k = kernel->dim[1] * kernel->dim[2] * kernel->dim[3];

#ifdef SHL_USE_DOT_INT8
    int k4 = (k % 4 != 0) ? ((k / 4 + 1) * 4) : k;
    params->conv_extra.kernel_tm->data = (int8_t *)shl_mem_alloc(group * m * k4 * sizeof(int8_t));
    int8_t *pa_reorder = (int8_t *)params->conv_extra.kernel_tm->data;

    for (int g = 0; g < group; g++) {
        shl_rvv_reorder_kernel_n8_int8_dot(kernel_data + g * m * k, pa_reorder + g * m * k4, m, k,
                                           k);
    }
#else
    params->conv_extra.kernel_tm->data = (int8_t *)shl_mem_alloc(group * m * k * sizeof(int8_t));
    int8_t *pa_reorder = (int8_t *)params->conv_extra.kernel_tm->data;

    for (int g = 0; g < group; g++) {
        shl_rvv_reorder_kernel_n4_int8_v128(kernel_data + g * m * k, pa_reorder + g * m * k, m, k,
                                            k);
    }
#endif  // SHL_USE_DOT_INT8
    // FIXME: free params->conv_extra.kernel_tm->data
    // memcpy(kernel_data, pa_reorder, group * m * k * sizeof(__fp16));
    // shl_mem_free(pa_reorder);
}

int shl_rvv_common_conv_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params,
                                  void (*reorder_input)(int8_t *, int8_t *, int, int, int),
                                  void (*gemm)(int8_t *, const int8_t *, const int8_t *, int32_t *,
                                               int, int, int, int, int32_t, int32_t *, int32_t *))
{
    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(input);
    }
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)params->conv_extra.kernel_tm->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];
    int32_t in_ch = input->dim[1];
    int32_t in_height = input->dim[2];
    int32_t in_width = input->dim[3];
    int32_t out_ch = kernel->dim[0];
    int32_t out_height = output->dim[2];
    int32_t out_width = output->dim[3];
    int32_t ksize_h = kernel->dim[2];
    int32_t ksize_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t pad_left = params->pad_left;
    int32_t pad_top = params->pad_top;
    int32_t dilation_h = params->dilation_height;
    int32_t dilation_w = params->dilation_width;

    int32_t m = out_ch / group;
    int32_t k = in_ch / group * ksize_h * ksize_w;
    int32_t n = out_height * out_width;

    int8_t *im2col_data = (int8_t *)shl_mem_alloc(k * n * sizeof(int8_t));
    int32_t *multiplier = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));

#ifdef SHL_USE_DOT_INT8
    int32_t k4 = (k % 4 != 0) ? ((k / 4 + 1) * 4) : k;
    int8_t *pb_reorder = (int8_t *)shl_mem_alloc(k4 * n * sizeof(int8_t));
#else
    int8_t *pb_reorder = (int8_t *)shl_mem_alloc(k * n * sizeof(int8_t));
#endif  // SHL_USE_DOT_INT8

    int j = 0;
    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            // im2col
            int8_t *data_col = im2col_data;
            int8_t *channel_data = input_data;
            for (int c = 0; c < in_ch / group; c++) {
                for (int kh = 0; kh < ksize_h; kh++) {
                    for (int kw = 0; kw < ksize_w; kw++) {
                        int in_row = -pad_top + kh * dilation_h;
                        for (int oh = 0; oh < out_height; oh++) {
                            if (in_row >= in_height || in_row < 0) {
                                for (int ow = 0; ow < out_width; ow++) {
                                    *data_col++ = input->qinfo->zero_point;
                                }
                            } else {
                                int in_col = -pad_left + kw * dilation_w;
                                for (int ow1 = 0; ow1 < out_width; ow1++) {
                                    int col_idx = (c * out_height + oh) * out_width + ow1;
                                    if (in_col < in_width && in_col >= 0) {
                                        *data_col++ = channel_data[in_row * in_width + in_col];
                                    } else {
                                        *data_col++ = input->qinfo->zero_point;
                                    }
                                    in_col += stride_w;
                                }
                            }
                            in_row += stride_h;
                        }
                    }
                }
                channel_data += in_height * in_width;
            }

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
            int8_t *pb = pb_reorder;
            int8_t *pc = output_data;

#ifdef SHL_USE_DOT_INT8
            int8_t *pa = kernel_data + g * m * k4;
            reorder_input(im2col_data, pb, k, n, n);
            gemm(pc, pa, pb, bias_data + g * m, m, k4, n, n, output->qinfo->zero_point, multiplier,
                 shift);
#else
            int8_t *pa = kernel_data + g * m * k;
            reorder_input(im2col_data, pb, k, n, n);
            gemm(pc, pa, pb, bias_data + g * m, m, k, n, n, output->qinfo->zero_point, multiplier,
                 shift);
#endif  // SHL_USE_DOT_INT8
            input_data += in_ch / group * in_height * in_width;
            output_data += m * n;
        }
    }
    shl_mem_free(pb_reorder);
    shl_mem_free(im2col_data);
    shl_mem_free(multiplier);
    shl_mem_free(shift);
    return CSINN_TRUE;
}

int shl_rvv_conv_im2col_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params)
{
#ifdef SHL_USE_DOT_INT8
    return shl_rvv_common_conv_gemm_int8(input, output, kernel, bias, params,
                                         shl_rvv_reorder_input_z8_int8_dot,
                                         shl_rvv_gemm_8x8_int8_dot);
#else
    return shl_rvv_common_conv_gemm_int8(input, output, kernel, bias, params,
                                         shl_rvv_reorder_input_z16_int8_v128,
                                         shl_rvv_gemm_4x16_int8_v128);
#endif  // SHL_USE_DOT_INT8
}
