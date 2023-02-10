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

/*************************************************************************************
 * reorder kernel_data inplace, means the origin kernel_data be destoried.
 * The reason to do this is that the packaging process must not consume more memory.
 **************************************************************************************/
void shl_c908_conv_im2col_gemm_reorder_kernel_fp16(struct csinn_tensor *kernel,
                                                   struct csinn_conv2d_params *params)
{
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    int group = params->group;

    int m = kernel->dim[0] / group;  // m = out_ch / group
    int k = kernel->dim[1] * kernel->dim[2] * kernel->dim[3];

    __fp16 *pa_reorder = (__fp16 *)shl_mem_alloc(group * m * k * sizeof(__fp16));
    for (int g = 0; g < group; g++) {
        shl_c908_reorder_kernel_n8_fp16(kernel_data + g * m * k, pa_reorder + g * m * k, m, k, k);
    }
    memcpy(kernel_data, pa_reorder, group * m * k * sizeof(__fp16));
    shl_mem_free(pa_reorder);
}

int shl_c908_conv_im2col_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params)
{
    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(input);
    }
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

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

    __fp16 *im2col_data = (__fp16 *)shl_mem_alloc(k * n * sizeof(__fp16));
    __fp16 *pb_reorder = (__fp16 *)shl_mem_alloc(k * n * sizeof(__fp16));

    const int vlen = csrr_vlenb() * 8;

    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            // im2col
            __fp16 *data_col = im2col_data;
            __fp16 *channel_data = input_data;
            for (int c = 0; c < in_ch / group; c++) {
                for (int kh = 0; kh < ksize_h; kh++) {
                    for (int kw = 0; kw < ksize_w; kw++) {
                        int in_row = -pad_top + kh * dilation_h;
                        for (int oh = 0; oh < out_height; oh++) {
                            if (in_row >= in_height || in_row < 0) {
                                for (int ow = 0; ow < out_width; ow++) {
                                    *data_col++ = 0.0f;
                                }
                            } else {
                                int in_col = -pad_left + kw * dilation_w;
                                for (int ow1 = 0; ow1 < out_width; ow1++) {
                                    int col_idx = (c * out_height + oh) * out_width + ow1;
                                    if (in_col < in_width && in_col >= 0) {
                                        *data_col++ = channel_data[in_row * in_width + in_col];
                                    } else {
                                        *data_col++ = 0.0f;
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

            __fp16 *pa = kernel_data + g * m * k;
            __fp16 *pb = pb_reorder;
            __fp16 *pc = output_data;
            if (vlen == 128) {
                // pack
                shl_c908_reorder_input_z24_fp16(im2col_data, pb, k, n, n);
                // GEMM
                shl_c908_gemm_8x24_fp16(pc, pa, pb, bias_data + g * m, m, k, n, n);
            } else if (vlen >= 256) {
                // pack
                shl_c908_reorder_input_z32_fp16_v256(im2col_data, pb, k, n, n);
                // GEMM
                shl_c908_gemm_8x32_fp16_v256(pc, pa, pb, bias_data + g * m, m, k, n, n);
            }
            input_data += in_ch / group * in_height * in_width;
            output_data += m * n;
        }
    }
    shl_mem_free(pb_reorder);
    shl_mem_free(im2col_data);
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}
