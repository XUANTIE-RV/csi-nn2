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

/* CSI-NN2 version 2.0.x */

#include "shl_thead_rvv.h"
#ifdef XTHEADV
void shl_rvv_conv_im2col_gemm_reorder_kernel_int8(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params)
{
    int8_t *kernel_data = (int8_t *)kernel->data;
    int group = params->group;

    int m = kernel->dim[0] / group;  // m = out_ch / group
    int k = kernel->dim[1] * kernel->dim[2] * kernel->dim[3];
    int k4 = (k % 4 != 0) ? ((k / 4 + 1) * 4) : k;

    params->conv_extra.kernel_tm->data = (int8_t *)shl_mem_alloc(group * m * k4 * sizeof(int8_t));
    int8_t *pa_reorder = (int8_t *)params->conv_extra.kernel_tm->data;

    for (int g = 0; g < group; g++) {
        shl_rvv_reorder_kernel_n8_int8(kernel_data + g * m * k, pa_reorder + g * m * k4, m, k, k);
    }
    // FIXME: free params->conv_extra.kernel_tm->data
    // memcpy(kernel_data, pa_reorder, group * m * k * sizeof(__fp16));
    // shl_mem_free(pa_reorder);
}

int shl_rvv_conv_im2col_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params)
{
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

    // im2col matrix_col = out_height * out_width
    // im2col matrix_row = channel_col
    int channel_col = in_ch / group * ksize_h * ksize_w;

    int32_t m = out_ch / group;
    int32_t k = channel_col;
    int32_t n = out_height * out_width;
    int32_t k4 = (k % 4 != 0) ? ((k / 4 + 1) * 4) : k;

    int8_t *im2col_data = (int8_t *)shl_mem_alloc(k * n * sizeof(int8_t));
    int8_t *pb_reorder = (int8_t *)shl_mem_alloc(k4 * n * sizeof(int8_t));

    int32_t *multiplier = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));

    int j = 0;
    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            // im2col
            for (int c = 0; c < channel_col; ++c) {
                int w_offset = c % ksize_w;
                int h_offset = c / ksize_w % ksize_h;
                int c_im = c / ksize_h / ksize_w;
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        int im_row = h_offset + h * stride_h;
                        int im_col = w_offset + w * stride_w;
                        int col_index =
                            (c * out_height + h) * out_width + w;  // [channel_col, out_h, out_w]
                        im_row = im_row - params->pad_top;
                        im_col = im_col - params->pad_left;
                        if (im_row < 0 || im_col < 0 || im_row >= in_height || im_col >= in_width) {
                            im2col_data[col_index] = input->qinfo->zero_point;
                        } else {
                            im2col_data[col_index] =
                                input_data[(c_im * input->dim[2] + im_row) * input->dim[3] +
                                           im_col];
                        }
                    }
                }
            }

            int8_t *pa = kernel_data + g * m * k4;
            int8_t *pb = pb_reorder;
            int8_t *pc = output_data;

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

            // pack
            shl_rvv_reorder_input_z8_int8(im2col_data, pb, k, n, n);
            // GEMM
            shl_rvv_gemm_8x8_int8(pc, pa, pb, bias_data + g * m, m, k4, n, n,
                                  output->qinfo->zero_point, multiplier, shift);

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
#endif
