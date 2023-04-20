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

#include "shl_thead_rvv.h"

/*************************************************************************************
 * reorder kernel_data inplace, means the origin kernel_data be destoried.
 * The reason to do this is that the packaging process must not consume more memory.
 **************************************************************************************/
void shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params)
{
    float *kernel_data = (float *)kernel->data;
    int group = params->group;

    int m = kernel->dim[0] / group;  // m = out_ch / group
    int k = kernel->dim[1] * kernel->dim[2] * kernel->dim[3];

    float *pa_reorder = (float *)shl_mem_alloc(group * m * k * sizeof(float));
    for (int g = 0; g < group; g++) {
        shl_rvv_reorder_kernel_n8_fp32(kernel_data + g * m * k, pa_reorder + g * m * k, m, k, k);
    }
    memcpy(kernel_data, pa_reorder, group * m * k * sizeof(float));
    shl_mem_free(pa_reorder);
}

int shl_rvv_conv_im2col_gemm_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

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

    float *im2col_data = (float *)shl_mem_alloc(k * n * sizeof(float));
    float *pb_reorder = (float *)shl_mem_alloc(k * n * sizeof(float));

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
                            im2col_data[col_index] = 0.0f;
                        } else {
                            im2col_data[col_index] =
                                input_data[(c_im * input->dim[2] + im_row) * input->dim[3] +
                                           im_col];
                        }
                    }
                }
            }

            float *pa = kernel_data + g * m * k;
            float *pb = pb_reorder;
            float *pc = output_data;

            // pack
            shl_rvv_reorder_input_z8_fp32(im2col_data, pb, k, n, n);
            // GEMM
            shl_rvv_gemm_8x8_fp32(pc, pa, pb, bias_data + g * m, m, k, n, n);
            input_data += in_ch / group * in_height * in_width;
            output_data += m * n;
        }
    }
    shl_mem_free(pb_reorder);
    shl_mem_free(im2col_data);
    return CSINN_TRUE;
}
