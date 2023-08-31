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

/*************************************************************************************
 * reorder kernel_data inplace, means the origin kernel_data be destoried.
 * The reason to do this is that the packaging process must not consume more memory.
 **************************************************************************************/
void shl_rvv_conv1d_im2col_gemm_reorder_kernel_fp32(struct csinn_tensor *kernel,
                                                    struct csinn_conv1d_params *params)
{
    float *kernel_data = (float *)kernel->data;
    int group = params->group;

    int m = kernel->dim[0] / group;
    int k = kernel->dim[1] * kernel->dim[2];

    float *pa_reorder = (float *)shl_mem_alloc(group * m * k * sizeof(float));
    for (int g = 0; g < group; g++) {
        shl_rvv_reorder_kernel_n8_fp32(kernel_data + g * m * k, pa_reorder + g * m * k, m, k, k);
    }
    memcpy(kernel_data, pa_reorder, group * m * k * sizeof(float));
    shl_mem_free(pa_reorder);
}

int shl_rvv_conv1d_im2col_gemm_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                    struct csinn_conv1d_params *params)
{
    if (input->layout == CSINN_LAYOUT_NC1WC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(input);
    }
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];
    int32_t in_ch = input->dim[1];
    int32_t in_width = input->dim[2];

    int32_t out_ch = kernel->dim[0];
    int32_t out_width = output->dim[2];

    int32_t kernel_w = kernel->dim[2];
    int32_t stride_w = params->stride_width;
    int32_t pad_left = params->pad_left;
    int32_t dilation_w = params->dilation_width;

    int32_t m = out_ch / group;
    int32_t k = in_ch / group * kernel_w;
    int32_t n = out_width;

    float *im2col_data = (float *)shl_mem_alloc(k * n * sizeof(float));
    float *pb_reorder = (float *)shl_mem_alloc(k * n * sizeof(float));
    const int vlen = csrr_vlenb() * 8;

    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            // im2col
            float *data_col = im2col_data;
            float *channel_data = input_data;
            for (int c = 0; c < in_ch / group; c++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int in_col = -pad_left + kw * dilation_w;
                    for (int ow1 = 0; ow1 < out_width; ow1++) {
                        if (in_col < in_width && in_col >= 0) {
                            *data_col++ = channel_data[in_col];
                        } else {
                            *data_col++ = 0.0f;
                        }
                        in_col += stride_w;
                    }
                }
                channel_data += in_width;
            }
            float *pa = kernel_data + g * m * k;
            float *pb = pb_reorder;
            float *pc = output_data;
            if (vlen == 128) {
                // pack
                shl_rvv_reorder_input_z8_fp32(im2col_data, pb, k, n, n);
                // GEMM
                shl_rvv_gemm_8x8_fp32(pc, pa, pb, bias_data + g * m, m, k, n, n);
            } else if (vlen >= 256) {
                shl_rvv256_reorder_input_z16_fp32(im2col_data, pb, k, n, n);
                shl_rvv256_gemm_8x16_fp32(pc, pa, pb, bias_data + g * m, m, k, n, n);
            }
            input_data += in_ch / group * in_width;
            output_data += m * n;
        }
    }
    shl_mem_free(pb_reorder);
    shl_mem_free(im2col_data);
    return CSINN_TRUE;
}
