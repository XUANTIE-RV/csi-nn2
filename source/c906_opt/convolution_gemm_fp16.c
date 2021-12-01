/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

/* CSI-NN2 version 1.10.x */

#include "csi_c906.h"

/*
    pack kernel_data inplace, means the origin kernel_data be destoried.
    The reason to do this is that the packaging process must not consume more memory.
*/
void csi_c906_conv_im2col_sgemm_transform_kernel_fp16(struct csi_tensor *kernel,
                                                      struct conv2d_params *params)
{
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    int group = params->group;

    int m = kernel->dim[0] / group;         // m = out_ch / group
    int k = kernel->dim[1] * kernel->dim[2] * kernel->dim[3];

    __fp16 *pa_reorder = (__fp16 *)csi_mem_alloc(group * m * k * sizeof(__fp16));
    for (int g = 0; g < group; g++) {
        csi_c906_reorder_kernel_fp16(kernel_data + g * m * k, pa_reorder + g * m * k, m, k, k);
    }
    memcpy(kernel_data, pa_reorder, group * m * k * sizeof(__fp16));
    csi_mem_free(pa_reorder);
}

int csi_c906_conv_im2col_sgemm_fp16(struct csi_tensor *input,
                                    struct csi_tensor *output,
                                    struct csi_tensor *kernel,
                                    struct csi_tensor *bias,
                                    struct conv2d_params *params)
{
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

    // im2col matrix_col = out_height * out_width
    // im2col matrix_row = channel_col
    int channel_col = in_ch / group * ksize_h * ksize_w;

    int32_t m = out_ch / group;
    int32_t k = channel_col;
    int32_t n = out_height * out_width;

    __fp16 *im2col_data = (__fp16 *)csi_mem_alloc(k * n * sizeof(__fp16));
    __fp16* pb_reorder = (__fp16 *)csi_mem_alloc(k * n * sizeof(__fp16));

    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {

            // im2col
            for(int c = 0; c < channel_col; ++c) {
                int w_offset = c % ksize_w;
                int h_offset = c / ksize_w % ksize_h;
                int c_im = c / ksize_h / ksize_w;
                for(int h = 0; h < out_height; ++h) {
                    for(int w = 0; w < out_width; ++w) {
                        int im_row = h_offset + h * stride_h;
                        int im_col = w_offset + w * stride_w;
                        int col_index = (c * out_height + h) * out_width + w;       // [channel_col, out_h, out_w]
                        im_row = im_row - params->pad_top;
                        im_col = im_col - params->pad_left;
                        if(im_row < 0 || im_col < 0 || im_row >= in_height || im_col >= in_width) {
                            im2col_data[col_index] = 0.0f;
                        } else {
                            im2col_data[col_index] = input_data[(c_im * input->dim[2] + im_row) * input->dim[3] + im_col];
                        }
                    }
                }
            }

            __fp16 *pa = kernel_data + g * m * k;
            __fp16 *pb = pb_reorder;
            __fp16 *pc = output_data;

            // pack
            csi_c906_reorder_input_fp16(im2col_data, pb, k, n, n);
            // GEMM
            csi_c906_sgemm_kernel_fp16(pc, pa, pb, m, k, n, n, bias_data + g * m);
            input_data += in_ch / group * in_height * in_width;
            output_data += m * n;
        }
    }
    csi_mem_free(pb_reorder);
    csi_mem_free(im2col_data);
    return CSINN_TRUE;
}
