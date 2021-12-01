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

/* CSI-NN2 version 1.8.x */

#include "sgemm.h"

/*
    pack kernel_data inplace, means the origin kernel_data be destoried.
    The reason to do this is that the packaging process must not consume more memory.
*/
static void conv_im2col_sgemm_transform_kernel(struct csi_tensor *kernel)
{
    float *kernel_data = (float *)kernel->data;
    int m = kernel->dim[0];         // m = out_channel
    int k = kernel->dim[1] * kernel->dim[2] * kernel->dim[3];

    float *pa_reorder = (float *)malloc(m * k * sizeof(float));
    reorder_a(kernel_data, pa_reorder, m, k, k);
    memcpy(kernel_data, pa_reorder, m * k * sizeof(float));
    free(pa_reorder);
}

static int conv_im2col_sgemm(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct csi_tensor *kernel,
                             struct csi_tensor *bias,
                             struct conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    int32_t batch = input->dim[0];
    int32_t in_channel = input->dim[1];
    int32_t in_height = input->dim[2];
    int32_t in_width = input->dim[3];
    int32_t out_channel = kernel->dim[0];
    int32_t ksize_h = kernel->dim[2];
    int32_t ksize_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t pad_left = params->pad_left;
    int32_t pad_right = params->pad_right;
    int32_t pad_top = params->pad_top;
    int32_t pad_down = params->pad_down;

    // im2col
    // im2col matrix_col = out_height * out_width
    int out_height = (in_height + params->pad_top + params->pad_down - ksize_h) / stride_h + 1;
    int out_width  = (in_width + params->pad_left + params->pad_right - ksize_w) / stride_w + 1;
    if(out_height != output->dim[2] || out_width != output->dim[3]) {
        printf("output dim don't match.\n");
        return CSINN_FALSE;
    }
    // im2col matrix_row = channel_col
    int channel_col = in_channel * ksize_h * ksize_w;

    float *im2col_data = (float *)malloc(batch * out_height * out_width * channel_col * sizeof(float));

    for(int b = 0; b < batch; ++b) {
        for(int c = 0; c < channel_col; ++c) {
            int w_offset = c % ksize_w;
            int h_offset = c / ksize_w % ksize_h;
            int c_im = c / ksize_h / ksize_w;
            for(int h = 0; h < out_height; ++h) {
                for(int w = 0; w < out_width; ++w) {
                    int im_row = h_offset + h * stride_h;
                    int im_col = w_offset + w * stride_w;
                    // int col_index = ((c * batch + b) * out_height + h) * out_width + w;      // [channel_col, batch, out_h, out_w]
                    int col_index = ((b * channel_col + c) * out_height + h) * out_width + w;       // [batch, channel_col, out_h, out_w]
                    im_row = im_row - params->pad_top;
                    im_col = im_col - params->pad_left;
                    if(im_row < 0 || im_col < 0 || im_row >= in_height || im_col >= in_width) {
                        im2col_data[col_index] = 0.0f;
                    } else {
                        im2col_data[col_index] = input_data[((b * input->dim[1] + c_im) * input->dim[2] + im_row) * input->dim[3] + im_col];
                    }
                }
            }
        }
    }

    int32_t m = out_channel;
    int32_t k = channel_col;
    int32_t n = out_height * out_width;

    float* pb_reorder = (float *)malloc(k * n * sizeof(float));
    const float *pa = kernel_data;

    for(int i = 0; i < batch; i++) {
        // pack
        reorder_b(im2col_data + i * k * n, pb_reorder, k, n, n);

        // GEMM
        const float *pb = pb_reorder;
        float *pc = output_data + i * m * n;    // bump output_point for each batch

        sgemm_kernel_f32(pc, pa, pb, m, k, n, n, bias_data);
    }

    free(pb_reorder);
    free(im2col_data);
    return CSINN_TRUE;
}
