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

#include "shl_thead_rvv.h"
#ifdef SHL_USE_DOT_INT4
void shl_rvv_conv_im2col_gemm_reorder_kernel_int4(struct csinn_tensor *kernel,
                                                  struct csinn_conv2d_params *params)
{
    int8_t *kernel_data = (int8_t *)kernel->data;
    int group = params->group;

    int n = kernel->dim[0] / group;  // m = out_ch / group
    int k = kernel->dim[1] * kernel->dim[2] * kernel->dim[3];

    int k_2 = (((k - 1) & -2) + 2) >> 1;
    int k4 = ((k_2 - 1) & -4) + 4;  // align of 4 for int8

    params->conv_extra.kernel_tm->data = (int8_t *)shl_mem_alloc(group * n * k4 * sizeof(int8_t));
    int8_t *pa_reorder = (int8_t *)params->conv_extra.kernel_tm->data;

    for (int g = 0; g < group; g++) {
        shl_rvv_reorder_kernel_n8_int8_dot(kernel_data + g * n * k_2, pa_reorder + g * n * k4, n,
                                           k_2, k_2);
    }
    // FIXME: free params->conv_extra.kernel_tm->data
}

int shl_rvv_conv_im2col_gemm_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)params->conv_extra.kernel_tm->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];
    int32_t in_height = input->dim[1];
    int32_t in_width = input->dim[2];
    int32_t in_ch = input->dim[3];
    int32_t out_ch = kernel->dim[0];
    int32_t out_height = output->dim[1];
    int32_t out_width = output->dim[2];
    int32_t ksize_h = kernel->dim[1];
    int32_t ksize_w = kernel->dim[2];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;
    int32_t pad_left = params->pad_left;
    int32_t pad_top = params->pad_top;

    // im2col matrix_col = out_height * out_width
    // im2col matrix_row = channel_col
    int channel_col = in_ch / group * ksize_h * ksize_w;

    int32_t m = out_height * out_width;
    int32_t k_2 = (channel_col - 1) / 2 + 1;
    int32_t n = out_ch / group;
    int32_t k4 = ((k_2 - 1) & -4) + 4;

    int32_t *multiplier = (int32_t *)shl_mem_alloc(n * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(n * sizeof(int32_t));

    int8_t *im2col_data = (int8_t *)shl_mem_alloc(m * k_2 * sizeof(int8_t));
    int8_t *pa_reorder = (int8_t *)shl_mem_alloc(m * k4 * sizeof(int8_t));

    int8_t *im2col_shadow = NULL;
    int8_t pad_value = 0;

    int j = 0;
    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            // im2col
            if (in_ch & 1) {
                int8_t *buffer_int4_to_int8 =
                    (int8_t *)shl_mem_alloc(in_height * in_width * in_ch * sizeof(int8_t));
                shl_rvv_int4_to_int8(input_data, buffer_int4_to_int8, in_height * in_width * in_ch);
                int8_t *buffer_im2col = (int8_t *)shl_mem_alloc(m * channel_col * sizeof(int8_t));
                im2col_shadow = buffer_im2col;
                pad_value = input->qinfo->zero_point & 0x0f;

                for (int i_out_h = 0; i_out_h < out_height; i_out_h++) {
                    for (int i_out_w = 0; i_out_w < out_width; i_out_w++) {
                        int ker_start_h = i_out_h * stride_h - pad_top;
                        int ker_start_w = i_out_w * stride_w - pad_left;
                        for (int i_ker_h = ker_start_h; i_ker_h < ker_start_h + ksize_h;
                             i_ker_h++) {
                            for (int i_ker_w = ker_start_w; i_ker_w < ker_start_w + ksize_w;
                                 i_ker_w++) {
                                if (i_ker_h < 0 || i_ker_h >= in_height || i_ker_w < 0 ||
                                    i_ker_w >= in_width) {
                                    memset(im2col_shadow, pad_value, in_ch * sizeof(int8_t));
                                } else {
                                    memcpy(im2col_shadow,
                                           buffer_int4_to_int8 +
                                               in_ch * (i_ker_h * in_width + i_ker_w),
                                           in_ch * sizeof(int8_t));
                                }
                                im2col_shadow += in_ch;
                            }
                        }
                    }
                }
                for (int k = 0; k < m; k++) {
                    shl_rvv_int8_to_int4(buffer_im2col + k * channel_col, im2col_data + k * k_2,
                                         channel_col);
                }
                shl_mem_free(buffer_int4_to_int8);
                shl_mem_free(buffer_im2col);

            } else {
                im2col_shadow = im2col_data;
                pad_value = (input->qinfo->zero_point << 4) | (input->qinfo->zero_point & 0x0f);

                for (int i_out_h = 0; i_out_h < out_height; i_out_h++) {
                    for (int i_out_w = 0; i_out_w < out_width; i_out_w++) {
                        int ker_start_h = i_out_h * stride_h - pad_top;
                        int ker_start_w = i_out_w * stride_w - pad_left;
                        for (int i_ker_h = ker_start_h; i_ker_h < ker_start_h + ksize_h;
                             i_ker_h++) {
                            for (int i_ker_w = ker_start_w; i_ker_w < ker_start_w + ksize_w;
                                 i_ker_w++) {
                                if (i_ker_h < 0 || i_ker_h >= in_height || i_ker_w < 0 ||
                                    i_ker_w >= in_width) {
                                    memset(im2col_shadow, pad_value, in_ch / 2 * sizeof(int8_t));
                                } else {
                                    memcpy(im2col_shadow,
                                           input_data + (i_ker_h * in_width + i_ker_w) * in_ch / 2,
                                           in_ch / 2 * sizeof(int8_t));
                                }
                                im2col_shadow += in_ch / 2;
                            }
                        }
                    }
                }
            }

            int8_t *pa = pa_reorder;
            int8_t *pb = kernel_data + g * n * k4;
            int8_t *pc = output_data;

            if (kernel->quant_channel > 1) {
                for (int c = 0; c < n; c++, j++) {
                    multiplier[c] = kernel->qinfo[j].multiplier;
                    shift[c] = kernel->qinfo[j].shift;
                }
            } else if (kernel->quant_channel == 1) {
                for (int c = 0; c < n; c++) {
                    multiplier[c] = kernel->qinfo[0].multiplier;
                    shift[c] = kernel->qinfo[0].shift;
                }
            }

            // pack
            shl_rvv_reorder_input_n8_int4_dot(im2col_data, pa, m, k_2, k_2);
            // GEMM
            shl_rvv_gemm_8x8_int4_dot(pc, pa, pb, m, k4, n, n / 2, bias_data + g * n,
                                      output->qinfo->zero_point, multiplier, shift);

            input_data += in_ch / group * in_height * in_width / 2;
            output_data += m * n / 2;
        }
    }
    shl_mem_free(pa_reorder);
    shl_mem_free(im2col_data);
    shl_mem_free(multiplier);
    shl_mem_free(shift);
    return CSINN_TRUE;
}
#endif
