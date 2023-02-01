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

/* SHL version 2.1.x */

#include "shl_thead_rvv.h"

void shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_int8(struct csinn_tensor *kernel,
                                                         struct csinn_conv2d_params *params)
{
    shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_int8(kernel, params);
}

int shl_rvv_conv1x1s1_gemm_packnto1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
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
    int32_t out_ch = kernel->dim[0];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int32_t m = out_ch / group;
    int32_t k = in_ch / group;
    int32_t n = out_h * out_w;

    int8_t *pb_reorder = (int8_t *)shl_mem_alloc(k * n * sizeof(int8_t));
    int32_t *multiplier = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));

    int8_t *output_ncxhwx = (int8_t *)shl_mem_alloc(m * n * sizeof(int8_t));

    for (int i = 0; i < batch; i++) {
        for (int g = 0, j = 0; g < group; g++) {
            int8_t *kernel_ptr = kernel_data + g * m * k;
            int8_t *in_ptr = pb_reorder;
            int8_t *out_ptr = output_data;
            int32_t *bias_ptr = bias_data + g * m;  // bias_data != NULL with fusing zp to bias

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

#ifdef SHL_USE_DOT_INT8
            shl_rvv_reorder_input_z12_packn_int8_dot(input_data, pb_reorder, k, n, n);
            shl_rvv_ncxhwx_gemm_12xpackn_int8_dot(output_ncxhwx, kernel_ptr, in_ptr, bias_ptr, m, k,
                                                  n, n, output->qinfo->zero_point, multiplier,
                                                  shift);
#else
            shl_rvv_reorder_input_z4_packn_int8(input_data, pb_reorder, k, n, n);
            shl_rvv_ncxhwx_gemm_4xpack2n_int8(output_ncxhwx, kernel_ptr, in_ptr, bias_ptr, m, k, n,
                                              n, output->qinfo->zero_point, multiplier, shift);
#endif  // SHL_USE_DOT_INT8
            shl_rvv_reorder_input_packnto1_int8(output_ncxhwx, output_data, m, out_h, out_w);

            input_data += k * n;
            output_data += m * n;
        }
    }
    shl_mem_free(pb_reorder);
    shl_mem_free(multiplier);
    shl_mem_free(shift);
    shl_mem_free(output_ncxhwx);
    return CSINN_TRUE;
}
