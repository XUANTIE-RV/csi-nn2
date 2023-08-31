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

void shl_rvv_conv1d_gemm_reorder_kernel_int8(struct csinn_tensor *kernel,
                                             struct csinn_conv1d_params *params)
{
    int8_t *kernel_data = (int8_t *)kernel->data;
    int group = params->group;

    int m = kernel->dim[0] / group;  // out_ch
    int k = kernel->dim[1] * kernel->dim[2];

    int8_t *pa_reorder = (int8_t *)shl_mem_alloc(group * m * k * sizeof(int8_t));
    for (int g = 0; g < group; g++) {
        shl_rvv_reorder_kernel_n4_int8_v128(kernel_data + g * m * k, pa_reorder + g * m * k, m, k,
                                            k);
    }
    memcpy(kernel_data, pa_reorder, group * m * k * sizeof(int8_t));
    shl_mem_free(pa_reorder);
}

int shl_rvv_conv1d_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv1d_params *params)
{
    if (input->layout == CSINN_LAYOUT_NC1WC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_int8(input);
    }

    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)kernel->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];  // assert(batch == 1);
    int32_t in_ch = input->dim[1];
    int32_t out_ch = kernel->dim[0];
    int32_t out_h = output->dim[2];

    int32_t m = out_ch / group;
    int32_t k = in_ch / group;
    int32_t n = out_h;

    int8_t *pb_reorder = (int8_t *)shl_mem_alloc(k * n * sizeof(int8_t));
    int32_t *multiplier = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));

    int j = 0;
    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
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
            int8_t *pa = kernel_data + g * m * k;
            shl_rvv_reorder_input_z16_int8_v128(input_data, pb, k, n, n);
            shl_rvv_gemm_4x16_int8_v128(pc, pa, pb, bias_data + g * m, m, k, n, n,
                                        output->qinfo->zero_point, multiplier, shift);
            input_data += k * n;
            output_data += m * n;
        }
    }
    shl_mem_free(pb_reorder);
    shl_mem_free(multiplier);
    shl_mem_free(shift);
    return CSINN_TRUE;
}
