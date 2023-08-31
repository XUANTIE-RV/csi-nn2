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

#include "c908/c908.h"

void shl_c908_conv_im2col_gemm_reorder_kernel_int8(struct csinn_tensor *kernel,
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
        shl_c908_reorder_kernel_n8_int8_dot(kernel_data + g * m * k, pa_reorder + g * m * k4, m, k,
                                            k);
    }
    // FIXME: free params->conv_extra.kernel_tm->data
    // memcpy(kernel_data, pa_reorder, group * m * k * sizeof(__fp16));
    // shl_mem_free(pa_reorder);
}

int shl_c908_conv_im2col_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params)
{
#ifdef SHL_USE_DOT_INT8
    const int vlen = csrr_vlenb() * 8;
    if (vlen == 128) {
        return shl_rvv_common_conv_gemm_int8(input, output, kernel, bias, params,
                                             shl_c908_reorder_input_z8_int8_dot,
                                             shl_c908_gemm_8x8_int8_dot);
    } else if (vlen >= 256) {
        return shl_rvv_common_conv_gemm_int8(input, output, kernel, bias, params,
                                             shl_c908_reorder_input_z16_int8_v256_dot,
                                             shl_c908_gemm_8x16_int8_v256_dot);
    }
#endif  // SHL_USE_DOT_INT8
}
