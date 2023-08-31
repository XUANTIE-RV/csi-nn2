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
    const int vlen = csrr_vlenb() * 8;
    if (vlen == 128) {
        return shl_rvv_common_conv_gemm_fp16(input, output, kernel, bias, params,
                                             shl_c908_reorder_input_z24_fp16,
                                             shl_c908_gemm_8x24_fp16);
    } else if (vlen >= 256) {
        return shl_rvv_common_conv_gemm_fp16(input, output, kernel, bias, params,
                                             shl_c908_reorder_input_z32_fp16_v256,
                                             shl_c908_gemm_8x32_fp16_v256);
    }
}
