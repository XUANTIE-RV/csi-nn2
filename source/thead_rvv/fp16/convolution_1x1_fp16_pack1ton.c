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
void shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp16(struct csinn_tensor *kernel,
                                                         struct csinn_conv2d_params *params)
{
    shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp16(kernel, params);
}

int shl_rvv_conv1x1s1_gemm_pack1ton_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];
    int32_t out_c = kernel->dim[0];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int32_t m = out_c / group;
    int32_t k = in_c / group;
    int32_t n = out_h * out_w;

    __fp16 *pb_reorder = (__fp16 *)shl_mem_alloc(k * n * sizeof(__fp16));
    __fp16 *input_ncxhwx = (__fp16 *)shl_mem_alloc(k * n * sizeof(__fp16));

    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            __fp16 *kernel_ptr = kernel_data + g * m * k;
            __fp16 *in_ptr = pb_reorder;
            __fp16 *out_ptr = output_data;
            __fp16 *bias_ptr = bias_data ? (bias_data + g * m) : NULL;

            shl_rvv_reorder_input_pack1ton_fp16(input_data, input_ncxhwx, k, out_h, out_w);

            // reorder(pack)
            shl_rvv_reorder_input_z12_pack1ton_fp16(input_ncxhwx, in_ptr, k, 1, n, n);

            // gemm
            shl_rvv_ncxhwx_gemm_12xpack2n_fp16(out_ptr, kernel_ptr, in_ptr, bias_ptr, m, k, n, n);

            input_data += k * n;
            output_data += m * n;
        }
    }
    shl_mem_free(pb_reorder);
    shl_mem_free(input_ncxhwx);
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}
