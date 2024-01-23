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

#include "c920/c920.h"
#include "c920/perf.h"

static struct shl_function_map shl_c920_kernel_map[] = {
    {shl_c920_conv_im2col_gemm_packn_fp32, "shl_c920_conv_im2col_gemm_packn_fp32"},
    {shl_c920_conv_im2col_gemm_packn_fp16, "shl_c920_conv_im2col_gemm_packn_fp16"},
    {shl_c920_conv1x1s1_gemm_packn_fp32, "shl_c920_conv1x1s1_gemm_packn_fp32"},
    {shl_c920_conv1x1s1_gemm_packn_fp16, "shl_c920_conv1x1s1_gemm_packn_fp16"},
    {shl_c920_wg_b4f3s1_packn_fp32, "shl_c920_wg_b4f3s1_packn_fp32"},
    {shl_c920_wg_b6f3s1_packn_fp32, "shl_c920_wg_b6f3s1_packn_fp32"},
    {shl_c920_wg_b4f3s1_packn_fp16, "shl_c920_wg_b4f3s1_packn_fp16"},
    {shl_c920_wg_b6f3s1_packn_fp16, "shl_c920_wg_b6f3s1_packn_fp16"},
    {shl_c920_ncxhwx_gemm_8xpack2n_fp32, "shl_c920_ncxhwx_gemm_8xpack2n_fp32"},
    {shl_c920_ncxhwx_gemm_8xpack2n_fp16, "shl_c920_ncxhwx_gemm_8xpack2n_fp16"},
    {shl_c920_reorder_a_block_8xk_fp32, "shl_c920_reorder_a_block_8xk_fp32"},
    {shl_c920_gemm_block_8xpack2n_fp32, "shl_c920_gemm_block_8xpack2n_fp32"},
    {shl_c920_reorder_a_block_8xk_fp16, "shl_c920_reorder_a_block_8xk_fp16"},
    {shl_c920_gemm_block_8xpack2n_fp16, "shl_c920_gemm_block_8xpack2n_fp16"},
    {shl_c920_gemm_a0b1_8xpack2n_fp32, "shl_c920_gemm_a0b1_8xpack2n_fp32"},
    {shl_c920_gemm_a0b1_8xpack2n_fp16, "shl_c920_gemm_a0b1_8xpack2n_fp16"},
    {shl_c920_fullyconnected_gemm_fp32, "shl_c920_fullyconnected_gemm_fp32"},
    {shl_c920_fullyconnected_gemm_fp16, "shl_c920_fullyconnected_gemm_fp16"},
    // {shl_c920_matmul_fp32, "shl_c920_matmul_fp32"},
    // {shl_c920_matmul_fp16, "shl_c920_matmul_fp16"},
    // {shl_c920_matmul_fp16_w_int8, "shl_c920_matmul_fp16_w_int8"},
    {NULL, NULL}};

char *shl_rvv_get_kernel_name(void *exec);

char *shl_c920_get_kernel_name(void *exec)
{
    char *name = shl_find_function_name(shl_c920_kernel_map, exec);
    if (name == NULL) {
        name = shl_rvv_get_kernel_name(exec);
    }
    return name;
}

int shl_c920_conv2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv2d_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c920_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c920_fullyconnected_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *weights, struct csinn_tensor *bias,
                                 struct csinn_fc_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c920_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c920_matmul_perf(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                         struct csinn_tensor *output, struct csinn_matmul_params *params,
                         struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c920_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}