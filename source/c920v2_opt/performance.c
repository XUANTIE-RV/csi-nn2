/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

#include "c920v2/c920v2.h"
#include "c920v2/perf.h"

static struct shl_function_map shl_c920v2_kernel_map[] = {
    {shl_c920v2_conv_im2col_gemm_packn_fp32, "shl_c920v2_conv_im2col_gemm_packn_fp32"},
    {shl_c920v2_conv_im2col_gemm_packn_fp16, "shl_c920v2_conv_im2col_gemm_packn_fp16"},
    {shl_c920v2_conv_im2col_gemm_pack1ton_fp32, "shl_c920v2_conv_im2col_gemm_pack1ton_fp32"},
    {shl_c920v2_conv_im2col_gemm_pack1ton_fp16, "shl_c920v2_conv_im2col_gemm_pack1ton_fp16"},
    {shl_c920v2_conv_im2col_gemm_packnto1_fp32, "shl_c920v2_conv_im2col_gemm_packnto1_fp32"},
    {shl_c920v2_conv_im2col_gemm_packnto1_fp16, "shl_c920v2_conv_im2col_gemm_packnto1_fp16"},
    {shl_c920v2_conv1x1s1_gemm_packn_fp32, "shl_c920v2_conv1x1s1_gemm_packn_fp32"},
    {shl_c920v2_conv1x1s1_gemm_packn_fp16, "shl_c920v2_conv1x1s1_gemm_packn_fp16"},
    {shl_c920v2_conv1x1s1_gemm_packn_int8, "shl_c920v2_conv1x1s1_gemm_packn_int8"},
    {shl_c920v2_conv1x1s1_gemm_pack1ton_fp32, "shl_c920v2_conv1x1s1_gemm_pack1ton_fp32"},
    {shl_c920v2_conv1x1s1_gemm_pack1ton_fp16, "shl_c920v2_conv1x1s1_gemm_pack1ton_fp16"},
    {shl_c920v2_conv1x1s1_gemm_pack1ton_int8, "shl_c920v2_conv1x1s1_gemm_pack1ton_int8"},
    {shl_c920v2_conv1x1s1_gemm_packnto1_fp32, "shl_c920v2_conv1x1s1_gemm_packnto1_fp32"},
    {shl_c920v2_conv1x1s1_gemm_packnto1_fp16, "shl_c920v2_conv1x1s1_gemm_packnto1_fp16"},
    {shl_c920v2_conv1x1s1_gemm_packnto1_int8, "shl_c920v2_conv1x1s1_gemm_packnto1_int8"},
    {shl_c920v2_ncxhwx_gemm_12xpack2n_fp32, "shl_c920v2_ncxhwx_gemm_12xpack2n_fp32"},
    {shl_c920v2_ncxhwx_gemm_12xpack2n_fp16, "shl_c920v2_ncxhwx_gemm_12xpack2n_fp16"},
    {shl_c920v2_ncxhwx_gemm_12xpackn_int8_dot, "shl_c920v2_ncxhwx_gemm_12xpackn_int8_dot"},
    {shl_c920v2_ncxhwx_gemm_4xpack2n_int8, "shl_c920v2_ncxhwx_gemm_4xpack2n_int8"},
    {NULL, NULL}};

char *shl_rvv_get_kernel_name(void *exec);

char *shl_c920v2_get_kernel_name(void *exec)
{
    char *name = shl_find_function_name(shl_c920v2_kernel_map, exec);
    if (name == NULL) {
        name = shl_rvv_get_kernel_name(exec);
    }
    return name;
}

int shl_c920v2_conv2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                           struct csinn_conv2d_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c920v2_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}