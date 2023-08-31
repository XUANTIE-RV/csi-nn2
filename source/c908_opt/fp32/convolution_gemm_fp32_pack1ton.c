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
void shl_c908_conv_im2col_gemm_reorder_kernel_pack1ton_fp32(struct csinn_tensor *kernel,
                                                            struct csinn_conv2d_params *params)
{
    shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp32(kernel, params);
}

int shl_c908_conv_im2col_gemm_pack1ton_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params)
{
    return shl_rvv_common_conv_gemm_pack1ton_fp32(input, output, kernel, bias, params,
                                                  shl_rvv_reorder_input_z12_pack1ton_fp32,
                                                  shl_c908_ncxhwx_gemm_12xpack2n_fp32);
}
