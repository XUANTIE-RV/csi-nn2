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
#ifdef SHL_USE_DOT_INT4
/*************************************************************
 * packn = vlenb / sizeof(int8_t) / 2

 ************************************************************/
static void im2col_gemm_reorder_kernel_packn_per_group_int4(int8_t *src, int8_t *dst, int out_c,
                                                            int in_c, int maxk)
{
}

void shl_rvv_conv_im2col_gemm_reorder_kernel_packn_int4(struct csinn_tensor *kernel,
                                                        struct csinn_conv2d_params *params)
{
}

int shl_rvv_conv_im2col_gemm_packn_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                        struct csinn_conv2d_params *params)
{
    return CSINN_TRUE;
}
#endif
