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

#include "c906/c906.h"

int shl_c906_conv1d_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv1d_params *params)
{
    int32_t out_c = kernel->dim[0];
    int32_t in_c = kernel->dim[1];
    int32_t in_w = input->dim[2];
    int32_t kernel_w = kernel->dim[2];
    int32_t stride_w = params->stride_width;
    int32_t dilation_w = params->dilation_width;
    struct csinn_callback *cb = params->base.cb;

    if (kernel_w == 1 && stride_w == 1 && dilation_w == 1) {
        shl_c906_conv1x1s1_sgemm_transform_kernel(kernel, (struct csinn_conv2d_params *)params);
        cb->exec = shl_c906_conv1x1s1_sgemm;
    } else {
        cb->exec = shl_ref_conv1d_f32;
    }
    return CSINN_TRUE;
}
