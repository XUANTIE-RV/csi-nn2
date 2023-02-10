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

#include "shl_thead_rvm.h"

int shl_rvm_avgpool2d_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params)
{
    int32_t in_h = input->dim[1];
    int32_t in_w = input->dim[2];
    int32_t kernel_h = params->filter_height;
    int32_t kernel_w = params->filter_width;

    struct csinn_callback *cb = params->base.cb;
    if (in_h == kernel_h && in_w == kernel_w) {
        cb->exec = shl_rvv_global_avgpool2d_nhwc_fp16;
        return CSINN_TRUE;
    }
    cb->exec = shl_rvv_avgpool_nhwc_fp16;
    return CSINN_TRUE;
}

int shl_rvm_avgpool2d_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params)
{
    int32_t in_h = input->dim[1];
    int32_t in_w = input->dim[2];
    int32_t kernel_h = params->filter_height;
    int32_t kernel_w = params->filter_width;

    struct csinn_callback *cb = params->base.cb;
    if (in_h == kernel_h && in_w == kernel_w) {
        cb->exec = shl_rvv_global_avgpool2d_nhwc_int8;
        return CSINN_TRUE;
    }
    cb->exec = shl_rvv_avgpool_nhwc_int8;
    return CSINN_TRUE;
}

int shl_rvm_global_avgpool2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    if (input->dtype == CSINN_DTYPE_FLOAT32) {
        cb->exec = shl_rvv_global_avgpool2d_nhwc_fp32;
    } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
        cb->exec = shl_rvv_global_avgpool2d_nhwc_fp16;
    } else if (input->dtype == CSINN_DTYPE_INT8) {
        cb->exec = shl_rvv_global_avgpool2d_nhwc_int8;
    }
    return CSINN_TRUE;
}