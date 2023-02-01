/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

#include "shl_ref.h"

int shl_ref_conv2d_relu6_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_tensor *kernel, struct csinn_tensor *bias,
                               struct csinn_conv2d_params *params)
{
    shl_ref_conv2d_quant(input, output, kernel, bias, params);
    struct csinn_relu_params *rp = shl_mem_alloc(sizeof(struct csinn_relu_params));
    memcpy(&(rp->base), &(params->base), sizeof(struct csinn_params_base));
    csinn_relu6_init(output, output, rp);
    csinn_relu6(output, output, rp);
    return CSINN_TRUE;
}

int shl_ref_depthwise_conv2d_relu6_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params)
{
    shl_ref_depthwise_conv2d_quant(input, output, kernel, bias, params);
    struct csinn_relu_params *rp = shl_mem_alloc(sizeof(struct csinn_relu_params));
    memcpy(&(rp->base), &(params->base), sizeof(struct csinn_params_base));
    csinn_relu6_init(output, output, rp);
    csinn_relu6(output, output, rp);
    return CSINN_TRUE;
}

int shl_ref_group_conv2d_relu6_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                     struct csinn_conv2d_params *params)
{
    shl_ref_group_conv2d_quant(input, output, kernel, bias, params);
    struct csinn_relu_params *rp = shl_mem_alloc(sizeof(struct csinn_relu_params));
    memcpy(&(rp->base), &(params->base), sizeof(struct csinn_params_base));
    csinn_relu6_init(output, output, rp);
    csinn_relu6(output, output, rp);

    return CSINN_TRUE;
}
