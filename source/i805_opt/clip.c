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

#include "i805_function.h"
#include "shl_i805.h"

int shl_i805_clip_init_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_clip_params *params)
{
    float real_scale = input->qinfo->scale / output->qinfo->scale;
    shl_quantize_multiplier(real_scale, &(output->qinfo->multiplier), &(output->qinfo->shift));
    struct csinn_callback *cb = params->base.cb;
    cb->exec = shl_i805_clip_u8;
    return CSINN_TRUE;
}

int shl_i805_clip_u8(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_clip_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;
    int32_t size = csinn_tensor_size(input);

    int32_t clip_qmin = floor(params->min_value / input->qinfo->scale) + input->qinfo->zero_point;
    int32_t clip_qmax = ceil(params->max_value / input->qinfo->scale) + input->qinfo->zero_point;

    shl_i805_clip_opt_u8(input_data, output_data, size, clip_qmin, clip_qmax,
                         input->qinfo->zero_point, output->qinfo->zero_point,
                         output->qinfo->multiplier, output->qinfo->shift);
    return CSINN_TRUE;
}
