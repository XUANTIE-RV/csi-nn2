/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

/* CSI-NN2 version 1.10.x */

#include "csi_i805.h"


int csi_i805_clip_init_u8(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct clip_params *params)
{
    float real_scale = input->qinfo->scale / output->qinfo->scale;
    csi_quantize_multiplier(real_scale, &(output->qinfo->multiplier), &(output->qinfo->shift));
    params->base.bc = csi_i805_clip_u8;
    return CSINN_TRUE;
}

int csi_i805_clip_u8(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct clip_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;
    int32_t size = csi_tensor_size(input);

    int32_t clip_qmin = floor(params->min_value / input->qinfo->scale) + input->qinfo->zero_point;
    int32_t clip_qmax = ceil(params->max_value / input->qinfo->scale) + input->qinfo->zero_point;

    csi_i805_clip_opt_u8(input_data, output_data, size, clip_qmin, clip_qmax, input->qinfo->zero_point, output->qinfo->zero_point,
                         output->qinfo->multiplier, output->qinfo->shift);
    return CSINN_TRUE;
}
