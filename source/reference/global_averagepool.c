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

#include "csi_ref.h"

int csi_ref_global_averagepool_f32(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct pool_params *params)
{
    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 0;
    params->pad_back = 0;
    params->pad_down = 0;
    params->pad_front = 0;
    params->pad_left = 0;
    params->pad_right = 0;
    params->pad_top = 0;
    if (params->base.layout == CSINN_NCHW) {
        params->filter_height = input->dim[2];
        params->filter_width = input->dim[3];
    } else if (params->base.layout == CSINN_NHWC) {
        params->filter_height = input->dim[1];
        params->filter_width = input->dim[2];
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    csi_ref_averagepool_f32(input, output, params);
}

int csi_ref_global_averagepool_quant(struct csi_tensor *input,
                                     struct csi_tensor *output,
                                     struct pool_params *params)
{
    return csi_ref_siso_callback_base(input, output, params, csi_ref_global_averagepool_f32);
}