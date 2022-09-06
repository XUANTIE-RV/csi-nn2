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

/* CSI-NN2 version 2.0.x */

#include "shl_ref.h"

int shl_ref_arange_f32(struct csinn_tensor *output, struct csinn_arange_params *params)
{
    float *data = output->data;
    int j = 0;
    float i = params->start;
    while (1) {
        if (params->step > FLT_EPSILON) {
            if (i - params->stop > FLT_EPSILON)  // i > stop
                break;
        } else {
            if (i - params->stop < FLT_EPSILON)  // i < stop
                break;
        }

        data[j] = i;
        i += params->step;
        j++;
    }
    return CSINN_TRUE;
}

int shl_ref_arange_quant(struct csinn_tensor *output, struct csinn_arange_params *params)
{
    struct csinn_quant_info qinfo;
    qinfo.zero_point = 0;
    qinfo.multiplier = params->start_multiplier;
    qinfo.shift = params->start_shift;
    float start = shl_ref_dequantize_u8_to_f32(1.0, &qinfo);
    qinfo.zero_point = 0;
    qinfo.multiplier = params->stop_multiplier;
    qinfo.shift = params->stop_shift;
    float stop = shl_ref_dequantize_u8_to_f32(1.0, &qinfo);
    qinfo.zero_point = 0;
    qinfo.multiplier = params->step_multiplier;
    qinfo.shift = params->step_shift;
    float step = shl_ref_dequantize_u8_to_f32(1.0, &qinfo);

    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    shl_ref_arange_f32(foutput, params);
    csinn_tensor_data_convert(output, foutput);
    shl_ref_tensor_transform_free_f32(foutput);

    return CSINN_TRUE;
}
