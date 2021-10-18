/*
 * Copyright (C) 2016-2020 C-SKY Limited. All rights reserved.
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

#include "csi_nn.h"
#include "csi_utils.h"

int csi_arange_f32(struct csi_tensor *output,
                   struct arange_params *params)
{
    float_t * data = output->data;
    int j = 0;
    float i = params->start;
    while (1) {
        if (params->step > FLT_EPSILON) {
            if (i - params->stop > FLT_EPSILON) //i > stop
                break;
        } else {
            if (i - params->stop < FLT_EPSILON) //i < stop
                break;
        }

        data[j] = i;
        i += params->step;
        j++;
    }
    return CSINN_TRUE;
}

int csi_arange_u8(struct csi_tensor *output,
                  struct arange_params *params)
{
    float start = csi_dequantize_u8_to_f32(1.0, 0, params->start_multiplier, params->start_shift);
    float stop = csi_dequantize_u8_to_f32(1.0, 0, params->stop_multiplier, params->stop_shift);
    float step = csi_dequantize_u8_to_f32(1.0, 0, params->step_multiplier, params->step_shift);

    uint8_t * data = output->data;
    int j = 0;
    float i = start;
    while (1) {
        if (step > FLT_EPSILON) {
            if (i - stop > FLT_EPSILON) //i > stop
                break;
        } else {
            if (i - stop < FLT_EPSILON) //i < stop
                break;
        }

        data[j] = csi_quantize_f32_to_u8(i, output->zero_point, output->multiplier, output->shift);
        i+=step;
        j++;
    }
    return CSINN_TRUE;
}

int csi_arange_init(struct csi_tensor *output,
                    struct arange_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_ARANGE, output->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_arange(struct csi_tensor *output,
               struct arange_params *params)
{
    if (params->bc != NULL) {
        params->bc(output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
