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

static int csi_less_equal_f32(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params)
{
    float *input0_data = input0->data;
    float *input1_data = input1->data;
    float *output_data = output->data;
    int size = 1;
    for (int i = 0; i < input0->dim_count; i++) {
        size = size * input0->dim[i];
    }

    for (int i = 0; i < size; i++) {
        output_data[i] = input0_data[i] <= input1_data[i];
    }
    return CSINN_TRUE;
}

static int csi_less_equal_u8(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params)
{
    uint8_t *input0_data = input0->data;
    uint8_t *input1_data = input1->data;
    uint8_t *output_data = output->data;
    int size = 1;
    for (int i = 0; i < input0->dim_count; i++) {
        size = size * input0->dim[i];
    }

    for (int i = 0; i < size; i++) {
        float input0_val = csi_dequantize_f32(input0_data[i], input0->offset, input0->multiplier,
                                               input0->shift);
        float input1_val = csi_dequantize_f32(input1_data[i], input1->offset, input1->multiplier,
                                               input1->shift);
        float res = input0_val <= input1_val;

        output_data[i] = csi_quantize_f32(res, output->offset, output->multiplier, output->shift);
    }
    return CSINN_TRUE;
}


int csi_less_equal_init(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params)
{
    if (input0->dtype == CSINN_DTYPE_UINT8) {
        params->bc = csi_less_equal_u8;
    } else if (input0->dtype == CSINN_DTYPE_FLOAT32) {
        params->bc = csi_less_equal_f32;
    } else {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_less_equal(struct csi_tensor *input0,
            struct csi_tensor *input1,
            struct csi_tensor *output,
            struct diso_params *params)
{
    if (params->bc != NULL) {
        params->bc(input0, input1, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}