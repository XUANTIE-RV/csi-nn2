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
#include <assert.h>


static int csi_or_u32(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params)
{
    uint32_t *input0_data = input0->data;
    uint32_t *input1_data = input1->data;
    uint32_t *output_data = output->data;
    int size = 1;
    for (int i = 0; i < output->dim_count; i++) {
        size = size * output->dim[i];
    }

    for (int i = 0; i < size; i++) {
        output_data[i] = input0_data[0] | input1_data[0];
    }
    return CSINN_TRUE;
}

static int csi_or_u8(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params)
{
    uint8_t *input0_data = input0->data;
    uint8_t *input1_data = input1->data;
    uint8_t *output_data = output->data;
    int size = 1;
    for (int i = 0; i < output->dim_count; i++) {
        size = size * output->dim[i];
    }

    for (int i = 0; i < size; i++) {
        output_data[i] = input0_data[0] | input1_data[0];
    }
    return CSINN_TRUE;
}

int csi_or_init(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params)
{
    if (input0->dtype == CSINN_DTYPE_UINT8) {
        params->bc = csi_or_u8;
    } else if (input0->dtype == CSINN_DTYPE_UINT32) {
        params->bc = csi_or_u32;
    } else {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_or(struct csi_tensor *input0,
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