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

static int csi_reshape_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct reshape_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    int size = 1;
    if (input_data != output_data) {
        for (int i = 0; i < input->dim_count; i++) {
            size *= input->dim[i];
        }
        memcpy(output_data, input_data, size * sizeof(float));
    }
    return CSINN_TRUE;
}

static int csi_reshape_u8(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reshape_params *params)
{
    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    int size = 1;
    if (input_data != output_data) {
        for (int i = 0; i < input->dim_count; i++) {
            size *= input->dim[i];
        }
        memcpy(output_data, input_data, size);
    }
    return CSINN_TRUE;
}

int csi_reshape_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct reshape_params *params)
{
    if (input->dtype == CSINN_DTYPE_UINT8) {
        params->bc = csi_reshape_u8;
    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        params->bc = csi_reshape_f32;
    } else {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_reshape(struct csi_tensor *input,
             struct csi_tensor *output,
             struct reshape_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}

