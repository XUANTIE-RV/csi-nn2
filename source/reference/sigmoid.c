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

int csi_sigmoid_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct sigmoid_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }

    for (int i = 0; i < size; i++) {
        float val = input_data[i];
        output_data[i] = 1.0f / (1.0f + exp(-val));
    }
    return CSINN_TRUE;
}

int csi_sigmoid_u8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct sigmoid_params *params)
{
    float *float_input_data;
    float *float_output_data;
    struct csi_tensor float_input;
    struct csi_tensor float_output;
    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    int size = 1;

    for (int i = 0; i < input->dim_count; i++) {
        size *= input->dim[i];
    }

    memcpy(&float_input, input, sizeof(struct csi_tensor));
    memcpy(&float_output, output, sizeof(struct csi_tensor));
    float_input_data = malloc(size * sizeof(float));
    float_output_data = malloc(size * sizeof(float));
    float_input.data = float_input_data;
    float_output.data = float_output_data;

    for (int i = 0; i < size; i++) {
        float_input_data[i] = csi_dequantize_u8_to_f32(input_data[i], input->zero_point,
                                                 input->multiplier, input->shift);
    }

    csi_sigmoid_f32(&float_input, &float_output, params);

    for (int i = 0; i < size; i++) {
        output_data[i] = csi_quantize_f32_to_u8(float_output_data[i], output->zero_point,
                                          output->multiplier, output->shift);
    }
    free(float_input_data);
    free(float_output_data);
    return CSINN_TRUE;
}


int csi_sigmoid_init(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct sigmoid_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_SIGMOID, input->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_sigmoid(struct csi_tensor *input,
             struct csi_tensor *output,
             struct sigmoid_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
