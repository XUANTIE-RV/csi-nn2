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
#include <assert.h>

int csi_atanh_f32(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct siso_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }

    for (int i = 0; i < size; i++) {
        output_data[i] = atanh(input_data[i]);
    }
    return CSINN_TRUE;
}

int csi_atanh_u8(struct csi_tensor *input,
                 struct csi_tensor *output,
                 struct siso_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }

    for (int i = 0; i < size; i++) {
        float input0_val = csi_dequantize_u8_to_f32(input_data[i], input->zero_point, input->multiplier,
                                               input->shift);
        float res = atanh(input0_val);

        output_data[i] = csi_quantize_f32_to_u8(res, output->zero_point, output->multiplier, output->shift);
    }
    return CSINN_TRUE;
}

int csi_atanh_init(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct siso_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_ATANH, input->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_atanh(struct csi_tensor *input,
              struct csi_tensor *output,
              struct siso_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
