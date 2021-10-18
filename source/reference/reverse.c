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

static int Multiplication(struct csi_tensor *input, int s, int e)
{
    int res = 1;
    for(int i=s; i<=e; i++) {
        res = res * input->dim[i];
    }
    return res;
}

int csi_reverse_f32(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct reverse_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }
    int axis = params->axis;
    int num = Multiplication(input, 0, axis) / (input->dim[axis]);
    int step = Multiplication(input, axis, input->dim_count-1) / (input->dim[axis]);
    int cnt = (input->dim[axis])/2;

    memcpy(output_data, input_data, size * sizeof(float));
    
    for(int i=0; i<num; i++) {
        float *start_addr = output_data + i*step*(input->dim[axis]);
        float *end_addr = start_addr + step*(input->dim[axis]) - 1;
        for(int j=0; j<cnt; j++) {
            float *temp = (float *)malloc(step * sizeof(float));
            memcpy(temp, start_addr, step*sizeof(float));
            memcpy(start_addr, end_addr-step+1, step*sizeof(float));
            memcpy(end_addr-step+1, temp, step*sizeof(float));
            start_addr += step;
            end_addr -= step;
            free(temp);
        }
    }
    return CSINN_TRUE;
}

int csi_reverse_u8(struct csi_tensor *input,
                   struct csi_tensor *output,
                   struct reverse_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;

    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }
    int axis = params->axis;
    int num = Multiplication(input, 0, axis) / (input->dim[axis]);
    int step = Multiplication(input, axis, input->dim_count-1) / (input->dim[axis]);
    int cnt = (input->dim[axis])/2;

    memcpy(output_data, input_data, size * sizeof(uint8_t));

    for(int i=0; i<num; i++) {
        uint8_t *start_addr = output_data + i*step*(input->dim[axis]);
        uint8_t *end_addr = start_addr + step*(input->dim[axis]) - 1;
        for(int j=0; j<cnt; j++) {
            uint8_t *temp = (uint8_t *)malloc(step * sizeof(uint8_t));
            memcpy(temp, start_addr, step*sizeof(uint8_t));
            memcpy(start_addr, end_addr-step+1, step*sizeof(uint8_t));
            memcpy(end_addr-step+1, temp, step*sizeof(uint8_t));
            start_addr += step;
            end_addr -= step;
            free(temp);
        }
    }
    return CSINN_TRUE;
}

int csi_reverse_init(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct reverse_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_REVERSE, input->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_reverse(struct csi_tensor *input,
                struct csi_tensor *output,
                struct reverse_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
