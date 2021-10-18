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

int csi_add_f32(struct csi_tensor *input0,
                struct csi_tensor *input1,
                struct csi_tensor *output,
                struct diso_params *params)
{
    float *input0_data = input0->data;
    float *input1_data = input1->data;
    float *output_data = output->data;
    int size0 = 1;
    for (int i = 0; i < input0->dim_count; i++) {
        size0 = size0 * input0->dim[i];
    }

    int size1 = 1;
    for (int i = 0; i < input1->dim_count; i++) {
        size1 = size1 * input1->dim[i];
    }

    if(size0 == size1){
        for (int i = 0; i < size0; i++) {
            output_data[i] = input0_data[i] + input1_data[i];
        }
    }
    else if(input1->dim[0] == input0->dim[3] && size1 == input1->dim[0]){
        for(int n = 0; n < input0->dim[0]; n++){
            for(int h = 0; h < input0->dim[1]; h++){
                for(int w = 0; w < input0->dim[2]; w++){
                    for(int c = 0; c < input0->dim[3]; c++){
                        int index = csi_get_index(input0->dim, n, h, w, c);
                        output_data[index] = input1_data[c] + input0_data[index];
                    }
                }
            }
        }
    }

    return CSINN_TRUE;
}

int csi_add_u8(struct csi_tensor *input0,
               struct csi_tensor *input1,
               struct csi_tensor *output,
               struct diso_params *params)
{
    uint8_t *input0_data = input0->data;
    uint8_t *input1_data = input1->data;
    uint8_t *output_data = output->data;

    int channel;
    if (params->layout == CSINN_NHWC){channel = input0->dim[3];}
    else if (params->layout == CSINN_NCHW){channel = input0->dim[1];}


    int size0 = 1;
    for (int i = 0; i < input0->dim_count; i++) {
        size0 = size0 * input0->dim[i];
    }

    int size1 = 1;
    int axis = 0;
    for (int i = 0; i < input1->dim_count; i++) {
        size1 = size1 * input1->dim[i];
        if (input1->dim[i] != 1){
            axis = i;
        }
    }

    if(size0 == size1){
        for (int i = 0; i < size0; i++) {
            float input0_val =
                csi_dequantize_u8_to_f32(input0_data[i], input0->zero_point, input0->multiplier, input0->shift);
            float input1_val =
                csi_dequantize_u8_to_f32(input1_data[i], input1->zero_point, input1->multiplier, input1->shift);
            float res = input0_val + input1_val;
            output_data[i] = csi_quantize_f32_to_u8(res, output->zero_point, output->multiplier, output->shift);
        }
    }
    else if(input1->dim[axis] == channel && size1 == input1->dim[axis]){
        for(int n = 0; n < input0->dim[0]; n++){
            for(int h = 0; h < input0->dim[1]; h++){
                for(int w = 0; w < input0->dim[2]; w++){
                    for(int c = 0; c < input0->dim[3]; c++){

                        if (params->layout == CSINN_NHWC){channel = c;}
                        else if (params->layout == CSINN_NCHW){channel = h;}

                        float input1_val =
                        csi_dequantize_u8_to_f32(input1_data[channel], input1->zero_point, input1->multiplier, input1->shift);

                        int index = csi_get_index(input0->dim, n, h, w, c);
                        float input0_val =
                        csi_dequantize_u8_to_f32(input0_data[index], input0->zero_point, input0->multiplier, input0->shift);
                        float res = input0_val + input1_val;
                        output_data[index] = csi_quantize_f32_to_u8(res, output->zero_point, output->multiplier, output->shift);
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

int csi_add_init(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_ADD, input0->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_add(struct csi_tensor *input0,
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