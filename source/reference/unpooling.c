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

static int csi_unpooling_nhwc_f32(struct csi_tensor *input,
                                  struct csi_tensor *mask,
                                  struct csi_tensor *output,
                                  struct unpooling_params *params)
{
    float *input_data = input->data;
    int *mask_data = mask->data;
    float *output_data = output->data;

    const int batches = input->dim[0];
    const int depth = input->dim[3];
    const int input_height = input->dim[1];
    const int input_width = input->dim[2];

    const int output_height = output->dim[1];
    const int output_width = output->dim[2];


    int size = 0;
    for(int i = 0; i < output->dim_count; i++){
        size *= output->dim[i];
    }
    memset(output_data, 0, size * sizeof(float));

    for(int b = 0; b < batches; b++){
        for(int h = 0; h < input_height; h++){
            for(int w = 0; w < input_width; w++){
                for(int c = 0; c < depth; c++){
                    int id = mask_data[csi_get_index(input->dim, b, h, w, c)];
                    if(id < output_height * output_width){
                        int id_h = id / output_width;
                        int id_w = id % output_width;
                        int o_index = csi_get_index(output->dim, b, id_h, id_w, c);
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_unpooling_nhwc_u8(struct csi_tensor *input,
                                 struct csi_tensor *mask,
                                 struct csi_tensor *output,
                                 struct unpooling_params *params)
{
    uint8_t *input_data = input->data;
    int32_t *mask_data = mask->data;
    uint8_t *output_data = output->data;

    const int batches = input->dim[0];
    const int depth = input->dim[3];
    const int input_height = input->dim[1];
    const int input_width = input->dim[2];

    const int output_height = output->dim[1];
    const int output_width = output->dim[2];


    int size = 1;
    for(int i = 0; i < output->dim_count; i++){
        size *= output->dim[i];
    }
    memset(output_data, output->zero_point, size * sizeof(uint8_t));

    for(int b = 0; b < batches; b++){
        for(int h = 0; h < input_height; h++){
            for(int w = 0; w < input_width; w++){
                for(int c = 0; c < depth; c++){
                    int index = csi_get_index(input->dim, b, h, w, c);
                    int id = mask_data[index];
                    if(id < output_height * output_width){
                        int id_h = id / output_width;
                        int id_w = id % output_width;
                        int o_index = csi_get_index(output->dim, b, id_h, id_w, c);
                        output_data[o_index] = input_data[index];
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_unpooling_nchw_f32(struct csi_tensor *input,
                                  struct csi_tensor *mask,
                                  struct csi_tensor *output,
                                  struct unpooling_params *params)
{
    float *input_data = input->data;
    int *mask_data = mask->data;
    float *output_data = output->data;

    const int batches = input->dim[0];
    const int depth = input->dim[3];
    const int input_height = input->dim[1];
    const int input_width = input->dim[2];

    const int output_height = output->dim[1];
    const int output_width = output->dim[2];


    int size = 0;
    for(int i = 0; i < output->dim_count; i++){
        size *= output->dim[i];
    }
    memset(output_data, 0, size * sizeof(float));

    for(int b = 0; b < batches; b++){
        for(int h = 0; h < input_height; h++){
            for(int w = 0; w < input_width; w++){
                for(int c = 0; c < depth; c++){
                    int id = mask_data[csi_get_index(input->dim, b, h, w, c)];
                    if(id < output_height * output_width){
                        int id_h = id / output_width;
                        int id_w = id % output_width;
                        int o_index = csi_get_index(output->dim, b, id_h, id_w, c);
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_unpooling_nchw_u8(struct csi_tensor *input,
                                 struct csi_tensor *mask,
                                 struct csi_tensor *output,
                                 struct unpooling_params *params)
{
    uint8_t *input_data = input->data;
    int32_t *mask_data = mask->data;
    uint8_t *output_data = output->data;

    const int batches = input->dim[0];
    const int depth = input->dim[1];
    const int input_height = input->dim[2];
    const int input_width = input->dim[3];

    const int output_height = output->dim[2];
    const int output_width = output->dim[3];


    int size = 1;
    for(int i = 0; i < output->dim_count; i++){
        size *= output->dim[i];
    }
    memset(output_data, output->zero_point, size * sizeof(uint8_t));

    for(int b = 0; b < batches; b++){
        for(int c = 0; c < depth; c++){
            for(int h = 0; h < input_height; h++){
                for(int w = 0; w < input_width; w++){
                    int index = csi_get_index(input->dim, b, c, h, w);
                    int id = mask_data[index];
                    if(id < output_height * output_width){
                        int id_h = id / output_width;
                        int id_w = id % output_width;
                        int o_index = csi_get_index(output->dim, b, c, id_h, id_w);
                        output_data[o_index] = input_data[index];
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

int csi_unpooling_init(struct csi_tensor *input,
                       struct csi_tensor *mask,
                       struct csi_tensor *output,
                       struct unpooling_params *params)
{
    if (params->layout == CSINN_NCHW) {
        if (input->dtype == CSINN_DTYPE_UINT8) {
            params->bc = csi_unpooling_nchw_u8;
        } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
            params->bc = csi_unpooling_nchw_f32;
        } else {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else if (params->layout = CSINN_NHWC) {
        if (input->dtype == CSINN_DTYPE_UINT8) {
            params->bc = csi_unpooling_nhwc_u8;
        } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
            params->bc = csi_unpooling_nhwc_f32;
        } else {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    return CSINN_TRUE;
}

int csi_unpooling(struct csi_tensor *input,
                  struct csi_tensor *mask,
                  struct csi_tensor *output,
                  struct unpooling_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, mask, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}

