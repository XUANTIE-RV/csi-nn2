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

static int csi_shuffle_channel_nchw_f32(struct csi_tensor *input,
                                        struct csi_tensor *output,
                                        struct shuffle_channel_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int batch   = input->dim[0];
    int channel = input->dim[1];
    int height  = input->dim[2];
    int width   = input->dim[3];
    int group = params->group;
    int group_channel = channel / group;
    int input_inner_size = input->dim[2] * input->dim[3];

    float *input_data_addr = input_data;
    for(int i = 0; i < batch; i++) {
        for(int j = 0; j < group_channel; j++) {
            for(int k = 0; k < group; k++) {
                float *input_data_addr1 = input_data_addr + (k * group_channel + j) * input_inner_size;
                memcpy(output_data, input_data_addr1, input_inner_size * sizeof(float));
                output_data += input_inner_size;
            }
        }
        input_data_addr += channel * input_inner_size;
    }
    return CSINN_TRUE;
}

static int csi_shuffle_channel_nchw_u8(struct csi_tensor *input,
                                       struct csi_tensor *output,
                                       struct shuffle_channel_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;

    int batch   = input->dim[0];
    int channel = input->dim[1];
    int height  = input->dim[2];
    int width   = input->dim[3];
    int group = params->group;
    int group_channel = channel / group;
    int input_inner_size = input->dim[2] * input->dim[3];

    uint8_t *input_data_addr = input_data;
    for(int i = 0; i < batch; i++) {
        for(int j = 0; j < group_channel; j++) {
            for(int k = 0; k < group; k++) {
                uint8_t *input_data_addr1 = input_data_addr + (k * group_channel + j) * input_inner_size;
                memcpy(output_data, input_data_addr1, input_inner_size * sizeof(uint8_t));
                output_data += input_inner_size;
            }
        }
        input_data_addr += channel * input_inner_size;
    }
    return CSINN_TRUE;
}

// defalut input_layout = NCHW
static int csi_shuffle_channel_nhwc_f32(struct csi_tensor *o_input,
                                        struct csi_tensor *o_output,
                                        struct shuffle_channel_params *params)
{
    struct csi_tensor *input;
    struct csi_tensor *output;
    input  = csi_nchw_to_nhwc_f32(o_input);
    output = csi_nchw_to_nhwc_f32(o_output);
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int batch   = input->dim[0];
    int height  = input->dim[1];
    int width   = input->dim[2];
    int channel = input->dim[3];
    int group = params->group;
    int group_channel = channel / group;
    int input_outer_size = input->dim[0] * input->dim[1] * input->dim[2];
    int input_inner_size = 1;

    float *input_data_addr = input_data;
    for(int i = 0; i < input_outer_size; i++) {
        for(int j = 0; j < group_channel; j++) {
            for(int k = 0; k < group; k++) {
                float *input_data_addr1 = input_data_addr + (k * group_channel + j) * input_inner_size;
                memcpy(output_data, input_data_addr1, input_inner_size * sizeof(float));
                output_data += input_inner_size;
            }
        }
        input_data_addr += channel * input_inner_size;
    }
    csi_nhwc_to_nchw_f32(o_output, output);
    return CSINN_TRUE;
}

// defalut input_layout = NCHW
static int csi_shuffle_channel_nhwc_u8(struct csi_tensor *o_input,
                                       struct csi_tensor *o_output,
                                       struct shuffle_channel_params *params)
{
    struct csi_tensor *input;
    struct csi_tensor *output;
    input  = csi_nchw_to_nhwc_8(o_input);
    output = csi_nchw_to_nhwc_8(o_output);
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;

    int batch   = input->dim[0];
    int height  = input->dim[1];
    int width   = input->dim[2];
    int channel = input->dim[3];
    int group = params->group;
    int group_channel = channel / group;
    int input_outer_size = input->dim[0] * input->dim[1] * input->dim[2];
    int input_inner_size = 1;

    uint8_t *input_data_addr = input_data;
    for(int i = 0; i < input_outer_size; i++) {
        for(int j = 0; j < group_channel; j++) {
            for(int k = 0; k < group; k++) {
                uint8_t *input_data_addr1 = input_data_addr + (k * group_channel + j) * input_inner_size;
                memcpy(output_data, input_data_addr1, input_inner_size * sizeof(uint8_t));
                output_data += input_inner_size;
            }
        }
        input_data_addr += channel * input_inner_size;
    }
    csi_nhwc_to_nchw_8(o_output, output);
    return CSINN_TRUE;
}

int csi_shuffle_channel_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct shuffle_channel_params *params)
{
    if (params->layout == CSINN_NCHW) {
        csi_shuffle_channel_nchw_f32(input, output, params);
    } else if (params->layout == CSINN_NHWC) {
        csi_shuffle_channel_nhwc_f32(input, output, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_shuffle_channel_u8(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct shuffle_channel_params *params)
{
    if (params->layout == CSINN_NCHW) {
        csi_shuffle_channel_nchw_u8(input, output, params);
    } else if (params->layout == CSINN_NHWC) {
        csi_shuffle_channel_nhwc_u8(input, output, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_shuffle_channel_init(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct shuffle_channel_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_SHUFFLE_CHANNEL, input->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_shuffle_channel(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct shuffle_channel_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}

