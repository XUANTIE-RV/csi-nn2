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

//tf.nn.space_to_batch:the input mast a  4-D Tensor with shape [batch, height, width, depth].

static int csi_space_to_batch_f32(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct space_to_batch_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int batch = input->dim[0];
    int in_channel = input->dim[1];
    int in_height = input->dim[2];
    int in_width = input->dim[3];

    int block_size = params->block_size;
    int block_size2 = block_size * block_size;

    int out_batch = output->dim[0];     //out_batch = in_batch * block_size * block_size;
    int out_channel = output->dim[1];   //out_channel = in_channel;
    int out_height = output->dim[2];    //out_height = (in_height) / block_size;
    int out_width = output->dim[3];     //out_width = (in_width = params->) / block_size;

    for(int in_b = 0; in_b < batch; ++in_b) {
        for(int out_h = 0; out_h < out_height * block_size; out_h = out_h + block_size) {
            for(int out_w = 0; out_w < out_width * block_size; out_w = out_w + block_size) {
                for(int out_c = 0; out_c < in_channel; ++out_c) {

                    float *temp = (float *)calloc(block_size2, sizeof(float));
                    int h_origin = out_h - params->pad_top;
                    int w_origin = out_w - params->pad_left;
                    for(int h = 0; h < block_size; ++h) {
                        for(int w = 0; w < block_size; ++w) {
                            int h_now = h_origin + h;
                            int w_now = w_origin + w;
                            if(h_now >= 0 && h_now < in_height && w_now >= 0 && w_now < in_width) {
                                int in_addr = csi_get_index(input->dim, in_b, out_c, h_now, w_now);
                                temp[h * block_size + w] = input_data[in_addr];
                            }
                        }
                    }
                    int out_start_addr = csi_get_index(output->dim, in_b, out_c, out_h / block_size, out_w / block_size);
                    for(int i = 0; i < block_size2; ++i) {
                        output_data[out_start_addr + i * batch * out_channel * out_height * out_width] = temp[i];
                    }
                    free(temp);
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_space_to_batch_u8(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct space_to_batch_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;

    int batch = input->dim[0];
    int in_channel = input->dim[1];
    int in_height = input->dim[2];
    int in_width = input->dim[3];

    int block_size = params->block_size;
    int block_size2 = block_size * block_size;

    int out_batch = output->dim[0];     //out_batch = in_batch * block_size * block_size;
    int out_channel = output->dim[1];   //out_channel = in_channel;
    int out_height = output->dim[2];    //out_height = in_height / block_size;
    int out_width = output->dim[3];     //out_width = in_width / block_size;

    for(int in_b = 0; in_b < batch; ++in_b) {
        for(int out_h = 0; out_h < out_height * block_size; out_h = out_h + block_size) {
            for(int out_w = 0; out_w < out_width * block_size; out_w = out_w + block_size) {
                for(int out_c = 0; out_c < in_channel; ++out_c) {

                    uint8_t *temp = (uint8_t *)calloc(block_size2, sizeof(uint8_t));
                    int h_origin = out_h - params->pad_top;
                    int w_origin = out_w - params->pad_left;
                    for(int h = 0; h < block_size; ++h) {
                        for(int w = 0; w < block_size; ++w) {
                            int h_now = h_origin + h;
                            int w_now = w_origin + w;
                            if(h_now >= 0 && h_now < in_height && w_now >= 0 && w_now < in_width) {
                                int in_addr = csi_get_index(input->dim, in_b, out_c, h_now, w_now);
                                temp[h * block_size + w] = input_data[in_addr];
                            }
                        }
                    }
                    int out_start_addr = csi_get_index(output->dim, in_b, out_c, out_h / block_size, out_w / block_size);
                    for(int i = 0; i < block_size2; ++i) {
                        output_data[out_start_addr + i * batch * out_channel * out_height * out_width] =
                            csi_requantize_u8(temp[i], input->offset, input->multiplier, input->shift,
                                                       output->offset, output->multiplier, output->shift);
                    }
                    free(temp);
                }
            }
        }
    }
    return CSINN_TRUE;
}


int csi_space_to_batch_init(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct space_to_batch_params *params)
{
    if (input->dtype == CSINN_DTYPE_UINT8) {
        params->bc = csi_space_to_batch_u8;
    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        params->bc = csi_space_to_batch_f32;
    } else {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_space_to_batch(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct space_to_batch_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}