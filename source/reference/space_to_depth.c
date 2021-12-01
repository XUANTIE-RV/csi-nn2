/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

/* CSI-NN2 version 1.8.x */

#include "csi_ref.h"
#include "csi_utils.h"

//the input->data is a 4-D Tensor with shape [batch, depth, height, width].
int csi_ref_space_to_depth_f32(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct space_to_depth_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int batch = input->dim[0];
    int in_channel = input->dim[1];
    int in_height = input->dim[2];
    int in_width = input->dim[3];

    int block_size = params->block_size;
    int block_size2 = block_size * block_size;
    assert(in_height%block_size==0 && in_width%block_size==0);

    int out_channel = output->dim[1];   //out_channel = in_channel * block_size * block_size;
    int out_height = output->dim[2];    //out_height = in_height / block_size;
    int out_width = output->dim[3];     //out_width = in_width / block_size;

    for(int out_b=0; out_b<batch; ++out_b) {
        for(int out_h=0; out_h<out_height; ++out_h) {
            for(int out_w=0; out_w<out_width; ++out_w) {
                for(int in_c=0; in_c<in_channel; ++in_c) {

                    float *temp = (float *)malloc(block_size2 * sizeof(float));
                    int in_start_addr = csi_ref_get_index(input->dim, out_b, in_c, out_h*block_size, out_w*block_size);
                    for(int h=0; h<block_size; h++) {
                        for(int w=0; w<block_size; w++) {
                            temp[h*block_size+w] = input_data[in_start_addr+h*in_width+w];
                        }
                    }
                    int out_start_addr = csi_ref_get_index(output->dim, out_b, in_c, out_h, out_w);
                    for(int i=0; i<block_size2; i++) {
                        output_data[out_start_addr + i*in_channel*out_height*out_width] = temp[i];
                    }
                    free(temp);
                }
            }
        }
    }
    return CSINN_TRUE;
}

int csi_ref_space_to_depth_quant(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct space_to_depth_params *params)
{
    return csi_ref_siso_callback_base(input, output, params, csi_ref_space_to_depth_f32);
}
