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


static int csi_ref_shuffle_channel_nhwc_f32(struct csi_tensor *input,
                                            struct csi_tensor *output,
                                            struct shuffle_channel_params *params)
{
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
    return CSINN_TRUE;
}


static int csi_ref_shuffle_channel_nchw_f32(struct csi_tensor *o_input,
                                            struct csi_tensor *o_output,
                                            struct shuffle_channel_params *params)
{
    struct csi_tensor *input;
    struct csi_tensor *output;
    input  = csi_ref_nchw_to_nhwc_f32(o_input);
    output = csi_ref_nchw_to_nhwc_f32(o_output);
    csi_ref_shuffle_channel_nhwc_f32(input, output, params);
    csi_ref_nhwc_to_nchw_f32(o_output, output);
    return CSINN_TRUE;
}

int csi_ref_shuffle_channel_f32(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct shuffle_channel_params *params)
{
    if (params->base.layout == CSINN_LAYOUT_NCHW) {
        csi_ref_shuffle_channel_nchw_f32(input, output, params);
    } else if (params->base.layout == CSINN_LAYOUT_NHWC) {
        csi_ref_shuffle_channel_nhwc_f32(input, output, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_ref_shuffle_channel_quant(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct shuffle_channel_params *params)
{
    return csi_ref_siso_callback_base(input, output, params, csi_ref_shuffle_channel_f32);
}
