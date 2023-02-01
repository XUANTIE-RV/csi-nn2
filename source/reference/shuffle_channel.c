/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

/* SHL version 2.1.x */

#include "shl_ref.h"

static int shl_ref_shuffle_channel_nhwc_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_shuffle_channel_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int batch = input->dim[0];
    int height = input->dim[1];
    int width = input->dim[2];
    int channel = input->dim[3];
    int group = params->group;
    int group_channel = channel / group;
    int input_outer_size = input->dim[0] * input->dim[1] * input->dim[2];
    int input_inner_size = 1;

    float *input_data_addr = input_data;
    for (int i = 0; i < input_outer_size; i++) {
        for (int j = 0; j < group_channel; j++) {
            for (int k = 0; k < group; k++) {
                float *input_data_addr1 =
                    input_data_addr + (k * group_channel + j) * input_inner_size;
                memcpy(output_data, input_data_addr1, input_inner_size * sizeof(float));
                output_data += input_inner_size;
            }
        }
        input_data_addr += channel * input_inner_size;
    }
    return CSINN_TRUE;
}

static int shl_ref_shuffle_channel_nchw_f32(struct csinn_tensor *o_input,
                                            struct csinn_tensor *o_output,
                                            struct csinn_shuffle_channel_params *params)
{
    struct csinn_tensor *input;
    struct csinn_tensor *output;
    input = shl_ref_nchw_to_nhwc_f32(o_input);
    output = shl_ref_nchw_to_nhwc_f32(o_output);
    shl_ref_shuffle_channel_nhwc_f32(input, output, params);
    shl_ref_nhwc_to_nchw_f32(o_output, output);
    shl_ref_free_float_tensor(input);
    return CSINN_TRUE;
}

int shl_ref_shuffle_channel_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_shuffle_channel_params *params)
{
    if (params->base.layout == CSINN_LAYOUT_NCHW) {
        shl_ref_shuffle_channel_nchw_f32(input, output, params);
    } else if (params->base.layout == CSINN_LAYOUT_NHWC) {
        shl_ref_shuffle_channel_nhwc_f32(input, output, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int shl_ref_shuffle_channel_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_shuffle_channel_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_shuffle_channel_f32);
}
