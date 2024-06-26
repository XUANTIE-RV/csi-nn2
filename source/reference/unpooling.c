/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

#include "reference/ref.h"

static int shl_ref_unpooling_nhwc_f32(struct csinn_tensor *input, struct csinn_tensor *mask,
                                      struct csinn_tensor *output,
                                      struct csinn_unpooling_params *params)
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

    int size = csinn_tensor_size(output);
    memset(output_data, 0, size * sizeof(float));

    for (int b = 0; b < batches; b++) {
        for (int h = 0; h < input_height; h++) {
            for (int w = 0; w < input_width; w++) {
                for (int c = 0; c < depth; c++) {
                    int index = shl_ref_get_index(input->dim, b, h, w, c);
                    int id = mask_data[index];
                    if (id < output_height * output_width) {
                        int id_h = id / output_width;
                        int id_w = id % output_width;
                        int o_index = shl_ref_get_index(output->dim, b, id_h, id_w, c);
                        output_data[o_index] = input_data[index];
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int shl_ref_unpooling_nchw_f32(struct csinn_tensor *input, struct csinn_tensor *mask,
                                      struct csinn_tensor *output,
                                      struct csinn_unpooling_params *params)
{
    float *input_data = input->data;
    int *mask_data = mask->data;
    float *output_data = output->data;

    const int batches = input->dim[0];
    const int depth = input->dim[1];
    const int input_height = input->dim[2];
    const int input_width = input->dim[3];

    const int output_height = output->dim[2];
    const int output_width = output->dim[3];

    int size = csinn_tensor_size(output);
    memset(output_data, 0, size * sizeof(float));

    for (int b = 0; b < batches; b++) {
        for (int c = 0; c < depth; c++) {
            for (int h = 0; h < input_height; h++) {
                for (int w = 0; w < input_width; w++) {
                    int index = shl_ref_get_index(input->dim, b, c, h, w);
                    int id = mask_data[index];
                    if (id < output_height * output_width) {
                        int id_h = id / output_width;
                        int id_w = id % output_width;
                        int o_index = shl_ref_get_index(output->dim, b, c, id_h, id_w);
                        output_data[o_index] = input_data[index];
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

int shl_ref_unpooling_f32(struct csinn_tensor *input, struct csinn_tensor *mask,
                          struct csinn_tensor *output, struct csinn_unpooling_params *params)
{
    if (params->base.layout == CSINN_LAYOUT_NCHW) {
        shl_ref_unpooling_nchw_f32(input, mask, output, params);
    } else if (params->base.layout == CSINN_LAYOUT_NHWC) {
        shl_ref_unpooling_nhwc_f32(input, mask, output, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    return CSINN_TRUE;
}

int shl_ref_unpooling_quant(struct csinn_tensor *input, struct csinn_tensor *mask,
                            struct csinn_tensor *output, struct csinn_unpooling_params *params)
{
    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    shl_ref_unpooling_f32(finput, mask, foutput, params);
    csinn_tensor_data_convert(output, foutput);
    shl_ref_tensor_transform_free_f32(finput);
    shl_ref_tensor_transform_free_f32(foutput);
    return CSINN_TRUE;
}
