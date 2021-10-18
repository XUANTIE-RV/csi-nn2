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
#include "float.h"

static int csi_global_maxpool_nhwc_u8(struct csi_tensor *input,
                                      struct csi_tensor *output,
                                      struct pool_params *params)
{
    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    const int batches = input->dim[0];
    const int depth = input->dim[3];
    const int input_height = input->dim[1];
    const int input_width = input->dim[2];
    const int output_height = output->dim[1];
    const int output_width = output->dim[2];

    const int32_t input_offset = input->offset;
    const int32_t input_multiplier = input->multiplier;
    const int32_t input_shift = input->shift;
    const int32_t output_offset = output->offset;
    const int32_t output_multiplier = output->multiplier;
    const int32_t output_shift = output->shift;

    int filter_height = input_height;
    int filter_width = input_width;
    int stride_height = 1;
    int stride_width = 1;
    int pad_height = 0;
    int pad_width = 0;

    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                for (int channel = 0; channel < depth; ++channel) {
                    const int in_x_origin = (out_x * stride_width) - pad_width;
                    const int in_y_origin = (out_y * stride_height) - pad_height;
                    // Compute the boundaries of the filter region clamped so as to
                    // ensure that the filter window fits in the input array.
                    const int filter_x_start = csi_max_internal_s32(0, -in_x_origin);
                    const int filter_x_end =
                        csi_min_internal_s32(filter_width, input_width - in_x_origin);
                    const int filter_y_start = csi_max_internal_s32(0, -in_y_origin);
                    const int filter_y_end =
                        csi_min_internal_s32(filter_height, input_height - in_y_origin);
                    float max_value = FLT_MIN;
                    float curr_value = 0;
                    for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
                        for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x) {
                            const int in_x = in_x_origin + filter_x;
                            const int in_y = in_y_origin + filter_y;
                            uint8_t input_val = input_data[csi_get_index(input->dim, batch, in_y,
                                                                         in_x, channel)];
                            curr_value = csi_dequantize_f32(input_val, input_offset, input_multiplier,
                                                        input_shift);
                            if (curr_value > max_value) {
                                max_value = curr_value;
                            }
                        }
                    }
                    output_data[csi_get_index(output->dim, batch, out_y, out_x, channel)] =
                        csi_quantize_f32(max_value, output_offset, output_multiplier, output_shift);
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_global_maxpool_nchw_u8(struct csi_tensor *o_input,
                                      struct csi_tensor *o_output,
                                      struct pool_params *params)
{
    struct csi_tensor* input;
    struct csi_tensor* output;
    input =  csi_nchw_to_nhwc_u8(o_input);
    output = csi_nchw_to_nhwc_u8(o_output);

    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    const int batches = input->dim[0];
    const int depth = input->dim[3];
    const int input_height = input->dim[1];
    const int input_width = input->dim[2];
    const int output_height = output->dim[1];
    const int output_width = output->dim[2];

    const int32_t input_offset = input->offset;
    const int32_t input_multiplier = input->multiplier;
    const int32_t input_shift = input->shift;
    const int32_t output_offset = output->offset;
    const int32_t output_multiplier = output->multiplier;
    const int32_t output_shift = output->shift;

    int filter_height = input_height;
    int filter_width = input_width;
    int stride_height = 1;
    int stride_width = 1;
    int pad_height = 0;
    int pad_width = 0;

    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                for (int channel = 0; channel < depth; ++channel) {
                    const int in_x_origin = (out_x * stride_width) - pad_width;
                    const int in_y_origin = (out_y * stride_height) - pad_height;
                    // Compute the boundaries of the filter region clamped so as to
                    // ensure that the filter window fits in the input array.
                    const int filter_x_start = csi_max_internal_s32(0, -in_x_origin);
                    const int filter_x_end =
                        csi_min_internal_s32(filter_width, input_width - in_x_origin);
                    const int filter_y_start = csi_max_internal_s32(0, -in_y_origin);
                    const int filter_y_end =
                        csi_min_internal_s32(filter_height, input_height - in_y_origin);
                    float max_value = FLT_MIN;
                    float curr_value = 0;
                    for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
                        for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x) {
                            const int in_x = in_x_origin + filter_x;
                            const int in_y = in_y_origin + filter_y;
                            uint8_t input_val = input_data[csi_get_index(input->dim, batch, in_y,
                                                                         in_x, channel)];
                            curr_value = csi_dequantize_f32(input_val, input_offset, input_multiplier,
                                                        input_shift);
                            if (curr_value > max_value) {
                                max_value = curr_value;
                            }
                        }
                    }
                    output_data[csi_get_index(output->dim, batch, out_y, out_x, channel)] =
                        csi_quantize_f32(max_value, output_offset, output_multiplier, output_shift);
                }
            }
        }
    }
    csi_nhwc_to_nchw_u8(o_output, output);
    return CSINN_TRUE;
}

int csi_global_maxpool_init(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct pool_params *params)
{
    if (params->layout == CSINN_NCHW) {
        if (input->dtype == CSINN_DTYPE_UINT8) {
            params->bc = csi_global_maxpool_nchw_u8;
        } else {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else if (params->layout = CSINN_NHWC) {
        if (input->dtype == CSINN_DTYPE_UINT8) {
            params->bc = csi_global_maxpool_nhwc_u8;
        } else {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    return CSINN_TRUE;
}

int csi_global_maxpool(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct pool_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
