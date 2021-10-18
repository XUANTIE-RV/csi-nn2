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

static int csi_maxpool2d_locat_nhwc_f32(struct csi_tensor *input,
                                        struct csi_tensor *output,
                                        struct pool_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    const int batches = input->dim[0];
    const int depth = input->dim[3];
    const int input_height = input->dim[1];
    const int input_width = input->dim[2];
    const int output_height = output->dim[1];
    const int output_width = output->dim[2];

    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                for (int channel = 0; channel < depth; ++channel) {
                    const int in_x_origin = (out_x * params->stride_width) - params->pad_left;
                    const int in_y_origin = (out_y * params->stride_height) - params->pad_top;
                    // Compute the boundaries of the filter region clamped so as to
                    // ensure that the filter window fits in the input array.
                    const int filter_x_start = csi_max_internal_s32(0, -in_x_origin);
                    const int filter_x_end =
                        csi_min_internal_s32(params->filter_width, input_width - in_x_origin);
                    const int filter_y_start = csi_max_internal_s32(0, -in_y_origin);
                    const int filter_y_end =
                        csi_min_internal_s32(params->filter_height, input_height - in_y_origin);
                    float max = FLT_MIN;
                    int locat = (in_y_origin + filter_y_start) * input->dim[2] + (in_x_origin + filter_x_start);
                    for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
                        for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x) {
                            const int in_x = in_x_origin + filter_x;
                            const int in_y = in_y_origin + filter_y;
                            if (input_data[
                            csi_get_index(input->dim, batch, in_y,in_x, channel)] > max){
                                max = input_data[csi_get_index(input->dim, batch, in_y, in_x, channel)];
                                locat = in_y * input->dim[2] + in_x;
                            }
                        }
                    }
                    output_data[csi_get_index(output->dim, batch, out_y, out_x, channel)] = locat;
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_maxpool2d_locat_nhwc_i32_u8(struct csi_tensor *input,
                                           struct csi_tensor *output,
                                           struct pool_params *params)
{
    uint8_t *input_data = input->data;
    int32_t *output_data = output->data;
    const int batches = input->dim[0];
    const int depth = input->dim[3];
    const int input_height = input->dim[1];
    const int input_width = input->dim[2];
    const int output_height = output->dim[1];
    const int output_width = output->dim[2];
    int flag = 1;

    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                for (int channel = 0; channel < depth; ++channel) {
                    const int in_x_origin = (out_x * params->stride_width) - params->pad_left;
                    const int in_y_origin = (out_y * params->stride_height) - params->pad_top;
                    // Compute the boundaries of the filter region clamped so as to
                    // ensure that the filter window fits in the input array.
                    const int filter_x_start = csi_max_internal_s32(0, -in_x_origin);
                    const int filter_x_end =
                        csi_min_internal_s32(params->filter_width, input_width - in_x_origin);
                    const int filter_y_start = csi_max_internal_s32(0, -in_y_origin);
                    const int filter_y_end =
                        csi_min_internal_s32(params->filter_height, input_height - in_y_origin);
                    float max = FLT_MIN;
                    int32_t locat = (in_y_origin + filter_y_start) * input->dim[2] + (in_x_origin + filter_x_start);
                    for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
                        for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x) {
                            const int in_x = in_x_origin + filter_x;
                            const int in_y = in_y_origin + filter_y;
                            if (input_data[csi_get_index(input->dim, batch, in_y, in_x, channel)] > max){
                                max = input_data[csi_get_index(input->dim, batch, in_y, in_x, channel)];
                                locat = in_y * input->dim[2] + in_x;
                            }
                        }
                    }
                    output_data[csi_get_index(output->dim, batch, out_y, out_x, channel)] = locat;
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_maxpool2d_locat_nchw_f32(struct csi_tensor *input,
                                        struct csi_tensor *output,
                                        struct pool_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    const int batches = input->dim[0];
    const int depth = input->dim[3];
    const int input_height = input->dim[1];
    const int input_width = input->dim[2];
    const int output_height = output->dim[1];
    const int output_width = output->dim[2];

    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                for (int channel = 0; channel < depth; ++channel) {
                    const int in_x_origin = (out_x * params->stride_width) - params->pad_left;
                    const int in_y_origin = (out_y * params->stride_height) - params->pad_top;
                    // Compute the boundaries of the filter region clamped so as to
                    // ensure that the filter window fits in the input array.
                    const int filter_x_start = csi_max_internal_s32(0, -in_x_origin);
                    const int filter_x_end =
                        csi_min_internal_s32(params->filter_width, input_width - in_x_origin);
                    const int filter_y_start = csi_max_internal_s32(0, -in_y_origin);
                    const int filter_y_end =
                        csi_min_internal_s32(params->filter_height, input_height - in_y_origin);
                    float max = FLT_MIN;
                    int locat = (in_y_origin + filter_y_start) * input->dim[2] + (in_x_origin + filter_x_start);
                    for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
                        for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x) {
                            const int in_x = in_x_origin + filter_x;
                            const int in_y = in_y_origin + filter_y;
                            if (input_data[
                            csi_get_index(input->dim, batch, in_y,in_x, channel)] > max){
                                max = input_data[csi_get_index(input->dim, batch, in_y, in_x, channel)];
                                locat = in_y * input->dim[2] + in_x;
                            }
                        }
                    }
                    output_data[csi_get_index(output->dim, batch, out_y, out_x, channel)] = locat;
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_maxpool2d_locat_nchw_i32_u8(struct csi_tensor *input,
                                           struct csi_tensor *output,
                                           struct pool_params *params)
{
    uint8_t *input_data = input->data;
    int32_t *output_data = output->data;
    const int batches = input->dim[0];
    const int depth = input->dim[1];
    const int input_height = input->dim[2];
    const int input_width = input->dim[3];
    const int output_height = output->dim[2];
    const int output_width = output->dim[3];
    int flag = 1;

    for (int batch = 0; batch < batches; ++batch) {
        for (int channel = 0; channel < depth; ++channel) {
            for (int out_y = 0; out_y < output_height; ++out_y) {
                for (int out_x = 0; out_x < output_width; ++out_x) {
                    const int in_x_origin = (out_x * params->stride_width) - params->pad_left;
                    const int in_y_origin = (out_y * params->stride_height) - params->pad_top;
                    // Compute the boundaries of the filter region clamped so as to
                    // ensure that the filter window fits in the input array.
                    const int filter_x_start = csi_max_internal_s32(0, -in_x_origin);
                    const int filter_x_end =
                        csi_min_internal_s32(params->filter_width, input_width - in_x_origin);
                    const int filter_y_start = csi_max_internal_s32(0, -in_y_origin);
                    const int filter_y_end =
                        csi_min_internal_s32(params->filter_height, input_height - in_y_origin);
                    float max = FLT_MIN;
                    int32_t locat = (in_y_origin + filter_y_start) * input->dim[3] + (in_x_origin + filter_x_start);
                    for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
                        for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x) {
                            const int in_x = in_x_origin + filter_x;
                            const int in_y = in_y_origin + filter_y;
                            if (input_data[csi_get_index(input->dim, batch, channel, in_y, in_x)] > max){
                                max = input_data[csi_get_index(input->dim, batch, channel, in_y, in_x)];
                                locat = in_y * input->dim[3] + in_x;
                            }
                        }
                    }
                    output_data[csi_get_index(output->dim, batch, channel, out_y, out_x)] = locat;
                }
            }
        }
    }
    return CSINN_TRUE;
}

int csi_maxpool2d_locat_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct pool_params *params)
{
    if (params->layout == CSINN_NCHW) {
        csi_maxpool2d_locat_nchw_f32(input, output, params);
    } else if (params->layout == CSINN_NHWC) {
        csi_maxpool2d_locat_nhwc_f32(input, output, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_maxpool2d_locat_i32_u8(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct pool_params *params)
{
    if (params->layout == CSINN_NCHW) {
        csi_maxpool2d_locat_nchw_i32_u8(input, output, params);
    } else if (params->layout == CSINN_NHWC) {
        csi_maxpool2d_locat_nhwc_i32_u8(input, output, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_maxpool2d_locat_init(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct pool_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_MAXPOOL2D_LOCAT, input->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_maxpool2d_locat(struct csi_tensor *input,
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

