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

/* CSI-NN2 version 1.12.x */

#include "csi_ref.h"

int csi_ref_avgpool2d_nhwc_f32(struct csi_tensor *input, struct csi_tensor *output,
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
                    const int filter_x_start = csi_ref_max_internal_s32(0, -in_x_origin);
                    const int filter_x_end =
                        csi_ref_min_internal_s32(params->filter_width, input_width - in_x_origin);
                    const int filter_y_start = csi_ref_max_internal_s32(0, -in_y_origin);
                    const int filter_y_end =
                        csi_ref_min_internal_s32(params->filter_height, input_height - in_y_origin);
                    float total = 0.f;
                    float filter_count = 0;
                    for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
                        for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x) {
                            const int in_x = in_x_origin + filter_x;
                            const int in_y = in_y_origin + filter_y;
                            total += input_data[csi_ref_get_index(input->dim, batch, in_y, in_x,
                                                                  channel)];
                            filter_count++;
                        }
                    }
                    if (params->count_include_pad) {
                        filter_count = params->filter_height * params->filter_width;
                    }
                    const float average = total / filter_count;
                    output_data[csi_ref_get_index(output->dim, batch, out_y, out_x, channel)] =
                        average;
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_ref_avgpool2d_nchw_f32(struct csi_tensor *input, struct csi_tensor *output,
                                      struct pool_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    const int batches = input->dim[0];
    const int depth = input->dim[1];
    const int input_height = input->dim[2];
    const int input_width = input->dim[3];
    const int output_height = output->dim[2];
    const int output_width = output->dim[3];

    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                for (int channel = 0; channel < depth; ++channel) {
                    const int in_x_origin = (out_x * params->stride_width) - params->pad_left;
                    const int in_y_origin = (out_y * params->stride_height) - params->pad_top;
                    // Compute the boundaries of the filter region clamped so as to
                    // ensure that the filter window fits in the input array.
                    const int filter_x_start = csi_ref_max_internal_s32(0, -in_x_origin);
                    const int filter_x_end =
                        csi_ref_min_internal_s32(params->filter_width, input_width - in_x_origin);
                    const int filter_y_start = csi_ref_max_internal_s32(0, -in_y_origin);
                    const int filter_y_end =
                        csi_ref_min_internal_s32(params->filter_height, input_height - in_y_origin);
                    float total = 0.f;
                    float filter_count = 0;
                    for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
                        for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x) {
                            const int in_x = in_x_origin + filter_x;
                            const int in_y = in_y_origin + filter_y;
                            total += input_data[csi_ref_get_index(input->dim, batch, channel, in_y,
                                                                  in_x)];
                            filter_count++;
                        }
                    }
                    if (params->count_include_pad) {
                        filter_count = params->filter_height * params->filter_width;
                    }
                    const float average = total / filter_count;
                    output_data[csi_ref_get_index(output->dim, batch, channel, out_y, out_x)] =
                        average;
                }
            }
        }
    }
    return CSINN_TRUE;
}

int csi_ref_avgpool2d_f32(struct csi_tensor *input, struct csi_tensor *output,
                          struct pool_params *params)
{
    if (params->base.layout == CSINN_LAYOUT_NCHW) {
        csi_ref_avgpool2d_nchw_f32(input, output, params);
    } else if (params->base.layout = CSINN_LAYOUT_NHWC) {
        csi_ref_avgpool2d_nhwc_f32(input, output, params);
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
}

int csi_ref_avgpool2d_quant(struct csi_tensor *input, struct csi_tensor *output,
                            struct pool_params *params)
{
    return csi_ref_siso_callback_base(input, output, params, csi_ref_avgpool2d_f32);
}
