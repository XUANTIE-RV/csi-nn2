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

/* CSI-NN2 version 1.10.x */

#include "csi_ref.h"
#include "csi_utils.h"

int csi_ref_l2pool_f32(struct csi_tensor *input,
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
                    const int filter_x_start = csi_ref_max_internal_s32(0, -in_x_origin);
                    const int filter_x_end =
                        csi_ref_min_internal_s32(params->filter_width, input_width - in_x_origin);
                    const int filter_y_start = csi_ref_max_internal_s32(0, -in_y_origin);
                    const int filter_y_end =
                        csi_ref_min_internal_s32(params->filter_height, input_height - in_y_origin);
                    float sum_squares = 0.f;
                    int filter_count = 0;
                    for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
                        for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x) {
                            const int in_x = in_x_origin + filter_x;
                            const int in_y = in_y_origin + filter_y;
                            const float val =
                                input_data[csi_ref_get_index(input->dim, batch, in_y, in_x, channel)];
                            sum_squares += val * val;
                            filter_count++;
                        }
                    }
                    const float l2pool_result = sqrt(sum_squares / filter_count);
                    output_data[csi_ref_get_index(output->dim, batch, out_y, out_x, channel)] = l2pool_result;
                }
            }
        }
    }
    return CSINN_TRUE;
}
