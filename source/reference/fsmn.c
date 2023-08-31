/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
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

static float fsmn(float x) { return x > 0 ? x : 0; }

int shl_ref_fsmn_f32(struct csinn_tensor *frame, struct csinn_tensor *l_filter,
                     struct csinn_tensor *r_filter, struct csinn_tensor *frame_sequence,
                     struct csinn_tensor *frame_counter, struct csinn_tensor *output,
                     struct csinn_fsmn_params *params)
{
    float *last_frame = frame->data;
    float *past_filter = l_filter->data;
    float *future_filter = r_filter->data;
    float *sequence_frame = frame_sequence->data;
    int32_t *frame_count = frame_counter->data;
    float *output_data = output->data;

    int len_order = frame_sequence->dim[0];
    int length = frame_sequence->dim[1];

    for (int i = 0; i < length; i++) output_data[i] = 0.0;

    frame_count[0]++;
    // set last frame to sequence tail.
    if (frame_count[0] > params->unavailable_frames) {
        for (int i = 0; i < len_order; i++) {
            for (int j = 0; j < length; j++) {
                int new_index = i * length + j;
                if (i == (len_order - 1)) {
                    sequence_frame[new_index] = last_frame[j];
                } else {
                    int original_index = (i + 1) * length + j;
                    sequence_frame[new_index] = sequence_frame[original_index];
                }
            }
        }
    }

    // past frame
    for (int k = 0; k < params->l_order; k++) {
        for (int l = 0; l < length; l++) {
            int in_index = k * params->l_stride * length + l;
            int filter_index = (params->l_order - k - 1) * length + l;
            output_data[l] = past_filter[filter_index] * sequence_frame[in_index] + output_data[l];
        }
    }

    //  current frame
    for (int m = 0; m < length; m++) {
        int in_index = (params->l_order - 1) * length * params->l_stride + m;
        output_data[m] = sequence_frame[in_index] + output_data[m];
    }

    // future frame
    for (int m = 0; m < params->r_order; m++) {
        for (int n = 0; n < length; n++) {
            int in_index =
                m * params->r_stride * length + n + params->l_order * params->l_stride * length;
            int filter_index = m * length + n;
            output_data[n] =
                future_filter[filter_index] * sequence_frame[in_index] + output_data[n];
        }
    }

    return CSINN_TRUE;
}

int shl_ref_fsmn_quant(struct csinn_tensor *frame, struct csinn_tensor *l_filter,
                       struct csinn_tensor *r_filter, struct csinn_tensor *frame_sequence,
                       struct csinn_tensor *frame_count, struct csinn_tensor *output,
                       struct csinn_fsmn_params *params)
{
    struct csinn_tensor *float_frame = shl_ref_tensor_transform_f32(frame);
    struct csinn_tensor *float_l_filter = shl_ref_tensor_transform_f32(l_filter);
    struct csinn_tensor *float_r_filter = shl_ref_tensor_transform_f32(r_filter);
    struct csinn_tensor *float_frame_sequence = shl_ref_tensor_transform_f32(frame_sequence);
    struct csinn_tensor *float_output = shl_ref_tensor_transform_f32(output);

    int ret = shl_ref_fsmn_f32(float_frame, float_l_filter, float_r_filter, float_frame_sequence,
                               frame_count, float_output, params);
    csinn_tensor_data_convert(output, float_output);
    csinn_tensor_data_convert(frame_sequence, float_frame_sequence);
    shl_ref_tensor_transform_free_f32(float_frame);
    shl_ref_tensor_transform_free_f32(float_output);
    shl_ref_tensor_transform_free_f32(float_l_filter);
    shl_ref_tensor_transform_free_f32(float_r_filter);
    shl_ref_tensor_transform_free_f32(float_frame_sequence);
    return ret;
}
