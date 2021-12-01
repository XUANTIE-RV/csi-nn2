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
#include <assert.h>

static float fsmn(float x){
	return x > 0 ? x : 0;
}

int csi_ref_fsmn_f32(struct csi_tensor *frame,
                     struct csi_tensor *l_filter,
                     struct csi_tensor *r_filter,
                     struct csi_tensor *frame_sequence,
                     struct csi_tensor *frame_counter,
                     struct csi_tensor *output,
                     struct fsmn_params *params)
{
    float *last_frame = frame->data;
    float *past_filter = l_filter->data;
    float *future_filter = r_filter->data;
    float *sequence_frame = frame_sequence->data;
    int32_t *frame_count = frame_counter->data;
    float *output_data = output->data;

    int len_order = frame_sequence->dim[0];
    int length  = frame_sequence->dim[1];

    for (int i = 0; i < length; i++)
        output_data[i] = 0.0;

    frame_count[0]++;
    // set last frame to sequence tail.
    if(frame_count[0] > params->unavailable_frames){
        for(int i = 0; i < len_order; i++){
            for (int j = 0; j < length; j++){
                int new_index = i * length + j;
                if(i == (len_order - 1)){
                    sequence_frame[new_index] = last_frame[j];
                }else{
                    int original_index = (i + 1) * length + j;
                    sequence_frame[new_index] = sequence_frame[original_index];
                }
            }
        }

    }

    // past frame
    for (int k = 0; k < params->l_order; k++){
        for( int l = 0; l < length; l++){
            int in_index = k * params->l_stride * length + l;
            int filter_index = (params->l_order - k - 1) * length + l;
            output_data[l] = past_filter[filter_index] * sequence_frame[in_index] + output_data[l];

        }

    }

    //  current frame
    for (int m = 0; m < length; m++){
        int in_index = (params->l_order - 1) * length * params->l_stride + m;
        output_data[m] = sequence_frame[in_index] + output_data[m];
    }

    // future frame
    for(int m = 0; m < params->r_order; m++){
        for(int n = 0; n < length; n++){
            int in_index = m * params->r_stride * length + n + params->l_order * params->l_stride * length;
            int filter_index = m * length + n;
            output_data[n] = future_filter[filter_index] * sequence_frame[in_index] + output_data[n];
        }
    }

    return CSINN_TRUE;
}

int csi_ref_fsmn_quant(struct csi_tensor *frame,
                     struct csi_tensor *l_filter,
                     struct csi_tensor *r_filter,
                     struct csi_tensor *frame_sequence,
                     struct csi_tensor *frame_count,
                     struct csi_tensor *output,
                     struct fsmn_params *params)
{
    struct csi_tensor *float_frame = csi_ref_tensor_transform_f32(frame);
    struct csi_tensor *float_l_filter = csi_ref_tensor_transform_f32(l_filter);
    struct csi_tensor *float_r_filter = csi_ref_tensor_transform_f32(r_filter);
    struct csi_tensor *float_frame_sequence = csi_ref_tensor_transform_f32(frame_sequence);
    struct csi_tensor *float_output = csi_ref_tensor_transform_f32(output);

    int ret = csi_ref_fsmn_f32(float_frame, float_l_filter, float_r_filter, float_frame_sequence, frame_count, float_output, params);
    csi_tensor_data_convert(output, float_output);
    csi_tensor_data_convert(frame_sequence, float_frame_sequence);
    csi_ref_tensor_transform_free_f32(float_frame);
    csi_ref_tensor_transform_free_f32(float_output);
    csi_ref_tensor_transform_free_f32(float_l_filter);
    csi_ref_tensor_transform_free_f32(float_r_filter);
    csi_ref_tensor_transform_free_f32(float_frame_sequence);
    return ret;
}
