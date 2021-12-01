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

int csi_ref_unsorted_segment_mean_f32(struct csi_tensor *input,
                                      struct csi_tensor *segment_ids,
                                      struct csi_tensor *output,
                                      struct segment_params *params)
{
    float *input_data  = input->data;
    int *segment_data  = segment_ids->data;
    float *output_data = output->data;

    int input_dim    = input->dim_count;
    int num_segments = params->num_segments;
    int index[input->dim[0]];

    for(int n = 0; n < num_segments; n++) {
        /* init the outputdata data */
        for(int h = 0; h < input->dim[1]; h++) {
            for(int w = 0; w < input->dim[2]; w++) {
                for(int c = 0; c < input->dim[3]; c++) {
                    int32_t output_index = csi_ref_get_index(input->dim, n, h, w, c);
                    output_data[output_index] = 0;
                }
            }
        }
        int num = 0;

        for(int i = 0; i < input->dim[0]; i++) {
            if (segment_data[i] == n) {
                index[num] = i;
                num++;
            }
        }
        int mean_n = num;
        if(num > 0) {
            for(int h = 0; h < input->dim[1]; h++) {
                for(int w = 0; w < input->dim[2]; w++) {
                    for(int c = 0; c < input->dim[3]; c++) {
                        int32_t output_index = csi_ref_get_index(input->dim, n, h, w, c);
                        for(int k = 0; k < num; k++) {
                            int32_t input_index = csi_ref_get_index(input->dim, index[k], h, w, c);
                            output_data[output_index] +=  input_data[input_index];
                        }
                        output_data[output_index] /= mean_n;
                    }
                }
            }
        }
    }

    return CSINN_TRUE;
}

int csi_ref_segment_mean_f32(struct csi_tensor *input,
                             struct csi_tensor *segment_ids,
                             struct csi_tensor *output,
                             struct segment_params *params)
{
    float *input_data  = input->data;
    int *segment_data  = segment_ids->data;
    float *output_data = output->data;

    int input_dim    = input->dim_count;
    int num_segments = params->num_segments;
    int index[input->dim[0]];
    int i = 0;

    for(int n = 0; n < num_segments; n++) {
        /* init the outputdata data */
        for(int h = 0; h < input->dim[1]; h++) {
            for(int w = 0; w < input->dim[2]; w++) {
                for(int c = 0; c < input->dim[3]; c++) {
                    int32_t output_index = csi_ref_get_index(input->dim, n, h, w, c);
                    output_data[output_index] = 0;
                }
            }
        }
        int num = 0;
        for(; i < input->dim[0]; i++) {
            if (segment_data[i] == n) {
                index[num] = i;
                num++;
            } else {
                break;
            }
        }
        int mean_n = num;
        if(num > 0) {
            for(int h = 0; h < input->dim[1]; h++) {
                for(int w = 0; w < input->dim[2]; w++) {
                    for(int c = 0; c < input->dim[3]; c++) {
                        int32_t output_index = csi_ref_get_index(input->dim, n, h, w, c);
                        for(int k = 0; k < num; k++) {
                            int32_t input_index = csi_ref_get_index(input->dim, index[k], h, w, c);
                            output_data[output_index] +=  input_data[input_index];
                        }
                        output_data[output_index] /= mean_n;
                    }
                }
            }
        }
    }

    return CSINN_TRUE;
}

int csi_ref_unsorted_segment_mean_quant(struct csi_tensor *input,
                                        struct csi_tensor *segment_ids,
                                        struct csi_tensor *output,
                                        struct segment_params *params)
{
    int ret;
    struct csi_tensor *finput = csi_ref_tensor_transform_f32(input);
    struct csi_tensor *foutput = csi_ref_tensor_transform_f32(output);
    ret = csi_ref_unsorted_segment_mean_f32(finput, segment_ids, foutput, params);
    csi_tensor_data_convert(output, foutput);
    csi_ref_tensor_transform_free_f32(finput);
    csi_ref_tensor_transform_free_f32(foutput);
    return ret;
}

int csi_ref_segment_mean_quant(struct csi_tensor *input,
                               struct csi_tensor *segment_ids,
                               struct csi_tensor *output,
                               struct segment_params *params)
{
    int ret;
    struct csi_tensor *finput = csi_ref_tensor_transform_f32(input);
    struct csi_tensor *foutput = csi_ref_tensor_transform_f32(output);
    ret = csi_ref_segment_mean_f32(finput, segment_ids, foutput, params);
    csi_tensor_data_convert(output, foutput);
    csi_ref_tensor_transform_free_f32(finput);
    csi_ref_tensor_transform_free_f32(foutput);
    return ret;
}
