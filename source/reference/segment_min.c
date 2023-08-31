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

int shl_ref_unsorted_segment_min_f32(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                     struct csinn_tensor *output,
                                     struct csinn_segment_params *params)
{
    float *input_data = input->data;
    int *segment_data = segment_ids->data;
    float *output_data = output->data;

    int input_dim = input->dim_count;
    int num_segments = params->num_segments;

    for (int n = 0; n < num_segments; n++) {
        /* init the outputdata data */
        for (int h = 0; h < input->dim[1]; h++) {
            for (int w = 0; w < input->dim[2]; w++) {
                for (int c = 0; c < input->dim[3]; c++) {
                    int32_t output_index = shl_ref_get_index(input->dim, n, h, w, c);
                    output_data[output_index] = FLT_MAX;
                }
            }
        }
        int flag = 0;
        for (int i = 0; i < input->dim[0]; i++) {
            if (segment_data[i] == n) {
                flag = 1;
            }
            if (flag) {
                for (int h = 0; h < input->dim[1]; h++) {
                    for (int w = 0; w < input->dim[2]; w++) {
                        for (int c = 0; c < input->dim[3]; c++) {
                            int32_t input_index = shl_ref_get_index(input->dim, i, h, w, c);
                            int32_t output_index = shl_ref_get_index(input->dim, n, h, w, c);
                            output_data[output_index] =
                                input_data[input_index] < output_data[output_index]
                                    ? input_data[input_index]
                                    : output_data[output_index];
                        }
                    }
                }
                flag = 0;
            }
        }
    }

    return CSINN_TRUE;
}

int shl_ref_segment_min_f32(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                            struct csinn_tensor *output, struct csinn_segment_params *params)
{
    float *input_data = input->data;
    int *segment_data = segment_ids->data;
    float *output_data = output->data;

    int input_dim = input->dim_count;
    int num_segments = params->num_segments;
    int i = 0;

    for (int n = 0; n < num_segments; n++) {
        /* init the outputdata data */
        for (int h = 0; h < input->dim[1]; h++) {
            for (int w = 0; w < input->dim[2]; w++) {
                for (int c = 0; c < input->dim[3]; c++) {
                    int32_t output_index = shl_ref_get_index(input->dim, n, h, w, c);
                    output_data[output_index] = FLT_MAX;
                }
            }
        }
        int flag = 0;
        for (; i < input->dim[0]; i++) {
            if (segment_data[i] == n) {
                flag = 1;
            } else {
                break;
            }
            if (flag) {
                for (int h = 0; h < input->dim[1]; h++) {
                    for (int w = 0; w < input->dim[2]; w++) {
                        for (int c = 0; c < input->dim[3]; c++) {
                            int32_t input_index = shl_ref_get_index(input->dim, i, h, w, c);
                            int32_t output_index = shl_ref_get_index(input->dim, n, h, w, c);
                            output_data[output_index] =
                                input_data[input_index] < output_data[output_index]
                                    ? input_data[input_index]
                                    : output_data[output_index];
                        }
                    }
                }
                flag = 0;
            }
        }
    }

    return CSINN_TRUE;
}

int shl_ref_unsorted_segment_min_quant(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                                       struct csinn_tensor *output,
                                       struct csinn_segment_params *params)
{
    int ret;
    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    ret = shl_ref_unsorted_segment_min_f32(finput, segment_ids, foutput, params);
    csinn_tensor_data_convert(output, foutput);
    shl_ref_tensor_transform_free_f32(finput);
    shl_ref_tensor_transform_free_f32(foutput);
    return ret;
}

int shl_ref_segment_min_quant(struct csinn_tensor *input, struct csinn_tensor *segment_ids,
                              struct csinn_tensor *output, struct csinn_segment_params *params)
{
    int ret;
    struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    ret = shl_ref_segment_min_f32(finput, segment_ids, foutput, params);
    csinn_tensor_data_convert(output, foutput);
    shl_ref_tensor_transform_free_f32(finput);
    shl_ref_tensor_transform_free_f32(foutput);
    return ret;
}
