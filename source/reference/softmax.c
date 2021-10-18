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

static int csi_softmax_nhwc_f32(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct softmax_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    const int trailing_dim = input->dim_count - 1;
    int outer_size = 1;
    const int depth = input->dim[trailing_dim];

    for (int i = 0; i < trailing_dim; i++) {
        outer_size *= input->dim[i];
    }

    for (int i = 0; i < outer_size; ++i) {
        // Find max element value which we'll use to ensure numerical stability
        // taking advantage of the following equality:
        // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
        float max = FLT_MIN;
        for (int c = 0; c < depth; ++c) {
            max = fmax(max, input_data[i * depth + c]);
        }

        // Compute sum.
        float sum = 0.f;
        for (int c = 0; c < depth; ++c) {
            sum += exp(input_data[i * depth + c] - max);
        }

        // Compute result.
        for (int c = 0; c < depth; ++c) {
            output_data[i * depth + c] = exp(input_data[i * depth + c] - max) / sum;
        }
    }
    return CSINN_TRUE;
}

static int csi_softmax_nhwc_u8(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct softmax_params *params)
{
    float *float_input_data;
    float *float_output_data;
    struct csi_tensor float_input;
    struct csi_tensor float_output;
    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    int size = 1;

    for (int i = 0; i < input->dim_count; i++) {
        size *= input->dim[i];
    }

    memcpy(&float_input, input, sizeof(struct csi_tensor));
    memcpy(&float_output, output, sizeof(struct csi_tensor));
    float_input_data = malloc(size * sizeof(float));
    float_output_data = malloc(size * sizeof(float));
    float_input.data = float_input_data;
    float_output.data = float_output_data;

    for (int i = 0; i < size; i++) {
        float_input_data[i] = csi_dequantize_f32(input_data[i], input->offset,
                                                 input->multiplier, input->shift);
    }

    csi_softmax_nhwc_f32(&float_input, &float_output, params);

    for (int i = 0; i < size; i++) {
        output_data[i] = csi_quantize_f32(float_output_data[i], output->offset,
                                          output->multiplier, output->shift);
    }
    free(float_input_data);
    free(float_output_data);
    return CSINN_TRUE;
}


static int csi_softmax_nchw_f32(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct softmax_params *params)
{
    // assert(input->dim_count - 1 >= axis);

    float *input_data = input->data;
    float *output_data = output->data;
    if (params->axis == -1) {
        params->axis = input->dim_count - 1;
    }
    const int depth = input->dim[params->axis];
    for (int i = input->dim_count; i < 4; i++) {
        input->dim[i] = 1;
    }
    input->dim_count = 4;
    int size[3];
    int tmp = 0;
    for (int i = 0; i < input->dim_count; i++) {
        if(i != params->axis){
            size[tmp] = input->dim[i];
            tmp++;
        }
    }
    tmp = 0;
    int dim[4];
    for (int n = 0; n < size[0]; ++n) {
        for (int i = 0; i < size[1]; ++i) {
            for (int j = 0; j < size[2]; ++j) {
                // Find max element value which we'll use to ensure numerical stability
                // taking advantage of the following equality:
                // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))

                int t_[] = {n, i, j};
                for (int k = 0; k < input->dim_count; k++) {
                    if(k != params->axis){
                        dim[k] = t_[tmp];
                        tmp++;
                    }
                }
                float max = FLT_MIN;
                for (int c = 0; c < depth; ++c) {
                    dim[params->axis] = c;
                    int32_t in_index = csi_get_index(input->dim, dim[0], dim[1], dim[2], dim[3]);
                    max = fmax(max, input_data[in_index]);
                }

                // Compute sum.
                float sum = 0.f;
                for (int c = 0; c < depth; ++c) {
                    dim[params->axis] = c;
                    int32_t in_index = csi_get_index(input->dim, dim[0], dim[1], dim[2], dim[3]);
                    sum += exp(input_data[in_index] - max);
                }

                // Compute result.
                for (int c = 0; c < depth; ++c) {
                    dim[params->axis] = c;
                    int32_t in_index = csi_get_index(input->dim, dim[0], dim[1], dim[2], dim[3]);
                    output_data[in_index] = exp(input_data[in_index] - max) / sum;
                }
                tmp = 0;
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_softmax_nchw_u8(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct softmax_params *params)
{
    // assert(input->dim_count - 1 == axis);
    float *float_input_data;
    float *float_output_data;
    struct csi_tensor float_input;
    struct csi_tensor float_output;
    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    int size = 1;

    for (int i = 0; i < input->dim_count; i++) {
        size *= input->dim[i];
    }

    memcpy(&float_input, input, sizeof(struct csi_tensor));
    memcpy(&float_output, output, sizeof(struct csi_tensor));
    float_input_data = malloc(size * sizeof(float));
    float_output_data = malloc(size * sizeof(float));
    float_input.data = float_input_data;
    float_output.data = float_output_data;

    for (int i = 0; i < size; i++) {
        float_input_data[i] = csi_dequantize_f32(input_data[i], input->offset,
                                                 input->multiplier, input->shift);
    }

    csi_softmax_nchw_f32(&float_input, &float_output, params);

    for (int i = 0; i < size; i++) {
        output_data[i] = csi_quantize_f32(float_output_data[i], output->offset,
                                          output->multiplier, output->shift);
    }
    free(float_input_data);
    free(float_output_data);
    return CSINN_TRUE;
}


int csi_softmax_init(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct softmax_params *params)
{
    if (params->layout == CSINN_NCHW) {
        if (input->dtype == CSINN_DTYPE_UINT8) {
            params->bc = csi_softmax_nchw_u8;
        } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
            params->bc = csi_softmax_nchw_f32;
        } else {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else if (params->layout = CSINN_NHWC) {
        if (input->dtype == CSINN_DTYPE_UINT8) {
            params->bc = csi_softmax_nhwc_u8;
        } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
            params->bc = csi_softmax_nhwc_f32;
        } else {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    return CSINN_TRUE;
}

int csi_softmax(struct csi_tensor *input,
                struct csi_tensor *output,
                struct softmax_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}