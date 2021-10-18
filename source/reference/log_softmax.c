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

// logsoftmax = logits - log(reduce_sum(exp(logits), axis))
// static int csi_log_softmax_f32(struct csi_tensor *input,
//                             struct csi_tensor *output,
//                             struct softmax_params *params)
// {
//     // now only support 2D input
//     assert(params->axis == 1 && input->dim_count == 2);
//     float *input_data = (float *)input->data;
//     float *output_data = (float *)output->data;

//     int in_size = 1, out_size = 1;
//     for(int i = 0; i < input->dim_count; i++) {
//         in_size *= input->dim[i];
//     }
//     out_size = in_size;
//     int input_outer_size = 1;
//     for(int i = 0; i < params->axis; i++) {
//         input_outer_size *= input->dim[i];
//     }
//     int input_inner_size = 1;
//     for(int i = params->axis + 1; i < input->dim_count; i++) {
//         input_inner_size *= input->dim[i];
//     }
//     int axis_dim = input->dim[params->axis];


//     struct csi_tensor *input_1 = (struct csi_tensor *)malloc(sizeof(struct csi_tensor));
//     memcpy(input_1, input, sizeof(struct csi_tensor));
//     struct csi_tensor output_1;
//     struct reduce_params rparams;

//     input_1->data = (float *)malloc(in_size * sizeof(float));
//     float *input_1_data = (float *)input_1->data;
//     memcpy(input_1->data, (float *)input->data, in_size * sizeof(float));

//     for(int i = 0; i < in_size; i++) {
//         input_1_data[i] = exp(input_1_data[i]);
//     }

//     output_1.data = (float *)malloc(in_size / axis_dim * sizeof(float));
//     float *output_1_data = (float *)output_1.data;

//     rparams.axis_count = 1;
//     rparams.axis = (int *)malloc(sizeof(int) * rparams.axis_count);
//     rparams.axis[0] = params->axis;
//     csi_reduce_sum_init(input_1, &output_1, &rparams);
//     csi_reduce_sum(input_1, &output_1, &rparams);

//     for(int i = 0; i < input_outer_size; i++) {
//         for(int j = 0; j < axis_dim; j++) {
//             for(int k = 0; k < input_inner_size; k++) {
//                 int index1 = (i * axis_dim + j) * input_inner_size + k;
//                 int index2 = i * input_inner_size + k;
//                 output_data[index1] = input_data[index1] - log(output_1_data[index2]);
//             }
//         }
//     }
//     free(input_1->data);
//     free(output_1.data);
//     free(rparams.axis);
//     return CSINN_TRUE;
// }


/* logsoftmax = logits - log(reduce_sum(exp(logits), axis)) */
static int csi_log_softmax_f32(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct softmax_params *params)
{
    // now only support 2D input
    assert(params->axis == 1 && input->dim_count == 2);
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int in_size = 1, out_size = 1;
    for(int i = 0; i < input->dim_count; i++) {
        in_size *= input->dim[i];
    }
    out_size = in_size;
    int input_outer_size = 1;
    for(int i = 0; i < params->axis; i++) {
        input_outer_size *= input->dim[i];
    }
    int input_inner_size = 1;
    for(int i = params->axis + 1; i < input->dim_count; i++) {
        input_inner_size *= input->dim[i];
    }
    int axis_dim = input->dim[params->axis];

    for(int i = 0; i < input_outer_size; i++) {
        for(int k = 0; k < input_inner_size; k++) {
            float acc = 0.0f;
            float input_val = 0.0f;
            for(int j = 0; j < axis_dim; j++) {
                input_val = *(input_data + j * input_inner_size + k);
                acc += exp(input_val);
            }
            acc = log(acc);
            for(int j = 0; j < axis_dim; j++) {
                *(output_data + j * input_inner_size + k) = *(input_data + j * input_inner_size + k) - acc;
            }
        }
        input_data += input_inner_size * axis_dim;
        output_data += input_inner_size * axis_dim;
    }
    return CSINN_TRUE;
}

static int csi_log_softmax_u8(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct softmax_params *params)
{
    // now only support 2D input
    assert(params->axis == 1 && input->dim_count == 2);
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;

    int in_size = 1, out_size = 1;
    for(int i = 0; i < input->dim_count; i++) {
        in_size *= input->dim[i];
    }
    out_size = in_size;
    int input_outer_size = 1;
    for(int i = 0; i < params->axis; i++) {
        input_outer_size *= input->dim[i];
    }
    int input_inner_size = 1;
    for(int i = params->axis + 1; i < input->dim_count; i++) {
        input_inner_size *= input->dim[i];
    }
    int axis_dim = input->dim[params->axis];

    for(int i = 0; i < input_outer_size; i++) {
        for(int k = 0; k < input_inner_size; k++) {
            float acc = 0.0f;
            float input_temp = 0.0f;
            for(int j = 0; j < axis_dim; j++) {
                uint8_t input_val = *(input_data + j * input_inner_size + k);
                input_temp = csi_dequantize_f32(input_val, input->offset, input->multiplier, input->shift);
                acc += exp(input_temp);
            }
            acc = log(acc);
            for(int j = 0; j < axis_dim; j++) {
                input_temp = csi_dequantize_f32(*(input_data + j * input_inner_size + k), input->offset, input->multiplier, input->shift);
                *(output_data + j * input_inner_size + k) = csi_quantize_f32(input_temp - acc, output->offset, output->multiplier, output->shift);
            }
        }
        input_data += input_inner_size * axis_dim;
        output_data += input_inner_size * axis_dim;
    }
    return CSINN_TRUE;
}

int csi_log_softmax_init(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct softmax_params *params)
{
    if(input->dtype == CSINN_DTYPE_UINT8) {
        params->bc = csi_log_softmax_u8;
    } else if(input->dtype == CSINN_DTYPE_FLOAT32) {
        params->bc = csi_log_softmax_f32;
    } else {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_log_softmax(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct softmax_params *params)
{
    if(params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}