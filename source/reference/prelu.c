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

static int csi_prelu_nhwc_f32(struct csi_tensor *input,
                              struct csi_tensor *alpha,
                              struct csi_tensor *output,
                              struct prelu_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    float *alpha_data = alpha->data;
    for (int b = 0; b < output->dim[0]; ++b) {
        for (int y = 0; y < output->dim[1]; ++y) {
            for (int x = 0; x < output->dim[2]; ++x) {
                for (int c = 0; c < output->dim[3]; ++c) {
                    int output_index = csi_get_index(output->dim, b, y, x, c);
                    int input_index = csi_get_index(input->dim, b, y, x, c);
                    float input_value = input_data[input_index];
                    if (input_value >= 0) {
                        output_data[output_index] = input_data[input_index];
                    } else {
                        output_data[output_index] = input_value * alpha_data[c];
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_prelu_nhwc_u8(struct csi_tensor *input,
                             struct csi_tensor *alpha,
                             struct csi_tensor *output,
                             struct prelu_params *params)
{
    int num_elements = 1;
    for (int i = 0; i < output->dim_count; i++) {
        num_elements *= output->dim[i];
    }

    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    uint8_t *alpha_data = alpha->data;
    const int32_t input_offset = input->offset;
    const int32_t alpha_offset = alpha->offset;

    for (int b = 0; b < output->dim[0]; ++b) {
        for (int y = 0; y < output->dim[1]; ++y) {
            for (int x = 0; x < output->dim[2]; ++x) {
                for (int c = 0; c < output->dim[3]; ++c) {
                    int index = csi_get_index(input->dim, b, y, x, c);
                    const float input_value = csi_dequantize_f32(input_data[index], input->offset, input->multiplier, input->shift);
                    if (input_value >= 0) {
                        output_data[index] = csi_quantize_f32(input_value,
                                                    output->offset, output->multiplier, output->shift);
                    } else {
                        float alpha_val =  csi_dequantize_f32(alpha_data[c], alpha->offset, alpha->multiplier, alpha->shift);
                        output_data[index] = csi_quantize_f32(input_value * alpha_val,
                                                    output->offset, output->multiplier, output->shift);
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_prelu_nchw_f32(struct csi_tensor *input,
                              struct csi_tensor *alpha,
                              struct csi_tensor *output,
                              struct prelu_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;
    float *alpha_data = alpha->data;
    for (int b = 0; b < output->dim[0]; ++b) {
        for (int y = 0; y < output->dim[1]; ++y) {
            for (int x = 0; x < output->dim[2]; ++x) {
                for (int c = 0; c < output->dim[3]; ++c) {
                    int output_index = csi_get_index(output->dim, b, y, x, c);
                    int input_index = csi_get_index(input->dim, b, y, x, c);
                    const int32_t input_value = input->offset + input_data[input_index];
                    if (input_value >= 0) {
                        output_data[output_index] = input_data[input_index];
                    } else {
                        output_data[output_index] = input_value * alpha_data[c];
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

static int csi_prelu_nchw_u8(struct csi_tensor *o_input,
                             struct csi_tensor *alpha,
                             struct csi_tensor *o_output,
                             struct prelu_params *params)
{
    struct csi_tensor* input = csi_nchw_to_nhwc_u8(o_input);;
    struct csi_tensor* output = csi_nchw_to_nhwc_u8(o_output);;
    int num_elements = 1;
    for (int i = 0; i < output->dim_count; i++) {
        num_elements *= output->dim[i];
    }

    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;
    uint8_t *alpha_data = alpha->data;
    const int32_t input_offset = input->offset;
    const int32_t alpha_offset = alpha->offset;

    for (int b = 0; b < output->dim[0]; ++b) {
        for (int y = 0; y < output->dim[1]; ++y) {
            for (int x = 0; x < output->dim[2]; ++x) {
                for (int c = 0; c < output->dim[3]; ++c) {
                    int index = csi_get_index(input->dim, b, y, x, c);
                    const float input_value = csi_dequantize_f32(input_data[index], input->offset, input->multiplier, input->shift);
                    if (input_value >= 0) {
                        output_data[index] = csi_quantize_f32(input_value,
                                                    output->offset, output->multiplier, output->shift);
                    } else {
                        float alpha_val =  csi_dequantize_f32(alpha_data[c], alpha->offset, alpha->multiplier, alpha->shift);
                        output_data[index] = csi_quantize_f32(input_value * alpha_val,
                                                    output->offset, output->multiplier, output->shift);
                    }
                }
            }
        }
    }
    csi_nhwc_to_nchw_u8(o_output, output);
    return CSINN_TRUE;
}

int csi_prelu_init(struct csi_tensor *input0,
                   struct csi_tensor *input1,
                   struct csi_tensor *output,
                   struct prelu_params *params)
{
    if (params->layout == CSINN_NCHW) {
        if (input0->dtype == CSINN_DTYPE_UINT8) {
            params->bc = csi_prelu_nchw_u8;
        } else if (input0->dtype == CSINN_DTYPE_FLOAT32) {
            params->bc = csi_prelu_nchw_f32;
        } else {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else if (params->layout = CSINN_NHWC) {
        if (input0->dtype == CSINN_DTYPE_UINT8) {
            params->bc = csi_prelu_nhwc_u8;
        } else if (input0->dtype == CSINN_DTYPE_FLOAT32) {
            params->bc = csi_prelu_nhwc_f32;
        } else {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    return CSINN_TRUE;
}

int csi_prelu(struct csi_tensor *input0,
              struct csi_tensor *input1,
              struct csi_tensor *output,
              struct prelu_params *params)
{
    if (params->bc != NULL) {
        params->bc(input0, input1, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}