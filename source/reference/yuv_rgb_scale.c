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

/* https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L3279-L3325 line 3279*/

static int csi_yuv_rgb_scale_f32(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct siso_params *params)
{
    float *input_data = input->data;
    float *output_data = output->data;

    for(int n = 0; n < input->dim[0]; n++){
        for(int h = 0; h < input->dim[1]; h++){
            for(int w = 0; w < input->dim[2]; w++){
                float y = input_data[0];
                float u = input_data[1];
                float v = input_data[2];

                float r = y + 1.13988303 * v;
                float g = y -0.394642334 * u - 0.58062185 * v;
                float b = y + 2.03206185 * u;

                input_data += 3;
                output_data[0] = r;
                output_data[1] = g;
                output_data[2] = b;
                output_data += 3;
            }
        }
    }

    return CSINN_TRUE;
}

static int csi_yuv_rgb_scale_u8(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct siso_params *params)
{

    uint8_t *input_data = input->data;
    uint8_t *output_data = output->data;

    for(int n = 0; n < input->dim[0]; n++){
        for(int h = 0; h < input->dim[1]; h++){
            for(int w = 0; w < input->dim[2]; w++){
                float y = csi_dequantize_f32(input_data[0], input->offset, input->multiplier, input->shift);
                float u = csi_dequantize_f32(input_data[1], input->offset, input->multiplier, input->shift);
                float v = csi_dequantize_f32(input_data[2], input->offset, input->multiplier, input->shift);

                float r = y + 1.13988303 * v;
                float g = y - 0.394642334 * u - 0.58062185 * v;
                float b = y + 2.03206185 * u;

                input_data += 3;
                output_data[0] = csi_quantize_f32(r, output->offset, output->multiplier, output->shift);
                output_data[1] = csi_quantize_f32(g, output->offset, output->multiplier, output->shift);
                output_data[2] = csi_quantize_f32(b, output->offset, output->multiplier, output->shift);
                output_data += 3;
            }
        }
    }

    return CSINN_TRUE;
}

int csi_yuv_rgb_scale_init(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct siso_params *params)
{
    if (input->dtype == CSINN_DTYPE_UINT8) {
        params->bc = csi_yuv_rgb_scale_u8;
    } else if (input->dtype == CSINN_DTYPE_FLOAT32) {
        params->bc = csi_yuv_rgb_scale_f32;
    } else {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_yuv_rgb_scale(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct siso_params *params)
{
    if (params->bc != NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}
