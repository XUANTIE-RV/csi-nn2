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

/* CSI-NN2 version 1.8.x */

#include "csi_ref.h"
#include "csi_utils.h"

/* https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L3279-L3325 line 3279*/

int csi_ref_yuv_rgb_scale_f32(struct csi_tensor *input,
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

int csi_ref_yuv_rgb_scale_quant(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct siso_params *params)
{
    return csi_ref_siso_callback_base(input, output, params, csi_ref_yuv_rgb_scale_f32);
}
