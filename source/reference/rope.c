/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

int shl_ref_rope_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_rope_params *params)
{
    float freq_base = params->freq_base;
    float freq_scale = params->freq_scale;
    float xpos_base = params->xpos_base;
    int32_t xpos_down = params->xpos_down;
    int n_dims = params->n_dims;

    float theta_scale = powf(freq_base, -2.0f / n_dims);

    float *src_data = input->data;
    float *dst_data = output->data;
    int32_t *pos = params->pos;

    if (!params->use_rope_cache) {
        for (int i3 = 0; i3 < input->dim[0]; i3++) {
            for (int i2 = 0; i2 < input->dim[1]; i2++) {
                int p = pos[i2];
                for (int i1 = 0; i1 < input->dim[2]; i1++) {
                    float theta = freq_scale * (float)p;

                    for (int i0 = 0; i0 < input->dim[3]; i0 += 2) {
                        float cos_theta = cosf(theta);
                        float sin_theta = sinf(theta);
                        // zeta scaling for xPos only:
                        float zeta =
                            xpos_base != 0.0f
                                ? powf((i0 + 0.4f * input->dim[0]) / (1.4f * input->dim[0]),
                                       p / xpos_base)
                                : 1.0f;
                        if (xpos_down) zeta = 1.0f / zeta;

                        theta *= theta_scale;

                        int index = i3 * (input->dim[3] * input->dim[2] * input->dim[1]) +
                                    i2 * (input->dim[3] * input->dim[2]) + i1 * input->dim[3] + i0;

                        float x0 = src_data[index];
                        float x1 = src_data[index + 1];

                        dst_data[index] = x0 * cos_theta * zeta - x1 * sin_theta * zeta;
                        dst_data[index + 1] = x0 * sin_theta * zeta + x1 * cos_theta * zeta;
                    }
                }
            }
        }
    } else {
        float *rope_cache = &((float *)params->rope_cache)[pos[0] * input->dim[2] * input->dim[3]];
        for (int i3 = 0; i3 < input->dim[0]; i3++) {
            for (int i2 = 0; i2 < input->dim[1]; i2++) {
                for (int i1 = 0; i1 < input->dim[2]; i1++) {
                    for (int i0 = 0; i0 < input->dim[3]; i0 += 2) {
                        int index = i3 * (input->dim[3] * input->dim[2] * input->dim[1]) +
                                    i2 * (input->dim[3] * input->dim[2]) + i1 * input->dim[3] + i0;

                        int rope_cache_index =
                            i2 * (input->dim[3] * input->dim[2]) + i1 * input->dim[3] + i0;

                        float x0 = src_data[index];
                        float x1 = src_data[index + 1];
                        float sin_theta = rope_cache[index];
                        float cos_theta = rope_cache[index + 1];

                        dst_data[index] = x0 * cos_theta - x1 * sin_theta;
                        dst_data[index + 1] = x0 * sin_theta + x1 * cos_theta;
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}

int shl_ref_rope_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_rope_params *params)
{
    return shl_ref_siso_callback_base(input, output, params, shl_ref_rope_f32);
}
