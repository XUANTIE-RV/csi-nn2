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

int shl_ref_instance_norm_f32(struct csinn_tensor *input, struct csinn_tensor *scales,
                              struct csinn_tensor *bias, struct csinn_tensor *output,
                              struct csinn_instance_norm_params *params)
{
    float sum, mean, var;
    int N = input->dim[0];
    int C = input->dim[1];
    int H = input->dim[2];
    int W = input->dim[3];

    float *input_data = input->data;
    float *scale_data = scales->data;
    float *bias_data = bias->data;
    float *output_data = output->data;

    for (int i = 0; i < N; i++) {
        for (int c = 0; c < C; c++) {
            // mean
            sum = 0.0f;
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    sum += input_data[(i * C * H * W) + (c * H * W) + (h * W) + w];
                }
            }
            mean = sum / (H * W);

            // var
            sum = 0.0f;
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    sum += powf(input_data[(i * C * H * W) + (c * H * W) + (h * W) + w] - mean, 2);
                }
            }
            var = sum / (H * W);

            // norm
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    output_data[(i * C * H * W) + (c * H * W) + (h * W) + w] =
                        (input_data[(i * C * H * W) + (c * H * W) + (h * W) + w] - mean) /
                        sqrtf(var + params->epsilon);
                }
            }

            // scale
            bool do_scale = scales->dim_count != 0;
            bool do_bias = bias->dim_count != 0;

            if (scales->dim_count == 1 && scale_data[0] == 1) {
                do_scale = false;
            }

            if (bias->dim_count == 1 && bias_data[0] == 0) {
                do_bias = false;
            }

            if (do_scale) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        output_data[(i * C * H * W) + (c * H * W) + (h * W) + w] *= scale_data[c];
                    }
                }
            }

            if (do_bias) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        output_data[(i * C * H * W) + (c * H * W) + (h * W) + w] += bias_data[c];
                    }
                }
            }
        }
    }

    return CSINN_TRUE;
}

int shl_ref_instance_norm_quant(struct csinn_tensor *input, struct csinn_tensor *scales,
                                struct csinn_tensor *bias, struct csinn_tensor *output,
                                struct csinn_instance_norm_params *params)
{
    struct csinn_tensor *finput0 = shl_ref_tensor_transform_f32(input);
    struct csinn_tensor *finput1 = shl_ref_tensor_transform_f32(scales);
    struct csinn_tensor *finput2 = shl_ref_tensor_transform_f32(bias);
    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    int ret = shl_ref_instance_norm_f32(finput0, finput1, finput2, foutput, params);
    csinn_tensor_data_convert(output, foutput);
    shl_ref_tensor_transform_free_f32(finput0);
    shl_ref_tensor_transform_free_f32(finput1);
    shl_ref_tensor_transform_free_f32(finput2);
    shl_ref_tensor_transform_free_f32(foutput);
    return ret;
}