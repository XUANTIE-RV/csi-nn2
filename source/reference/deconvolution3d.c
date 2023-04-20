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

/* SHL version 2.1.x */

#include "shl_ref.h"

// input:  NCDHW
// kernel: IODHW
// output: NODHW
int shl_ref_deconv3d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv3d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    const int32_t batch = input->dim[0];
    const int32_t in_channel = input->dim[1];
    const int32_t in_depth = input->dim[2];
    const int32_t in_height = input->dim[3];
    const int32_t in_width = input->dim[4];

    // const int32_t filter_inchannel = kernel->dim[0];
    // const int32_t filter_outchannel = kernel->dim[1];
    const int32_t filter_depth = kernel->dim[2];
    const int32_t filter_height = kernel->dim[3];
    const int32_t filter_width = kernel->dim[4];

    // const int32_t output_batch = output->dim[0];
    const int32_t output_channel = output->dim[1];
    const int32_t output_depth = output->dim[2];
    const int32_t output_height = output->dim[3];
    const int32_t output_width = output->dim[4];

    int num_elements = 1;
    for (int i = 0; i < output->dim_count; ++i) {
        num_elements *= output->dim[i];
    }
    // We need to initialize scratch_buffer to all 0s
    float *scratch_buffer = shl_mem_alloc(num_elements * sizeof(float));

    // Loop through input elements one at a time.
    for (int out_b = 0; out_b < batch; ++out_b) {
        for (int in_ch = 0; in_ch < in_channel; ++in_ch) {
            for (int in_d = 0; in_d < in_depth; ++in_d) {
                for (int in_h = 0; in_h < in_height; ++in_h) {
                    for (int in_w = 0; in_w < in_width; ++in_w) {
                        // Loop through the output elements it will influence.
                        const int out_d_origin = (in_d * params->stride_depth) - params->pad_front;
                        const int out_h_origin = (in_h * params->stride_height) - params->pad_top;
                        const int out_w_origin = (in_w * params->stride_width) - params->pad_left;

                        for (int out_ch = 0; out_ch < output_channel; ++out_ch) {
                            for (int filter_d = 0; filter_d < filter_depth; ++filter_d) {
                                for (int filter_h = 0; filter_h < filter_height; ++filter_h) {
                                    for (int filter_w = 0; filter_w < filter_width; ++filter_w) {
                                        // Compute output element location.
                                        const int out_d = out_d_origin + filter_d;
                                        const int out_h = out_h_origin + filter_h;
                                        const int out_w = out_w_origin + filter_w;
                                        // Cannot accumulate out of bounds.
                                        if ((out_d >= 0) && (out_d < output_depth) &&
                                            (out_h >= 0) && (out_h < output_height) &&
                                            (out_w >= 0) && (out_w < output_width)) {
                                            int32_t input_idx = shl_ref_get_index_5(
                                                input->dim, out_b, in_ch, in_d, in_h, in_w);
                                            float input_val = input_data[input_idx];
                                            int32_t filter_idx =
                                                shl_ref_get_index_5(kernel->dim, in_ch, out_ch,
                                                                    filter_d, filter_h, filter_w);
                                            float filter_val = kernel_data[filter_idx];
                                            int32_t output_idx = shl_ref_get_index_5(
                                                output->dim, out_b, out_ch, out_d, out_h, out_w);
                                            scratch_buffer[output_idx] += input_val * filter_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (bias->dim_count != 0) {
        for (int out_b = 0; out_b < batch; ++out_b) {
            for (int out_ch = 0; out_ch < output_channel; ++out_ch) {
                for (int out_d = 0; out_d < output_depth; ++out_d) {
                    for (int out_h = 0; out_h < output_height; ++out_h) {
                        for (int out_w = 0; out_w < output_width; ++out_w) {
                            int32_t out_idx = shl_ref_get_index_5(output->dim, out_b, out_ch, out_d,
                                                                  out_h, out_w);
                            scratch_buffer[out_idx] += bias_data[out_ch];
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < num_elements; ++i) {
        output_data[i] = scratch_buffer[i];
    }
    shl_mem_free(scratch_buffer);
    return CSINN_TRUE;
}

int shl_ref_deconv3d_quant(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                           struct csinn_conv3d_params *params)
{
    return shl_ref_conv_callback_base(input, output, kernel, bias, params, shl_ref_deconv3d_f32);
}