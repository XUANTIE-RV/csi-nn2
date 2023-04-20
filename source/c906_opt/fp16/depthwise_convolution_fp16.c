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

#include "shl_c906.h"

int shl_c906_dwconv2d_s1_pad0_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

    const int32_t dilation_width_factor = params->dilation_width;
    const int32_t dilation_height_factor = params->dilation_height;
    const int32_t batches = input->dim[0];
    const int32_t input_depth = input->dim[1];
    const int32_t output_depth = output->dim[1];
    const int32_t input_height = input->dim[2];
    const int32_t input_width = input->dim[3];
    const int32_t filter_height = kernel->dim[2];
    const int32_t filter_width = kernel->dim[3];
    const int32_t output_height = output->dim[2];
    const int32_t output_width = output->dim[3];  // input_depth = output_depth;

    for (int32_t b = 0; b < batches; ++b) {
        int output_dim_pos = 0;
        for (int32_t ic = 0; ic < input_depth; ++ic) {
            int kernel_dim_pos_tmp = (ic * kernel->dim[1]) * filter_height * filter_width;
            int input_dim_pos_tmp = (b * input_depth + ic) * input_height * input_width;
            for (int32_t out_y = 0; out_y < output_height; ++out_y) {
                for (int32_t out_x = 0; out_x < output_width; ++out_x) {
                    __fp16 acc = 0;
                    vfloat16m1_t _acc = vfmv_v_f_f16m1(0.0f, 8);
                    for (int32_t filter_y = 0; filter_y < filter_height; ++filter_y) {
                        int kernel_dim_pos_tmp2 = kernel_dim_pos_tmp + filter_y * filter_width;
                        int32_t filter_x = 0;
                        for (; filter_x + 7 < filter_width; filter_x += 8) {
                            const int32_t in_x = out_x + filter_x;
                            const int32_t in_y = out_y + filter_y;
                            int input_dim_pos = input_dim_pos_tmp + in_y * input_width + in_x;
                            int kernel_dim_pos = kernel_dim_pos_tmp2 + filter_x;
                            vfloat16m1_t _input_val = vle16_v_f16m1(input_data + input_dim_pos, 8);
                            vfloat16m1_t _kernel_data =
                                vle16_v_f16m1(kernel_data + kernel_dim_pos, 8);
                            _acc = vfmacc_vv_f16m1(_acc, _input_val, _kernel_data, 8);
                        }

                        vfloat16m1_t _0_f = vfmv_v_f_f16m1(0.0f, 8);
                        vfloat16m1_t _sum2 = vfredosum_vs_f16m1_f16m1(_0_f, _acc, _0_f, 16);
                        acc = vfmv_f_s_f16m1_f16(_sum2);
                        for (; filter_x < filter_width; ++filter_x) {
                            const int32_t in_x = out_x + filter_x;
                            const int32_t in_y = out_y + filter_y;
                            int input_dim_pos = input_dim_pos_tmp + in_y * input_width + in_x;
                            int kernel_dim_pos = kernel_dim_pos_tmp2 + filter_x;
                            __fp16 input_val = input_data[input_dim_pos];
                            __fp16 filter_val = kernel_data[kernel_dim_pos];
                            acc += (filter_val) * (input_val);
                        }
                    }
                    acc += bias_data[ic];
                    output_data[output_dim_pos] = acc;
                    output_dim_pos++;
                }
            }
        }
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}
