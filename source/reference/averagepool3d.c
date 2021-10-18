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

int csi_averagepool3d_f32(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct pool_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    const int batch = input->dim[0];
    const int channel = input->dim[1];
    const int in_depth = input->dim[2];
    const int in_height = input->dim[3];
    const int in_width = input->dim[4];
    const int out_depth = output->dim[2];
    const int out_height = output->dim[3];
    const int out_width = output->dim[4];

    int filter_cnt = 0;
    filter_cnt = (params->filter_depth) * (params->filter_height) * (params->filter_width);

    for(int in_ch=0; in_ch<batch; ++in_ch) {
        for(int out_ch=0; out_ch<channel; ++out_ch) {
            for(int out_d=0; out_d<out_depth; ++out_d) {
                for(int out_h=0; out_h<out_height; ++out_h) {
                    for(int out_w=0; out_w<out_width; ++out_w) {

                        const int in_d_origin = (out_d * params->stride_depth) - params->pad_front;
                        const int in_h_origin = (out_h * params->stride_height) - params->pad_top;
                        const int in_w_origin = (out_w * params->stride_width) - params->pad_left;

                        const int filter_d_begin = csi_max_internal_s32(0, -in_d_origin);
                        const int filter_d_end = csi_min_internal_s32(params->filter_depth, in_depth - in_d_origin);
                        const int filter_h_begin = csi_max_internal_s32(0, -in_h_origin);
                        const int filter_h_end = csi_min_internal_s32(params->filter_height, in_height - in_h_origin);
                        const int filter_w_begin = csi_max_internal_s32(0, -in_w_origin);
                        const int filter_w_end = csi_min_internal_s32(params->filter_width, in_width - in_w_origin);

                        float total = 0.0f;
                        for(int filter_d=filter_d_begin; filter_d<filter_d_end; ++filter_d) {
                            for(int filter_h=filter_h_begin; filter_h<filter_h_end; ++filter_h) {
                                for(int filter_w=filter_w_begin; filter_w<filter_w_end; ++filter_w) {
                                    int in_d = in_d_origin + filter_d;
                                    int in_h = in_h_origin + filter_h;
                                    int in_w = in_w_origin + filter_w;
                                    total += input_data[csi_get_index_5(input->dim, in_ch, out_ch, in_d, in_h, in_w)];
                                    // filter_cnt++;
                                }
                            }
                        }
                        // float average = filter_cnt==0 ? total : total/filter_cnt;
                        float average = total/filter_cnt;
                        output_data[csi_get_index_5(output->dim, in_ch, out_ch, out_d, out_h, out_w)] = average;
                    }
                }
            }
        }
    }

    return CSINN_TRUE;
}

int csi_averagepool3d_u8(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct pool_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;

    const int batch = input->dim[0];
    const int channel = input->dim[1];
    const int in_depth = input->dim[2];
    const int in_height = input->dim[3];
    const int in_width = input->dim[4];
    const int out_depth = output->dim[2];
    const int out_height = output->dim[3];
    const int out_width = output->dim[4];

    int filter_cnt = 0;
    filter_cnt = (params->filter_depth) * (params->filter_height) * (params->filter_width);

    for(int in_ch=0; in_ch<batch; ++in_ch) {
        for(int out_ch=0; out_ch<channel; ++out_ch) {
            for(int out_d=0; out_d<out_depth; ++out_d) {
                for(int out_h=0; out_h<out_height; ++out_h) {
                    for(int out_w=0; out_w<out_width; ++out_w) {

                        const int in_d_origin = (out_d * params->stride_depth) - params->pad_front;
                        const int in_h_origin = (out_h * params->stride_height) - params->pad_top;
                        const int in_w_origin = (out_w * params->stride_width) - params->pad_left;

                        const int filter_d_begin = csi_max_internal_s32(0, -in_d_origin);
                        const int filter_d_end = csi_min_internal_s32(params->filter_depth, in_depth - in_d_origin);
                        const int filter_h_begin = csi_max_internal_s32(0, -in_h_origin);
                        const int filter_h_end = csi_min_internal_s32(params->filter_height, in_height - in_h_origin);
                        const int filter_w_begin = csi_max_internal_s32(0, -in_w_origin);
                        const int filter_w_end = csi_min_internal_s32(params->filter_width, in_width - in_w_origin);

                        float total = 0.0f;
                        for(int filter_d=filter_d_begin; filter_d<filter_d_end; ++filter_d) {
                            for(int filter_h=filter_h_begin; filter_h<filter_h_end; ++filter_h) {
                                for(int filter_w=filter_w_begin; filter_w<filter_w_end; ++filter_w) {
                                    int in_d = in_d_origin + filter_d;
                                    int in_h = in_h_origin + filter_h;
                                    int in_w = in_w_origin + filter_w;
                                    uint8_t input_val = input_data[csi_get_index_5(input->dim, in_ch, out_ch, in_d, in_h, in_w)];
                                    total += csi_dequantize_u8_to_f32(input_val, input->zero_point, input->multiplier, input->shift);
                                    // filter_cnt++;
                                }
                            }
                        }
                        // float average = filter_cnt==0 ? total : total/filter_cnt;
                        float average = total/filter_cnt;
                        uint8_t output_val = csi_quantize_f32_to_u8(average, output->zero_point, output->multiplier, output->shift);
                        output_data[csi_get_index_5(output->dim, in_ch, out_ch, out_d, out_h, out_w)] = output_val;
                    }
                }
            }
        }
    }

    return CSINN_TRUE;
}

int csi_averagepool3d_init(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct pool_params *params)
{
    params->bc = csi_bc_map(params->api, CSINN_OP_AVGPOOL3D, input->dtype);
    if (params->bc == NULL) {
        return CSINN_UNSUPPORT_DTYPE;
    }
    return CSINN_TRUE;
}

int csi_averagepool3d(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct pool_params *params)
{
    if(params->bc !=NULL) {
        params->bc(input, output, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}