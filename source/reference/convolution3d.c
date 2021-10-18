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

static int csi_conv3d_ncdhw_f32(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *kernel,
                                struct csi_tensor *bias,
                                struct conv3d_params *params)
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

    //const int filter_outchannel = kernel->dim[0];
    //const int filter_inchannel = kernel->dim[1];
    const int32_t filter_depth = kernel->dim[2];
    const int32_t filter_height = kernel->dim[3];
    const int32_t filter_width = kernel->dim[4];

    //int output_batch = output->dim[0];
    const int32_t output_channel = output->dim[1];
    const int32_t output_depth = output->dim[2];
    const int32_t output_height = output->dim[3];
    const int32_t output_width = output->dim[4];

    const int32_t dilation_depth = params->dilation_depth;
    const int32_t dilation_height = params->dilation_height;
    const int32_t dilation_width = params->dilation_width;

    for(int32_t out_b=0; out_b<batch; ++out_b) {
        for(int32_t out_ch=0; out_ch<output_channel; ++out_ch) {
            for(int32_t out_d=0; out_d<output_depth; ++out_d) {
                for(int32_t out_h=0; out_h<output_height; ++out_h) {
                    for(int32_t out_w=0; out_w<output_width; ++out_w) {

                        const int32_t in_d_origin = (out_d * params->stride_depth) - params->pad_front;
                        const int32_t in_h_origin = (out_h * params->stride_height) - params->pad_top;
                        const int32_t in_w_origin = (out_w * params->stride_width) - params->pad_left;

                        float acc = 0.0f;
                        for(int32_t in_ch=0; in_ch<in_channel; ++in_ch) {
                            for(int32_t filter_d=0; filter_d<filter_depth; ++filter_d) {
                                for(int32_t filter_h=0; filter_h<filter_height; ++filter_h) {
                                    for(int32_t filter_w=0; filter_w<filter_width; ++filter_w) {
                                        int32_t in_d = in_d_origin + dilation_depth * filter_d;
                                        int32_t in_h = in_h_origin + dilation_height * filter_h;
                                        int32_t in_w = in_w_origin + dilation_width * filter_w;
                                        // If the location is outside the bounds of the input image,
                                        // use zero as a default value.
                                        if((in_d>=0)&&(in_d<in_depth) && (in_h>=0)&&(in_h<in_height) &&
                                           (in_w>=0)&&(in_w<in_width)) {
                                            int32_t input_idx = csi_get_index_5(input->dim, out_b, in_ch, in_d, in_h, in_w);
                                            float input_val = input_data[input_idx];
                                            int32_t filter_idx = csi_get_index_5(kernel->dim, out_ch, in_ch, filter_d, filter_h, filter_w);
                                            float filter_val = kernel_data[filter_idx];
                                            acc += input_val * filter_val;
                                        }
                                    }
                                }
                            }
                        }
                        float bias_val = 0.0f;
                        if(bias_data!=NULL) {
                            bias_val = bias_data[out_ch];
                        }
                        int32_t output_idx = csi_get_index_5(output->dim, out_b, out_ch, out_d, out_h, out_w);
                        output_data[output_idx] = acc + bias_val;
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}


static int csi_conv3d_ncdhw_u8(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *kernel,
                                struct csi_tensor *bias,
                                struct conv3d_params *params)
{
    uint8_t *input_data = (uint8_t *)input->data;
    uint8_t *output_data = (uint8_t *)output->data;
    uint8_t *kernel_data = (uint8_t *)kernel->data;
    int32_t *bias_data = (uint32_t *)bias->data;

    const int32_t batch = input->dim[0];
    const int32_t in_channel = input->dim[1];
    const int32_t in_depth = input->dim[2];
    const int32_t in_height = input->dim[3];
    const int32_t in_width = input->dim[4];

    //const int32_t filter_outchannel = kernel->dim[0];
    //const int32_t filter_inchannel = kernel->dim[1];
    const int32_t filter_depth = kernel->dim[2];
    const int32_t filter_height = kernel->dim[3];
    const int32_t filter_width = kernel->dim[4];

    //const int32_t output_batch = output->dim[0];
    const int32_t output_channel = output->dim[1];
    const int32_t output_depth = output->dim[2];
    const int32_t output_height = output->dim[3];
    const int32_t output_width = output->dim[4];

    const int32_t dilation_depth = params->dilation_depth;
    const int32_t dilation_height = params->dilation_height;
    const int32_t dilation_width = params->dilation_width;

    const int32_t input_offset = input->offset;
    const int32_t filter_offset = kernel->offset;
    const int32_t output_offset = output->offset;
    const int32_t output_multiplier = output->multiplier;
    const int32_t output_shift = output->shift;

    for(int32_t out_b=0; out_b<batch; ++out_b) {
        for(int32_t out_ch=0; out_ch<output_channel; ++out_ch) {
            for(int32_t out_d=0; out_d<output_depth; ++out_d) {
                for(int32_t out_h=0; out_h<output_height; ++out_h) {
                    for(int32_t out_w=0; out_w<output_width; ++out_w) {

                        const int32_t in_d_origin = (out_d * params->stride_depth) - params->pad_front;
                        const int32_t in_h_origin = (out_h * params->stride_height) - params->pad_top;
                        const int32_t in_w_origin = (out_w * params->stride_width) - params->pad_left;

                        int64_t acc = 0.0f;
                        for(int32_t in_ch=0; in_ch<in_channel; ++in_ch) {
                            for(int32_t filter_d=0; filter_d<filter_depth; ++filter_d) {
                                for(int32_t filter_h=0; filter_h<filter_height; ++filter_h) {
                                    for(int32_t filter_w=0; filter_w<filter_width; ++filter_w) {
                                        int32_t in_d = in_d_origin + dilation_depth * filter_d;
                                        int32_t in_h = in_h_origin + dilation_height * filter_h;
                                        int32_t in_w = in_w_origin + dilation_width * filter_w;
                                        // If the location is outside the bounds of the input image,
                                        // use zero as a default value.
                                        if((in_d>=0)&&(in_d<in_depth) && (in_h>=0)&&(in_h<in_height) &&
                                           (in_w>=0)&&(in_w<in_width)) {
                                            int32_t input_idx = csi_get_index_5(input->dim, out_b, in_ch, in_d, in_h, in_w);
                                            int32_t input_val = input_data[input_idx];
                                            int32_t filter_idx = csi_get_index_5(kernel->dim, out_ch, in_ch, filter_d, filter_h, filter_w);
                                            int32_t filter_val = kernel_data[filter_idx];
                                            acc += (input_val+input_offset) * (filter_val+filter_offset);
                                        }
                                    }
                                }
                            }
                        }
                        float bias_val = 0.0f;
                        if(bias_data!=NULL) {
                            acc += bias_data[out_ch];
                        }
                        acc = csi_quantize_u8(acc, output_offset, output_multiplier, output_shift);
                        uint32_t output_idx = csi_get_index_5(output->dim, out_b, out_ch, out_d, out_h, out_w);
                        output_data[output_idx] = acc;
                    }
                }
            }
        }
    }
    return CSINN_TRUE;
}


static int csi_conv3d_ndhwc_f32(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *kernel,
                                struct csi_tensor *bias,
                                struct conv3d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    return CSINN_FALSE;
}


static int csi_conv3d_ndhwc_u8(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *kernel,
                                struct csi_tensor *bias,
                                struct conv3d_params *params)
{

    return CSINN_FALSE;
}

int csi_conv3d_init(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct csi_tensor *bias,
                    struct conv3d_params *params)
{
    if(params->layout == CSINN_NCDHW) {
        if(input->dtype == CSINN_DTYPE_UINT8) {
            params->bc = csi_conv3d_ncdhw_u8;
        } else if(input->dtype == CSINN_DTYPE_FLOAT32) {
            params->bc = csi_conv3d_ncdhw_f32;
        } else {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else if(params->layout == CSINN_NDHWC) {
        if(input->dtype == CSINN_DTYPE_UINT8) {
            params->bc = csi_conv3d_ndhwc_u8;
        } else if(input->dtype == CSINN_DTYPE_FLOAT32) {
            params->bc = csi_conv3d_ndhwc_f32;
        } else {
            return CSINN_UNSUPPORT_DTYPE;
        }
    } else {
        return CSINN_UNSUPPORT_LAYOUT;
    }
    return CSINN_TRUE;
}

int csi_conv3d(struct csi_tensor *input,
                struct csi_tensor *output,
                struct csi_tensor *kernel,
                struct csi_tensor *bias,
                struct conv3d_params *params)
{
    if(params->bc != NULL) {
        params->bc(input, output, kernel, bias, params);
    } else {
        return CSINN_CALLBACK_UNSET;
    }
    return CSINN_TRUE;
}