/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

/* CSI-NN2 version 1.12.x */

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"
#include "../valid_data/convolution_u8.dat"


void verify_conv2d_u8(float *input_data,
                      float *kernel_data,
                      float *bias_data,
                      float *ref_data,
                      uint16_t batch,
                      uint16_t in_h,
                      uint16_t in_w,
                      uint16_t in_c,
                      uint16_t out_h,
                      uint16_t out_w,
                      uint16_t out_c,
                      uint16_t kernel_h,
                      uint16_t kernel_w,
                      uint16_t stride_h,
                      uint16_t stride_w,
                      uint16_t pad_x,
                      uint16_t pad_y,
                      float difference)
{
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size, out_size, kernel_size = 0, bias_size = 0;

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    input->dim[0] = batch;  // N
    input->dim[1] = in_h;   // H
    input->dim[2] = in_w;   // W
    input->dim[3] = in_c;   // C
    input->dim_count = 4;
    input->dtype = CSINN_DTYPE_UINT8;
    input->layout = CSINN_LAYOUT_NHWC;
    input->name = "input";
    input->data = (float *)input_data;
    get_quant_info(input);
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];

    uint8_t *input_tmp = malloc(in_size * sizeof(char));
    for(int i = 0; i < in_size; i++) {
        input_tmp[i] = csi_ref_quantize_f32_to_u8(input_data[i], input->qinfo);
    }
    input->data = input_tmp;


    struct csi_tensor *kernel = csi_alloc_tensor(NULL);
    kernel->dim[0] = out_c;     // O
    kernel->dim[1] = kernel_h;  // H
    kernel->dim[2] = kernel_w;  // W
    kernel->dim[3] = in_c;      // I
    kernel->dim_count = 4;
    kernel->dtype = CSINN_DTYPE_UINT8;
    kernel->layout = CSINN_LAYOUT_OHWI;
    kernel->name = "kernel";
    kernel->data = (float *)kernel_data;
    get_quant_info(kernel);
    kernel_size = kernel->dim[0] * kernel->dim[1] * kernel->dim[2] * kernel->dim[3];

    uint8_t *kernel_tmp = malloc(kernel_size * sizeof(char));
    for(int i = 0; i < kernel_size; i++) {
        kernel_tmp[i] = csi_ref_quantize_f32_to_u8(kernel_data[i], kernel->qinfo);
        // printf("%d, ", kernel_tmp[i]);
    }
    kernel->data = kernel_tmp;


    struct csi_tensor *bias = csi_alloc_tensor(NULL);
    bias->dim[0] = out_c;   // O
    bias->dim_count = 1;
    bias->dtype = CSINN_DTYPE_INT32;
    bias->layout = CSINN_LAYOUT_O;
    bias->name = "bias";
    bias_size = bias->dim[0];
    bias->data = (float *)bias_data;

    int32_t *bias_tmp = malloc(bias_size * sizeof(int32_t));
    for(int i = 0; i < bias_size; i++) {
        bias_tmp[i] = (int32_t)(bias_data[i] / (input->qinfo->scale * kernel->qinfo->scale));
    }
    bias->qinfo->scale = input->qinfo->scale * kernel->qinfo->scale;
    bias->data = bias_tmp;


    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim[0] = batch;
    output->dim[1] = out_h;
    output->dim[2] = out_w;
    output->dim[3] = out_c;
    output->dim_count = 4;
    output->dtype = CSINN_DTYPE_UINT8;
    output->layout = CSINN_LAYOUT_NHWC;
    output->name = "output";
    output->data = (float *)ref_data;
    get_quant_info(output);
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    output->data = malloc(out_size);


    struct conv2d_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NHWC;
    params.base.run_mode = CSINN_RM_LAYER;
    params.stride_height = stride_h;
    params.stride_width  = stride_w;
    params.pad_left   = pad_x;
    params.pad_right  = pad_x;
    params.pad_top    = pad_y;
    params.pad_down   = pad_y;
    params.dilation_width  = 1;
    params.dilation_height = 1;
    params.group      = 1;
    params.conv_extra.kernel_tm = NULL;
    params.conv_extra.conv_mode = CSINN_DIRECT;

    if (csi_conv2d_init(input, output, kernel, bias, &params) == CSINN_TRUE) {
        csi_conv2d(input, output, kernel, bias, &params);
    }

    reference->data  = (float *)ref_data;
    result_verify_8(reference->data, output, input->data, difference, out_size, false);

    free(input);
    free(kernel);
    free(bias);
    free(output->data);
    free(output);
    free(reference);
    free(input_tmp);
    free(kernel_tmp);
    free(bias_tmp);
}



int main(int argc, char** argv)
{
    init_testsuite("Testing function of convolution(u8) for i805.\n");

    verify_conv2d_u8(conv_input_0, conv_kernel_0, conv_bias_0, conv_output_0,
                     1, 7, 7, 5, 7, 7, 11, 3, 3, 1, 1, 1, 1, 0.0f);
}
