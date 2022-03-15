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


int main(int argc, char** argv)
{
    init_testsuite("Testing function of convolution3d i8.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct csi_tensor *kernel = csi_alloc_tensor(NULL);
    struct csi_tensor *bias = csi_alloc_tensor(NULL);
    struct conv3d_params params;
    int in_size, out_size, weight_size, bias_size;
    int zp, quantized_multiplier, shift;
    float max_value, min_value, scale, scale1, scale2, scale3;
    float error[2] = {0};
    float max_error;

    if (argc == 1) {
        printf("please assign the input data.\n");
        return 0;
    }

    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0]   = buffer[0];     //batch
    input->dim[1]   = buffer[1];     //in_channel
    input->dim[2]   = buffer[2];     //in_depth
    input->dim[3]   = buffer[3];     //in_height
    input->dim[4]   = buffer[4];     //in_width

    kernel->dim[0] = buffer[5];      //out_channel
    kernel->dim[1] = buffer[1];      //in_channel
    kernel->dim[2] = buffer[6];      //filter_depth
    kernel->dim[3] = buffer[7];      //filter_height
    kernel->dim[4] = buffer[8];      //filter_width

    bias->dim[0]   = buffer[5];

    output->dim[0] = buffer[0];      //batch
    output->dim[1] = buffer[5];      //out_channel
    output->dim[2] = buffer[9];      //out_depth
    output->dim[3] = buffer[10];     //out_height
    output->dim[4] = buffer[11];     //out_width

    params.stride_depth  = buffer[12];
    params.stride_height = buffer[13];
    params.stride_width  = buffer[14];
    params.pad_left   = buffer[15];
    params.pad_right  = buffer[16];
    params.pad_top    = buffer[17];
    params.pad_down   = buffer[18];
    params.pad_front  = buffer[19];
    params.pad_back   = buffer[20];

    params.dilation_depth  = buffer[21];
    params.dilation_height = buffer[22];
    params.dilation_width  = buffer[23];
    params.base.layout     = CSINN_LAYOUT_NCDHW;
    params.group      = 1;

    input->dim_count = 5;
    kernel->dim_count = 5;
    bias->dim_count = 1;
    output->dim_count = 5;
    input->dtype = CSINN_DTYPE_INT8;
    input->layout = CSINN_LAYOUT_NCDHW;
    input->is_const = 0;
    input->quant_channel = 1;

    kernel->dtype = CSINN_DTYPE_INT8;
    kernel->layout = CSINN_LAYOUT_OIDHW;
    kernel->is_const = 1;
    kernel->quant_channel = 1;

    bias->dtype = CSINN_DTYPE_INT8;
    bias->layout = CSINN_LAYOUT_O;
    bias->is_const = 1;
    bias->quant_channel = 1;

    output->dtype = CSINN_DTYPE_INT8;
    output->layout = CSINN_LAYOUT_NCDHW;
    output->is_const = 0;
    output->quant_channel = 1;

    in_size     = input->dim[0]  * input->dim[1]  * input->dim[2]  * input->dim[3]  * input->dim[4];
    out_size    = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3] * output->dim[4];
    weight_size = kernel->dim[0] * kernel->dim[1] * kernel->dim[2] * kernel->dim[3] * kernel->dim[4];
    bias_size   = output->dim[1];
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    float *src_in   = (float *)(buffer + 24);
    float *kernel_in  = (float *)(buffer + 24 + in_size);
    float *bias_in   = (float *)(buffer + 24 + in_size + weight_size);
    float *ref      = (float *)(buffer + 24 + in_size + weight_size + bias_size);
    int8_t *input_tmp = malloc(in_size * sizeof(char));
    int8_t *kernel_tmp  = malloc(weight_size * sizeof(char));
    int32_t *bias_tmp   = (int32_t *)malloc(bias_size * sizeof(int32_t));

    input->data = src_in;
    get_quant_info(input);
    scale1 = input->qinfo->scale;

    for(int i = 0; i < in_size; i++) {
        input_tmp[i] = csi_ref_quantize_f32_to_i8(src_in[i], input->qinfo);
    }

    /* compute the max quantize error */
    for(int i = 0; i < in_size; i++) {
        float error1;
        float output_tmp  = csi_ref_dequantize_i8_to_f32(input_tmp[i], input->qinfo);
        if(isinf(src_in[i]) || isnan(src_in[i])){
            continue;
        } else {
            error1 = fabs(src_in[i] -output_tmp);
            if(error1 > 1e-6) {
                error1 = fabs(src_in[i] - output_tmp)/fabs(src_in[i] + 1e-9);
            }
        }
        if(error1 > error[0]) {
            error[0] = error1;
        }
    }


    kernel->data = kernel_in;
    get_quant_info(kernel);
    scale2 = kernel->qinfo->scale;

    for(int i = 0; i < weight_size; i++) {
        kernel_tmp[i] = csi_ref_quantize_f32_to_i8(kernel_in[i], kernel->qinfo);
    }

    /* compute the max quantize error */
    for(int i = 0; i < weight_size; i++) {
        float error1;
        float output_tmp  = csi_ref_dequantize_i8_to_f32(kernel_tmp[i], kernel->qinfo);
        if(isinf(kernel_in[i]) || isnan(kernel_in[i])){
            continue;
        } else {
            error1 = fabs(kernel_in[i] -output_tmp);
            if(error1 > 1e-6) {
                error1 = fabs(kernel_in[i] - output_tmp)/fabs(kernel_in[i] + 1e-9);
            }
        }
        if(error1 > error[1]) {
            error[1] = error1;
        }
    }

    max_error = (error[0] + error[1]);

    scale=scale1*scale2;
    for(int i = 0; i < bias_size; i++) {
        bias_tmp[i] =(int32_t)(bias_in[i]/scale);
    }

    output->data = ref;
    get_quant_info(output);
    scale3=output->qinfo->scale;
    scale=(scale1*scale2)/scale3;
    csi_quantize_multiplier(scale, &quantized_multiplier, &shift);
    output->qinfo->multiplier = quantized_multiplier;
    output->qinfo->shift      = shift;

    input->data     = input_tmp;
    kernel->data      = kernel_tmp;
    bias->data  = bias_tmp;
    reference->data = ref;
    output->data    = malloc(out_size * sizeof(char));



    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csi_conv3d_init(input, output, kernel, bias, &params) == CSINN_TRUE) {
        csi_conv3d(input, output, kernel, bias, &params);
    }


    csi_quantize_multiplier(scale3, &quantized_multiplier, &shift);
    output->qinfo->multiplier = quantized_multiplier;
    output->qinfo->shift      = shift;
    result_verify_8(reference->data, output, input->data, difference, out_size, false);

    free(buffer);
    free(input_tmp);
    free(kernel_tmp);
    free(bias_tmp);
    free(output->data);
    return done_testing();
}
