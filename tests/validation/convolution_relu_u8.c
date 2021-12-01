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

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of convolution relu u8.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct csi_tensor *kernel = csi_alloc_tensor(NULL);
    struct csi_tensor *bias = csi_alloc_tensor(NULL);
    struct conv2d_params params;
    int in_size, out_size, weight_size;
    int zp, quantized_multiplier, shift;
    float max_value, min_value, scale, scale1, scale2, scale3;
    float error[2] = {0};
    float max_error;
    

    if (argc == 1) {
        printf("please assign the input data.\n");
        return 0;
    }

    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0]   = buffer[0];          // batch
    input->dim[1]   = buffer[1];          // height
    input->dim[2]   = buffer[2];          // width
    input->dim[3]   = buffer[3];          // in_channel
    kernel->dim[1]  = buffer[6];
    kernel->dim[2]  = buffer[7];
    kernel->dim[3]  = buffer[3];
    bias->dim[0]    = buffer[12];
    output->dim[0]  = buffer[0];         // batch
    output->dim[1]  = buffer[16];        // height
    output->dim[2]  = buffer[15];        // width
    output->dim[3]  = buffer[12];        // out_channel
    params.stride_height = buffer[4];
    params.stride_width  = buffer[5];
    params.pad_left   = buffer[8];
    params.pad_right  = buffer[9];
    params.pad_top    = buffer[10];
    params.pad_down   = buffer[11];
    params.dilation_width  = buffer[13];
    params.dilation_height = buffer[14];
    params.base.layout     = CSINN_NHWC;
    params.group      = 1;

    input->dim_count = 4;
    kernel->dim_count = 4;
    bias->dim_count = 1;
    output->dim_count = 4;
    input->dtype = CSINN_DTYPE_UINT8;
    kernel->dtype = CSINN_DTYPE_UINT8;
    bias->dtype = CSINN_DTYPE_UINT8;
    output->dtype = CSINN_DTYPE_UINT8;
    in_size  = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    weight_size = output->dim[3] * input->dim[3] *  kernel->dim[1] *  kernel->dim[2];
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    float *src_in   = (float *)(buffer + 17);
    float *kernel_in  = (float *)(buffer + 17 + in_size);
    float *bias_in   = (float *)(buffer + 17 + in_size + weight_size);
    float *ref      = (float *)(buffer + 17 + in_size + weight_size + output->dim[3]);
    uint8_t *input_tmp = malloc(in_size * sizeof(char));
    uint8_t *kernel_tmp  = malloc(weight_size * sizeof(char));
    int32_t *bias_tmp   = (int32_t *)malloc(output->dim[3] * sizeof(int32_t));


    input->qinfo = get_quant_info(src_in, in_size);
    scale1 = input->qinfo->scale;

    for(int i = 0; i < in_size; i++) {
        input_tmp[i] = csi_ref_quantize_f32_to_u8(src_in[i], input->qinfo);
    }

    /* compute the max quantize error */
    for(int i = 0; i < in_size; i++) {
        float error1;
        float output_tmp  = csi_ref_dequantize_u8_to_f32(input_tmp[i], input->qinfo);
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



    kernel->qinfo = get_quant_info(kernel_in, weight_size);
    scale2 = kernel->qinfo->scale;

    for(int i = 0; i < weight_size; i++) {
        kernel_tmp[i] = csi_ref_quantize_f32_to_u8(kernel_in[i], kernel->qinfo);
    }

    /* compute the max quantize error */
    for(int i = 0; i < weight_size; i++) {
        float error1;
        float output_tmp  = csi_ref_dequantize_u8_to_f32(kernel_tmp[i], kernel->qinfo);
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
    for(int i = 0; i < output->dim[3]; i++) {
        bias_tmp[i] =(int32_t)(bias_in[i]/scale);
    }

    output->qinfo = get_quant_info(ref, out_size);
    scale3=output->qinfo->scale; 
    scale=(scale1*scale2)/scale3;
    quantize_multiplier(scale, &quantized_multiplier, &shift);
    output->qinfo->multiplier = quantized_multiplier;
    output->qinfo->shift      = shift;

    input->data     = input_tmp;
    kernel->data      = kernel_tmp;
    bias->data  = bias_tmp;
    reference->data = ref;
    output->data    = malloc(out_size * sizeof(char));

    float difference = argc > 2 ? atof(argv[2]) : max_error;

    if (csi_conv2d_relu_init(input, output, kernel, bias, &params) == CSINN_TRUE) {
        csi_conv2d_relu(input, output, kernel, bias, &params);
    } 


    quantize_multiplier(scale3, &quantized_multiplier, &shift);
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
