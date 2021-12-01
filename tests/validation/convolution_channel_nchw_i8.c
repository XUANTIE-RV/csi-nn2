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

/* CSI-NN2 version 1.10.x */

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of convolution channel nchw i8.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct csi_tensor *kernel = csi_alloc_tensor(NULL);
    struct csi_tensor *bias = csi_alloc_tensor(NULL);
    struct conv2d_params params;
    int in_size, out_size, weight_size, per_weight_size;
    int zp, quantized_multiplier, shift;
    float max_value, min_value, scale, scale1, scale2, scale3;
    float max_error = 0.5f;

    if (argc == 1) {
        printf("please assign the input data.\n");
        return 0;
    }

    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0]   = buffer[0];          // batch
    input->dim[1]   = buffer[1];          // in_channel
    input->dim[2]   = buffer[2];          // height
    input->dim[3]   = buffer[3];          // width
    kernel->dim[1]  = buffer[1];
    kernel->dim[2]  = buffer[6];
    kernel->dim[3]  = buffer[7];
    kernel->dim[0]  = buffer[12];
    bias->dim[0]    = buffer[12];
    output->dim[0]  = buffer[0];         // batch
    output->dim[1]  = buffer[12];        // out_channel
    output->dim[2]  = buffer[16];        // height
    output->dim[3]  = buffer[15];        // width

    params.stride_height = buffer[4];
    params.stride_width  = buffer[5];
    params.pad_left   = buffer[8];
    params.pad_right  = buffer[9];
    params.pad_top    = buffer[10];
    params.pad_down   = buffer[11];
    params.dilation_width  = buffer[13];
    params.dilation_height = buffer[14];
    params.base.layout     = CSINN_LAYOUT_NCHW;
    params.group      = 1;
    struct ScaleZp szp[kernel->dim[0]];

    input->dim_count = 4;
    kernel->dim_count = 4;
    bias->dim_count = 1;
    output->dim_count = 4;
    input->dtype = CSINN_DTYPE_INT8;
    kernel->dtype = CSINN_DTYPE_INT8;
    bias->dtype = CSINN_DTYPE_INT8;
    output->dtype = CSINN_DTYPE_INT8;
    in_size  = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    weight_size = output->dim[1] * input->dim[1] *  kernel->dim[2] *  kernel->dim[3];
    per_weight_size = input->dim[1] *  kernel->dim[2] *  kernel->dim[3];
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    float *src_in   = (float *)(buffer + 17);
    float *bias_in   = (float *)(buffer + 17 + in_size + weight_size);
    float *ref      = (float *)(buffer + 17 + in_size + weight_size + output->dim[1]);
    int8_t *input_tmp = malloc(in_size * sizeof(char));
    int8_t *kernel_tmp  = malloc(weight_size * sizeof(char));
    int32_t *bias_tmp   = (int32_t *)malloc(output->dim[1] * sizeof(int32_t));


    input->data = src_in;
    get_quant_info(input);
    scale1 = input->qinfo->scale;

    for(int i = 0; i < in_size; i++) {
        input_tmp[i] = csi_ref_quantize_f32_to_i8(src_in[i], input->qinfo);
    }

    for(int i = 0; i < kernel->dim[0]; i++){
        float *kernel_in  = (float *)(buffer + 17 + in_size + i*per_weight_size);
        kernel->qinfo = get_quant_info_i8(kernel_in, per_weight_size);
        scale2 = kernel->qinfo->scale;
        zp = kernel->qinfo->zero_point;

        for(int j = 0; j < per_weight_size; j++) {
            kernel_tmp[i*per_weight_size + j] = csi_ref_quantize_f32_to_i8(kernel_in[j], kernel->qinfo);
        }

        szp[i].zero_point = zp;
        szp[i].scale = scale2;

    }
    params.scale_zp = szp;

    output->data = ref;
    get_quant_info(output);
    scale3=output->qinfo->scale;
    csi_quantize_multiplier(scale3, &quantized_multiplier, &shift);
    output->qinfo->multiplier = quantized_multiplier;
    output->qinfo->shift      = shift;

    input->data     = input_tmp;
    kernel->data      = kernel_tmp;
    bias->data  = bias_tmp;
    reference->data = ref;
    output->data    = malloc(out_size * sizeof(char));


    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csi_conv2d_init(input, output, kernel, bias, &params) == CSINN_TRUE) {
        csi_conv2d(input, output, kernel, bias, &params);
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
