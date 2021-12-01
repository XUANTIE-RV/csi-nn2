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
#include "csi_ovx.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of convolution nchw(anole).\n");

    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_ANOLE;
    sess->base_dtype = CSINN_DTYPE_UINT8;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);

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
    params.base.layout     = CSINN_NCHW;
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
    weight_size = output->dim[1] * input->dim[1] *  kernel->dim[2] *  kernel->dim[3];
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.base.name = "params";
    params.conv_extra.kernel_tm = NULL;
    params.conv_extra.conv_mode = CSINN_DIRECT;

    float *src_in   = (float *)(buffer + 17);
    float *kernel_in  = (float *)(buffer + 17 + in_size);
    float *bias_in   = (float *)(buffer + 17 + in_size + weight_size);
    float *ref      = (float *)(buffer + 17 + in_size + weight_size + output->dim[1]);
    uint8_t *input_tmp = malloc(in_size * sizeof(char));
    uint8_t *kernel_tmp  = malloc(weight_size * sizeof(char));
    int32_t *bias_tmp   = (int32_t *)malloc(output->dim[1] * sizeof(int32_t));


    input->qinfo = get_quant_info(src_in, in_size);
    scale1 = input->qinfo->scale;

    for(int i = 0; i < in_size; i++) {
        input_tmp[i] = csi_ref_quantize_f32_to_u8(src_in[i], input->qinfo);
    }
    input->name = "input";


    kernel->qinfo = get_quant_info(kernel_in, weight_size);
    scale2 = kernel->qinfo->scale;

    for(int i = 0; i < weight_size; i++) {
        kernel_tmp[i] = csi_ref_quantize_f32_to_u8(kernel_in[i], kernel->qinfo);
    }
    kernel->name = "kernel";



    scale=scale1*scale2;
    for(int i = 0; i < output->dim[1]; i++) {
        bias_tmp[i] =(int32_t)(bias_in[i]/scale);
    }
    bias->name = "bias";

    output->qinfo = get_quant_info(ref, out_size);
    scale3=output->qinfo->scale; 

    kernel->data      = kernel_tmp;
    bias->data  = bias_tmp;
    reference->data = ref;

    if (csi_conv2d_init(input, output, kernel, bias, &params) != CSINN_TRUE) {
        printf("conv2d init fail.\n\t");
        return -1;
    }
    else{
        printf("conv2d init pass.\n\t");
    }

    csi_ovx_set_tensor(input, sess);
    csi_set_input(0, input, sess);

    csi_conv2d(input, output, kernel, bias, &params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);


    struct csi_tensor *input_tensor = csi_alloc_tensor(NULL);
    input_tensor->data = input_tmp;
    csi_update_input(0, input_tensor, sess);
    csi_session_run(sess);

    struct csi_tensor *output_tensor = csi_alloc_tensor(NULL);
    output_tensor->is_const = 0;
    int output_num = csi_get_output_number(sess);
    printf("output_num = %d\n", output_num);
    csi_get_output(0, output_tensor, sess);    // output_num = 1


    quantize_multiplier(scale3, &quantized_multiplier, &shift);
    output_tensor->qinfo->multiplier = quantized_multiplier;
    output_tensor->qinfo->shift      = shift; 
    output_tensor->qinfo->zero_point = output->qinfo->zero_point;
    output_tensor->dtype == CSINN_DTYPE_UINT8;

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    result_verify_8(reference->data, output_tensor, input->data, difference, out_size, false);

    /* free alloced memory */
    free(buffer);
    free(input_tensor->qinfo);
    free(input_tensor);
    free(output_tensor->qinfo);
    free(output_tensor);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();


}
