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

/* CSI-NN2 version 1.8.x */

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of deconv2d(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size = 0, out_size = 0, weight_size = 0, bias_size = 0;

    /* session configuration */
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_API;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);


    /* input tensor configuration */
    struct csi_tensor *input  = csi_alloc_tensor(sess);
    input->dim[0]   = buffer[0];          // batch
    input->dim[1]   = buffer[1];          // in_channel
    input->dim[2]   = buffer[2];          // height
    input->dim[3]   = buffer[3];          // width
    input->dim_count = 4;
    in_size  = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    input->name = "input";
    float *input_data = (float *)(buffer + 17);
    input->data = input_data;
    get_quant_info(input);

    void *src_tmp = malloc(in_size * sizeof(char));
    for(int i = 0; i < in_size; i++) {
        if (sess->base_dtype == CSINN_DTYPE_UINT8) {
            *((uint8_t *)src_tmp + i) = csi_ref_quantize_f32_to_u8(input_data[i], input->qinfo);
        } else if (sess->base_dtype == CSINN_DTYPE_INT8) {
            *((int8_t *)src_tmp + i) = csi_ref_quantize_f32_to_i8(input_data[i], input->qinfo);
        }
    }


    /* kernel tensor configuration */
    struct csi_tensor *kernel  = csi_alloc_tensor(sess);
    kernel->dim[0]  = buffer[1];    // i
    kernel->dim[1]  = buffer[14];   // o
    kernel->dim[2]  = buffer[6];    // h
    kernel->dim[3]  = buffer[7];    // w
    kernel->dim_count = 4;
    weight_size = kernel->dim[0] * kernel->dim[1] *  kernel->dim[2] *  kernel->dim[3];
    kernel->name = "kernel";
    float *kernel_data = (float *)(buffer + 17 + in_size);
    kernel->data = kernel_data;
    get_quant_info(kernel);

    void *kernel_tmp = malloc(weight_size * sizeof(char));
    for(int i = 0; i < weight_size; i++) {
        if (sess->base_dtype == CSINN_DTYPE_UINT8) {
            *((uint8_t *)kernel_tmp + i) = csi_ref_quantize_f32_to_u8(kernel_data[i], kernel->qinfo);
        } else if (sess->base_dtype == CSINN_DTYPE_INT8) {
            *((int8_t *)kernel_tmp + i) = csi_ref_quantize_f32_to_i8(kernel_data[i], kernel->qinfo);
        }
    }
    if (sess->base_dtype == CSINN_DTYPE_UINT8 || sess->base_dtype == CSINN_DTYPE_INT8) {
        kernel->data = kernel_tmp;
    }


    /* bias tensor configuratioin */
    struct csi_tensor *bias  = csi_alloc_tensor(sess);
    bias->dim[0] = buffer[14];
    bias->dim_count = 1;
    bias_size = bias->dim[0];
    bias->name = "bias";
    float *bias_data = (float *)(buffer + 17 + in_size + weight_size);
    bias->data = bias_data;
    // get_quant_info(bias);

    /* FIX ME */
    void *bias_tmp = malloc(bias_size * sizeof(int32_t));
    for(int i = 0; i < bias_size; i++) {
        if (sess->base_dtype == CSINN_DTYPE_UINT8) {
            // *((int32_t *)bias_tmp + i) = csi_ref_quantize_f32_to_u8(bias_data[i], bias->qinfo);
            *((int32_t *)bias_tmp + i) = (int32_t)(bias_data[i] / (input->qinfo->scale * kernel->qinfo->scale));
        }
    }
    if (sess->base_dtype == CSINN_DTYPE_UINT8 || sess->base_dtype == CSINN_DTYPE_INT8) {
        bias->data = bias_tmp;
    }


    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim[0]  = buffer[0];         // batch
    output->dim[1]  = buffer[14];        // out_channel
    output->dim[2]  = buffer[16];        // height
    output->dim[3]  = buffer[15];        // width
    output->dim_count = 4;
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    reference->data = (float *)(buffer + 17 + in_size + weight_size + bias->dim[0]);
    output->data = reference->data;
    output->name = "output";
    get_quant_info(output);


    /* operator parameter configuration */
    struct conv2d_params params;
    params.stride_height = buffer[4];
    params.stride_width  = buffer[5];
    params.pad_left   = buffer[8];
    params.pad_right  = buffer[9];
    params.pad_top    = buffer[10];
    params.pad_down   = buffer[11];
    params.dilation_width  = buffer[12];
    params.dilation_height = buffer[13];
    params.group      = 1;
    params.base.api = CSINN_API;
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.base.name = "params";


    if (csi_deconv2d_init(input, output, kernel, bias, &params) != CSINN_TRUE) {
        printf("deconv2d init fail.\n\t");
        return -1;
    }

    csi_set_tensor_entry(input, sess);
    csi_set_input(0, input, sess);

    csi_deconv2d(input, output, kernel, bias, &params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);


    struct csi_tensor *input_tensor = csi_alloc_tensor(NULL);
    if (sess->base_dtype == CSINN_DTYPE_FLOAT32) {
        input_tensor->data = input_data;
    } else if (sess->base_dtype == CSINN_DTYPE_UINT8 || sess->base_dtype == CSINN_DTYPE_INT8) {
        input_tensor->data = src_tmp;
    }
    csi_update_input(0, input_tensor, sess);
    csi_session_run(sess);

    struct csi_tensor *output_tensor = csi_alloc_tensor(NULL);
    output_tensor->data = NULL;
    output_tensor->dtype = sess->base_dtype;
    output_tensor->is_const = 0;
    int output_num = csi_get_output_number(sess);
    printf("output_num = %d\n", output_num);
    csi_get_output(0, output_tensor, sess);
    memcpy(output_tensor->qinfo, output->qinfo, sizeof(struct csi_quant_info));


    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    if (sess->base_dtype == CSINN_DTYPE_UINT8 || sess->base_dtype == CSINN_DTYPE_INT8) {
        result_verify_8(reference->data, output_tensor, input->data, difference, out_size, false);
    } else if (sess->base_dtype == CSINN_DTYPE_FLOAT32) {
        result_verify_f32(reference->data, output_tensor->data, input->data, difference, out_size, false);
    }

    /* free alloced memory */
    free(buffer);
    free(input_tensor->qinfo);
    free(input_tensor);
    free(output_tensor->qinfo);
    free(output_tensor);
    free(reference->qinfo);
    free(reference);
    free(src_tmp);
    free(kernel_tmp);
    free(bias_tmp);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}
