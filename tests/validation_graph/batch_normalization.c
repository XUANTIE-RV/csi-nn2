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
    init_testsuite("Testing function of batch normalization(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int channel_size = buffer[4];

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size = 0, out_size = 0;

    /* session configuration */
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_API;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);


    /* input tensor configuration */
    struct csi_tensor *input  = csi_alloc_tensor(sess);
    input->dim[0] = buffer[1];          // batch
    input->dim[1] = buffer[4];          // channel
    input->dim[2] = buffer[2];          // height
    input->dim[3] = buffer[3];          // width
    input->dim_count = 4;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    input->name = "input";
    float *input_data = (float *)(buffer + 6);
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

    /* mean tensor configuration */
    struct csi_tensor *mean  = csi_alloc_tensor(sess);
    mean->dim[0] = channel_size;
    mean->dim_count = 1;
    mean->name = "mean";
    float *mean_data = (float *)(buffer + 6 + in_size);
    mean->data = mean_data;
    get_quant_info(mean);

    void *mean_tmp = malloc(channel_size * sizeof(char));
    for(int i = 0; i < channel_size; i++) {
        if (sess->base_dtype == CSINN_DTYPE_UINT8) {
            *((uint8_t *)mean_tmp + i) = csi_ref_quantize_f32_to_u8(mean_data[i], mean->qinfo);
        } else if (sess->base_dtype == CSINN_DTYPE_INT8) {
            *((int8_t *)mean_tmp + i) = csi_ref_quantize_f32_to_i8(mean_data[i], mean->qinfo);
        }
    }
    if (sess->base_dtype == CSINN_DTYPE_UINT8 || sess->base_dtype == CSINN_DTYPE_INT8) {
        mean->data = mean_tmp;
    }


    /* variance tensor configuration */
    struct csi_tensor *variance  = csi_alloc_tensor(sess);
    variance->dim[0] = channel_size;
    variance->dim_count = 1;
    variance->name = "variance";
    float *variance_data = (float *)(buffer + 6 + in_size + channel_size);
    variance->data = variance_data;
    get_quant_info(variance);

    void *variance_tmp = malloc(channel_size * sizeof(char));
    for(int i = 0; i < channel_size; i++) {
        if (sess->base_dtype == CSINN_DTYPE_UINT8) {
            *((uint8_t *)variance_tmp + i) = csi_ref_quantize_f32_to_u8(variance_data[i], variance->qinfo);
        } else if (sess->base_dtype == CSINN_DTYPE_INT8) {
            *((int8_t *)variance_tmp + i) = csi_ref_quantize_f32_to_i8(variance_data[i], variance->qinfo);
        }
    }
    if (sess->base_dtype == CSINN_DTYPE_UINT8 || sess->base_dtype == CSINN_DTYPE_INT8) {
        variance->data = variance_tmp;
    }

    /* gamma tensor configuration */
    struct csi_tensor *gamma  = csi_alloc_tensor(sess);
    gamma->dim[0] = channel_size;
    gamma->dim_count = 1;
    gamma->name = "gamma";
    float *gamma_data = (float *)(buffer + 6 + in_size + channel_size * 2);
    gamma->data = gamma_data;
    get_quant_info(gamma);

    void *gamma_tmp = malloc(channel_size * sizeof(char));
    for(int i = 0; i < channel_size; i++) {
        if (sess->base_dtype == CSINN_DTYPE_UINT8) {
            *((uint8_t *)gamma_tmp + i) = csi_ref_quantize_f32_to_u8(gamma_data[i], gamma->qinfo);
        } else if (sess->base_dtype == CSINN_DTYPE_INT8) {
            *((int8_t *)gamma_tmp + i) = csi_ref_quantize_f32_to_i8(gamma_data[i], gamma->qinfo);
        }
    }
    if (sess->base_dtype == CSINN_DTYPE_UINT8 || sess->base_dtype == CSINN_DTYPE_INT8) {
        gamma->data = gamma_tmp;
    }

    /* beta tensor configuration */
    struct csi_tensor *beta  = csi_alloc_tensor(sess);
    beta->dim[0] = channel_size;
    beta->dim_count = 1;
    beta->name = "beta";
    float *beta_data = (float *)(buffer + 6 + in_size + channel_size * 3);
    beta->data = beta_data;
    get_quant_info(beta);

    /* FIX ME */
    void *beta_tmp = malloc(channel_size * sizeof(int));
    for(int i = 0; i < channel_size; i++) {
        if (sess->base_dtype == CSINN_DTYPE_UINT8) {
            *((int32_t *)beta_tmp + i) = csi_ref_quantize_f32_to_u8(beta_data[i], beta->qinfo);
        } else if (sess->base_dtype == CSINN_DTYPE_INT8) {
            *((int32_t *)beta_tmp + i) = csi_ref_quantize_f32_to_i8(beta_data[i], beta->qinfo);
        }
    }
    if (sess->base_dtype == CSINN_DTYPE_UINT8 || sess->base_dtype == CSINN_DTYPE_INT8) {
        beta->data = beta_tmp;
    }

    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim[0] = input->dim[0];
    output->dim[1] = input->dim[1];
    output->dim[2] = input->dim[2];
    output->dim[3] = input->dim[3];
    output->dim_count = 4;
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    reference->data = (float *)(buffer + 6 + in_size + channel_size * 4);
    output->data = reference->data;
    output->name = "output";
    get_quant_info(output);


    /* operator parameter configuration */
    struct bn_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.epsilon = *((float *)buffer + 5);


    if (csi_batch_normalization_init(input, mean, variance, gamma, beta, output, &params) != CSINN_TRUE) {
        printf("batch normalization init fail.\n\t");
        return -1;
    }

    csi_set_tensor_entry(input, sess);
    csi_set_input(0, input, sess);

    csi_batch_normalization(input, mean, variance, gamma, beta, output, &params);

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
    free(mean_tmp);
    free(variance_tmp);
    free(gamma_tmp);
    free(beta_tmp);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}
