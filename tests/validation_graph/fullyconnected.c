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
    init_testsuite("Testing function of fullyconnected(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size = 0, weights_size = 0, bias_size = 0, out_size = 0;

    /* session configuration */
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_API;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);


    /* input tensor configuration */
    struct csi_tensor *input  = csi_alloc_tensor(sess);
    input->dim[0] = buffer[0];          // batch
    input->dim[1] = buffer[1];          // in_nodes
    input->dim_count = 2;
    in_size = input->dim[0] * input->dim[1];
    input->name = "input";
    float *input_data = (float *)(buffer + 3);
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


    /* weight tensor configuration */
    struct csi_tensor *weights  = csi_alloc_tensor(sess);
    weights->dim[0] = buffer[2];    // out_nodes
    weights->dim[1] = buffer[1];    // in_nodes
    weights->dim_count = 2;
    weights_size = weights->dim[0] * weights->dim[1];
    weights->name = "weights";
    float *weight_data = (float *)(buffer + 3 + in_size);
    weights->data = weight_data;
    get_quant_info(weights);

    void *weight_tmp = malloc(weights_size * sizeof(char));
    for(int i = 0; i < weights_size; i++) {
        if (sess->base_dtype == CSINN_DTYPE_UINT8) {
            *((uint8_t *)weight_tmp + i) = csi_ref_quantize_f32_to_u8(weight_data[i], weights->qinfo);
        } else if (sess->base_dtype == CSINN_DTYPE_INT8) {
            *((int8_t *)weight_tmp + i) = csi_ref_quantize_f32_to_i8(weight_data[i], weights->qinfo);
        }
    }
    if (sess->base_dtype == CSINN_DTYPE_UINT8 || sess->base_dtype == CSINN_DTYPE_INT8) {
        weights->data = weight_tmp;
    }


    /* bias tensor configuration */
    struct csi_tensor *bias  = csi_alloc_tensor(sess);
    bias->dim[0] = buffer[2];    // out_nodes
    bias->dim_count = 1;
    bias_size = bias->dim[0];
    bias->name = "bias";
    float *bias_data = (float *)(buffer + 3 + in_size + weights_size);
    bias->data = bias_data;
    // get_quant_info(bias);    // anole:segentation fault

    /* FIX ME */
    int32_t *bias_tmp = malloc(bias_size * sizeof(int32_t));
    for(int i = 0; i < bias_size; i++) {
        if (sess->base_dtype == CSINN_DTYPE_UINT8) {
            // *((int32_t *)bias_tmp + i) = csi_ref_quantize_f32_to_u8(bias_data[i], bias->qinfo);
            *((int32_t *)bias_tmp + i) = (int32_t)(bias_data[i] / (input->qinfo->scale * weights->qinfo->scale));
        }
    }
    if (sess->base_dtype == CSINN_DTYPE_UINT8 || sess->base_dtype == CSINN_DTYPE_INT8) {
        bias->data = bias_tmp;
    }


    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim[0] = buffer[0];     // batch
    output->dim[1] = buffer[2];     // out_nodes
    output->dim_count = 2;
    out_size = output->dim[0] * output->dim[1];
    reference->data = (float *)(buffer + 3 + in_size + weights_size + bias_size);
    output->data = reference->data;
    output->name = "output";
    get_quant_info(output);

    /* operator parameter configuration */
    struct fc_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.units = buffer[2];   // out_nodes


    if (csi_fullyconnected_init(input, output, weights, bias, &params) != CSINN_TRUE) {
        printf("fullyconnected init fail.\n\t");
        return -1;
    }

    csi_set_tensor_entry(input, sess);
    csi_set_input(0, input, sess);

    csi_fullyconnected(input, output, weights, bias, &params);

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
    free(weight_tmp);
    free(bias_tmp);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}
