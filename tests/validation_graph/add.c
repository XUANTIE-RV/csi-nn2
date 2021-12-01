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
    init_testsuite("Testing function of add(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int flag  = buffer[4];

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in0_size = 0, in1_size = 0, out_size = 0;

    /* session configuration */
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_API;
    csi_session_init(sess);
    csi_set_input_number(2, sess);
    csi_set_output_number(1, sess);

    /* input0 tensor configuration */
    struct csi_tensor *input0  = csi_alloc_tensor(sess);
    input0->dim[0] = buffer[0];          // batch
    input0->dim[1] = buffer[1];          // channel
    input0->dim[2] = buffer[2];          // height
    input0->dim[3] = buffer[3];          // width
    input0->dim_count = 4;
    in0_size = input0->dim[0] * input0->dim[1] * input0->dim[2] * input0->dim[3];
    input0->name = "input0";
    float *input0_data = (float *)(buffer + 5);
    input0->data   = input0_data;
    get_quant_info(input0);

    void *src0_tmp = malloc(in0_size * sizeof(char));
    for(int i = 0; i < in0_size; i++) {
        if (sess->base_dtype == CSINN_DTYPE_UINT8) {
            *((uint8_t *)src0_tmp + i) = csi_ref_quantize_f32_to_u8(input0_data[i], input0->qinfo);
        } else if (sess->base_dtype == CSINN_DTYPE_INT8) {
            *((int8_t *)src0_tmp + i) = csi_ref_quantize_f32_to_i8(input0_data[i], input0->qinfo);
        }
    }

    /* input1 tensor configuration */
    struct csi_tensor *input1  = csi_alloc_tensor(sess);
    if(flag) {
        input1->dim[0] = input0->dim[3];
        input1->dim_count = 1;
        in1_size = input1->dim[0];
    } else {
        input1->dim[0] = input0->dim[0];
        input1->dim[1] = input0->dim[1];
        input1->dim[2] = input0->dim[2];
        input1->dim[3] = input0->dim[3];
        input1->dim_count = 4;
        in1_size = in0_size;
    }
    input1->name = "input1";
    float *input1_data = (float *)(buffer + 5 + in0_size);
    input1->data  = input1_data;
    get_quant_info(input1);

    void *src1_tmp = malloc(in1_size * sizeof(char));
    for(int i = 0; i < in1_size; i++) {
        if (sess->base_dtype == CSINN_DTYPE_UINT8) {
            *((uint8_t *)src1_tmp + i) = csi_ref_quantize_f32_to_u8(input1_data[i], input1->qinfo);
        } else if (sess->base_dtype == CSINN_DTYPE_INT8) {
            *((int8_t *)src1_tmp + i) = csi_ref_quantize_f32_to_i8(input1_data[i], input1->qinfo);
        }
    }

    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim[0] = input0->dim[0];
    output->dim[1] = input0->dim[1];
    output->dim[2] = input0->dim[2];
    output->dim[3] = input0->dim[3];
    output->dim_count = 4;
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    reference->data= (float *)(buffer + 5 + in0_size + in1_size);
    output->data = reference->data;
    output->name = "output";
    get_quant_info(output);

    /* operator parameter configuration */
    struct diso_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;

    if (csi_add_init(input0, input1, output, &params) != CSINN_TRUE) {
        printf("add init fail.\n\t");
        return -1;
    }

    csi_set_tensor_entry(input0, sess);
    csi_set_tensor_entry(input1, sess);
    csi_set_input(0, input0, sess);
    csi_set_input(1, input1, sess);

    csi_add(input0, input1, output, &params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);


    struct csi_tensor *input0_tensor = csi_alloc_tensor(NULL);
    struct csi_tensor *input1_tensor = csi_alloc_tensor(NULL);
    if (sess->base_dtype == CSINN_DTYPE_FLOAT32) {
        input0_tensor->data = input0_data;
        input1_tensor->data = input1_data;
    } else if (sess->base_dtype == CSINN_DTYPE_UINT8 || sess->base_dtype == CSINN_DTYPE_UINT8) {
        input0_tensor->data = src0_tmp;
        input1_tensor->data = src1_tmp;
    }
    csi_update_input(0, input0_tensor, sess);
    csi_update_input(1, input1_tensor, sess);
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
        result_verify_8(reference->data, output_tensor, input0->data, difference, out_size, false);
    } else if (sess->base_dtype == CSINN_DTYPE_FLOAT32) {
        result_verify_f32(reference->data, output_tensor->data, input0->data, difference, out_size, false);
    }

    /* free alloced memory */
    free(buffer);
    free(input0_tensor->qinfo);
    free(input0_tensor);
    free(input1_tensor->qinfo);
    free(input1_tensor);
    free(output_tensor->qinfo);
    free(output_tensor);
    free(reference->qinfo);
    free(reference);
    free(src0_tmp);
    free(src1_tmp);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}
