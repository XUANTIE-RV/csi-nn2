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
    init_testsuite("Testing function of equal (anole).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int in_out_dims = buffer[0];

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in0_size = 1, in1_size = 1, out_size = 1;

    /* session configuration */
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_ANOLE;
    csi_session_init(sess);
    csi_set_input_number(2, sess);
    csi_set_output_number(1, sess);


    /* input0 tensor configuration */
    struct csi_tensor *input0 = csi_alloc_tensor(NULL);
    input0->dim_count = in_out_dims;
    for (int i = 0; i < input0->dim_count; i++ ) {
        input0->dim[i] = buffer[1 + i];
        in0_size *= input0->dim[i];
    }
    input0->name = "input0";
    float *input0_data = (float *)(buffer + 1 + in_out_dims);
    input0->data = input0_data;
    get_quant_info(input0);

    uint8_t *src0_tmp = malloc(in0_size * sizeof(uint8_t));
    for (int i = 0; i < in0_size; i++) {
        src0_tmp[i] = csi_ref_quantize_f32_to_u8(input0_data[i], input0->qinfo);
    }


    /* input1 tensor configuration */
    struct csi_tensor *input1 = csi_alloc_tensor(NULL);
    input1->dim_count = in_out_dims;
    for (int i = 0; i < input1->dim_count; i++ ) {
        input1->dim[i] = buffer[1 + i];
        in1_size *= input1->dim[i];
    }
    input1->name = "input1";
    float *input1_data = (float *)(buffer + 1 + in_out_dims + in0_size);
    input1->data = input1_data;
    get_quant_info(input0);

    uint8_t *src1_tmp = malloc(in1_size * sizeof(uint8_t));
    for (int i = 0; i < in1_size; i++) {
        src1_tmp[i] = csi_ref_quantize_f32_to_u8(input1_data[i], input1->qinfo);
    }


    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim_count = in_out_dims;
    for (int i = 0; i < output->dim_count; i++ ) {
        output->dim[i] = buffer[1 + i];
        out_size *= output->dim[i];
    }
    reference->data = (float *)(buffer + 1 + in_out_dims + in0_size + in1_size);
    output->data = reference->data;
    output->name = "output";
    get_quant_info(output);


    /* operator parameter configuration */
    struct diso_params params;
    params.base.api = CSINN_API;
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.base.name = "params";


    if (csi_equal_init(input0, input1, output, &params) != CSINN_TRUE) {
        printf("equal init fail.\n\t");
        return -1;
    }

    csi_set_tensor_entry(input0, sess);
    csi_set_tensor_entry(input1, sess);
    csi_set_input(0, input0, sess);
    csi_set_input(1, input1, sess);

    csi_equal(input0, input1, output, &params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);


    struct csi_tensor *input0_tensor = csi_alloc_tensor(NULL);
    struct csi_tensor *input1_tensor = csi_alloc_tensor(NULL);
    input0_tensor->data = src0_tmp;
    input1_tensor->data = src1_tmp;
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
    result_verify_8(reference->data, output_tensor, input0->data, difference, out_size, false);

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
