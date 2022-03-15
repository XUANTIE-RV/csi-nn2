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
    init_testsuite("Testing function of softmax(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int axis = buffer[4];

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size = 0, out_size = 0;
    enum csinn_dtype_enum test_dtype = CSINN_TEST_DTYPE;
    /* session configuration */
    struct csi_session *sess = csi_alloc_session();
    sess->base_api = CSINN_API;
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);


    /* input tensor configuration */
    struct csi_tensor *input  = csi_alloc_tensor(sess);
    input->dim[0] = buffer[0];          //
    input->dim[1] = buffer[1];          //
    input->dim[2] = buffer[2];          //
    input->dim[3] = buffer[3];          //
    input->dim_count = 4;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    input->name = "input";
    float *input_data = (float *)(buffer + 5);
    input->data = input_data;
    get_quant_info(input);
    input->dtype = CSINN_DTYPE_FLOAT32;

    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(sess);
    output->dim[0] = input->dim[0];
    output->dim[1] = input->dim[1];
    output->dim[2] = input->dim[2];
    output->dim[3] = input->dim[3];
    output->dim_count = 4;
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    reference->data = (float *)(buffer + 5 + in_size);
    output->data = reference->data;
    output->name = "output";
    get_quant_info(output);


    /* operator parameter configuration */
    struct softmax_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.axis = axis;

    struct csi_tensor *input_tensor = convert_input(input, test_dtype);
    input->dtype = sess->base_dtype;
    /*
    light:
        software layer. axis = 0 or 1 or 2 or 3
        At the moment algorithm expects 4D tensors and gives wrong values for 1-3D
    */
    if (csi_softmax_init(input, output, &params) != CSINN_TRUE) {
        printf("softmax init fail.\n\t");
        return -1;
    }

    csi_set_tensor_entry(input, sess);
    csi_set_input(0, input, sess);

    csi_softmax(input, output, &params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);

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
    } else if (sess->base_dtype == CSINN_DTYPE_FLOAT32 && output_tensor->dtype == CSINN_DTYPE_INT8) {
        struct csi_tensor *foutput = csi_ref_tensor_transform_f32(output_tensor);
        result_verify_f32(reference->data, foutput->data, input->data, difference, out_size, false);
    }

    /* free alloced memory */
    free(buffer);
    free(input_tensor->qinfo);
    free(input_tensor);
    free(output_tensor->qinfo);
    free(output_tensor);
    free(reference->qinfo);
    free(reference);

    csi_session_deinit(sess);
    csi_free_session(sess);
    return done_testing();
}
