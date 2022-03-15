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

void op_test_run(struct csi_tensor *input, struct csi_tensor *kernel, struct csi_tensor *bias,
                 struct csi_tensor *output, struct fc_params *params, struct csi_session *sess,
                 struct csi_tensor *real_input, float *output_data, float diff)
{
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);
    csi_fullyconnected_init(input, output, kernel, bias, params);

    csi_set_tensor_entry(input, sess);
    csi_set_input(0, input, sess);

    csi_fullyconnected(input, output, kernel, bias, params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);

    csi_update_input(0, real_input, sess);
    csi_session_run(sess);
    csi_get_output(0, output, sess);

    struct csi_tensor *foutput = csi_ref_tensor_transform_f32(output);
    result_verify_f32(output_data, foutput->data, input->data, diff, csi_tensor_size(output),
                      false);

    free_input(real_input);
    csi_ref_tensor_transform_free_f32(foutput);
    csi_session_deinit(sess);
    csi_free_session(sess);
}

void test_fc(struct csi_tensor *input, struct csi_tensor *weights, struct csi_tensor *bias,
             struct csi_tensor *output, struct fc_params *params, float difference);

int main(int argc, char** argv)
{
    init_testsuite("Testing function of fullyconnected(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size = 0, weights_size = 0, bias_size = 0, out_size = 0;

    /* input tensor configuration */
    struct csi_tensor *input  = csi_alloc_tensor(NULL);
    input->dim[0] = buffer[0];          // batch
    input->dim[1] = buffer[1];          // in_nodes
    input->dim_count = 2;
    in_size = input->dim[0] * input->dim[1];
    input->name = "input";
    float *input_data = (float *)(buffer + 3);
    input->data = input_data;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;

    /* weight tensor configuration */
    struct csi_tensor *weights  = csi_alloc_tensor(NULL);
    weights->dim[0] = buffer[2];    // out_nodes
    weights->dim[1] = buffer[1];    // in_nodes
    weights->dim_count = 2;
    weights_size = weights->dim[0] * weights->dim[1];
    weights->name = "weights";
    float *weight_data = (float *)(buffer + 3 + in_size);
    weights->data = weight_data;
    weights->is_const = true;
    weights->dtype = CSINN_DTYPE_FLOAT32;
    weights->layout = CSINN_LAYOUT_OIHW;


    /* bias tensor configuration */
    struct csi_tensor *bias  = csi_alloc_tensor(NULL);
    bias->dim[0] = buffer[2];    // out_nodes
    bias->dim_count = 1;
    bias_size = bias->dim[0];
    bias->name = "bias";
    float *bias_data = (float *)(buffer + 3 + in_size + weights_size);
    bias->data = bias_data;
    bias->is_const = true;
    bias->dtype = CSINN_DTYPE_FLOAT32;
    bias->layout = CSINN_LAYOUT_O;

    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim[0] = buffer[0];     // batch
    output->dim[1] = buffer[2];     // out_nodes
    output->dim_count = 2;
    out_size = output->dim[0] * output->dim[1];
    reference->data = (float *)(buffer + 3 + in_size + weights_size + bias_size);
    output->data = reference->data;
    output->name = "output";
    output->layout = CSINN_LAYOUT_NCHW;
    output->dtype = CSINN_DTYPE_FLOAT32;

    /* operator parameter configuration */
    struct fc_params params;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.units = buffer[2];   // out_nodes

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    test_fc(input, weights, bias, output, &params, difference);

    return done_testing();
}
