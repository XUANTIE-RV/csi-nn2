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

/* CSI-NN2 version 2.0.x */

#include "csi_nn.h"
#include "math_snr.h"
#include "test_utils.h"

void op_test_run(struct csinn_tensor *input0, struct csinn_tensor *input1,
                 struct csinn_tensor *output, struct csinn_diso_params *params,
                 struct csinn_session *sess, struct csinn_tensor *real_input0,
                 struct csinn_tensor *real_input1, float *output_data, float diff)
{
    csinn_session_init(sess);
    csinn_set_input_number(2, sess);
    csinn_set_output_number(1, sess);
    csinn_add_init(input0, input1, output, params);

    csinn_set_tensor_entry(input0, sess);
    csinn_set_tensor_entry(input1, sess);
    csinn_set_input(0, input0, sess);
    csinn_set_input(1, input1, sess);

    csinn_add(input0, input1, output, params);

    csinn_set_output(0, output, sess);
    csinn_session_setup(sess);

    csinn_update_input(0, real_input0, sess);
    csinn_update_input(1, real_input1, sess);
    csinn_session_run(sess);

    csinn_get_output(0, output, sess);

    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    result_verify_f32(output_data, foutput->data, input0->data, diff, csinn_tensor_size(output),
                      false);

    free_input(real_input0);
    free_input(real_input1);
    shl_ref_tensor_transform_free_f32(foutput);
    csinn_session_deinit(sess);
    csinn_free_session(sess);
}

void test_add(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params, float difference);

int main(int argc, char **argv)
{
    init_testsuite("Testing function of add(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int flag = buffer[4];

    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    int in0_size = 0, in1_size = 0, out_size = 0;

    /* input0 tensor configuration */
    struct csinn_tensor *input0 = csinn_alloc_tensor(NULL);
    input0->dim[0] = buffer[0];  // batch
    input0->dim[1] = buffer[1];  // channel
    input0->dim[2] = buffer[2];  // height
    input0->dim[3] = buffer[3];  // width
    input0->dim_count = 4;
    in0_size = input0->dim[0] * input0->dim[1] * input0->dim[2] * input0->dim[3];
    input0->name = "input0";
    float *input0_data = (float *)(buffer + 5);
    input0->data = input0_data;
    input0->dtype = CSINN_DTYPE_FLOAT32;
    input0->layout = CSINN_LAYOUT_NCHW;

    /* input1 tensor configuration */
    struct csinn_tensor *input1 = csinn_alloc_tensor(NULL);
    if (flag) {
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
    input1->data = input1_data;
    input1->dtype = CSINN_DTYPE_FLOAT32;
    input1->layout = CSINN_LAYOUT_NCHW;

    /* output tensor configuration */
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[0] = input0->dim[0];
    output->dim[1] = input0->dim[1];
    output->dim[2] = input0->dim[2];
    output->dim[3] = input0->dim[3];
    output->dim_count = 4;
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    reference->data = (float *)(buffer + 5 + in0_size + in1_size);
    output->data = reference->data;
    output->name = "output";
    output->layout = CSINN_LAYOUT_NCHW;
    output->dtype = CSINN_DTYPE_FLOAT32;

    /* operator parameter configuration */
    struct csinn_diso_params *params = csinn_alloc_params(sizeof(struct csinn_diso_params), NULL);
    params->base.name = "params";
    params->base.layout = CSINN_LAYOUT_NCHW;

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;

    test_add(input0, input1, output, params, difference);

    return done_testing();
}
