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

void op_test_run(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
                 struct diso_params *params, struct csi_session *sess,
                 struct csi_tensor *real_input0, struct csi_tensor *real_input1, float *output_data,
                 float diff)
{
    csi_session_init(sess);
    csi_set_input_number(2, sess);
    csi_set_output_number(1, sess);
    csi_maximum_init(input0, input1, output, params);

    csi_set_tensor_entry(input0, sess);
    csi_set_tensor_entry(input1, sess);
    csi_set_input(0, input0, sess);
    csi_set_input(1, input1, sess);

    csi_maximum(input0, input1, output, params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);

    csi_update_input(0, real_input0, sess);
    csi_update_input(1, real_input1, sess);
    csi_session_run(sess);

    csi_get_output(0, output, sess);

    struct csi_tensor *foutput = csi_ref_tensor_transform_f32(output);
    result_verify_f32(output_data, foutput->data, input0->data, diff, csi_tensor_size(output),
                      false);

    free_input(real_input0);
    free_input(real_input1);
    csi_ref_tensor_transform_free_f32(foutput);
    csi_session_deinit(sess);
    csi_free_session(sess);
}

void test_maximum(struct csi_tensor *input0, struct csi_tensor *input1, struct csi_tensor *output,
                  struct diso_params *params, float difference);

int main(int argc, char** argv)
{
    init_testsuite("Testing function of maximum(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in0_size = 1, in1_size = 1, out_size = 1;

    /* input0 tensor configuration */
    struct csi_tensor *input0  = csi_alloc_tensor(NULL);
    input0->dim_count = buffer[0];
    for(int i = 0; i < input0->dim_count; i++) {
        input0->dim[i] = buffer[1 + i];
        in0_size *= input0->dim[i];
    }
    input0->name = "input0";
    float *input0_data = (float *)(buffer + 1 + input0->dim_count);
    input0->data = input0_data;
    input0->dtype = CSINN_DTYPE_FLOAT32;
    input0->layout = CSINN_LAYOUT_NCHW;
    /* input1 tensor configuration */
    struct csi_tensor *input1  = csi_alloc_tensor(NULL);
    input1->dim_count = input0->dim_count;
    for(int i = 0; i < input1->dim_count; i++) {
        input1->dim[i] = input0->dim[i];
        in1_size *= input1->dim[i];
    }
    input1->name = "input1";
    float *input1_data = (float *)(buffer + 1 + input0->dim_count + in0_size);
    input1->data = input1_data;
    input1->dtype = CSINN_DTYPE_FLOAT32;
    input1->layout = CSINN_LAYOUT_NCHW;
    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim_count = input0->dim_count;
    for(int i = 0; i < output->dim_count; i++) {
        output->dim[i] = csi_ref_max_internal_s32(input0->dim[i], input1->dim[i]);  // in fact, ouput->dim[i] are always equal to input0->dim[i]
        out_size *= output->dim[i];
    }
    reference->data = (float *)(buffer + 1 + input0->dim_count + in0_size + in1_size);
    output->data = reference->data;
    output->name = "output";
    output->layout = CSINN_LAYOUT_NCHW;
    output->dtype = CSINN_DTYPE_FLOAT32;

    /* operator parameter configuration */
    struct diso_params params;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;

    test_maximum(input0, input1, output, &params, difference);

    return done_testing();
}
