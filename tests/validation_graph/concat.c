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

/* CSI-NN2 version 1.10.x */

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"

void op_test_run(struct csi_tensor **input, struct csi_tensor *output, struct concat_params *params,
                 struct csi_session *sess, struct csi_tensor **real_input, float *output_data,
                 float diff)
{
    csi_session_init(sess);
    csi_set_input_number(params->inputs_count, sess);
    csi_set_output_number(1, sess);
    csi_concat_init(input, output, params);

    for(int i = 0; i < params->inputs_count; i++) {
        csi_set_tensor_entry(input[i], sess);
        csi_set_input(i, input[i], sess);
    }

    csi_concat(input, output, params);

    csi_set_output(0, output, sess);
    csi_session_setup(sess);

    for(int i = 0; i < params->inputs_count; i++) {
        csi_update_input(i, real_input[i], sess);
    }
    csi_session_run(sess);
    csi_get_output(0, output, sess);

    struct csi_tensor *foutput = csi_ref_tensor_transform_f32(output);
    result_verify_f32(output_data, foutput->data, input[0]->data, diff, csi_tensor_size(output),
                      false);

    // free_input(real_input);
    csi_ref_tensor_transform_free_f32(foutput);
    csi_session_deinit(sess);
    csi_free_session(sess);
}

void test_concat(struct csi_tensor **input, struct csi_tensor *output, struct concat_params *params,
                 float difference);

int main(int argc, char** argv)
{
    init_testsuite("Testing function of concat(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int input_cnt = buffer[4];
    int axis = buffer[5];

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size = 0, out_size = 1;

    /* input tensor configuration */
    struct csi_tensor *input[input_cnt];
    float *input_data[input_cnt];
    void **src_tmp = malloc(input_cnt * sizeof(void *));
    char input_name[input_cnt][10];
    for(int i = 0; i < input_cnt; i++) {
        input[i]  = csi_alloc_tensor(NULL);
        input[i]->dim[0] = buffer[0];          // batch
        input[i]->dim[1] = buffer[1];          // in_channel
        input[i]->dim[2] = buffer[2];          // height
        input[i]->dim[3] = buffer[3];          // width
        input[i]->dim_count = 4;
        in_size = input[i]->dim[0] * input[i]->dim[1] * input[i]->dim[2] * input[i]->dim[3];

        input_data[i] = (float *)(buffer + 6 + in_size * i);
        input[i]->data = input_data[i];
        input[i]->dtype = CSINN_DTYPE_FLOAT32;
        input[i]->layout = CSINN_LAYOUT_NCHW;
    }

    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    for(int i = 0; i < 4; i++) {
        if(i == axis) {
            output->dim[i] = input_cnt * buffer[i];
        } else {
            output->dim[i] = buffer[i];
        }
        out_size *= output->dim[i];
    }
    output->dim_count = 4;

    reference->data = (float *)(buffer + 6 + in_size * input_cnt);
    output->data = reference->data;
    output->name = "output";
    output->layout = CSINN_LAYOUT_NCHW;
    output->dtype = CSINN_DTYPE_FLOAT32;

    /* operator parameter configuration */
    struct concat_params params;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.axis = axis;
    params.inputs_count = input_cnt;

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    test_concat(input, output, &params, difference);

    return done_testing();
}
