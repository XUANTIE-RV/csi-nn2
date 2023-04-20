/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
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

/* SHL version 2.1.x */

#include "csi_nn.h"
#include "test_utils.h"

void op_test_run(struct csinn_tensor **input, struct csinn_tensor *output,
                 struct csinn_concat_params *params, struct csinn_session *sess,
                 struct csinn_tensor **real_input, float *output_data, float diff)
{
    csinn_session_init(sess);
    csinn_set_input_number(params->inputs_count, sess);
    csinn_set_output_number(1, sess);
    csinn_concat_init(input, output, params);

    for (int i = 0; i < params->inputs_count; i++) {
        csinn_set_tensor_entry(input[i], sess);
        csinn_set_input(i, input[i], sess);
    }

    csinn_concat(input, output, params);

    csinn_set_output(0, output, sess);
    csinn_session_setup(sess);

    for (int i = 0; i < params->inputs_count; i++) {
        csinn_update_input(i, real_input[i], sess);
    }
    csinn_session_run(sess);
    csinn_get_output(0, output, sess);

    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    result_verify_f32(output_data, foutput->data, input[0]->data, diff, csinn_tensor_size(output),
                      false);

    // free_input(real_input);
    shl_ref_tensor_transform_free_f32(foutput);
    csinn_session_deinit(sess);
    csinn_free_session(sess);
}

void test_concat(struct csinn_tensor **input, struct csinn_tensor *output,
                 struct csinn_concat_params *params, float difference);

int main(int argc, char **argv)
{
    init_testsuite("Testing function of concat(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int input_cnt = buffer[4];
    int axis = buffer[5];

    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    int in_size = 0, out_size = 1;

    /* input tensor configuration */
    struct csinn_tensor *input[input_cnt];
    float *input_data[input_cnt];
    void **src_tmp = malloc(input_cnt * sizeof(void *));
    char input_name[input_cnt][10];
    for (int i = 0; i < input_cnt; i++) {
        input[i] = csinn_alloc_tensor(NULL);
        input[i]->dim[0] = buffer[0];  // batch
        input[i]->dim[1] = buffer[1];  // in_channel
        input[i]->dim[2] = buffer[2];  // height
        input[i]->dim[3] = buffer[3];  // width
        input[i]->dim_count = 4;
        in_size = input[i]->dim[0] * input[i]->dim[1] * input[i]->dim[2] * input[i]->dim[3];

        input_data[i] = (float *)(buffer + 6 + in_size * i);
        input[i]->data = input_data[i];
        input[i]->dtype = CSINN_DTYPE_FLOAT32;
        input[i]->layout = CSINN_LAYOUT_NCHW;
    }

    /* output tensor configuration */
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    for (int i = 0; i < 4; i++) {
        if (i == axis) {
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
    struct csinn_concat_params *params =
        csinn_alloc_params(sizeof(struct csinn_concat_params), NULL);
    params->base.name = "params";
    params->base.layout = CSINN_LAYOUT_NCHW;
    params->axis = axis;
    params->inputs_count = input_cnt;

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    test_concat(input, output, params, difference);

    return done_testing();
}
