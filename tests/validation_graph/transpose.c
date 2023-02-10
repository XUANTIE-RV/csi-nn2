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

#include "csi_nn.h"
#include "test_utils.h"

void op_test_run(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_transpose_params *params, struct csinn_session *sess,
                 struct csinn_tensor *real_input, float *output_data, float diff)
{
    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);
    csinn_transpose_init(input, output, params);

    csinn_set_tensor_entry(input, sess);
    csinn_set_input(0, input, sess);

    csinn_transpose(input, output, params);

    csinn_set_output(0, output, sess);
    csinn_session_setup(sess);

    csinn_update_input(0, real_input, sess);
    csinn_session_run(sess);
    csinn_get_output(0, output, sess);

    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    result_verify_f32(output_data, foutput->data, input->data, diff, csinn_tensor_size(output),
                      false);

    free_input(real_input);
    shl_ref_tensor_transform_free_f32(foutput);
    csinn_session_deinit(sess);
    csinn_free_session(sess);
}

void test_transpose(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_transpose_params *params, float difference);

int main(int argc, char **argv)
{
    init_testsuite("Testing function of transpose(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int32_t *permute = (int32_t *)malloc(buffer[0] * sizeof(int32_t));
    for (int i = 0; i < buffer[0]; i++) {
        permute[i] = buffer[1 + buffer[0] + i];
    }

    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    int in_size = 1, out_size = 1;

    /* input tensor configuration */
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim_count = buffer[0];
    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[1 + i];
        in_size *= input->dim[i];
    }
    input->name = "input";
    float *input_data = (float *)(buffer + 1 + 3 * input->dim_count);
    input->data = input_data;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    /* output tensor configuration */
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim_count = input->dim_count;
    for (int i = 0; i < output->dim_count; i++) {
        output->dim[i] = input->dim[permute[i]];
        out_size *= output->dim[i];
    }
    reference->data = (float *)(buffer + 1 + 3 * input->dim_count + in_size);
    output->data = reference->data;
    output->name = "output";
    output->layout = CSINN_LAYOUT_NCHW;
    output->dtype = CSINN_DTYPE_FLOAT32;

    /* operator parameter configuration */
    struct csinn_transpose_params *params =
        csinn_alloc_params(sizeof(struct csinn_transpose_params), NULL);
    params->base.name = "params";
    params->base.layout = CSINN_LAYOUT_NCHW;
    params->permute = permute;
    params->permute_num = input->dim_count;

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    test_transpose(input, output, params, difference);

    return done_testing();
}
