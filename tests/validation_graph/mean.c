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
                 struct csinn_reduce_params *params, struct csinn_session *sess,
                 struct csinn_tensor *real_input, float *output_data, float diff)
{
    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);
    csinn_mean_init(input, output, params);

    csinn_set_tensor_entry(input, sess);
    csinn_set_input(0, input, sess);

    csinn_mean(input, output, params);

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

void test_mean(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_reduce_params *params, float difference);

bool find_axis(int *axis, int axis_cnt, int index)
{
    for (int i = 0; i < axis_cnt; i++) {
        if (axis[i] == index) {
            return true;
        }
    }
    return false;
}

int main(int argc, char **argv)
{
    init_testsuite("Testing function of mean(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);
    bool keep_dim = buffer[4];
    int axis_count = buffer[5];
    int *axis = (int *)malloc(axis_count * sizeof(int));
    for (int i = 0; i < axis_count; i++) {
        axis[i] = buffer[6 + i];
    }

    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    int in_size = 0, out_size = 1;

    /* input tensor configuration */
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = buffer[0];  // batch
    input->dim[1] = buffer[1];  // channel
    input->dim[2] = buffer[2];  // height
    input->dim[3] = buffer[3];  // width
    input->dim_count = 4;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    input->name = "input";
    float *input_data = (float *)(buffer + 6 + axis_count);
    input->data = input_data;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;

    /* output tensor configuration */
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[2] = input->dim[2];
    output->dim[3] = input->dim[3];
    if (keep_dim) {
        output->dim_count = input->dim_count;
        output->dim[0] = input->dim[0];  // can not reduce on batch and channel axis
        output->dim[1] = input->dim[1];
        for (int i = 2; i < output->dim_count; i++) {
            if (find_axis(axis, axis_count, i) == true) {
                output->dim[i] = 1;
            } else {
                output->dim[i] = input->dim[i];
            }
        }
    } else {
        output->dim_count = input->dim_count - axis_count;
        output->dim[0] = input->dim[0];  // can not reduce on batch and channel axis
        output->dim[1] = input->dim[1];
        int j = 2;
        for (int i = 2; i < input->dim_count; i++) {
            if (find_axis(axis, axis_count, i) == false) {
                output->dim[j] = input->dim[i];
                j++;
            }
        }
    }
    for (int i = 0; i < output->dim_count; i++) {
        out_size *= output->dim[i];
    }
    reference->data = (float *)(buffer + 6 + axis_count + in_size);
    output->data = reference->data;
    output->name = "output";
    output->layout = CSINN_LAYOUT_NCHW;
    output->dtype = CSINN_DTYPE_FLOAT32;

    /* operator parameter configuration */
    struct csinn_reduce_params *params =
        csinn_alloc_params(sizeof(struct csinn_reduce_params), NULL);
    params->base.name = "params";
    params->base.layout = CSINN_LAYOUT_NCHW;
    params->axis = axis;
    params->axis_count = axis_count;
    params->keepdims = keep_dim;

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;

    test_mean(input, output, params, difference);

    return done_testing();
}
