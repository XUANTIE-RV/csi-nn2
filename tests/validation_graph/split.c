/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

void op_test_run(struct csinn_tensor *input, struct csinn_tensor **output,
                 struct csinn_split_params *params, struct csinn_session *sess,
                 struct csinn_tensor *real_input, float **output_data, float diff)
{
    int output_cnt = params->output_num;
    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(output_cnt, sess);
    csinn_split_init(input, output, params);

    csinn_set_tensor_entry(input, sess);
    csinn_set_input(0, input, sess);

    csinn_split(input, output, params);

    for (int i = 0; i < output_cnt; i++) {
        csinn_set_output(i, output[i], sess);
    }
    csinn_session_setup(sess);

    csinn_update_input(0, real_input, sess);
    csinn_session_run(sess);
    for (int i = 0; i < output_cnt; i++) {
        csinn_get_output(i, output[i], sess);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output[i]);
        result_verify_f32(output_data[i], foutput->data, input->data, diff,
                          csinn_tensor_size(output[i]), false);
    }

    free_input(real_input);
    csinn_session_deinit(sess);
    csinn_free_session(sess);
}

void test_split(struct csinn_tensor *input, struct csinn_tensor **output,
                struct csinn_split_params *params, float difference);

int main(int argc, char **argv)
{
    init_testsuite("Testing function of split(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int axis = buffer[4];
    int output_cnt = buffer[5];
    int32_t *split_index = (int32_t *)malloc(output_cnt * sizeof(int32_t));
    for (int i = 0; i < output_cnt; i++) {
        split_index[i] = buffer[axis] / output_cnt;
    }

    struct csinn_tensor *reference[output_cnt];
    for (int i = 0; i < output_cnt; i++) {
        reference[i] = csinn_alloc_tensor(NULL);
    }
    float min_value, max_value;
    int in_size = 0;
    int out_size[output_cnt];
    int acc_out_size = 0;  // in fact, different output tensor may has different out_size

    /* input tensor configuration */
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = buffer[0];  // batch
    input->dim[1] = buffer[1];  // channel
    input->dim[2] = buffer[2];  // height
    input->dim[3] = buffer[3];  // width
    input->dim_count = 4;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    input->name = "input";
    float *input_data = (float *)(buffer + 6);
    input->data = input_data;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;

    /* output tensor configuration */
    struct csinn_tensor *output[output_cnt];
    char output_name[output_cnt][10];
    for (int i = 0; i < output_cnt; i++) {
        output[i] = csinn_alloc_tensor(NULL);
        for (int j = 0; j < 4; j++) {
            if (j == axis) {
                output[i]->dim[j] = split_index[i];
            } else {
                output[i]->dim[j] = input->dim[j];
            }
        }
        output[i]->dim_count = 4;
        out_size[i] = output[i]->dim[0] * output[i]->dim[1] * output[i]->dim[2] * output[i]->dim[3];
        reference[i]->data = (float *)(buffer + 6 + in_size + acc_out_size);
        acc_out_size += out_size[i];
        output[i]->data = reference[i]->data;
        sprintf(output_name[i], "output_%d", i);
        output[i]->name = output_name[i];
        output[i]->is_const = 0;
        output[i]->layout = CSINN_LAYOUT_NCHW;
        output[i]->dtype = CSINN_DTYPE_FLOAT32;
    }

    /* operator parameter configuration */
    struct csinn_split_params *params = csinn_alloc_params(sizeof(struct csinn_split_params), NULL);
    params->base.name = "params";
    params->base.layout = CSINN_LAYOUT_NCHW;
    params->axis = axis;
    params->output_num = output_cnt;
    int temp = 0;
    for (int i = 0; i < output_cnt; i++) {
        temp += split_index[i];
        split_index[i] = temp;
    }
    params->split_index = split_index;

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    test_split(input, output, params, difference);

    return done_testing();
}
