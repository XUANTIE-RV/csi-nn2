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

void op_test_run(struct csi_tensor *input, struct csi_tensor *output, struct reduce_params *params,
                 struct csi_session *sess, struct csi_tensor *real_input, float *output_data,
                 float diff)
{
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);
    csi_mean_init(input, output, params);

    csi_set_tensor_entry(input, sess);
    csi_set_input(0, input, sess);

    csi_mean(input, output, params);

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

void test_mean(struct csi_tensor *input, struct csi_tensor *output, struct reduce_params *params,
               float difference);

bool find_axis(int *axis, int axis_cnt, int index)
{
    for(int i = 0; i < axis_cnt; i++) {
        if(axis[i] == index) {
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv)
{
    init_testsuite("Testing function of mean(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);
    bool keep_dim = buffer[4];
    int axis_count = buffer[5];
    int *axis = (int *)malloc(axis_count * sizeof(int));
    for(int i = 0; i < axis_count; i++) {
        axis[i] = buffer[6 + i];
    }

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size = 0, out_size = 1;

    /* input tensor configuration */
    struct csi_tensor *input  = csi_alloc_tensor(NULL);
    input->dim[0] = buffer[0];          // batch
    input->dim[1] = buffer[1];          // channel
    input->dim[2] = buffer[2];          // height
    input->dim[3] = buffer[3];          // width
    input->dim_count = 4;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    input->name = "input";
    float *input_data = (float *)(buffer + 6 + axis_count);
    input->data = input_data;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;

    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim[2] = input->dim[2];
    output->dim[3] = input->dim[3];
    if(keep_dim) {
        output->dim_count = input->dim_count;
        output->dim[0] = input->dim[0];     // can not reduce on batch and channel axis
        output->dim[1] = input->dim[1];
        for(int i = 2; i < output->dim_count; i++) {
            if(find_axis(axis, axis_count, i) == true) {
                output->dim[i] = 1;
            } else {
                output->dim[i] = input->dim[i];
            }
        }
    } else {
        output->dim_count = input->dim_count - axis_count;
        output->dim[0] = input->dim[0];     // can not reduce on batch and channel axis
        output->dim[1] = input->dim[1];
        int j = 2;
        for(int i = 2; i < input->dim_count; i++) {
            if(find_axis(axis, axis_count, i) == false) {
                output->dim[j] = input->dim[i];
                j++;
            }
        }
    }
    for(int i = 0; i < output->dim_count; i++) {
        out_size *= output->dim[i];
    }
    reference->data = (float *)(buffer + 6 + axis_count + in_size);
    output->data = reference->data;
    output->name = "output";
    output->layout = CSINN_LAYOUT_NCHW;
    output->dtype = CSINN_DTYPE_FLOAT32;

    /* operator parameter configuration */
    struct reduce_params params;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.axis = axis;
    params.axis_count = axis_count;
    params.keepdims = keep_dim;

    /* verify result */
    float difference = argc > 2 ? atof(argv[2]) : 1e-4;

    test_mean(input, output, &params, difference);

    return done_testing();
}
