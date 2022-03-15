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

#include "csi_nn.h"
#include "math_snr.h"
#include "test_utils.h"

void op_test_run(struct csi_tensor *input, struct csi_tensor *kernel, struct csi_tensor *bias,
                 struct csi_tensor *output, struct conv2d_params *params, struct csi_session *sess,
                 struct csi_tensor *real_input, float *output_data, float diff)
{
    csi_session_init(sess);
    csi_set_input_number(1, sess);
    csi_set_output_number(1, sess);
    csi_conv2d_init(input, output, kernel, bias, params);

    csi_set_tensor_entry(input, sess);
    csi_set_input(0, input, sess);

    csi_conv2d(input, output, kernel, bias, params);

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

void test_depthwise_conv2d(struct csi_tensor *input, struct csi_tensor *kernel,
                           struct csi_tensor *bias, struct csi_tensor *output,
                           struct conv2d_params *params, float difference);

int main(int argc, char **argv)
{
    init_testsuite("Testing function of depthwise conv2d(graph).\n");

    int *buffer = read_input_data_f32(argv[1]);
    int group = buffer[1];  // group = in_channel

    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size = 0, out_size = 0, weight_size = 0, bias_size = 0;

    /* input tensor configuration */
    struct csi_tensor *input = csi_alloc_tensor(NULL);
    input->dim[0] = buffer[0];  // batch
    input->dim[1] = buffer[1];  // in_channel
    input->dim[2] = buffer[2];  // height
    input->dim[3] = buffer[3];  // width
    input->dim_count = 4;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    input->name = "input";

    float *input_data = (float *)(buffer + 17);
    input->data = input_data;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;

    /* kernel tensor configuration */
    struct csi_tensor *kernel = csi_alloc_tensor(NULL);
    kernel->dim[0] = buffer[1];           // i
    kernel->dim[1] = buffer[12] / group;  // o
    kernel->dim[2] = buffer[6];           // h
    kernel->dim[3] = buffer[7];           // w
    kernel->dim_count = 4;
    weight_size = kernel->dim[0] * kernel->dim[1] * kernel->dim[2] * kernel->dim[3];
    kernel->name = "kernel";
    float *kernel_data = (float *)(buffer + 17 + in_size);
    kernel->data = kernel_data;
    kernel->is_const = true;
    kernel->dtype = CSINN_DTYPE_FLOAT32;
    kernel->layout = CSINN_LAYOUT_OIHW;

    /* bias tensor configuratioin */
    struct csi_tensor *bias = csi_alloc_tensor(NULL);
    bias->dim[0] = buffer[12];
    bias->dim_count = 1;
    bias_size = bias->dim[0];
    bias->name = "bias";
    float *bias_data = (float *)(buffer + 17 + in_size + weight_size);
    bias->data = bias_data;
    bias->is_const = true;
    bias->dtype = CSINN_DTYPE_FLOAT32;
    bias->layout = CSINN_LAYOUT_O;

    /* output tensor configuration */
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim[0] = buffer[0];   // batch
    output->dim[1] = buffer[12];  // out_channel
    output->dim[2] = buffer[15];  // height
    output->dim[3] = buffer[16];  // width
    output->dim_count = 4;
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    reference->data = (float *)(buffer + 17 + in_size + weight_size + output->dim[1]);
    output->data = reference->data;
    output->name = "output";
    output->layout = CSINN_LAYOUT_NCHW;
    output->dtype = CSINN_DTYPE_FLOAT32;

    /* operator parameter configuration */
    struct conv2d_params params;
    params.stride_height = buffer[4];
    params.stride_width = buffer[5];
    params.pad_left = buffer[8];
    params.pad_right = buffer[9];
    params.pad_top = buffer[10];
    params.pad_down = buffer[11];
    params.dilation_width = buffer[14];
    params.dilation_height = buffer[13];
    params.group = group;
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_NPU_GRAPH;
    params.base.name = "params";
    params.conv_extra.kernel_tm = NULL;
    params.conv_extra.conv_mode = CSINN_DIRECT;

    float difference = argc > 2 ? atof(argv[2]) : 1e-4;
    test_depthwise_conv2d(input, kernel, bias, output, &params, difference);

    return done_testing();
}
