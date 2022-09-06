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

int main(int argc, char **argv)
{
    init_testsuite("Testing function of group convolution nchw f32.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_tensor *kernel = csinn_alloc_tensor(NULL);
    struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
    struct csinn_conv2d_params *params =
        csinn_alloc_params(sizeof(struct csinn_conv2d_params), NULL);
    int in_size = 0, out_size = 0, weight_size = 0, bias_size = 0;

    if (argc == 1) {
        printf("please assign the input data.\n");
        return 0;
    }

    int *buffer = read_input_data_f32(argv[1]);
    int group = buffer[17];

    input->dim[0] = buffer[0];  // batch
    input->dim[1] = buffer[1];  // in_channel
    input->dim[2] = buffer[2];  // height
    input->dim[3] = buffer[3];  // width
    input->dim_count = 4;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    input->name = "input";

    float *input_data = (float *)(buffer + 18);
    input->data = input_data;

    kernel->dim[0] = buffer[12];         // o
    kernel->dim[1] = buffer[1] / group;  // i
    kernel->dim[2] = buffer[6];          // h
    kernel->dim[3] = buffer[7];          // w
    kernel->dim_count = 4;
    weight_size = kernel->dim[0] * kernel->dim[1] * kernel->dim[2] * kernel->dim[3];
    kernel->name = "kernel";
    float *kernel_data = (float *)(buffer + 18 + in_size);
    kernel->data = kernel_data;

    bias->dim[0] = buffer[12];
    bias->dim_count = 1;
    bias_size = bias->dim[0];
    bias->name = "bias";
    float *bias_data = (float *)(buffer + 18 + in_size + weight_size);
    bias->data = bias_data;

    output->dim[0] = buffer[0];   // batch
    output->dim[1] = buffer[12];  // out_channel
    output->dim[2] = buffer[16];  // height
    output->dim[3] = buffer[15];  // width
    output->dim_count = 4;
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    reference->data = (float *)(buffer + 18 + in_size + weight_size + output->dim[1]);
    output->data = reference->data;
    output->name = "output";

    params->stride_height = buffer[4];
    params->stride_width = buffer[5];
    params->pad_left = buffer[8];
    params->pad_right = buffer[9];
    params->pad_top = buffer[10];
    params->pad_down = buffer[11];
    params->dilation_width = buffer[13];
    params->dilation_height = buffer[14];
    params->group = group;
    params->base.api = CSINN_API;
    params->base.layout = CSINN_LAYOUT_NCHW;
    params->base.name = "params";
    params->conv_extra.kernel_tm = NULL;
    params->conv_extra.conv_mode = CSINN_DIRECT;

    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_conv2d_init(input, output, kernel, bias, params) != CSINN_TRUE) {
        printf("group conv2d init fail.\n\t");
        return -1;
    }

    result_verify_f32(reference->data, output->data, input->data, difference, out_size, false);

    free(buffer);
    return done_testing();
}
