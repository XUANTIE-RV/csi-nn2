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

int main(int argc, char **argv)
{
    init_testsuite("Testing function of convolution f32.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_tensor *kernel = csinn_alloc_tensor(NULL);
    struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
    struct csinn_conv2d_params *params =
        csinn_alloc_params(sizeof(struct csinn_conv2d_params), NULL);
    int in_size, out_size, weight_size;

    if (argc == 1) {
        printf("please assign the input data.\n");
        return 0;
    }

    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0] = buffer[0];  // batch
    input->dim[1] = buffer[1];  // height
    input->dim[2] = buffer[2];  // width
    input->dim[3] = buffer[3];  // in_channel
    kernel->dim[1] = buffer[6];
    kernel->dim[2] = buffer[7];
    kernel->dim[3] = buffer[3];
    bias->dim[0] = buffer[12];
    output->dim[0] = buffer[0];   // batch
    output->dim[1] = buffer[16];  // height
    output->dim[2] = buffer[15];  // width
    output->dim[3] = buffer[12];  // out_channel
    params->stride_height = buffer[4];
    params->stride_width = buffer[5];
    params->pad_left = buffer[8];
    params->pad_right = buffer[9];
    params->pad_top = buffer[10];
    params->pad_down = buffer[11];
    params->dilation_width = buffer[13];
    params->dilation_height = buffer[14];
    params->base.layout = CSINN_LAYOUT_NHWC;
    params->group = 1;

    input->dim_count = 4;
    kernel->dim_count = 4;
    bias->dim_count = 1;
    output->dim_count = 4;
    input->dtype = CSINN_DTYPE_FLOAT32;
    kernel->dtype = CSINN_DTYPE_FLOAT32;
    bias->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    weight_size = output->dim[3] * input->dim[3] * kernel->dim[1] * kernel->dim[2];
    params->base.api = CSINN_API;

    input->data = (float *)(buffer + 17);
    kernel->data = (float *)(buffer + 17 + in_size);
    bias->data = (float *)(buffer + 17 + in_size + weight_size);
    reference->data = (float *)(buffer + 17 + in_size + weight_size + output->dim[3]);
    output->data = malloc(out_size * sizeof(float));
    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_conv2d_init(input, output, kernel, bias, params) == CSINN_TRUE) {
        csinn_conv2d(input, output, kernel, bias, params);
    }

    result_verify_f32(reference->data, output->data, input->data, difference, out_size, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
