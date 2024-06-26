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

#include "testutil.h"

int main(int argc, char **argv)
{
    init_testsuite("Testing function of im2col(layer).\n");

    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_im2col_params *params =
        csinn_alloc_params(sizeof(struct csinn_im2col_params), sess);
    int in_size = 1, out_size = 1;

    int *buffer = read_input_data_f32(argv[1]);

    input->dim[0] = buffer[0];  // batch
    input->dim[1] = buffer[1];  // in_channel
    input->dim[2] = buffer[2];  // in_height
    input->dim[3] = buffer[3];  // in_width
    input->dim_count = 4;

    params->kernel_h = buffer[4];
    params->kernel_w = buffer[5];
    params->stride_h = buffer[6];
    params->stride_w = buffer[7];
    params->pad_left = buffer[8];
    params->pad_right = buffer[9];
    params->pad_top = buffer[10];
    params->pad_down = buffer[11];

    for (int i = 0; i < input->dim_count; i++) {
        in_size *= input->dim[i];
    }

    int out_h =
        (input->dim[2] + params->pad_top + params->pad_down - params->kernel_h) / params->stride_h +
        1;
    int out_w = (input->dim[3] + params->pad_left + params->pad_right - params->kernel_w) /
                    params->stride_w +
                1;

    output->dim[0] = input->dim[1] * params->kernel_h * params->kernel_w;
    output->dim[1] = input->dim[0] * out_h * out_w;
    output->dim_count = 2;

    out_size = input->dim[0] * input->dim[1] * params->kernel_h * params->kernel_w * out_h * out_w;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;
    params->base.layout = CSINN_LAYOUT_NCHW;
    params->base.api = CSINN_API;

    input->data = (float *)(buffer + 12);
    reference->data = (float *)(buffer + 12 + in_size);
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

    test_im2col_CSINN_QUANT_FLOAT32(input, output, params, &difference);
    test_im2col_CSINN_QUANT_UINT8_ASYM(input, output, params, &difference);
    test_im2col_CSINN_QUANT_INT8_SYM(input, output, params, &difference);

    return done_testing();
}
