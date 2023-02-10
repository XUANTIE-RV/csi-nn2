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
#include "shl_thead_rvv.h"
#include "test_utils.h"
#include "testutil.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of convolution1d(layer).\n");

    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_tensor *kernel = csinn_alloc_tensor(sess);
    struct csinn_tensor *bias = csinn_alloc_tensor(sess);
    struct csinn_conv1d_params *params = (csinn_conv1d_params *)csinn_alloc_params(sizeof(struct csinn_conv1d_params), sess);
    int in_size, out_size, kernel_size;

    if (argc == 1) {
        printf("please assign the input data.\n");
        return 0;
    }

    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0]   = buffer[0];    // batch
    input->dim[1]   = buffer[1];    // in_channel
    input->dim[2]   = buffer[2];    // width
    kernel->dim[0]  = buffer[7];
    kernel->dim[1]  = 1;
    kernel->dim[2]  = buffer[4];
    bias->dim[0]    = buffer[7];
    output->dim[0]  = buffer[0];    // batch
    output->dim[1]  = buffer[7];    // out_channel
    output->dim[2]  = buffer[9];    // width

    params->stride_width = buffer[3];
    params->pad_left  = buffer[5];
    params->pad_right = buffer[6];
    params->dilation_width = buffer[8];
    params->base.layout = CSINN_LAYOUT_NCW;
    params->group = buffer[1];

    input->dim_count = 3;
    input->layout = CSINN_LAYOUT_NCW;
    input->is_const = 0;
    input->quant_channel = 1;

    kernel->dim_count = 3;
    kernel->layout = CSINN_LAYOUT_OIW;
    kernel->is_const = 1;
    kernel->quant_channel = 1;

    bias->dim_count = 1;
    bias->layout = CSINN_LAYOUT_O;
    bias->is_const = 1;
    bias->quant_channel = 1;

    output->dim_count = 3;
    output->layout = CSINN_LAYOUT_NCW;
    output->is_const = 0;
    output->quant_channel = 1;

    input->dtype = CSINN_DTYPE_FLOAT32;
    kernel->dtype = CSINN_DTYPE_FLOAT32;
    bias->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;

    in_size  = input->dim[0] * input->dim[1] * input->dim[2];
    out_size = output->dim[0] * output->dim[1] * output->dim[2];
    kernel_size = kernel->dim[0] * kernel->dim[1] * kernel->dim[2];
    params->base.api = CSINN_API;

    input->data     = (float *)(buffer + 10);
    kernel->data    = (float *)(buffer + 10 + in_size);
    bias->data      = (float *)(buffer + 10 + in_size + kernel_size);
    reference->data = (float *)(buffer + 10 + in_size + kernel_size + output->dim[1]);
    output->data    = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

#if (DTYPE==32)
    test_fully_op(input, output, kernel, bias, params, CSINN_QUANT_FLOAT32,
                  csinn_conv1d_init, csinn_conv1d, &difference);
#elif (DTYPE==16)
    test_fully_op(input, output, kernel, bias, params, CSINN_QUANT_FLOAT16,
                  csinn_conv1d_init, csinn_conv1d, &difference);
#elif (DTYPE==8)
    test_fully_op(input, output, kernel, bias, params, CSINN_QUANT_INT8_SYM,
                  csinn_conv1d_init, csinn_conv1d, &difference);

#endif

    return done_testing();
}
