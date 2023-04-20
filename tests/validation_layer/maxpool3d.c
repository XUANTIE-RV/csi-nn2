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

int main(int argc, char **argv)
{
    init_testsuite("Testing function of maxpool3d(layer).\n");

    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_pool_params *params = csinn_alloc_params(sizeof(struct csinn_pool_params), sess);
    int in_size = 1;
    int out_size = 1;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0] = buffer[0];  // batch
    input->dim[1] = buffer[1];  // channel
    input->dim[2] = buffer[2];  // depth
    input->dim[3] = buffer[3];  // height
    input->dim[4] = buffer[4];  // width

    output->dim[0] = buffer[0];
    output->dim[1] = buffer[1];
    output->dim[2] = buffer[17];
    output->dim[3] = buffer[18];
    output->dim[4] = buffer[19];

    params->stride_depth = buffer[5];
    params->stride_height = buffer[6];
    params->stride_width = buffer[7];
    params->filter_depth = buffer[8];
    params->filter_height = buffer[9];
    params->filter_width = buffer[10];

    params->pad_left = buffer[11];
    params->pad_right = buffer[12];
    params->pad_top = buffer[13];
    params->pad_down = buffer[14];
    params->pad_front = buffer[15];
    params->pad_back = buffer[16];
    params->base.layout = CSINN_LAYOUT_NCDHW;

    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCDHW;
    input->is_const = 0;
    input->quant_channel = 1;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCDHW;
    output->is_const = 0;
    output->quant_channel = 1;
    input->dim_count = 5;
    output->dim_count = 5;

    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3] * input->dim[4];
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3] * output->dim[4];
    params->base.api = CSINN_API;

    input->data = (float *)(buffer + 20);
    reference->data = (float *)(buffer + 20 + in_size);
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

    test_maxpool3d_CSINN_QUANT_FLOAT32(input, output, params, &difference);
    test_maxpool3d_CSINN_QUANT_UINT8_ASYM(input, output, params, &difference);
    test_maxpool3d_CSINN_QUANT_INT8_SYM(input, output, params, &difference);

    return done_testing();
}
