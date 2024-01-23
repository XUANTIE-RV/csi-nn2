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

#include "testutil.h"

int main(int argc, char **argv)
{
    init_testsuite("Testing function of select(layer).\n");
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_tensor *input0 = csinn_alloc_tensor(sess);
    struct csinn_tensor *input1 = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_tensor *condition = csinn_alloc_tensor(sess);
    struct csinn_select_params *params =
        csinn_alloc_params(sizeof(struct csinn_select_params), sess);
    int in_size;

    int *buffer = read_input_data_f32(argv[1]);
    int flag = buffer[4];
    input0->dim[0] = input1->dim[0] = buffer[0];
    input0->dim[1] = input1->dim[1] = buffer[1];
    input0->dim[2] = input1->dim[2] = buffer[2];
    input0->dim[3] = input1->dim[3] = buffer[3];

    condition->dim[0] = buffer[0];
    condition->dim[1] = buffer[1];
    condition->dim[2] = buffer[2];
    condition->dim[3] = buffer[3];

    output->dim[0] = input0->dim[0];
    output->dim[1] = input0->dim[1];
    output->dim[2] = input0->dim[2];
    output->dim[3] = input0->dim[3];

    in_size = input0->dim[0] * input0->dim[1] * input0->dim[2] * input0->dim[3];
    input0->dim_count = 4;
    input1->dim_count = 4;
    condition->dim_count = 4;
    output->dim_count = 4;
    input0->dtype = CSINN_DTYPE_FLOAT32;
    input1->dtype = CSINN_DTYPE_FLOAT32;
    condition->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;

    input0->layout = CSINN_LAYOUT_NCHW;
    input1->layout = CSINN_LAYOUT_NCHW;
    condition->layout = CSINN_LAYOUT_NCHW;
    output->layout = CSINN_LAYOUT_NCHW;
    params->base.api = CSINN_API;

    input0->data = (float *)(buffer + 4);
    input1->data = (float *)(buffer + 4 + in_size);
    condition->data = (float *)(buffer + 4 + 2 * in_size);
    reference->data = (float *)(buffer + 4 + 3 * in_size);
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

    test_select_CSINN_QUANT_FLOAT32(condition, input0, input1, output, params, &difference);
    test_select_CSINN_QUANT_UINT8_ASYM(condition, input0, input1, output, params, &difference);
    test_select_CSINN_QUANT_INT8_SYM(condition, input0, input1, output, params, &difference);

    return done_testing();
}
