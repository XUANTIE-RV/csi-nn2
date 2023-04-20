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
    init_testsuite("Testing function of stack(layer).\n");

    int in_size = 1;
    int out_size = 1;
    int *buffer = read_input_data_f32(argv[1]);
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_stack_params *params = csinn_alloc_params(sizeof(struct csinn_stack_params), sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);

    params->inputs_count = buffer[0];
    params->axis = buffer[1];
    output->dim_count = buffer[2];
    for (int i = 0; i < output->dim_count; i++) {
        output->dim[i] = buffer[3 + i];
        out_size *= output->dim[i];
    }
    in_size = out_size / params->inputs_count;

    struct csinn_tensor *input[params->inputs_count];
    for (int i = 0; i < params->inputs_count; i++) {
        input[i] = csinn_alloc_tensor(sess);
        input[i]->data = (float *)(buffer + 3 + output->dim_count + in_size * i);
        input[i]->dim_count = buffer[2] - 1;
        input[i]->layout = CSINN_LAYOUT_NCHW;
        input[i]->is_const = 0;
        input[i]->quant_channel = 1;
        input[i]->dtype = CSINN_DTYPE_FLOAT32;
        for (int j = 0; j < input[i]->dim_count; j++) {
            if (j < params->axis) {
                input[i]->dim[j] = buffer[3 + j];  // input[i]->dim[j] = output->dim[j]
            } else {
                input[i]->dim[j] = buffer[3 + j + 1];  // input[i]->dim[j] = output->dim[j + 1]
            }
        }
    }

    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;
    params->base.api = CSINN_API;
    reference->data = (float *)(buffer + 3 + output->dim_count + in_size * params->inputs_count);
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

    test_stack_CSINN_QUANT_FLOAT32((struct csinn_tensor **)input, output, params, &difference);
    test_stack_CSINN_QUANT_UINT8_ASYM((struct csinn_tensor **)input, output, params, &difference);
    test_stack_CSINN_QUANT_INT8_SYM((struct csinn_tensor **)input, output, params, &difference);

    return done_testing();
}
