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
    init_testsuite("Testing function of unstack(layer).\n");

    int in_size = 1;
    int out_size = 1;
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    int *buffer = read_input_data_f32(argv[1]);
    struct csinn_unstack_params *params =
        csinn_alloc_params(sizeof(struct csinn_unstack_params), sess);
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    params->axis = buffer[0];
    input->dim_count = buffer[1];
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[2 + i];
        in_size *= input->dim[i];
    }
    params->outputs_count = input->dim[params->axis];

    struct csinn_tensor *reference[params->outputs_count];
    for (int i = 0; i < params->outputs_count; i++) {
        reference[i] = csinn_alloc_tensor(sess);
    }

    out_size = in_size / params->outputs_count;
    params->base.api = CSINN_API;

    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;
    input->data = (float *)(buffer + 2 + input->dim_count);

    struct csinn_tensor *output[params->outputs_count];
    for (int i = 0; i < params->outputs_count; i++) {
        output[i] = csinn_alloc_tensor(sess);
        output[i]->dim_count = input->dim_count - 1;
        output[i]->dtype = CSINN_DTYPE_FLOAT32;
        output[i]->layout = CSINN_LAYOUT_NCHW;
        output[i]->is_const = 0;
        output[i]->quant_channel = 1;
        for (int j = 0; j < input->dim_count; j++) {
            if (j < params->axis) {
                output[i]->dim[j] = input->dim[j];
            } else if (j > params->axis) {
                output[i]->dim[j - 1] = input->dim[j];
            }
        }

        reference[i]->data = (float *)(buffer + 2 + input->dim_count + in_size + out_size * i);
        output[i]->data = reference[i]->data;
    }
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

    test_unstack_CSINN_QUANT_FLOAT32(input, (struct csinn_tensor **)output, params, &difference);
    test_unstack_CSINN_QUANT_UINT8_ASYM(input, (struct csinn_tensor **)output, params, &difference);
    test_unstack_CSINN_QUANT_INT8_SYM(input, (struct csinn_tensor **)output, params, &difference);

    return done_testing();
}
