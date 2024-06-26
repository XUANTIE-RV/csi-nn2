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
    init_testsuite("Testing function of tile(layer).\n");

    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_tile_params *params = csinn_alloc_params(sizeof(struct csinn_tile_params), sess);
    int in_size = 1;
    int out_size = 1;

    int *buffer = read_input_data_f32(argv[1]);

    input->dim_count = buffer[0];
    output->dim_count = input->dim_count;
    params->reps_num = buffer[0];

    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i + 1];
        in_size *= input->dim[i];
    }
    params->reps = (int *)malloc(params->reps_num * sizeof(int));
    for (int i = 0; i < params->reps_num; i++) {
        params->reps[i] = buffer[i + 1 + input->dim_count];
        output->dim[i] = input->dim[i] * params->reps[i];
        out_size *= params->reps[i];
    }
    out_size = out_size * in_size;
    params->base.api = CSINN_API;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;

    input->data = (float *)(buffer + 1 + input->dim_count + input->dim_count);
    reference->data = (float *)(buffer + 1 + input->dim_count + input->dim_count + in_size);
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

    test_tile_CSINN_QUANT_FLOAT32(input, output, params, &difference);
    test_tile_CSINN_QUANT_UINT8_ASYM(input, output, params, &difference);
    test_tile_CSINN_QUANT_INT8_SYM(input, output, params, &difference);

    return done_testing();
}
