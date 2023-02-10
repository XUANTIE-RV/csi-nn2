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
    init_testsuite("Testing function of gather_nd f32.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *indices = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_gather_nd_params *params =
        csinn_alloc_params(sizeof(struct csinn_gather_nd_params), NULL);
    int in_size = 1, out_size = 1, indices_size = 1;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim_count = buffer[0];
    output->dim_count = 0;  // init output->dim_count = 0
    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i + 1];
        in_size *= input->dim[i];
    }
    indices->dim_count = buffer[1 + input->dim_count];
    for (int i = 0; i < indices->dim_count; i++) {
        indices->dim[i] = buffer[i + 2 + input->dim_count];
        indices_size *= indices->dim[i];
        if (i < indices->dim_count - 1) {
            output->dim_count++;
            output->dim[i] = indices->dim[i];
        }
    }

    int axis = indices->dim[indices->dim_count - 1];

    int indices_outer_size = 1;
    indices_outer_size = indices_size / indices->dim[indices->dim_count - 1];

    int input_inner_size = 1;
    for (int i = axis; i < input->dim_count; i++) {
        input_inner_size *= input->dim[i];
        output->dim[output->dim_count] = input->dim[i];
        output->dim_count++;
    }

    out_size = indices_outer_size * input_inner_size;
    input->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;
    params->base.api = CSINN_API;

    indices->data = (uint32_t *)(buffer + 2 + input->dim_count + indices->dim_count);
    input->data = (float *)(buffer + 2 + input->dim_count + indices->dim_count + indices_size);
    reference->data =
        (float *)(buffer + 2 + input->dim_count + indices->dim_count + indices_size + in_size);
    output->data = (float *)malloc(out_size * sizeof(float));
    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_gather_nd_init(input, indices, output, params) == CSINN_TRUE) {
        csinn_gather_nd(input, indices, output, params);
    }

    result_verify_f32(reference->data, output->data, input->data, difference, out_size, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
