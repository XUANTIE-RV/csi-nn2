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
    init_testsuite("Testing function of isnan f32.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_siso_params *params = csinn_alloc_params(sizeof(struct csinn_siso_params), NULL);
    int in_size = 1, out_size = 1;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim_count = buffer[0];
    output->dim_count = input->dim_count;
    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i + 1];
        output->dim[i] = input->dim[i];
        in_size *= input->dim[i];
    }

    out_size = in_size;
    input->dtype = CSINN_DTYPE_FLOAT32;
    params->base.api = CSINN_API;

    input->data = (float *)(buffer + 1 + input->dim_count);
    reference->data = (bool *)(buffer + 1 + input->dim_count + in_size);
    output->data = (bool *)malloc(out_size * sizeof(bool));
    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_isnan_bool_init(input, output, params) == CSINN_TRUE) {
        csinn_isnan_bool(input, output, params);
    }

    result_verify_bool(reference->data, output->data, input->data, difference, out_size, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
