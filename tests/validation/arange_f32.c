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
    init_testsuite("Testing function of arange f32.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_arange_params *params =
        csinn_alloc_params(sizeof(struct csinn_arange_params), NULL);
    int out_size = 1;

    int *buffer = read_input_data_f32(argv[1]);

    out_size = buffer[3];
    params->start = buffer[0];
    params->stop = buffer[1];
    params->step = buffer[2];
    output->dim_count = 1;
    output->dim[0] = out_size;
    output->dtype = CSINN_DTYPE_FLOAT32;
    params->base.api = CSINN_API;
    input->data = 0;

    reference->data = (float *)(buffer + 4);
    output->data = (float *)malloc(out_size * sizeof(float));
    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_arange_init(output, params) == CSINN_TRUE) {
        csinn_arange(output, params);
    }

    result_verify_f32(reference->data, output->data, input->data, difference, out_size, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
