/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

/* CSI-NN2 version 2.0.x */

#include "csi_nn.h"
#include "math_snr.h"
#include "test_utils.h"

int main(int argc, char **argv)
{
    init_testsuite("Testing function of prelu nhwc f32.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_tensor *alpha_data = csinn_alloc_tensor(NULL);
    struct csinn_prelu_params *params = csinn_alloc_params(sizeof(struct csinn_prelu_params), NULL);
    int in_size = 1;
    int out_size = 1;

    int *buffer = read_input_data_f32(argv[1]);
    output->dim[0] = input->dim[0] = buffer[0];  // batch
    output->dim[1] = input->dim[1] = buffer[1];  // height
    output->dim[2] = input->dim[2] = buffer[2];  // width
    output->dim[3] = input->dim[3] = buffer[3];  // channel
    input->dim_count = 4;
    output->dim_count = 4;
    input->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;
    params->base.layout = CSINN_LAYOUT_NHWC;

    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    out_size = in_size;
    params->base.api = CSINN_API;

    input->data = (float *)(buffer + 4);
    alpha_data->data = (float *)(buffer + 4 + in_size);
    reference->data = (float *)(buffer + 4 + in_size + input->dim[3]);
    output->data = malloc(in_size * sizeof(float));
    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_prelu_init(input, alpha_data, output, params) == CSINN_TRUE) {
        csinn_prelu(input, alpha_data, output, params);
    }

    result_verify_f32(reference->data, output->data, input->data, difference, out_size, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
