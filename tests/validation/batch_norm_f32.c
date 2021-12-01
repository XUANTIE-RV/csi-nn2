/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of batch normalization f32.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *mean = csi_alloc_tensor(NULL);
    struct csi_tensor *variance = csi_alloc_tensor(NULL);
    struct csi_tensor *beta = csi_alloc_tensor(NULL);
    struct csi_tensor *gamma = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct bn_params params;
    int size = 1;

    int *buffer = read_input_data_f32(argv[1]);
    /* get the dim para */
    output->dim_count = input->dim_count = buffer[0];
    for (int i = 0; i < input->dim_count; ++i) {
        output->dim[i] = input->dim[i] = buffer[1 + i];
    }

    for (int i = 0; i < input->dim_count; ++i) {
        size *= input->dim[i];
    }

    input->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;
    params.base.layout = CSINN_NHWC;
    params.epsilon = *((float *)buffer + 1 + input->dim_count);
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    input->data     = (float *)(buffer + 2 + input->dim_count);
    mean->data      = (float *)(buffer + 2 + input->dim_count + size);
    variance->data  = (float *)(buffer + 2 + input->dim_count + size + input->dim[input->dim_count - 1]);
    gamma->data     = (float *)(buffer + 2 + input->dim_count + size + 2 * input->dim[input->dim_count - 1]);
    beta->data      = (float *)(buffer + 2 + input->dim_count + size + 3 * input->dim[input->dim_count - 1]);
    reference->data = (float *)(buffer + 2 + input->dim_count + size + 4 * input->dim[input->dim_count - 1]);
    output->data    = malloc(size * sizeof(float));
    float difference = argc > 2 ? atof(argv[2]) : 1e-1;

    if (csi_batch_normalization_init(input, mean, variance, gamma, beta, output, &params) == CSINN_TRUE) {
        csi_batch_normalization(input, mean, variance, gamma, beta, output, &params);
    }


    result_verify_f32(reference->data, output->data, input->data, difference, size, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
