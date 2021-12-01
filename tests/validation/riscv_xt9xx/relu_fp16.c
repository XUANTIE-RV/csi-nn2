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

/* CSI-NN2 version 1.10.x */

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"
#include "csi_c906.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of relu fp16.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct relu_params params;
    int in_size;

    char *buffer = read_input_data_fp16(argv[1], 4);

    int *int_buffer = (int *)buffer;
    __fp16 *fp16_buffer = (__fp16 *)(buffer + 4 * 4);

    input->dim[0] = int_buffer[0];
    input->dim[1] = int_buffer[1];
    input->dim[2] = int_buffer[2];
    input->dim[3] = int_buffer[3];

    output->dim[0] = input->dim[0];
    output->dim[1] = input->dim[1];
    output->dim[2] = input->dim[2];
    output->dim[3] = input->dim[3];

    input->dim_count = 4;
    output->dim_count = 4;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    params.base.api = CSINN_API;

    input->data      = (__fp16 *)(fp16_buffer);
    reference->data  = (__fp16 *)(fp16_buffer + in_size);
    output->data     = malloc(in_size * sizeof(__fp16));
    float difference = argc > 2 ? atof(argv[2]) : 0.1;

    csi_c906_relu_fp16(input, output, &params);    // TODO: use nn2_api

    result_verify_fp16(output->data, reference->data, input->data, difference, in_size, false);

    /* free alloced memory */
    free(buffer);
    free(input->qinfo);
    free(input);
    free(output->qinfo);
    free(output);
    free(reference->qinfo);
    free(reference);
    return done_testing();
}
