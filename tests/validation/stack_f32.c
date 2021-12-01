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

/* CSI-NN2 version 1.8.x */

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of stack f32.\n");   
    int in_size = 1;
    int out_size = 1;
    int *buffer = read_input_data_f32(argv[1]);

    struct stack_params params;

    params.inputs_count = buffer[0];
    params.axis = buffer[1];

    struct csi_tensor *input[params.inputs_count];
    for (int i = 0; i < params.inputs_count; i++) {
        input[i] = csi_alloc_tensor(NULL);
        input[i]->dim_count = buffer[2] - 1;
        input[i]->dtype = CSINN_DTYPE_FLOAT32;
        for (int j = 0; j < input[i]->dim_count; j++) {
            if (j < params.axis) {
                input[i]->dim[j] = buffer[3+j];     // input[i]->dim[j] = output->dim[j]
            } else {
                input[i]->dim[j] = buffer[3+j+1];   // input[i]->dim[j] = output->dim[j + 1]
            }
        }
    }
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);

    output->dim_count = buffer[2];

    for(int i = 0; i < output->dim_count; i++) {
        output->dim[i] = buffer[3+i];
        out_size *= output->dim[i];
    }
    in_size = out_size / params.inputs_count;
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    for(int i = 0; i < params.inputs_count; i++) {
        input[i]->data = (float *)(buffer + 3 + output->dim_count + in_size * i);
    }
    reference->data = (float *)(buffer + 3 + output->dim_count + in_size * params.inputs_count);
    output->data  = (float *)malloc(out_size * sizeof(float));
    float difference = argc > 2 ? atof(argv[2]) : 1e-6;

    if (csi_stack_init(input, output, &params) == CSINN_TRUE) {
        csi_stack(input, output, &params);
    }

    result_verify_f32(reference->data, output->data, input[0]->data, difference, out_size, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
