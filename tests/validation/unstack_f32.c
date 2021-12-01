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
    init_testsuite("Testing function of unstack f32.\n");

    int in_size = 1;
    int out_size = 1;
    int *buffer = read_input_data_f32(argv[1]);

    struct unstack_params params;
    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    params.axis = buffer[0];
    input->dim_count = buffer[1];
    for(int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[2+i];
        in_size *= input->dim[i];
    }
    params.outputs_count = input->dim[params.axis];
    struct csi_tensor *output[params.outputs_count];
    for (int i = 0; i < params.outputs_count; i++) {
        output[i] = csi_alloc_tensor(NULL);
        output[i]->dim_count = input->dim_count - 1;
        output[i]->dtype = CSINN_DTYPE_FLOAT32;
        for(int j = 0; j < input->dim_count; j++) {
            if(j < params.axis) {
                output[i]->dim[j] = input->dim[j];
            } else if(j > params.axis) {
                output[i]->dim[j-1] = input->dim[j];
            }
        }
    }
    out_size = in_size / params.outputs_count;
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    input->data = (float *)(buffer + 2 + input->dim_count);
    reference->data = (float *)(buffer + 2 + input->dim_count + in_size);
    input->dtype = CSINN_DTYPE_FLOAT32;

    for(int i = 0; i < params.outputs_count; i++) {
        output[i]->data  = (float *)malloc(out_size * sizeof(float));
    }
    float difference = argc > 2 ? atof(argv[2]) : 1e-6;

    if (csi_unstack_init(input, output, &params) == CSINN_TRUE) {
        csi_unstack(input, output, &params);
    }

    float *ref_addr = (float *)reference->data;
    for(int i = 0; i < params.outputs_count; i++) {
        result_verify_f32(ref_addr, output[i]->data, input->data, difference, out_size, false);
        ref_addr += out_size;
    }

    free(buffer);
    for(int i = 0; i < params.outputs_count; i++) {
        free(output[i]->data);
        output[i]->data = NULL;
    }
    return done_testing();
}
