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
    init_testsuite("Testing function of mod f32.\n");

    struct csi_tensor *input0 = csi_alloc_tensor(NULL);
    struct csi_tensor *input1 = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct diso_params params;
    int in_size0, in_size1;

    int *buffer = read_input_data_f32(argv[1]);
    int flag  = buffer[4];
    input0->dim[0] = buffer[0];          // batch
    input0->dim[1] = buffer[1];          // height
    input0->dim[2] = buffer[2];          // width
    input0->dim[3] = buffer[3];          // channel

    output->dim[0] = input0->dim[0];
    output->dim[1] = input0->dim[1];
    output->dim[2] = input0->dim[2];
    output->dim[3] = input0->dim[3];

    in_size0 = input0->dim[0] * input0->dim[1] * input0->dim[2] * input0->dim[3];
    input0->dim_count = 4;
    output->dim_count = 4;
    input0->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;
    if(flag) {
        input1->dim[0] = input0->dim[3];
        input1->dim_count = 1;
        in_size1 = input1->dim[0];
    } else {
        input1->dim[0] = input0->dim[0];
        input1->dim[1] = input0->dim[1];
        input1->dim[2] = input0->dim[2];
        input1->dim[3] = input0->dim[3];
        input1->dim_count = 4;
        in_size1 = in_size0;
    }
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    input0->data    = (float *)(buffer + 5);
    input1->data    = (float *)(buffer + 5 + in_size0);
    reference->data = (float *)(buffer + 5 + in_size0 + in_size1);
    output->data    = malloc(in_size0 * sizeof(float));
    float difference = argc > 2 ? *argv[2] : 0;

    if (csi_mod_init(input0, input1, output, &params) == CSINN_TRUE) {
        csi_mod(input0, input1, output, &params);
    }

    result_verify_f32(reference->data, output->data, input0->data, difference, in_size0, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
