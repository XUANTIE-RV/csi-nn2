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
    init_testsuite("Testing function of fullyconnected f32.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_tensor *weight = csinn_alloc_tensor(NULL);
    struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
    struct csinn_fc_params *params = csinn_alloc_params(sizeof(struct csinn_fc_params), NULL);
    int in_size0, in_size1, out_size;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0] = buffer[0];   // batch
    input->dim[1] = buffer[1];   // in_size
    weight->dim[0] = buffer[2];  // out_size
    weight->dim[1] = buffer[1];  // in_size
    bias->dim[0] = buffer[2];
    output->dim[0] = buffer[0];
    output->dim[1] = buffer[2];
    input->dim_count = 2;
    weight->dim_count = 2;
    bias->dim_count = 1;
    output->dim_count = 2;
    in_size0 = input->dim[0] * input->dim[1];
    in_size1 = weight->dim[0] * weight->dim[1];
    out_size = output->dim[0] * output->dim[1];
    input->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;
    params->base.api = CSINN_API;

    input->data = (float *)(buffer + 3);
    weight->data = (float *)(buffer + 3 + in_size0);
    bias->data = (float *)(buffer + 3 + in_size0 + in_size1);
    reference->data = (float *)(buffer + 3 + in_size0 + in_size1 + buffer[2]);
    output->data = malloc(out_size * sizeof(float));
    float difference = argc > 2 ? atof(argv[2]) : 0.9;
    ;

    if (csinn_fullyconnected_init(input, output, weight, bias, params) == CSINN_TRUE) {
        csinn_fullyconnected(input, output, weight, bias, params);
    }

    result_verify_f32(reference->data, output->data, input->data, difference, out_size, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
