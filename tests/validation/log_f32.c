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

/* CSI-NN2 version 1.12.x */

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of log f32.\n");

    struct csi_tensor *input0 = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct siso_params params;
    int in_size0;

    int *buffer = read_input_data_f32(argv[1]);

    input0->dim[0] = buffer[0];         
    input0->dim[1] = buffer[1];         
    input0->dim[2] = buffer[2];         
    input0->dim[3] = buffer[3];         

    output->dim[0] = input0->dim[0];
    output->dim[1] = input0->dim[1];
    output->dim[2] = input0->dim[2];
    output->dim[3] = input0->dim[3];

    in_size0 = input0->dim[0] * input0->dim[1] * input0->dim[2] * input0->dim[3];
    input0->dim_count = 4;
    output->dim_count = 4;
    input0->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    input0->data    = (float *)(buffer + 4);
    reference->data = (float *)(buffer + 4 + in_size0);
    output->data = (float *)malloc(in_size0 * sizeof(float));
    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csi_log_init(input0, output, &params) == CSINN_TRUE) {
        csi_log(input0, output, &params);
    }

    result_verify_f32(reference->data, output->data, input0->data, difference, in_size0, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
