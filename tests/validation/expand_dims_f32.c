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
    init_testsuite("Testing function of expand_dims f32.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct expand_dims_params params;
    int in_size = 1;
    int out_size = 1;
    int *buffer = read_input_data_f32(argv[1]);

    int dim_count = buffer[0];
    int axis = buffer[1];
    for(int i = 0; i < dim_count; i++) {
        input->dim[i] = buffer[2 + i];
        in_size *= input->dim[i];
    }
    input->dim_count = dim_count;
    output->dim_count = input->dim_count + 1;   // axis is 0-D scalar

    for(int i = 0; i < output->dim_count; i++) {
        if(i < axis) {
            output->dim[i] = input->dim[i];
        } else if(i == axis) {
            output->dim[i] = 1;
        } else {
            output->dim[i] = input->dim[i - 1];
        }
    }

    input->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;
    out_size = in_size;
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    input->data = (float *)(buffer + 2 + dim_count);
    reference->data = (float *)(buffer + 2 + dim_count + in_size);
    output->data = (float *)malloc(sizeof(float) * out_size);
    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csi_expand_dims_init(input, output, &params) == CSINN_TRUE) {
        csi_expand_dims(input, output, &params);
    }

    result_verify_f32(reference->data, output->data, input->data, difference, in_size, false);

    free(buffer);
    free(output->data);
    return done_testing();
}