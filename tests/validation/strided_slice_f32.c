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
    init_testsuite("Testing function of strided_slice f32.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct strided_slice_params params;
    int in_size = 1;
    int out_size = 1;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim_count = buffer[0];
    for(int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i+1];
        in_size *= input->dim[i];
    }
    params.slice_count = buffer[1+input->dim_count];
    params.begin = (int *)malloc(params.slice_count * sizeof(int));
    params.end = (int *)malloc(params.slice_count * sizeof(int));
    params.stride = (int *)malloc(params.slice_count * sizeof(int));
    for(int i = 0; i < params.slice_count; i++) {
        params.begin[i] = buffer[2+input->dim_count+3*i];
        params.end[i] = buffer[3+input->dim_count+3*i];
        params.stride[i] = buffer[4+input->dim_count+3*i];
    }
    output->dim_count = input->dim_count;
    for(int i = 0; i < output->dim_count; i++) {
        if(i < params.slice_count) {
            output->dim[i] = (params.end[i] - params.begin[i]) / params.stride[i];
        } else {
            output->dim[i] = input->dim[i];
        }
    }
    out_size = buffer[2+input->dim_count+3*params.slice_count];
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    input->data = (float *)(buffer + 3 + input->dim_count + 3*params.slice_count);
    reference->data = (float *)(buffer + 3 + input->dim_count + 3*params.slice_count + in_size); //input->data + in_size
    input->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->data  = (float *)malloc(out_size * sizeof(float));
    float difference = argc > 2 ? atof(argv[2]) : 1e-6;

    if (csi_strided_slice_init(input, output, &params) == CSINN_TRUE) {
        csi_strided_slice(input, output, &params);
    }

    result_verify_f32(reference->data, output->data, input->data, difference, out_size, false);

    free(buffer);
    free(output->data);
    free(params.begin);
    free(params.end);
    free(params.stride);

    return done_testing();
}
