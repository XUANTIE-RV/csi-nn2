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
    init_testsuite("Testing function of unsorted segment mean f32.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct csi_tensor *segment = csi_alloc_tensor(NULL);
    struct segment_params params;
    int in_size, out_size;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0] = buffer[0];          // batch
    input->dim[1] = buffer[1];          // height
    input->dim[2] = buffer[2];          // width
    input->dim[3] = buffer[3];          // channel
    output->dim[0] = buffer[4];          // batch
    output->dim[1] = buffer[1];          // height
    output->dim[2] = buffer[2];          // width
    output->dim[3] = buffer[3];          // channel

    input->dim_count = 4;
    output->dim_count = 4;
    input->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;
    params.num_segments = buffer[4];
    params.unsorted = CSINN_TRUE;
    in_size  = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    input->data      = (float *)(buffer + 5);
    segment->data    = (int *)(buffer + 5 + in_size);
    reference->data  = (float *)(buffer + 5 + in_size + buffer[0]);
    output->data     = malloc(out_size * sizeof(float));
    float difference = argc > 2 ? *argv[2] : 1e-6;

    if (csi_segment_mean_init(input, segment, output, &params) == CSINN_TRUE) {
        csi_segment_mean(input, segment, output, &params);
    } 

    result_verify_f32(reference->data, output->data, input->data, difference, out_size, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
