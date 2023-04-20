/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
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

/* SHL version 2.1.x */

#include "csi_nn.h"
#include "test_utils.h"

int main(int argc, char **argv)
{
    init_testsuite("Testing function of resize nearestneighbor nchw u8.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_resize_params *params =
        csinn_alloc_params(sizeof(struct csinn_resize_params), NULL);
    int in_size, out_size;
    int zp, quantized_multiplier, shift;
    float scale, min_value, max_value;
    float max_error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0] = buffer[0];  // batch
    input->dim[1] = buffer[1];  // channel
    input->dim[2] = buffer[2];  // height
    input->dim[3] = buffer[3];  // width

    output->dim[0] = buffer[0];  // batch
    output->dim[1] = buffer[1];  // channel
    output->dim[2] = buffer[4];  // height
    output->dim[3] = buffer[5];  // width
    input->dim_count = 4;
    output->dim_count = 4;
    params->resize_mode = CSINN_RESIZE_NEAREST_NEIGHBOR;
    params->align_corners = buffer[6];
    input->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;
    params->base.layout = CSINN_LAYOUT_NCHW;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    params->base.api = CSINN_API;

    input->data = (float *)(buffer + 7);
    reference->data = (float *)(buffer + 7 + in_size);
    output->data = malloc(out_size * sizeof(float));

    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_resize_init(input, output, params) == CSINN_TRUE) {
        csinn_resize(input, output, params);
    }

    result_verify_f32(reference->data, output->data, input->data, difference, out_size, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
