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
    init_testsuite("Testing function of l2 normalization f32.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct l2n_params params;
    int size = 1;

    int *buffer = read_input_data_f32(argv[1]);
    /* get the dim para */
    output->dim_count = input->dim_count = buffer[0];
    params.epsilon = *(float *)&buffer[1];
    int32_t axis[] = {1};
    params.axis = axis;
    params.n = 1;
    
    for (int i = 0; i < input->dim_count; ++i) {
        output->dim[i] = input->dim[i] = buffer[2 + i];
    }

    for (int i = 0; i < input->dim_count; ++i) {
        size *= input->dim[i];
    }

    input->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;
    //params.epsilon = *(float *)&buffer[1 + input->dim_count];
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    input->data     = (float *)(buffer + 2 + input->dim_count);
    reference->data = (float *)(buffer + 2 + input->dim_count + size);
    output->data    = malloc(size * sizeof(float));
    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csi_l2_normalization_init(input, output, &params) == CSINN_TRUE) {
        csi_l2_normalization(input, output, &params);
    }

    result_verify_f32(reference->data, output->data, input->data, difference, size, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
