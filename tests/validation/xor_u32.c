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
    init_testsuite("Testing function of xor u32.\n");

    struct csi_tensor *input_0 = csi_alloc_tensor(NULL);
    struct csi_tensor *input_1 = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct diso_params params;
    int in_size = 1, out_size = 1;

    int *buffer = read_input_data_f32(argv[1]);
    input_0->dim_count = buffer[0];
    input_1->dim_count = buffer[0];
    output->dim_count = input_0->dim_count;
    for(int i = 0; i < input_0->dim_count; i++) {
        input_0->dim[i] = buffer[i + 1];
        input_1->dim[i] = buffer[i + 1];
        output->dim[i] = input_0->dim[i];
        in_size *= input_0->dim[i];
    }

    out_size = in_size;
    input_0->dtype = CSINN_DTYPE_UINT32;
    input_1->dtype = CSINN_DTYPE_UINT32;
    output->dtype = CSINN_DTYPE_UINT32;
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    input_0->data    = (uint32_t *)(buffer + 1 + input_0->dim_count);
    input_1->data    = (uint32_t *)(buffer + 1 + input_0->dim_count + in_size);
    reference->data = (uint32_t *)(buffer + 1 + input_0->dim_count + 2 * in_size);
    output->data    = (uint32_t *)malloc(out_size * sizeof(uint32_t));
    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csi_xor_init(input_0, input_1, output, &params) == CSINN_TRUE) {
        csi_xor(input_0, input_1, output, &params);
    }

    result_verify_int32(reference->data, output->data, input_0->data, difference, out_size, false);

    free(buffer);
    free(output->data);
    return done_testing();
}
