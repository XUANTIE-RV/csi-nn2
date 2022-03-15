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
    init_testsuite("Testing function of minimum(layer).\n");

    struct csi_tensor *input0 = csi_alloc_tensor(NULL);
    struct csi_tensor *input1 = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct diso_params params;
    int in_size = 1, out_size = 1;

    int *buffer = read_input_data_f32(argv[1]);
    input0->dim_count = buffer[0];
    output->dim_count = input0->dim_count;
    for(int i = 0; i < input0->dim_count; i++) {
        input0->dim[i] = buffer[i + 1];
        output->dim[i] = input0->dim[i];
        in_size *= input0->dim[i];
    }

    out_size = in_size;

    input0->dtype = CSINN_DTYPE_FLOAT32;
    input0->layout = CSINN_LAYOUT_NCHW;
    input0->is_const = 0;
    input0->quant_channel = 1;
    input1->dtype = CSINN_DTYPE_FLOAT32;
    input1->layout = CSINN_LAYOUT_NCHW;
    input1->is_const = 0;
    input1->quant_channel = 1;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    input0->data    = (float *)(buffer + 1 + input0->dim_count);
    input1->data    = (float *)(buffer + 1 + input0->dim_count + in_size);
    reference->data = (float *)(buffer + 1 + input0->dim_count + 2*in_size);
    output->data    = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

    test_minimum_CSINN_QUANT_FLOAT32(input0, input1, output, &params, &difference);
    test_minimum_CSINN_QUANT_UINT8_ASYM(input0, input1, output, &params, &difference);
    test_minimum_CSINN_QUANT_INT8_SYM(input0, input1, output, &params, &difference);

    return done_testing();
}