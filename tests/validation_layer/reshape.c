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
    init_testsuite("Testing function of reshape(layer).\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct reshape_params params;
    int in_size, out_size;

    int *buffer = read_input_data_f32(argv[1]);
    int reshape_count = buffer[4];
    int *reshape = (int *)malloc(reshape_count * sizeof(int));
    for(int i = 0; i < reshape_count; i++) {
        reshape[i] = buffer[5 + i];
    }

    input->dim[0] = buffer[0];          // batch
    input->dim[1] = buffer[1];          // channel
    input->dim[2] = buffer[2];          // height
    input->dim[3] = buffer[3];          // width   
    input->dim_count = 4;
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    input->name = "input";
    float *input_data = (float *)(buffer + 5 + reshape_count);
    input->data = input_data; 
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    output->dim_count = reshape_count;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;
    out_size = in_size;
    for(int i = 0; i < output->dim_count; i++) {
        output->dim[i] = reshape[i];
        // out_size *= output->dim[i];
    }

    reference->data = (float *)(buffer + 5 + reshape_count + in_size);
    output->data = reference->data;
    output->name = "output";
    output->dtype = CSINN_DTYPE_FLOAT32;

    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.shape = reshape;
    params.shape_num = output->dim_count;
    
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

    test_reshape_CSINN_QUANT_FLOAT32(input, output, &params, &difference);
    test_reshape_CSINN_QUANT_UINT8_ASYM(input, output, &params, &difference);
    test_reshape_CSINN_QUANT_INT8_SYM(input, output, &params, &difference);

    return done_testing();
}
