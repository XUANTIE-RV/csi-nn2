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
    init_testsuite("Testing function of batch normalization(layer).\n");
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_tensor *mean = csinn_alloc_tensor(sess);
    struct csinn_tensor *variance = csinn_alloc_tensor(sess);
    struct csinn_tensor *beta = csinn_alloc_tensor(sess);
    struct csinn_tensor *gamma = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_bn_params *params = csinn_alloc_params(sizeof(struct csinn_bn_params), sess);
    int size = 1;

    int *buffer = read_input_data_f32(argv[1]);
    /* get the dim para */
    output->dim_count = input->dim_count = buffer[0];
    for (int i = 0; i < input->dim_count; ++i) {
        output->dim[i] = input->dim[i] = buffer[1 + i];
    }

    for (int i = 0; i < input->dim_count; ++i) {
        size *= input->dim[i];
    }

    mean->dim_count = 1;
    variance->dim_count = 1;
    gamma->dim_count = 1;
    beta->dim_count = 1;

    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NHWC;
    input->is_const = 0;
    input->quant_channel = 1;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NHWC;
    output->is_const = 0;
    output->quant_channel = 1;
    mean->dtype = CSINN_DTYPE_FLOAT32;
    mean->layout = CSINN_LAYOUT_O;
    mean->is_const = 0;
    mean->quant_channel = 1;
    variance->dtype = CSINN_DTYPE_INT8;
    variance->layout = CSINN_LAYOUT_O;
    variance->is_const = 0;
    variance->quant_channel = 1;
    gamma->dtype = CSINN_DTYPE_FLOAT32;
    gamma->layout = CSINN_LAYOUT_O;
    gamma->is_const = 0;
    gamma->quant_channel = 1;
    beta->dtype = CSINN_DTYPE_FLOAT32;
    beta->layout = CSINN_LAYOUT_O;
    beta->is_const = 0;
    beta->quant_channel = 1;
    params->base.layout = CSINN_LAYOUT_NHWC;
    params->epsilon = *((float *)buffer + 1 + input->dim_count);
    params->base.api = CSINN_API;

    input->data = (float *)(buffer + 2 + input->dim_count);
    mean->data = (float *)(buffer + 2 + input->dim_count + size);
    variance->data =
        (float *)(buffer + 2 + input->dim_count + size + input->dim[input->dim_count - 1]);
    gamma->data =
        (float *)(buffer + 2 + input->dim_count + size + 2 * input->dim[input->dim_count - 1]);
    beta->data =
        (float *)(buffer + 2 + input->dim_count + size + 3 * input->dim[input->dim_count - 1]);
    reference->data =
        (float *)(buffer + 2 + input->dim_count + size + 4 * input->dim[input->dim_count - 1]);
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

    test_batch_normalization_CSINN_QUANT_FLOAT32(input, mean, variance, gamma, beta, output, params,
                                                 &difference);
    test_batch_normalization_CSINN_QUANT_UINT8_ASYM(input, mean, variance, gamma, beta, output,
                                                    params, &difference);
    test_batch_normalization_CSINN_QUANT_INT8_SYM(input, mean, variance, gamma, beta, output,
                                                  params, &difference);

    return done_testing();
}
