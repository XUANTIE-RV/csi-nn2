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

#include "testutil.h"

int main(int argc, char **argv)
{
    init_testsuite("Testing function of prelu(layer).\n");

    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_CPU_GRAPH;
    sess->model.save_mode = CSINN_RUN_ONLY;
    sess->dynamic_shape = CSINN_FALSE;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_tensor *alpha_data = csinn_alloc_tensor(sess);
    struct csinn_prelu_params *params =
        (csinn_prelu_params *)csinn_alloc_params(sizeof(struct csinn_prelu_params), sess);
    int in_size, out_size, alpha_size;

    if (argc == 1) {
        printf("please assign the input data.\n");
        return 0;
    }

    int *buffer = read_input_data_f32(argv[1]);
    output->dim[0] = input->dim[0] = buffer[0];  // batch
    output->dim[1] = input->dim[1] = buffer[1];  // channel
    output->dim[2] = input->dim[2] = buffer[2];  // height
    output->dim[3] = input->dim[3] = buffer[3];  // width
    alpha_data->dim[0] = buffer[1];
    input->dim_count = 4;
    alpha_data->dim_count = 1;
    output->dim_count = 4;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    alpha_data->dtype = CSINN_DTYPE_FLOAT32;
    alpha_data->layout = CSINN_LAYOUT_O;
    alpha_data->is_const = 0;
    alpha_data->quant_channel = 1;

    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;
    params->base.layout = CSINN_LAYOUT_NCHW;

    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    out_size = in_size;
    alpha_size = input->dim[1];
    params->base.api = CSINN_API;
    params->axis = 1;  // NCHW

    input->data = (float *)(buffer + 4);
    alpha_data->data = (float *)(buffer + 4 + in_size);
    reference->data = (float *)(buffer + 4 + in_size + input->dim[1]);
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

#if (DTYPE == 32)
    test_binary_op(input, alpha_data, output, params, CSINN_DTYPE_FLOAT32, CSINN_QUANT_FLOAT32,
                   sess, csinn_prelu_init, csinn_prelu, &difference);
#elif (DTYPE == 16)
    test_binary_op(input, alpha_data, output, params, CSINN_DTYPE_FLOAT16, CSINN_QUANT_FLOAT16,
                   sess, csinn_prelu_init, csinn_prelu, &difference);
#elif (DTYPE == 8)
    test_binary_op(input, alpha_data, output, params, CSINN_DTYPE_INT8, CSINN_QUANT_INT8_ASYM, sess,
                   csinn_prelu_init, csinn_prelu, &difference);
#endif

    return done_testing();
}
