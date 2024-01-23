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
    init_testsuite("Testing function of div(layer).\n");

    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_CPU_GRAPH;
    sess->model.save_mode = CSINN_RUN_ONLY;
    sess->dynamic_shape = CSINN_FALSE;
    struct csinn_tensor *input0 = csinn_alloc_tensor(sess);
    struct csinn_tensor *input1 = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_diso_params *params =
        (csinn_diso_params *)csinn_alloc_params(sizeof(struct csinn_diso_params), sess);
    int in_size0, in_size1, out_size;

    int *buffer = read_input_data_f32(argv[1]);

    input0->dim_count = buffer[0];
    input1->dim_count = buffer[1];
    output->dim_count = buffer[2];
    in_size0 = 1;
    in_size1 = 1;
    out_size = 1;
    for (int i = 0; i < input0->dim_count; i++) {
        input0->dim[i] = buffer[3 + i];
        in_size0 *= input0->dim[i];
    }
    for (int i = 0; i < input1->dim_count; i++) {
        input1->dim[i] = buffer[3 + input0->dim_count + i];
        in_size1 *= input1->dim[i];
    }
    for (int i = 0; i < output->dim_count; i++) {
        output->dim[i] = buffer[3 + input0->dim_count + input1->dim_count + i];
        out_size *= output->dim[i];
    }

    input0->dtype = CSINN_DTYPE_FLOAT32;
    input0->is_const = 0;
    input0->quant_channel = 1;
    set_layout(input0);

    input1->dtype = CSINN_DTYPE_FLOAT32;
    input1->is_const = 0;
    input1->quant_channel = 1;
    set_layout(input1);

    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;
    params->base.api = CSINN_API;

    int start = 3 + input0->dim_count + input1->dim_count + output->dim_count;
    input0->data = (float *)(buffer + start);
    input1->data = (float *)(buffer + start + in_size0);
    reference->data = (float *)(buffer + start + in_size0 + in_size1);
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.9;

#if (DTYPE == 32)
    test_binary_op(input0, input1, output, params, CSINN_DTYPE_FLOAT32, CSINN_QUANT_FLOAT32, sess,
                   csinn_div_init, csinn_div, &difference);
#elif (DTYPE == 16)
    test_binary_op(input0, input1, output, params, CSINN_DTYPE_FLOAT16, CSINN_QUANT_FLOAT16, sess,
                   csinn_div_init, csinn_div, &difference);
#elif (DTYPE == 8)
    test_binary_op(input0, input1, output, params, CSINN_DTYPE_INT8, CSINN_QUANT_INT8_ASYM, sess,
                   csinn_div_init, csinn_div, &difference);
#endif

    return done_testing();
}