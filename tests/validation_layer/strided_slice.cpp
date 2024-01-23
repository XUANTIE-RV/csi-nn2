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

// list begin/end/stride has same length with input->dim_count
int main(int argc, char **argv)
{
    init_testsuite("Testing function of strided_slice(layer).\n");

    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_CPU_GRAPH;
    sess->model.save_mode = CSINN_RUN_ONLY;
    sess->dynamic_shape = CSINN_FALSE;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_strided_slice_params *params = (csinn_strided_slice_params *)csinn_alloc_params(
        sizeof(struct csinn_strided_slice_params), sess);
    int in_size = 1;
    int out_size = 1;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim_count = buffer[0];
    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i + 1];
        in_size *= input->dim[i];
    }
    params->slice_count = buffer[1 + input->dim_count];
    params->begin = (int *)malloc(input->dim_count * sizeof(int));
    params->end = (int *)malloc(input->dim_count * sizeof(int));
    params->stride = (int *)malloc(input->dim_count * sizeof(int));
    for (int i = 0; i < input->dim_count; i++) {
        params->begin[i] = buffer[2 + input->dim_count + 3 * i];
        params->end[i] = buffer[3 + input->dim_count + 3 * i];
        params->stride[i] = buffer[4 + input->dim_count + 3 * i];
    }
    output->dim_count = input->dim_count;
    params->slice_count = input->dim_count;  // slice_count constrain

    for (int i = 0; i < output->dim_count; i++) {
        if (i < params->slice_count) {
            output->dim[i] = ceil((float)(params->end[i] - params->begin[i]) / params->stride[i]);
        } else {
            output->dim[i] = input->dim[i];
        }
    }

    out_size = buffer[2 + input->dim_count + 3 * params->slice_count];
    params->base.api = CSINN_API;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;

    input->data = (float *)(buffer + 3 + input->dim_count + 3 * params->slice_count);
    reference->data = (float *)(buffer + 3 + input->dim_count + 3 * params->slice_count +
                                in_size);  // input->data + in_size
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

#if (DTYPE == 32)
    test_unary_op(input, output, params, CSINN_DTYPE_FLOAT32, CSINN_QUANT_FLOAT32, sess,
                  csinn_strided_slice_init, csinn_strided_slice, &difference);
#elif (DTYPE == 16)
    test_unary_op(input, output, params, CSINN_DTYPE_FLOAT16, CSINN_QUANT_FLOAT16, sess,
                  csinn_strided_slice_init, csinn_strided_slice, &difference);
#elif (DTYPE == 8)
    test_unary_op(input, output, params, CSINN_DTYPE_INT8, CSINN_QUANT_INT8_ASYM, sess,
                  csinn_strided_slice_init, csinn_strided_slice, &difference);
#endif

    return done_testing();
}
