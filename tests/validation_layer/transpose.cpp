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
    init_testsuite("Testing function of transpose(layer).\n");
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_transpose_params *params =
        (csinn_transpose_params *)csinn_alloc_params(sizeof(struct csinn_transpose_params), sess);
    int in_size = 1, out_size = 1;
    float max_error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim_count = buffer[0];  // input->dim_count == 4
    output->dim_count = input->dim_count;

    int32_t *perm = (int32_t *)malloc(input->dim_count * sizeof(int32_t));

    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i + 1];
        perm[i] = buffer[input->dim_count + i + 1];
        output->dim[i] = buffer[2 * input->dim_count + i + 1];
        in_size *= input->dim[i];
    }
    out_size = in_size;

    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;
    params->base.api = CSINN_API;
    params->permute = perm;
    params->permute_num = input->dim_count;
    params->base.layout = CSINN_LAYOUT_NCHW;

    input->data = (float *)(buffer + 1 + input->dim_count * 3);
    reference->data = (float *)(buffer + 1 + input->dim_count * 3 + in_size);
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

#if (DTYPE == 32)
    test_unary_op(input, output, params, CSINN_QUANT_FLOAT32, csinn_transpose_init, csinn_transpose,
                  &difference);
#elif (DTYPE == 16)
    test_unary_op(input, output, params, CSINN_QUANT_FLOAT16, csinn_transpose_init, csinn_transpose,
                  &difference);
#elif (DTYPE == 8)
    test_unary_op(input, output, params, CSINN_QUANT_INT8_SYM, csinn_transpose_init,
                  csinn_transpose, &difference);
#endif

    return done_testing();
}
