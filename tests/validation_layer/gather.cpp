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
    init_testsuite("Testing function of gather(layer).\n");
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_tensor *indices = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_gather_params *params =
        (csinn_gather_params *)csinn_alloc_params(sizeof(struct csinn_gather_params), sess);
    int in_size = 1, indices_size = 1, out_size = 1;

    int *buffer = read_input_data_f32(argv[1]);
    int axis = buffer[0];
    input->dim_count = buffer[1];
    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i + 2];
        in_size *= input->dim[i];
    }

    indices->dim_count = buffer[2 + input->dim_count];
    for (int i = 0; i < indices->dim_count; i++) {
        indices->dim[i] = buffer[3 + input->dim_count + i];
        indices_size *= indices->dim[i];
    }

    output->dim_count = input->dim_count + indices->dim_count - 1;
    int j = 0;
    for (int i = 0; i < axis; i++) {
        output->dim[j] = input->dim[i];
        out_size *= output->dim[j];
        j++;
    }
    for (int i = 0; i < indices->dim_count; i++) {
        output->dim[j] = indices->dim[i];
        out_size *= output->dim[j];
        j++;
    }
    for (int i = axis + 1; i < input->dim_count; i++) {
        output->dim[j] = input->dim[i];
        out_size *= output->dim[j];
        j++;
    }

    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;
    indices->dtype = CSINN_DTYPE_INT64;
    indices->layout = CSINN_LAYOUT_NCHW;
    indices->is_const = 0;
    indices->quant_channel = 1;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;
    params->base.api = CSINN_API;
    params->axis = axis;

    input->data = (float *)(buffer + 3 + input->dim_count + indices->dim_count);
    indices->data = (int32_t *)(buffer + 3 + input->dim_count + indices->dim_count + in_size);
    reference->data =
        (float *)(buffer + 3 + input->dim_count + indices->dim_count + in_size + indices_size);
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

    int32_t *indices_data = (int32_t *)indices->data;
    int64_t *data_i64 = (int64_t *)malloc(indices_size * sizeof(int64_t));
    for (int i = 0; i < indices_size; i++) {
        data_i64[i] = (int64_t)indices_data[i];
    }
    indices->data = data_i64;

#if (DTYPE == 32)
    test_gather_op(input, indices, output, params, CSINN_QUANT_FLOAT32, csinn_gather_init,
                   csinn_gather, &difference);
#elif (DTYPE == 16)
    test_gather_op(input, indices, output, params, CSINN_QUANT_FLOAT16, csinn_gather_init,
                   csinn_gather, &difference);
#elif (DTYPE == 8)
    test_gather_op(input, indices, output, params, CSINN_QUANT_INT8_ASYM, csinn_gather_init,
                   csinn_gather, &difference);
#endif

    return done_testing();
}
