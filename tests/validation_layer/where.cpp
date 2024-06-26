/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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
    init_testsuite("Testing function of where(layer).\n");

    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_CPU_GRAPH;
    sess->model.save_mode = CSINN_RUN_ONLY;
    sess->dynamic_shape = CSINN_FALSE;
    struct csinn_tensor *condition = csinn_alloc_tensor(sess);
    struct csinn_tensor *x = csinn_alloc_tensor(sess);
    struct csinn_tensor *y = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_where_params *params =
        (csinn_where_params *)csinn_alloc_params(sizeof(struct csinn_where_params), sess);

    if (argc == 1) {
        printf("please assign the input data.\n");
        return 0;
    }

    int *buffer = read_input_data_f32(argv[1]);
    int shape_rank = buffer[0];

    condition->dim_count = shape_rank;
    x->dim_count = shape_rank;
    y->dim_count = shape_rank;
    output->dim_count = shape_rank;

    // Only support same shape
    int in_size = 1;
    for (int i = 0; i < shape_rank; i++) {
        condition->dim[i] = buffer[i + 1];
        x->dim[i] = condition->dim[i];
        y->dim[i] = condition->dim[i];
        output->dim[i] = condition->dim[i];
        in_size *= condition->dim[i];
    }

    condition->dtype = CSINN_DTYPE_BOOL;
    condition->layout = CSINN_LAYOUT_N;
    x->dtype = CSINN_DTYPE_FLOAT32;
    x->layout = CSINN_LAYOUT_N;
    y->dtype = CSINN_DTYPE_FLOAT32;
    y->layout = CSINN_LAYOUT_N;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_N;

    params->base.api = CSINN_API;

    condition->data = (float *)(buffer + 1 + shape_rank);
    x->data = (float *)(buffer + 1 + shape_rank + in_size);
    y->data = (float *)(buffer + 1 + shape_rank + in_size * 2);
    reference->data = (float *)(buffer + 1 + shape_rank + in_size * 3);
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

    float *c_data = (float *)condition->data;
    uint8_t *data_u8 = (uint8_t *)malloc(in_size * sizeof(uint8_t));
    for (int i = 0; i < in_size; i++) {
        data_u8[i] = (uint8_t)c_data[i];
    }
    condition->data = data_u8;

#if (DTYPE == 32)
    test_where_op(condition, x, y, output, params, CSINN_DTYPE_FLOAT32, CSINN_QUANT_FLOAT32, sess,
                  csinn_where_init, csinn_where, &difference);
#elif (DTYPE == 16)
    test_where_op(condition, x, y, output, params, CSINN_DTYPE_FLOAT16, CSINN_QUANT_FLOAT16, sess,
                  csinn_where_init, csinn_where, &difference);
#elif (DTYPE == 8)
    test_where_op(condition, x, y, output, params, CSINN_DTYPE_INT8, CSINN_QUANT_INT8_ASYM, sess,
                  csinn_where_init, csinn_where, &difference);
#endif

    return done_testing();
}
