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
    init_testsuite("Testing function of rms_norm(layer)\n");

    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_CPU_GRAPH;
    sess->model.save_mode = CSINN_RUN_ONLY;
    sess->dynamic_shape = CSINN_FALSE;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_tensor *weight = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_rms_norm_params *params =
        (csinn_rms_norm_params *)csinn_alloc_params(sizeof(struct csinn_rms_norm_params), sess);

    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0] = buffer[0];
    input->dim[1] = buffer[1];
    input->dim[2] = buffer[2];
    input->dim[3] = buffer[3];
    input->dim_count = 4;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    params->axis = buffer[4];
    int eps_i = buffer[5];
    float *eps_f = (float *)(&eps_i);
    params->epsilon = *eps_f;

    int norm_size = 1;
    int axis = params->axis >= 0 ? params->axis : (params->axis + input->dim_count);
    weight->dim_count = input->dim_count - axis;
    for (int i = 0; i < input->dim_count - axis; i++) {
        weight->dim[i] = input->dim[axis + i];
        norm_size *= weight->dim[i];
    }
    weight->dtype = CSINN_DTYPE_FLOAT32;
    weight->layout = CSINN_LAYOUT_NCHW;
    weight->is_const = 1;
    weight->quant_channel = 1;

    output->dim[0] = input->dim[0];
    output->dim[1] = input->dim[1];
    output->dim[2] = input->dim[2];
    output->dim[3] = input->dim[3];
    output->dim_count = 4;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;

    params->base.layout = CSINN_LAYOUT_NCHW;
    int in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    int out_size = in_size;
    params->base.api = CSINN_API;

    input->data = (float *)(buffer + 6);
    weight->data = (float *)(buffer + 6 + in_size);
    reference->data = (float *)(buffer + 6 + in_size + norm_size);
    output->data = reference->data;

    float difference = argc > 2 ? atof(argv[2]) : 0.99;

#if (DTYPE == 32)
    test_binary2_op(input, output, weight, params, CSINN_DTYPE_FLOAT32, CSINN_QUANT_FLOAT32, sess,
                    csinn_rms_norm_init, csinn_rms_norm, &difference);
#elif (DTYPE == 16)
    test_binary2_op(input, output, weight, params, CSINN_DTYPE_FLOAT16, CSINN_QUANT_FLOAT16, sess,
                    csinn_rms_norm_init, csinn_rms_norm, &difference);
#elif (DTYPE == 8)
    test_binary2_op(input, output, weight, params, CSINN_DTYPE_INT8, CSINN_QUANT_INT8_ASYM, sess,
                    csinn_rms_norm_init, csinn_rms_norm, &difference);
#endif

    return done_testing();
}
