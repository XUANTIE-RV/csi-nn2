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

#include "csi_nn.h"
#include "test_utils.h"
#include "testutil.h"

int main(int argc, char **argv)
{
    init_testsuite("Testing function of clip(layer).\n");

    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *reference = csinn_alloc_tensor(sess);
    struct csinn_clip_params *params = (csinn_clip_params *)csinn_alloc_params(sizeof(struct csinn_clip_params), sess);
    int in_size = 0;
    int out_size = 0;

    int *buffer = read_input_data_f32(argv[1]);

    input->dim[0] = buffer[0];  // batch
    input->dim[1] = buffer[1];  // channel
    input->dim[2] = buffer[2];  // height
    input->dim[3] = buffer[3];  // width

    output->dim[0] = input->dim[0];
    output->dim[1] = input->dim[1];
    output->dim[2] = input->dim[2];
    output->dim[3] = input->dim[3];

    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    out_size = in_size;
    input->dim_count = 4;
    output->dim_count = 4;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;

    params->min_value = buffer[4];
    params->max_value = buffer[5];
    params->base.api = CSINN_API;

    input->data = (float *)(buffer + 6);
    reference->data = (float *)(buffer + 6 + in_size);
    output->data = reference->data;
    float difference = argc > 2 ? atof(argv[2]) : 0.99;

#if (DTYPE==32)
    test_unary_op(input, output, params, CSINN_QUANT_FLOAT32, csinn_clip_init, csinn_clip,
                   &difference);
#elif (DTYPE==16)
    test_unary_op(input, output, params, CSINN_QUANT_FLOAT16, csinn_clip_init, csinn_clip,
                   &difference);
#elif (DTYPE==8)
    test_unary_op(input, output, params, CSINN_QUANT_INT8_ASYM, csinn_clip_init, csinn_clip,
                   &difference);
#endif

    return done_testing();
}