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

void op_test_run(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_flatten_params *params, struct csinn_session *sess,
                 struct csinn_tensor *real_input, float *output_data, float diff);

void test_i8_sym(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_flatten_params *params, float difference)
{
    printf("test flatten i8 sym\n");
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_api = CSINN_TH1520;
    sess->base_quant_type = CSINN_QUANT_INT8_SYM;
    // sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
    enum csinn_dtype_enum test_dtype = CSINN_DTYPE_FLOAT32;

    struct csinn_tensor *qinput = convert_f32_input(input, test_dtype, sess);
    struct csinn_tensor *qoutput = convert_f32_input(output, test_dtype, sess);
    struct csinn_tensor *real_input = convert_f32_input(input, CSINN_DTYPE_INT8, sess);

    op_test_run(qinput, qoutput, params, sess, real_input, output->data, difference);
}

void test_i16_sym(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_flatten_params *params, float difference)
{
    printf("test flatten i16 sym\n");
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_api = CSINN_TH1520;
    sess->base_quant_type = CSINN_QUANT_INT16_SYM;
    // sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
    enum csinn_dtype_enum test_dtype = CSINN_DTYPE_FLOAT32;

    struct csinn_tensor *qinput = convert_f32_input(input, test_dtype, sess);
    struct csinn_tensor *qoutput = convert_f32_input(output, test_dtype, sess);
    struct csinn_tensor *real_input = convert_f32_input(input, CSINN_DTYPE_INT16, sess);

    op_test_run(qinput, qoutput, params, sess, real_input, output->data, difference);
}

void test_i8_asym(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_flatten_params *params, float difference)
{
    printf("test flatten i8 sym\n");
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_api = CSINN_TH1520;
    sess->base_quant_type = CSINN_QUANT_INT8_ASYM;
    // sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
    enum csinn_dtype_enum test_dtype = CSINN_DTYPE_INT8;

    struct csinn_tensor *qinput = convert_f32_input(input, test_dtype, sess);
    struct csinn_tensor *qoutput = convert_f32_input(output, test_dtype, sess);
    struct csinn_tensor *real_input = convert_f32_input(input, test_dtype, sess);

    op_test_run(qinput, qoutput, params, sess, real_input, output->data, difference);
}

void test_u8_asym(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_flatten_params *params, float difference)
{
    printf("test flatten u8 asym\n");
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_api = CSINN_TH1520;
    sess->base_quant_type = CSINN_QUANT_UINT8_ASYM;
    // sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
    enum csinn_dtype_enum test_dtype = CSINN_DTYPE_UINT8;

    struct csinn_tensor *qinput = convert_f32_input(input, test_dtype, sess);
    struct csinn_tensor *qoutput = convert_f32_input(output, test_dtype, sess);
    struct csinn_tensor *real_input = convert_f32_input(input, test_dtype, sess);

    op_test_run(qinput, qoutput, params, sess, real_input, output->data, difference);
}

void test_flatten(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_flatten_params *params, float difference)
{
    params->base.api = CSINN_TH1520;

    test_i8_sym(input, output, params, difference);
    test_i16_sym(input, output, params, difference);
    test_i8_asym(input, output, params, difference);
    test_u8_asym(input, output, params, difference);
}
