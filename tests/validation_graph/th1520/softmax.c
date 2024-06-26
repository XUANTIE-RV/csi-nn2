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

#include "csi_nn.h"
#include "test_utils.h"

void op_test_run_th1520(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_softmax_params *params, struct csinn_session *sess,
                        struct csinn_tensor *real_input, float *output_data, float diff)
{
    csinn_session_init(sess);
    csinn_set_input_number(2, sess);
    csinn_set_output_number(1, sess);

    struct csinn_tensor *zero = csinn_alloc_tensor(sess);
    struct csinn_tensor *add_output = csinn_alloc_tensor(sess);
    struct csinn_diso_params *add_params =
        csinn_alloc_params(sizeof(struct csinn_diso_params), sess);
    add_params->base.name = "add_params";
    add_params->base.layout = CSINN_LAYOUT_NCHW;
    add_params->base.run_mode = CSINN_RM_NPU_GRAPH;
    add_params->base.api = CSINN_TH1520;

    csinn_tensor_copy(zero, input);
    zero->qinfo->scale = 1;
    zero->qinfo->zero_point = 0;

    csinn_tensor_copy(add_output, input);

    csinn_add_init(input, zero, add_output, add_params);
    csinn_softmax_init(add_output, output, params);

    csinn_set_tensor_entry(input, sess);
    csinn_set_input(0, input, sess);
    csinn_set_tensor_entry(zero, sess);
    csinn_set_input(1, zero, sess);

    csinn_add(input, zero, add_output, add_params);
    csinn_softmax(add_output, output, params);

    csinn_set_output(0, output, sess);
    csinn_session_setup(sess);

    csinn_update_input(0, real_input, sess);
    zero->data = calloc(1, csinn_tensor_byte_size(zero));
    csinn_update_input(1, zero, sess);
    csinn_session_run(sess);
    csinn_get_output(0, output, sess);

    struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
    result_verify_f32(output_data, foutput->data, input->data, diff, csinn_tensor_size(output),
                      false);

    free_input(real_input);
    shl_ref_tensor_transform_free_f32(foutput);
    csinn_session_deinit(sess);
    csinn_free_session(sess);
}

void test_i8_sym(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_softmax_params *params, float difference)
{
    printf("test softmax i8 sym\n");
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_api = CSINN_TH1520;
    sess->base_quant_type = CSINN_QUANT_INT8_SYM;
    // sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
    enum csinn_dtype_enum test_dtype = CSINN_DTYPE_FLOAT32;

    struct csinn_tensor *qinput = convert_f32_input(input, test_dtype, sess);
    struct csinn_tensor *qoutput = convert_f32_input(output, test_dtype, sess);
    struct csinn_tensor *real_input = convert_f32_input(input, CSINN_DTYPE_INT8, sess);

    op_test_run_th1520(qinput, qoutput, params, sess, real_input, output->data, difference);
}

void test_i16_sym(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_softmax_params *params, float difference)
{
    printf("test softmax i16 sym\n");
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_api = CSINN_TH1520;
    sess->base_quant_type = CSINN_QUANT_INT16_SYM;
    // sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
    enum csinn_dtype_enum test_dtype = CSINN_DTYPE_FLOAT32;

    struct csinn_tensor *qinput = convert_f32_input(input, test_dtype, sess);
    struct csinn_tensor *qoutput = convert_f32_input(output, test_dtype, sess);
    struct csinn_tensor *real_input = convert_f32_input(input, CSINN_DTYPE_INT16, sess);

    op_test_run_th1520(qinput, qoutput, params, sess, real_input, output->data, difference);
}

void test_i8_asym(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_softmax_params *params, float difference)
{
    printf("test softmax i8 asym\n");
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_api = CSINN_TH1520;
    sess->base_quant_type = CSINN_QUANT_INT8_ASYM;
    // sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
    enum csinn_dtype_enum test_dtype = CSINN_DTYPE_INT8;

    struct csinn_tensor *qinput = convert_f32_input(input, test_dtype, sess);
    struct csinn_tensor *qoutput = convert_f32_input(output, test_dtype, sess);
    struct csinn_tensor *real_input = convert_f32_input(input, test_dtype, sess);

    op_test_run_th1520(qinput, qoutput, params, sess, real_input, output->data, difference);
}

void test_u8_asym(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_softmax_params *params, float difference)
{
    printf("test softmax u8 asym\n");
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_api = CSINN_TH1520;
    sess->base_quant_type = CSINN_QUANT_UINT8_ASYM;
    // sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
    enum csinn_dtype_enum test_dtype = CSINN_DTYPE_UINT8;

    struct csinn_tensor *qinput = convert_f32_input(input, test_dtype, sess);
    struct csinn_tensor *qoutput = convert_f32_input(output, test_dtype, sess);
    struct csinn_tensor *real_input = convert_f32_input(input, test_dtype, sess);

    op_test_run_th1520(qinput, qoutput, params, sess, real_input, output->data, difference);
}

void test_softmax(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_softmax_params *params, float difference)
{
    params->base.api = CSINN_TH1520;

    test_i8_sym(input, output, params, difference);
    test_i16_sym(input, output, params, difference);
    test_i8_asym(input, output, params, difference);
    test_u8_asym(input, output, params, difference);
}
