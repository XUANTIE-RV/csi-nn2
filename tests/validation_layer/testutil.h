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

// #include "common.h"

#include <stddef.h>
#include <string.h>

#include "csi_nn.h"
#include "shl_ref.h"
#include "test_utils.h"

void set_layout(struct csinn_tensor *t)
{
    if (t->dim_count == 1) {
        t->layout = CSINN_LAYOUT_N;
    } else if (t->dim_count == 2) {
        t->layout = CSINN_LAYOUT_NC;
    } else if (t->dim_count == 3) {
        t->layout = CSINN_LAYOUT_NCW;
    } else if (t->dim_count == 4) {
        t->layout = CSINN_LAYOUT_NCHW;
    } else if (t->dim_count == 5) {
        t->layout = CSINN_LAYOUT_NCDHW;
    }
}

template <typename T>
void test_unary_op(struct csinn_tensor *input, struct csinn_tensor *output, T *params,
                   enum csinn_dtype_enum dtype, enum csinn_quant_enum quant_type,
                   struct csinn_session *sess,
                   int (*init_op)(struct csinn_tensor *, struct csinn_tensor *, T *),
                   int (*unary_op)(struct csinn_tensor *, struct csinn_tensor *, T *),
                   float *difference)
{
    sess->base_dtype = dtype;
    sess->base_quant_type = quant_type;
    int test_api = params->base.api;
    struct csinn_tensor *qinput =
        convert_f32_layer(input, quant_type, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput =
        convert_f32_layer(output, quant_type, (enum csinn_api_enum)test_api);

    struct csinn_tensor *real_input =
        convert_f32_layer(input, quant_type, (enum csinn_api_enum)test_api);
    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);

    if (init_op(qinput, qoutput, params) == CSINN_TRUE) {
        csinn_set_tensor_entry(qinput, sess);
        csinn_set_input(0, qinput, sess);
        unary_op(qinput, qoutput, params);
        csinn_set_output(0, qoutput, sess);
        csinn_session_setup(sess);
        csinn_update_input(0, real_input, sess);
        csinn_session_run(sess);

        csinn_get_output(0, qoutput, sess);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input->data,
                          *difference, csinn_tensor_size(output), false);

        free_input(real_input);
        shl_ref_tensor_transform_free_f32(foutput);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_maxpool_op(struct csinn_tensor *input, struct csinn_tensor *output, T *params,
                     enum csinn_dtype_enum dtype, enum csinn_quant_enum quant_type,
                     struct csinn_session *sess,
                     int (*init_op)(struct csinn_tensor *, struct csinn_tensor *, T *),
                     int (*unary_op)(struct csinn_tensor *, struct csinn_tensor *, T *),
                     float *difference)
{
    sess->base_dtype = dtype;
    sess->base_quant_type = quant_type;
    int test_api = params->base.api;
    struct csinn_tensor *qinput =
        convert_f32_layer(input, quant_type, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput = broadcast_quant_info(qinput, output, dtype);

    struct csinn_tensor *real_input =
        convert_f32_layer(input, quant_type, (enum csinn_api_enum)test_api);
    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);

    if (init_op(qinput, qoutput, params) == CSINN_TRUE) {
        csinn_set_tensor_entry(qinput, sess);
        csinn_set_input(0, qinput, sess);
        unary_op(qinput, qoutput, params);
        csinn_set_output(0, qoutput, sess);
        csinn_session_setup(sess);
        csinn_update_input(0, real_input, sess);
        csinn_session_run(sess);

        csinn_get_output(0, qoutput, sess);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input->data,
                          *difference, csinn_tensor_size(output), false);

        free_input(real_input);
        shl_ref_tensor_transform_free_f32(foutput);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_binary_op(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, T *params, enum csinn_dtype_enum dtype,
                    enum csinn_quant_enum quant_type, struct csinn_session *sess,
                    int (*init_op)(struct csinn_tensor *, struct csinn_tensor *,
                                   struct csinn_tensor *, T *),
                    int (*binary_op)(struct csinn_tensor *, struct csinn_tensor *,
                                     struct csinn_tensor *, T *),
                    float *difference)
{
    sess->base_dtype = dtype;
    sess->base_quant_type = quant_type;
    int test_api = params->base.api;

    struct csinn_tensor *qinput0;
    struct csinn_tensor *qinput1;

    struct csinn_tensor *qoutput;
    struct csinn_tensor *real_input0;
    struct csinn_tensor *real_input1;

    if (quant_type == CSINN_QUANT_FLOAT16_W_INT8) {
        qinput0 = convert_f32_layer(input0, CSINN_QUANT_FLOAT16, (enum csinn_api_enum)test_api);
        qinput1 = convert_f32_layer(input1, CSINN_QUANT_INT8_SYM, (enum csinn_api_enum)test_api);

        qoutput = convert_f32_layer(output, CSINN_QUANT_FLOAT16, (enum csinn_api_enum)test_api);
        real_input0 = convert_f32_layer(input0, CSINN_QUANT_FLOAT16, (enum csinn_api_enum)test_api);
        real_input1 =
            convert_f32_layer(input1, CSINN_QUANT_INT8_SYM, (enum csinn_api_enum)test_api);

    } else {
        qinput0 = convert_f32_layer(input0, quant_type, (enum csinn_api_enum)test_api);
        qinput1 = convert_f32_layer(input1, quant_type, (enum csinn_api_enum)test_api);
        qoutput = convert_f32_layer(output, quant_type, (enum csinn_api_enum)test_api);

        real_input0 = convert_f32_layer(input0, quant_type, (enum csinn_api_enum)test_api);
        real_input1 = convert_f32_layer(input1, quant_type, (enum csinn_api_enum)test_api);
    }

    csinn_session_init(sess);
    csinn_set_input_number(2, sess);
    csinn_set_output_number(1, sess);

    if (init_op(qinput0, qinput1, qoutput, params) == CSINN_TRUE) {
        csinn_set_tensor_entry(qinput0, sess);
        csinn_set_tensor_entry(qinput1, sess);
        csinn_set_input(0, qinput0, sess);
        csinn_set_input(1, qinput1, sess);
        binary_op(qinput0, qinput1, qoutput, params);
        csinn_set_output(0, qoutput, sess);
        csinn_session_setup(sess);
        csinn_update_input(0, real_input0, sess);
        csinn_update_input(1, real_input1, sess);
        csinn_session_run(sess);

        csinn_get_output(0, qoutput, sess);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input0->data,
                          *difference, csinn_tensor_size(output), false);

        free_input(real_input0);
        free_input(real_input1);
        shl_ref_tensor_transform_free_f32(foutput);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_matmul_op(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, T *params, enum csinn_dtype_enum dtype,
                    enum csinn_quant_enum quant_type, struct csinn_session *sess,
                    int (*init_op)(struct csinn_tensor *, struct csinn_tensor *,
                                   struct csinn_tensor *, T *),
                    int (*binary_op)(struct csinn_tensor *, struct csinn_tensor *,
                                     struct csinn_tensor *, T *),
                    float *difference)
{
    sess->base_dtype = dtype;
    sess->base_quant_type = quant_type;
    int test_api = params->base.api;

    struct csinn_tensor *qinput0;
    struct csinn_tensor *qinput1;

    struct csinn_tensor *qoutput;
    struct csinn_tensor *real_input0;
    struct csinn_tensor *real_input1;

    if (quant_type == CSINN_QUANT_FLOAT16_W_INT8) {
        qinput0 = convert_f32_layer(input0, CSINN_QUANT_FLOAT16, (enum csinn_api_enum)test_api);
        qinput1 = convert_f32_layer(input1, CSINN_QUANT_INT8_SYM, (enum csinn_api_enum)test_api);

        qoutput = convert_f32_layer(output, CSINN_QUANT_FLOAT16, (enum csinn_api_enum)test_api);
        real_input0 = convert_f32_layer(input0, CSINN_QUANT_FLOAT16, (enum csinn_api_enum)test_api);

    } else {
        qinput0 = convert_f32_layer(input0, quant_type, (enum csinn_api_enum)test_api);
        qinput1 = convert_f32_layer(input1, quant_type, (enum csinn_api_enum)test_api);

        qoutput = convert_f32_layer(output, quant_type, (enum csinn_api_enum)test_api);
        real_input0 = convert_f32_layer(input0, quant_type, (enum csinn_api_enum)test_api);
    }

    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);

    if (init_op(qinput0, qinput1, qoutput, params) == CSINN_TRUE) {
        csinn_set_tensor_entry(qinput0, sess);
        csinn_set_input(0, qinput0, sess);

        binary_op(qinput0, qinput1, qoutput, params);
        csinn_set_output(0, qoutput, sess);
        csinn_session_setup(sess);
        csinn_update_input(0, real_input0, sess);
        csinn_session_run(sess);

        csinn_get_output(0, qoutput, sess);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input0->data,
                          *difference, csinn_tensor_size(output), false);

        free_input(real_input0);
        shl_ref_tensor_transform_free_f32(foutput);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_binary2_op(
    struct csinn_tensor *input, struct csinn_tensor *output, struct csinn_tensor *weight, T *params,
    enum csinn_dtype_enum dtype, enum csinn_quant_enum quant_type, struct csinn_session *sess,
    int (*init_op)(struct csinn_tensor *, struct csinn_tensor *, struct csinn_tensor *, T *),
    int (*binary2_op)(struct csinn_tensor *, struct csinn_tensor *, struct csinn_tensor *, T *),
    float *difference)
{
    sess->base_dtype = dtype;
    sess->base_quant_type = quant_type;
    int test_api = params->base.api;
    struct csinn_tensor *qinput =
        convert_f32_layer(input, quant_type, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput =
        convert_f32_layer(output, quant_type, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qweight =
        convert_f32_layer(weight, quant_type, (enum csinn_api_enum)test_api);

    struct csinn_tensor *real_input =
        convert_f32_layer(input, quant_type, (enum csinn_api_enum)test_api);
    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);

    if (init_op(qinput, qweight, qoutput, params) == CSINN_TRUE) {
        csinn_set_tensor_entry(qinput, sess);
        csinn_set_input(0, qinput, sess);

        binary2_op(qinput, qweight, qoutput, params);
        csinn_set_output(0, qoutput, sess);
        csinn_session_setup(sess);
        csinn_update_input(0, real_input, sess);
        csinn_session_run(sess);

        csinn_get_output(0, qoutput, sess);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input->data,
                          *difference, csinn_tensor_size(output), false);

        free_input(real_input);
        shl_ref_tensor_transform_free_f32(foutput);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_ternary_op(struct csinn_tensor *input0, struct csinn_tensor *output,
                     struct csinn_tensor *input1, struct csinn_tensor *input2, T *params,
                     enum csinn_dtype_enum dtype, enum csinn_quant_enum quant_type,
                     struct csinn_session *sess,
                     int (*init_op)(struct csinn_tensor *, struct csinn_tensor *,
                                    struct csinn_tensor *, struct csinn_tensor *, T *),
                     int (*ternary_op)(struct csinn_tensor *, struct csinn_tensor *,
                                       struct csinn_tensor *, struct csinn_tensor *, T *),
                     float *difference)
{
    sess->base_dtype = dtype;
    sess->base_quant_type = quant_type;
    int test_api = params->base.api;
    struct csinn_tensor *qinput0 =
        convert_f32_layer(input0, quant_type, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qinput1 =
        convert_f32_layer(input1, quant_type, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qinput2 =
        convert_f32_layer(input2, quant_type, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput =
        convert_f32_layer(output, quant_type, (enum csinn_api_enum)test_api);

    struct csinn_tensor *real_input =
        convert_f32_layer(input0, quant_type, (enum csinn_api_enum)test_api);

    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);
    if (init_op(qinput0, qoutput, qinput1, qinput2, params) == CSINN_TRUE) {
        csinn_set_tensor_entry(qinput0, sess);
        csinn_set_input(0, qinput0, sess);
        ternary_op(qinput0, qoutput, qinput1, qinput2, params);
        csinn_set_output(0, qoutput, sess);
        csinn_session_setup(sess);
        csinn_update_input(0, real_input, sess);
        csinn_session_run(sess);

        csinn_get_output(0, qoutput, sess);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input0->data,
                          *difference, csinn_tensor_size(output), false);
        free_input(real_input);
        shl_ref_tensor_transform_free_f32(foutput);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_concat_op(struct csinn_tensor **input, struct csinn_tensor *output, T *params,
                    enum csinn_dtype_enum dtype, enum csinn_quant_enum quant_type,
                    struct csinn_session *sess,
                    int (*init_op)(struct csinn_tensor **, struct csinn_tensor *, T *),
                    int (*unary_op)(struct csinn_tensor **, struct csinn_tensor *, T *),
                    float *difference)
{
    sess->base_dtype = dtype;
    sess->base_quant_type = quant_type;
    int test_api = params->base.api;
    struct csinn_tensor *qinput[params->inputs_count];
    struct csinn_tensor *real_input[params->inputs_count];
    for (int i = 0; i < params->inputs_count; i++) {
        qinput[i] = convert_f32_layer(input[i], quant_type, (enum csinn_api_enum)test_api);
        real_input[i] = convert_f32_layer(input[i], quant_type, (enum csinn_api_enum)test_api);
    }
    struct csinn_tensor *qoutput =
        convert_f32_layer(output, quant_type, (enum csinn_api_enum)test_api);
    csinn_session_init(sess);
    csinn_set_input_number(params->inputs_count, sess);
    csinn_set_output_number(1, sess);

    if (init_op((struct csinn_tensor **)qinput, qoutput, params) == CSINN_TRUE) {
        for (int i = 0; i < params->inputs_count; i++) {
            csinn_set_tensor_entry(qinput[i], sess);
            csinn_set_input(i, qinput[i], sess);
        }
        unary_op((struct csinn_tensor **)qinput, qoutput, params);

        csinn_set_output(0, qoutput, sess);
        csinn_session_setup(sess);

        for (int i = 0; i < params->inputs_count; i++) {
            csinn_update_input(i, real_input[i], sess);
        }
        csinn_session_run(sess);
        csinn_get_output(0, qoutput, sess);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input[0]->data,
                          *difference, csinn_tensor_size(output), false);

        for (int i = 0; i < params->inputs_count; i++) {
            free_input(real_input[i]);
        }

        shl_ref_tensor_transform_free_f32(foutput);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_split_op(struct csinn_tensor *input, struct csinn_tensor **output, T *params,
                   enum csinn_dtype_enum dtype, enum csinn_quant_enum quant_type,
                   struct csinn_session *sess,
                   int (*init_op)(struct csinn_tensor *, struct csinn_tensor **, T *),
                   int (*unary_op)(struct csinn_tensor *, struct csinn_tensor **, T *),
                   float *difference)
{
    sess->base_dtype = dtype;
    sess->base_quant_type = quant_type;
    int test_api = params->base.api;
    struct csinn_tensor *qinput =
        convert_f32_layer(input, quant_type, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput[params->output_num];
    int output_size = 0;
    int o_size[params->output_num];
    for (int i = 0; i < params->output_num; i++) {
        qoutput[i] = broadcast_quant_info(qinput, output[i], dtype);
        o_size[i] = csinn_tensor_size(output[i]);
        output_size += o_size[i];
    }
    struct csinn_tensor *real_input =
        convert_f32_layer(input, quant_type, (enum csinn_api_enum)test_api);
    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(params->output_num, sess);

    if (init_op(qinput, (struct csinn_tensor **)qoutput, params) == CSINN_TRUE) {
        csinn_set_tensor_entry(qinput, sess);
        csinn_set_input(0, qinput, sess);

        unary_op(qinput, (struct csinn_tensor **)qoutput, params);

        for (int i = 0; i < params->output_num; i++) {
            csinn_set_output(i, qoutput[i], sess);
        }
        csinn_session_setup(sess);

        csinn_update_input(0, real_input, sess);
        csinn_session_run(sess);
        struct csinn_tensor *foutput[params->output_num];
        float *output_data = (float *)malloc(output_size * sizeof(float));
        float *foutput_data = (float *)malloc(output_size * sizeof(float));
        int acc_size = 0;
        for (int i = 0; i < params->output_num; i++) {
            csinn_get_output(i, qoutput[i], sess);
            foutput[i] = shl_ref_tensor_transform_f32(qoutput[i]);
            memcpy(output_data + acc_size, output[i]->data, o_size[i] * sizeof(float));
            memcpy(foutput_data + acc_size, foutput[i]->data, o_size[i] * sizeof(float));
            acc_size += o_size[i];
            shl_ref_tensor_transform_free_f32(foutput[i]);
        }
        result_verify_f32(output_data, foutput_data, (float *)input->data, *difference, output_size,
                          false);
        free(output_data);
        free(foutput_data);
        free_input(real_input);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_conv2d_op(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_tensor *kernel, struct csinn_tensor *bias, T *params,
                    enum csinn_dtype_enum dtype, enum csinn_quant_enum quant_type,
                    struct csinn_session *sess,
                    int (*init_op)(struct csinn_tensor *, struct csinn_tensor *,
                                   struct csinn_tensor *, struct csinn_tensor *, T *),
                    int (*conv2d_op)(struct csinn_tensor *, struct csinn_tensor *,
                                     struct csinn_tensor *, struct csinn_tensor *, T *),
                    float *difference)
{
    sess->base_dtype = dtype;
    sess->base_quant_type = quant_type;
    params->base.quant_type = quant_type;
    int test_api = params->base.api;
    struct csinn_tensor *qbias;
    struct csinn_tensor *qinput;
    struct csinn_tensor *qkernel;

    struct csinn_tensor *qoutput;
    struct csinn_tensor *real_input;

    if (quant_type == CSINN_QUANT_INT8_ASYM_W_SYM) {
        if (!params->conv_extra.fuse_zp2bias) {
            qkernel =
                convert_f32_layer(kernel, CSINN_QUANT_INT8_SYM, (enum csinn_api_enum)test_api);
            qinput = convert_f32_layer(input, CSINN_QUANT_INT8_ASYM, (enum csinn_api_enum)test_api);
            qbias = convert_f32_bias(input, kernel, bias, (enum csinn_api_enum)test_api);
        } else {
            qkernel =
                convert_f32_layer(kernel, CSINN_QUANT_INT8_SYM, (enum csinn_api_enum)test_api);
            qbias = fuse_zp_to_bias(input, kernel, bias, (enum csinn_api_enum)test_api);
            qinput = convert_f32_layer(input, CSINN_QUANT_INT8_ASYM, (enum csinn_api_enum)test_api);
            qinput->qinfo->zero_point = 0;
        }

        qoutput = convert_f32_layer(output, CSINN_QUANT_INT8_ASYM, (enum csinn_api_enum)test_api);
        real_input = convert_f32_layer(input, CSINN_QUANT_INT8_ASYM, (enum csinn_api_enum)test_api);

    } else if (quant_type == CSINN_QUANT_FLOAT16_W_INT8) {
        qkernel = convert_f32_layer(kernel, CSINN_QUANT_INT8_SYM, (enum csinn_api_enum)test_api);
        qinput = convert_f32_layer(input, CSINN_QUANT_FLOAT16, (enum csinn_api_enum)test_api);
        qbias = convert_f32_layer(bias, CSINN_QUANT_FLOAT16, (enum csinn_api_enum)test_api);

        qoutput = convert_f32_layer(output, CSINN_QUANT_FLOAT16, (enum csinn_api_enum)test_api);
        real_input = convert_f32_layer(input, CSINN_QUANT_FLOAT16, (enum csinn_api_enum)test_api);

    } else {
        qkernel = convert_f32_layer(kernel, quant_type, (enum csinn_api_enum)test_api);
        qbias = convert_f32_layer(bias, quant_type, (enum csinn_api_enum)test_api);
        qinput = convert_f32_layer(input, quant_type, (enum csinn_api_enum)test_api);
        qoutput = convert_f32_layer(output, quant_type, (enum csinn_api_enum)test_api);
        real_input = convert_f32_layer(input, quant_type, (enum csinn_api_enum)test_api);
    }

    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);

    if (init_op(qinput, qoutput, qkernel, qbias, params) == CSINN_TRUE) {
        csinn_set_tensor_entry(qinput, sess);
        csinn_set_input(0, qinput, sess);
        conv2d_op(qinput, qoutput, qkernel, qbias, params);
        csinn_set_output(0, qoutput, sess);
        csinn_session_setup(sess);
        csinn_update_input(0, real_input, sess);
        csinn_session_run(sess);

        csinn_get_output(0, qoutput, sess);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input->data,
                          *difference, csinn_tensor_size(output), false);
        free_input(real_input);
        shl_ref_tensor_transform_free_f32(foutput);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_fully_op(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_tensor *kernel, struct csinn_tensor *bias, T *params,
                   enum csinn_dtype_enum dtype, enum csinn_quant_enum quant_type,
                   struct csinn_session *sess,
                   int (*init_op)(struct csinn_tensor *, struct csinn_tensor *,
                                  struct csinn_tensor *, struct csinn_tensor *, T *),
                   int (*conv2d_op)(struct csinn_tensor *, struct csinn_tensor *,
                                    struct csinn_tensor *, struct csinn_tensor *, T *),
                   float *difference)
{
    sess->base_dtype = dtype;
    sess->base_quant_type = quant_type;
    params->base.quant_type = quant_type;
    int test_api = params->base.api;
    struct csinn_tensor *qbias;
    struct csinn_tensor *qinput;
    struct csinn_tensor *qkernel;

    struct csinn_tensor *qoutput;
    struct csinn_tensor *real_input;

    if (quant_type == CSINN_QUANT_INT8_ASYM_W_SYM) {
        qkernel = convert_f32_layer(kernel, CSINN_QUANT_INT8_SYM, (enum csinn_api_enum)test_api);
        qbias = fuse_zp_to_bias(input, kernel, bias, (enum csinn_api_enum)test_api);
        qinput = convert_f32_layer(input, CSINN_QUANT_INT8_ASYM, (enum csinn_api_enum)test_api);
        qinput->qinfo->zero_point = 0;

        qoutput = convert_f32_layer(output, CSINN_QUANT_INT8_ASYM, (enum csinn_api_enum)test_api);
        real_input = convert_f32_layer(input, CSINN_QUANT_INT8_ASYM, (enum csinn_api_enum)test_api);

    } else if (quant_type == CSINN_QUANT_FLOAT16_W_INT8) {
        qkernel = convert_f32_layer(kernel, CSINN_QUANT_INT8_SYM, (enum csinn_api_enum)test_api);
        qinput = convert_f32_layer(input, CSINN_QUANT_FLOAT16, (enum csinn_api_enum)test_api);
        qbias = convert_f32_layer(bias, CSINN_QUANT_FLOAT16, (enum csinn_api_enum)test_api);

        qoutput = convert_f32_layer(output, CSINN_QUANT_FLOAT16, (enum csinn_api_enum)test_api);
        real_input = convert_f32_layer(input, CSINN_QUANT_FLOAT16, (enum csinn_api_enum)test_api);

    } else {
        qbias = convert_f32_layer(bias, quant_type, (enum csinn_api_enum)test_api);
        qinput = convert_f32_layer(input, quant_type, (enum csinn_api_enum)test_api);
        qkernel = convert_f32_layer(kernel, quant_type, (enum csinn_api_enum)test_api);

        qoutput = convert_f32_layer(output, quant_type, (enum csinn_api_enum)test_api);
        real_input = convert_f32_layer(input, quant_type, (enum csinn_api_enum)test_api);
    }

    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);

    if (init_op(qinput, qoutput, qkernel, qbias, params) == CSINN_TRUE) {
        csinn_set_tensor_entry(qinput, sess);
        csinn_set_input(0, qinput, sess);
        conv2d_op(qinput, qoutput, qkernel, qbias, params);
        csinn_set_output(0, qoutput, sess);
        csinn_session_setup(sess);
        csinn_update_input(0, real_input, sess);
        csinn_session_run(sess);

        csinn_get_output(0, qoutput, sess);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input->data,
                          *difference, csinn_tensor_size(output), false);
        free_input(real_input);
        shl_ref_tensor_transform_free_f32(foutput);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_where_op(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *input2, struct csinn_tensor *output, T *params,
                   enum csinn_dtype_enum dtype, enum csinn_quant_enum quant_type,
                   struct csinn_session *sess,
                   int (*init_op)(struct csinn_tensor *, struct csinn_tensor *,
                                  struct csinn_tensor *, struct csinn_tensor *, T *),
                   int (*trinary_op)(struct csinn_tensor *, struct csinn_tensor *,
                                     struct csinn_tensor *, struct csinn_tensor *, T *),
                   float *difference)
{
    sess->base_dtype = dtype;
    sess->base_quant_type = quant_type;
    int test_api = params->base.api;
    struct csinn_tensor *qinput1 =
        convert_f32_layer(input1, quant_type, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qinput2 =
        convert_f32_layer(input2, quant_type, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput =
        convert_f32_layer(output, quant_type, (enum csinn_api_enum)test_api);

    struct csinn_tensor *real_input = input0;
    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);

    if (init_op(input0, qinput1, qinput2, qoutput, params) == CSINN_TRUE) {
        csinn_set_tensor_entry(input0, sess);
        csinn_set_input(0, input0, sess);
        trinary_op(input0, qinput1, qinput2, qoutput, params);

        csinn_set_output(0, qoutput, sess);
        csinn_session_setup(sess);
        csinn_update_input(0, real_input, sess);
        csinn_session_run(sess);

        csinn_get_output(0, qoutput, sess);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input0->data,
                          *difference, csinn_tensor_size(output), false);
        free_input(real_input);
        shl_ref_tensor_transform_free_f32(foutput);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_where_softmax_op(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, T *params, enum csinn_dtype_enum dtype,
                           enum csinn_quant_enum quant_type, struct csinn_session *sess,
                           int (*init_op)(struct csinn_tensor *, struct csinn_tensor *,
                                          struct csinn_tensor *, T *),
                           int (*binary_op)(struct csinn_tensor *, struct csinn_tensor *,
                                            struct csinn_tensor *, T *),
                           float *difference)
{
    sess->base_dtype = dtype;
    sess->base_quant_type = quant_type;
    int test_api = params->base.api;
    struct csinn_tensor *qinput1 =
        convert_f32_layer(input1, quant_type, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput =
        convert_f32_layer(output, quant_type, (enum csinn_api_enum)test_api);

    struct csinn_tensor *real_input = input0;
    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);
    if (init_op(input0, qinput1, qoutput, params) == CSINN_TRUE) {
        csinn_set_tensor_entry(input0, sess);
        csinn_set_input(0, input0, sess);
        binary_op(input0, qinput1, qoutput, params);

        csinn_set_output(0, qoutput, sess);
        csinn_session_setup(sess);
        csinn_update_input(0, real_input, sess);
        csinn_session_run(sess);

        csinn_get_output(0, qoutput, sess);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input0->data,
                          *difference, csinn_tensor_size(output), false);
        free_input(real_input);
        shl_ref_tensor_transform_free_f32(foutput);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_gather_op(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, T *params, enum csinn_dtype_enum dtype,
                    enum csinn_quant_enum quant_type, struct csinn_session *sess,
                    int (*init_op)(struct csinn_tensor *, struct csinn_tensor *,
                                   struct csinn_tensor *, T *),
                    int (*binary_op)(struct csinn_tensor *, struct csinn_tensor *,
                                     struct csinn_tensor *, T *),
                    float *difference)
{
    sess->base_dtype = dtype;
    sess->base_quant_type = quant_type;
    int test_api = params->base.api;
    struct csinn_tensor *qinput0 =
        convert_f32_layer(input0, quant_type, (enum csinn_api_enum)test_api);

    struct csinn_tensor *qoutput = broadcast_quant_info(qinput0, output, dtype);

    struct csinn_tensor *real_input =
        convert_f32_layer(input0, quant_type, (enum csinn_api_enum)test_api);

    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);

    if (init_op(qinput0, input1, qoutput, params) == CSINN_TRUE) {
        csinn_set_tensor_entry(qinput0, sess);
        csinn_set_input(0, qinput0, sess);

        binary_op(qinput0, input1, qoutput, params);
        csinn_set_output(0, qoutput, sess);
        csinn_session_setup(sess);
        csinn_update_input(0, real_input, sess);
        csinn_session_run(sess);

        csinn_get_output(0, qoutput, sess);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input0->data,
                          *difference, csinn_tensor_size(output), false);
        free_input(real_input);
        shl_ref_tensor_transform_free_f32(foutput);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_matmul_op_hybrid_quant(
    struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
    T *params, enum csinn_dtype_enum dtype, enum csinn_quant_enum quant_type,
    enum csinn_quant_enum quant_type_w, struct csinn_session *sess,
    int (*init_op)(struct csinn_tensor *, struct csinn_tensor *, struct csinn_tensor *, T *),
    int (*binary_op)(struct csinn_tensor *, struct csinn_tensor *, struct csinn_tensor *, T *),
    float *difference)
{
    sess->base_dtype = dtype;
    sess->base_quant_type = quant_type;
    int test_api = params->base.api;

    struct csinn_tensor *qinput0 =
        convert_f32_layer(input0, quant_type, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qinput1 =
        convert_f32_layer(input1, quant_type_w, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput =
        convert_f32_layer(output, quant_type, (enum csinn_api_enum)test_api);
    struct csinn_tensor *real_input0 =
        convert_f32_layer(input0, quant_type, (enum csinn_api_enum)test_api);

    csinn_session_init(sess);
    csinn_set_input_number(1, sess);
    csinn_set_output_number(1, sess);

    if (init_op(qinput0, qinput1, qoutput, params) == CSINN_TRUE) {
        csinn_set_tensor_entry(qinput0, sess);
        csinn_set_input(0, qinput0, sess);

        binary_op(qinput0, qinput1, qoutput, params);
        csinn_set_output(0, qoutput, sess);
        csinn_session_setup(sess);
        csinn_update_input(0, real_input0, sess);
        csinn_session_run(sess);

        csinn_get_output(0, qoutput, sess);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input0->data,
                          *difference, csinn_tensor_size(output), false);

        free_input(real_input0);
        shl_ref_tensor_transform_free_f32(foutput);
        csinn_session_deinit(sess);
        csinn_free_session(sess);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_conv2d_layer(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias, T *params,
                       enum csinn_quant_enum quant_dtype, enum csinn_quant_enum quant_dtype_w,
                       int (*init_op)(struct csinn_tensor *, struct csinn_tensor *,
                                      struct csinn_tensor *, struct csinn_tensor *, T *),
                       int (*conv2d_op)(struct csinn_tensor *, struct csinn_tensor *,
                                        struct csinn_tensor *, struct csinn_tensor *, T *),
                       float *difference)
{
    int test_api = params->base.api;
    struct csinn_tensor *qbias;
    struct csinn_tensor *qinput;

    struct csinn_tensor *qkernel =
        convert_f32_layer(kernel, quant_dtype_w, (enum csinn_api_enum)test_api);

    if (quant_dtype == CSINN_QUANT_INT8_ASYM) {
        params->base.quant_type = CSINN_QUANT_INT8_ASYM_W_SYM;
        if (!params->conv_extra.fuse_zp2bias) {
            qinput = convert_f32_layer(input, quant_dtype, (enum csinn_api_enum)test_api);
            qbias = convert_f32_bias(input, kernel, bias, (enum csinn_api_enum)test_api);
        } else {
            qbias = fuse_zp_to_bias(input, kernel, bias, (enum csinn_api_enum)test_api);
            qinput = convert_f32_layer(input, quant_dtype, (enum csinn_api_enum)test_api);
            qinput->qinfo->zero_point = 0;
        }
    } else {
        qbias = convert_f32_layer(bias, quant_dtype, (enum csinn_api_enum)test_api);
        qinput = convert_f32_layer(input, quant_dtype, (enum csinn_api_enum)test_api);
    }

    struct csinn_tensor *qoutput =
        convert_f32_layer(output, quant_dtype, (enum csinn_api_enum)test_api);

    if (init_op(qinput, qoutput, qkernel, qbias, params) == CSINN_TRUE) {
        conv2d_op(qinput, qoutput, qkernel, qbias, params);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input->data,
                          *difference, csinn_tensor_size(output), false);
        shl_ref_tensor_transform_free_f32(foutput);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_unary_layer(struct csinn_tensor *input, struct csinn_tensor *output, T *params,
                      enum csinn_quant_enum quant_dtype,
                      int (*init_op)(struct csinn_tensor *, struct csinn_tensor *, T *),
                      int (*unary_op)(struct csinn_tensor *, struct csinn_tensor *, T *),
                      float *difference)
{
    int test_api = params->base.api;
    struct csinn_tensor *qinput =
        convert_f32_layer(input, quant_dtype, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput =
        convert_f32_layer(output, quant_dtype, (enum csinn_api_enum)test_api);
    if (init_op(qinput, qoutput, params) == CSINN_TRUE) {
        unary_op(qinput, qoutput, params);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input->data,
                          *difference, csinn_tensor_size(output), false);
        shl_ref_tensor_transform_free_f32(foutput);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_maxpool_layer(struct csinn_tensor *input, struct csinn_tensor *output, T *params,
                        enum csinn_quant_enum quant_dtype,
                        int (*init_op)(struct csinn_tensor *, struct csinn_tensor *, T *),
                        int (*unary_op)(struct csinn_tensor *, struct csinn_tensor *, T *),
                        float *difference)
{
    int test_api = params->base.api;
    struct csinn_tensor *qinput =
        convert_f32_layer(input, quant_dtype, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput = broadcast_quant_info(qinput, output, qinput->dtype);
    if (init_op(qinput, qoutput, params) == CSINN_TRUE) {
        unary_op(qinput, qoutput, params);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input->data,
                          *difference, csinn_tensor_size(output), false);
        shl_ref_tensor_transform_free_f32(foutput);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}
