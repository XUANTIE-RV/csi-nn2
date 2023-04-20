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

/* SHL version 2.1.x */

// #include "common.h"

#include <stddef.h>
#include <string.h>

#include "csi_nn.h"
#include "test_utils.h"

template <typename T>
void test_unary_op(struct csinn_tensor *input, struct csinn_tensor *output, T *params,
                   enum csinn_quant_enum quant_dtype,
                   int (*init_op)(struct csinn_tensor *, struct csinn_tensor *, T *),
                   int (*unary_op)(struct csinn_tensor *, struct csinn_tensor *, T *),
                   float *difference)
{
    enum csinn_quant_enum test_dtype = quant_dtype;
    int test_api = params->base.api;
    struct csinn_tensor *qinput =
        convert_f32_layer(input, test_dtype, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput =
        convert_f32_layer(output, test_dtype, (enum csinn_api_enum)test_api);
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
void test_binary_op(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, T *params, enum csinn_quant_enum quant_dtype,
                    int (*init_op)(struct csinn_tensor *, struct csinn_tensor *,
                                   struct csinn_tensor *, T *),
                    int (*binary_op)(struct csinn_tensor *, struct csinn_tensor *,
                                     struct csinn_tensor *, T *),
                    float *difference)
{
    enum csinn_quant_enum test_dtype = quant_dtype;
    int test_api = params->base.api;
    struct csinn_tensor *qinput0 =
        convert_f32_layer(input0, test_dtype, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qinput1 =
        convert_f32_layer(input1, test_dtype, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput =
        convert_f32_layer(output, test_dtype, (enum csinn_api_enum)test_api);
    if (init_op(qinput0, qinput1, qoutput, params) == CSINN_TRUE) {
        binary_op(qinput0, qinput1, qoutput, params);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input0->data,
                          *difference, csinn_tensor_size(output), false);
        shl_ref_tensor_transform_free_f32(foutput);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_concat_op(struct csinn_tensor **input, struct csinn_tensor *output, T *params,
                    enum csinn_quant_enum quant_dtype,
                    int (*init_op)(struct csinn_tensor **, struct csinn_tensor *, T *),
                    int (*unary_op)(struct csinn_tensor **, struct csinn_tensor *, T *),
                    float *difference)
{
    enum csinn_quant_enum test_dtype = quant_dtype;
    int test_api = params->base.api;
    struct csinn_tensor *qinput[params->inputs_count];
    for (int i = 0; i < params->inputs_count; i++) {
        qinput[i] = convert_f32_layer(input[i], test_dtype, (enum csinn_api_enum)test_api);
    }
    struct csinn_tensor *qoutput =
        convert_f32_layer(output, test_dtype, (enum csinn_api_enum)test_api);
    if (init_op((struct csinn_tensor **)qinput, qoutput, params) == CSINN_TRUE) {
        unary_op((struct csinn_tensor **)qinput, qoutput, params);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input[0]->data,
                          *difference, csinn_tensor_size(output), false);
        shl_ref_tensor_transform_free_f32(foutput);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_split_op(struct csinn_tensor *input, struct csinn_tensor **output, T *params,
                   enum csinn_quant_enum quant_dtype,
                   int (*init_op)(struct csinn_tensor *, struct csinn_tensor **, T *),
                   int (*unary_op)(struct csinn_tensor *, struct csinn_tensor **, T *),
                   float *difference)
{
    enum csinn_quant_enum test_dtype = quant_dtype;
    int test_api = params->base.api;
    struct csinn_tensor *qinput =
        convert_f32_layer(input, test_dtype, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput[params->output_num];
    int output_size = 0;
    int o_size[params->output_num];
    for (int i = 0; i < params->output_num; i++) {
        qoutput[i] = convert_f32_layer(output[i], test_dtype, (enum csinn_api_enum)test_api);
        o_size[i] = csinn_tensor_size(output[i]);
        output_size += o_size[i];
    }
    if (init_op(qinput, (struct csinn_tensor **)qoutput, params) == CSINN_TRUE) {
        unary_op(qinput, (struct csinn_tensor **)qoutput, params);
        struct csinn_tensor *foutput[params->output_num];
        float *output_data = (float *)malloc(output_size * sizeof(float));
        float *foutput_data = (float *)malloc(output_size * sizeof(float));
        int acc_size = 0;
        for (int i = 0; i < params->output_num; i++) {
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
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_conv2d_op(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_tensor *kernel, struct csinn_tensor *bias, T *params,
                    enum csinn_quant_enum quant_dtype,
                    int (*init_op)(struct csinn_tensor *, struct csinn_tensor *,
                                   struct csinn_tensor *, struct csinn_tensor *, T *),
                    int (*conv2d_op)(struct csinn_tensor *, struct csinn_tensor *,
                                     struct csinn_tensor *, struct csinn_tensor *, T *),
                    float *difference)
{
    enum csinn_quant_enum test_dtype = quant_dtype;
    int test_api = params->base.api;
    struct csinn_tensor *qbias;
    struct csinn_tensor *qinput;

    struct csinn_tensor *qkernel =
        convert_f32_layer(kernel, test_dtype, (enum csinn_api_enum)test_api);

    if (test_dtype == CSINN_QUANT_INT8_SYM) {
        if (!params->conv_extra.fuse_zp2bias) {
            qinput = convert_f32_layer(input, CSINN_QUANT_INT8_ASYM, (enum csinn_api_enum)test_api);
            qbias = convert_f32_bias(input, kernel, bias, (enum csinn_api_enum)test_api);
        } else {
            qbias = fuse_zp_to_bias(input, kernel, bias, (enum csinn_api_enum)test_api);
            qinput = convert_f32_layer(input, CSINN_QUANT_INT8_ASYM, (enum csinn_api_enum)test_api);
            qinput->qinfo->zero_point = 0;
        }

    } else {
        qbias = convert_f32_layer(bias, test_dtype, (enum csinn_api_enum)test_api);
        qinput = convert_f32_layer(input, test_dtype, (enum csinn_api_enum)test_api);
    }

    struct csinn_tensor *qoutput =
        convert_f32_layer(output, test_dtype, (enum csinn_api_enum)test_api);

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
void test_fully_op(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_tensor *kernel, struct csinn_tensor *bias, T *params,
                   enum csinn_quant_enum quant_dtype,
                   int (*init_op)(struct csinn_tensor *, struct csinn_tensor *,
                                  struct csinn_tensor *, struct csinn_tensor *, T *),
                   int (*conv2d_op)(struct csinn_tensor *, struct csinn_tensor *,
                                    struct csinn_tensor *, struct csinn_tensor *, T *),
                   float *difference)
{
    enum csinn_quant_enum test_dtype = quant_dtype;
    int test_api = params->base.api;
    struct csinn_tensor *qbias;
    struct csinn_tensor *qinput;

    struct csinn_tensor *qkernel =
        convert_f32_layer(kernel, test_dtype, (enum csinn_api_enum)test_api);

    if (test_dtype == CSINN_QUANT_INT8_SYM) {
        qbias = fuse_zp_to_bias(input, kernel, bias, (enum csinn_api_enum)test_api);
        qinput = convert_f32_layer(input, CSINN_QUANT_INT8_ASYM, (enum csinn_api_enum)test_api);
        qinput->qinfo->zero_point = 0;

    } else {
        qbias = convert_f32_layer(bias, test_dtype, (enum csinn_api_enum)test_api);
        qinput = convert_f32_layer(input, test_dtype, (enum csinn_api_enum)test_api);
    }

    struct csinn_tensor *qoutput =
        convert_f32_layer(output, test_dtype, (enum csinn_api_enum)test_api);

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
void test_where_op(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *input2, struct csinn_tensor *output, T *params,
                   enum csinn_quant_enum quant_dtype,
                   int (*init_op)(struct csinn_tensor *, struct csinn_tensor *,
                                  struct csinn_tensor *, struct csinn_tensor *, T *),
                   int (*trinary_op)(struct csinn_tensor *, struct csinn_tensor *,
                                     struct csinn_tensor *, struct csinn_tensor *, T *),
                   float *difference)
{
    enum csinn_quant_enum test_dtype = quant_dtype;
    int test_api = params->base.api;
    struct csinn_tensor *qinput1 =
        convert_f32_layer(input1, test_dtype, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qinput2 =
        convert_f32_layer(input2, test_dtype, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput =
        convert_f32_layer(output, test_dtype, (enum csinn_api_enum)test_api);
    if (init_op(input0, qinput1, qinput2, qoutput, params) == CSINN_TRUE) {
        trinary_op(input0, qinput1, qinput2, qoutput, params);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input0->data,
                          *difference, csinn_tensor_size(output), false);
        shl_ref_tensor_transform_free_f32(foutput);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_where_softmax_op(
    struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
    T *params, enum csinn_quant_enum quant_dtype,
    int (*init_op)(struct csinn_tensor *, struct csinn_tensor *, struct csinn_tensor *, T *),
    int (*binary_op)(struct csinn_tensor *, struct csinn_tensor *, struct csinn_tensor *, T *),
    float *difference)
{
    enum csinn_quant_enum test_dtype = quant_dtype;
    int test_api = params->base.api;
    struct csinn_tensor *qinput1 =
        convert_f32_layer(input1, test_dtype, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput =
        convert_f32_layer(output, test_dtype, (enum csinn_api_enum)test_api);
    if (init_op(input0, qinput1, qoutput, params) == CSINN_TRUE) {
        binary_op(input0, qinput1, qoutput, params);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input0->data,
                          *difference, csinn_tensor_size(output), false);
        shl_ref_tensor_transform_free_f32(foutput);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}

template <typename T>
void test_gather_op(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, T *params, enum csinn_quant_enum quant_dtype,
                    int (*init_op)(struct csinn_tensor *, struct csinn_tensor *,
                                   struct csinn_tensor *, T *),
                    int (*binary_op)(struct csinn_tensor *, struct csinn_tensor *,
                                     struct csinn_tensor *, T *),
                    float *difference)
{
    enum csinn_quant_enum test_dtype = quant_dtype;
    int test_api = params->base.api;
    struct csinn_tensor *qinput0 =
        convert_f32_layer(input0, test_dtype, (enum csinn_api_enum)test_api);
    struct csinn_tensor *qoutput =
        convert_f32_layer(output, test_dtype, (enum csinn_api_enum)test_api);
    if (init_op(qinput0, input1, qoutput, params) == CSINN_TRUE) {
        binary_op(qinput0, input1, qoutput, params);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);
        result_verify_f32((float *)output->data, (float *)foutput->data, (float *)input0->data,
                          *difference, csinn_tensor_size(output), false);
        shl_ref_tensor_transform_free_f32(foutput);
    } else {
        printf("Function init failed\n");
        exit(-1);
    }
}
