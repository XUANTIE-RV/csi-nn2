/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

/* CSI-NN2 version 2.0.x */

#include "common.h"

#include <stddef.h>
#include <string.h>

#include "csi_nn.h"
#include "math_snr.h"
#include "test_utils.h"

#define LAYER_TEST_DISO(OP, STYPE, SPARAMS)                                             \
    void test_##OP##_##STYPE(struct csinn_tensor *input0, struct csinn_tensor *input1,  \
                             struct csinn_tensor *output, struct SPARAMS *params,       \
                             float *difference)                                         \
    {                                                                                   \
        enum csinn_dtype_enum test_dtype = STYPE;                                       \
        enum csinn_api_enum test_api = params->base.api;                                \
        struct csinn_tensor *qinput0 = convert_f32_layer(input0, test_dtype, test_api); \
        struct csinn_tensor *qinput1 = convert_f32_layer(input1, test_dtype, test_api); \
        struct csinn_tensor *qoutput = convert_f32_layer(output, test_dtype, test_api); \
        if (csinn_##OP##_init(qinput0, qinput1, qoutput, params) == CSINN_TRUE) {       \
            csinn_##OP(qinput0, qinput1, qoutput, params);                              \
        }                                                                               \
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);           \
        result_verify_f32(output->data, foutput->data, input0->data, *difference,       \
                          csinn_tensor_size(output), false);                            \
        shl_ref_tensor_transform_free_f32(foutput);                                     \
    }

#define LAYER_TEST_SEGMENT(OP, STYPE, SPARAMS)                                          \
    void test_##OP##_##STYPE(struct csinn_tensor *input0, struct csinn_tensor *segment, \
                             struct csinn_tensor *output, struct SPARAMS *params,       \
                             float *difference)                                         \
    {                                                                                   \
        enum csinn_dtype_enum test_dtype = STYPE;                                       \
        enum csinn_api_enum test_api = params->base.api;                                \
        struct csinn_tensor *qinput0 = convert_f32_layer(input0, test_dtype, test_api); \
        struct csinn_tensor *qoutput = convert_f32_layer(output, test_dtype, test_api); \
        if (csinn_##OP##_init(qinput0, segment, qoutput, params) == CSINN_TRUE) {       \
            csinn_##OP(qinput0, segment, qoutput, params);                              \
        }                                                                               \
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);           \
        result_verify_f32(output->data, foutput->data, input0->data, *difference,       \
                          csinn_tensor_size(output), false);                            \
        shl_ref_tensor_transform_free_f32(foutput);                                     \
    }

#define LAYER_TEST_SISO(OP, STYPE, SPARAMS)                                             \
    void test_##OP##_##STYPE(struct csinn_tensor *input, struct csinn_tensor *output,   \
                             struct SPARAMS *params, float *difference)                 \
    {                                                                                   \
        enum csinn_dtype_enum test_dtype = STYPE;                                       \
        enum csinn_api_enum test_api = params->base.api;                                \
        struct csinn_tensor *qinput = convert_f32_layer(input, test_dtype, test_api);   \
        struct csinn_tensor *qoutput = convert_f32_layer(output, test_dtype, test_api); \
        if (csinn_##OP##_init(qinput, qoutput, params) == CSINN_TRUE) {                 \
            csinn_##OP(qinput, qoutput, params);                                        \
        }                                                                               \
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);           \
        result_verify_f32(output->data, foutput->data, input->data, *difference,        \
                          csinn_tensor_size(output), false);                            \
        shl_ref_tensor_transform_free_f32(foutput);                                     \
    }

#define LAYER_TEST_CONCAT(OP, STYPE, SPARAMS)                                                   \
    void test_##OP##_##STYPE(struct csinn_tensor **input, struct csinn_tensor *output,          \
                             struct SPARAMS *params, float *difference)                         \
    {                                                                                           \
        enum csinn_dtype_enum test_dtype = STYPE;                                               \
        enum csinn_api_enum test_api = params->base.api;                                        \
        struct csinn_tensor *qinput[params->inputs_count];                                      \
        for (int i = 0; i < params->inputs_count; i++) {                                        \
            qinput[i] = convert_f32_layer(input[i], test_dtype, test_api);                      \
        }                                                                                       \
        struct csinn_tensor *qoutput = convert_f32_layer(output, test_dtype, test_api);         \
        if (csinn_##OP##_init((struct csinn_tensor **)qinput, qoutput, params) == CSINN_TRUE) { \
            csinn_##OP((struct csinn_tensor **)qinput, qoutput, params);                        \
        }                                                                                       \
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);                   \
        result_verify_f32(output->data, foutput->data, input[0]->data, *difference,             \
                          csinn_tensor_size(output), false);                                    \
        shl_ref_tensor_transform_free_f32(foutput);                                             \
    }

#define LAYER_TEST_SPLIT(OP, STYPE, SPARAMS)                                                    \
    void test_##OP##_##STYPE(struct csinn_tensor *input, struct csinn_tensor **output,          \
                             struct SPARAMS *params, float *difference)                         \
    {                                                                                           \
        enum csinn_dtype_enum test_dtype = STYPE;                                               \
        enum csinn_api_enum test_api = params->base.api;                                        \
        struct csinn_tensor *qoutput[params->output_num];                                       \
        int num = params->output_num;                                                           \
        struct csinn_tensor *qinput = convert_f32_layer(input, test_dtype, test_api);           \
        for (int i = 0; i < num; i++) {                                                         \
            qoutput[i] = convert_f32_layer(output[i], test_dtype, test_api);                    \
        }                                                                                       \
        if (csinn_##OP##_init(qinput, (struct csinn_tensor **)qoutput, params) == CSINN_TRUE) { \
            csinn_##OP(qinput, (struct csinn_tensor **)qoutput, params);                        \
        }                                                                                       \
        for (int i = 0; i < num; i++) {                                                         \
            struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput[i]);            \
            result_verify_f32(output[i]->data, foutput->data, input->data, *difference,         \
                              csinn_tensor_size(output[i]), false);                             \
            shl_ref_tensor_transform_free_f32(foutput);                                         \
        }                                                                                       \
    }

#define LAYER_TEST_UNSTACK(OP, STYPE, SPARAMS)                                                  \
    void test_##OP##_##STYPE(struct csinn_tensor *input, struct csinn_tensor **output,          \
                             struct SPARAMS *params, float *difference)                         \
    {                                                                                           \
        enum csinn_dtype_enum test_dtype = STYPE;                                               \
        enum csinn_api_enum test_api = params->base.api;                                        \
        struct csinn_tensor *qoutput[params->outputs_count];                                    \
        int num = params->outputs_count;                                                        \
        struct csinn_tensor *qinput = convert_f32_layer(input, test_dtype, test_api);           \
        for (int i = 0; i < num; i++) {                                                         \
            qoutput[i] = convert_f32_layer(output[i], test_dtype, test_api);                    \
        }                                                                                       \
        if (csinn_##OP##_init(qinput, (struct csinn_tensor **)qoutput, params) == CSINN_TRUE) { \
            csinn_##OP(qinput, (struct csinn_tensor **)qoutput, params);                        \
        }                                                                                       \
        for (int i = 0; i < num; i++) {                                                         \
            struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput[i]);            \
            result_verify_f32(output[i]->data, foutput->data, input->data, *difference,         \
                              csinn_tensor_size(output[i]), false);                             \
            shl_ref_tensor_transform_free_f32(foutput);                                         \
        }                                                                                       \
    }

#define LAYER_TEST_CONV2D(OP, STYPE, SPARAMS)                                           \
    void test_##OP##_##STYPE(struct csinn_tensor *input, struct csinn_tensor *output,   \
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,    \
                             struct SPARAMS *params, float *difference)                 \
    {                                                                                   \
        enum csinn_dtype_enum test_dtype = STYPE;                                       \
        enum csinn_api_enum test_api = params->base.api;                                \
        struct csinn_tensor *qinput = convert_f32_layer(input, test_dtype, test_api);   \
        struct csinn_tensor *qoutput = convert_f32_layer(output, test_dtype, test_api); \
        struct csinn_tensor *qkernel = convert_f32_layer(kernel, test_dtype, test_api); \
        struct csinn_tensor *qbias = convert_f32_layer(bias, test_dtype, test_api);     \
        if (csinn_##OP##_init(qinput, qoutput, qkernel, qbias, params) == CSINN_TRUE) { \
            csinn_##OP(qinput, qoutput, qkernel, qbias, params);                        \
        }                                                                               \
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);           \
        result_verify_f32(output->data, foutput->data, input->data, *difference,        \
                          csinn_tensor_size(output), false);                            \
        shl_ref_tensor_transform_free_f32(foutput);                                     \
    }

#define LAYER_TEST_BATCHNORM(OP, STYPE, SPARAMS)                                            \
    void test_##OP##_##STYPE(struct csinn_tensor *input, struct csinn_tensor *mean,         \
                             struct csinn_tensor *variance, struct csinn_tensor *gamma,     \
                             struct csinn_tensor *beta, struct csinn_tensor *output,        \
                             struct SPARAMS *params, float *difference)                     \
    {                                                                                       \
        enum csinn_dtype_enum test_dtype = STYPE;                                           \
        enum csinn_api_enum test_api = params->base.api;                                    \
        struct csinn_tensor *qinput = convert_f32_layer(input, test_dtype, test_api);       \
        struct csinn_tensor *qmean = convert_f32_layer(mean, test_dtype, test_api);         \
        struct csinn_tensor *qvariance = convert_f32_layer(variance, test_dtype, test_api); \
        struct csinn_tensor *qgamma = convert_f32_layer(gamma, test_dtype, test_api);       \
        struct csinn_tensor *qbeta = convert_f32_layer(beta, test_dtype, test_api);         \
        struct csinn_tensor *qoutput = convert_f32_layer(output, test_dtype, test_api);     \
        if (csinn_##OP##_init(qinput, qmean, qvariance, qgamma, qbeta, qoutput, params) ==  \
            CSINN_TRUE) {                                                                   \
            csinn_##OP(qinput, qmean, qvariance, qgamma, qbeta, qoutput, params);           \
        }                                                                                   \
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);               \
        result_verify_f32(output->data, foutput->data, input->data, *difference,            \
                          csinn_tensor_size(output), false);                                \
        shl_ref_tensor_transform_free_f32(foutput);                                         \
    }

#define LAYER_TEST_TISO(OP, STYPE, SPARAMS)                                                \
    void test_##OP##_##STYPE(struct csinn_tensor *input0, struct csinn_tensor *input1,     \
                             struct csinn_tensor *input2, struct csinn_tensor *output,     \
                             struct SPARAMS *params, float *difference)                    \
    {                                                                                      \
        enum csinn_dtype_enum test_dtype = STYPE;                                          \
        enum csinn_api_enum test_api = params->base.api;                                   \
        struct csinn_tensor *qinput0 = convert_f32_layer(input0, test_dtype, test_api);    \
        struct csinn_tensor *qinput1 = convert_f32_layer(input1, test_dtype, test_api);    \
        struct csinn_tensor *qinput2 = convert_f32_layer(input2, test_dtype, test_api);    \
        struct csinn_tensor *qoutput = convert_f32_layer(output, test_dtype, test_api);    \
        if (csinn_##OP##_init(qinput0, qinput1, qinput2, qoutput, params) == CSINN_TRUE) { \
            csinn_##OP(qinput0, qinput1, qinput2, qoutput, params);                        \
        }                                                                                  \
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);              \
        result_verify_f32(output->data, foutput->data, input1->data, *difference,          \
                          csinn_tensor_size(output), false);                               \
        shl_ref_tensor_transform_free_f32(foutput);                                        \
    }

#define LAYER_TEST_ARANGE(OP, STYPE, SPARAMS)                                           \
    void test_##OP##_##STYPE(struct csinn_tensor *output, struct SPARAMS *params,       \
                             float *difference)                                         \
    {                                                                                   \
        enum csinn_dtype_enum test_dtype = STYPE;                                       \
        enum csinn_api_enum test_api = params->base.api;                                \
        struct csinn_tensor *qoutput = convert_f32_layer(output, test_dtype, test_api); \
        if (csinn_##OP##_init(qoutput, params) == CSINN_TRUE) {                         \
            csinn_##OP(qoutput, params);                                                \
        }                                                                               \
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(qoutput);           \
        result_verify_f32(output->data, foutput->data, output->data, *difference,       \
                          csinn_tensor_size(output), false);                            \
        shl_ref_tensor_transform_free_f32(foutput);                                     \
    }

LAYER_QUANT_TEST_DISO(LAYER_TEST_DISO)
LAYER_QUANT_TEST_SISO(LAYER_TEST_SISO)
LAYER_QUANT_TEST_BATCHNORM(LAYER_TEST_BATCHNORM)
LAYER_QUANT_TEST_CONCAT(LAYER_TEST_CONCAT)
LAYER_QUANT_TEST_CONV2D(LAYER_TEST_CONV2D)
LAYER_QUANT_TEST_TISO(LAYER_TEST_TISO)
LAYER_QUANT_TEST_SEGMENT(LAYER_TEST_SEGMENT)
LAYER_QUANT_TEST_SPLIT(LAYER_TEST_SPLIT)
LAYER_QUANT_TEST_UNSTACK(LAYER_TEST_UNSTACK)
LAYER_QUANT_TEST_ARANGE(LAYER_TEST_ARANGE)
