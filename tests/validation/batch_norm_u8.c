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

int main(int argc, char **argv)
{
    init_testsuite("Testing function of batch normalization u8.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *mean = csinn_alloc_tensor(NULL);
    struct csinn_tensor *variance = csinn_alloc_tensor(NULL);
    struct csinn_tensor *beta = csinn_alloc_tensor(NULL);
    struct csinn_tensor *gamma = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_bn_params *params = csinn_alloc_params(sizeof(struct csinn_bn_params), NULL);
    int size = 1;
    int zp, quantized_multiplier, shift;
    float max_value, min_value, scale;
    float error[5] = {0};
    float max_error;

    int *buffer = read_input_data_f32(argv[1]);
    /* get the dim para */
    output->dim_count = input->dim_count = buffer[0];
    for (int i = 0; i < input->dim_count; ++i) {
        output->dim[i] = input->dim[i] = buffer[1 + i];
    }

    for (int i = 0; i < input->dim_count; ++i) {
        size *= input->dim[i];
    }

    mean->dim_count = 1;
    variance->dim_count = 1;
    gamma->dim_count = 1;
    beta->dim_count = 1;

    mean->dim[0] = input->dim[input->dim_count - 1];
    variance->dim[0] = input->dim[input->dim_count - 1];
    gamma->dim[0] = input->dim[input->dim_count - 1];
    beta->dim[0] = input->dim[input->dim_count - 1];

    input->dtype = CSINN_DTYPE_UINT8;
    input->layout = CSINN_LAYOUT_NHWC;
    input->is_const = 0;
    input->quant_channel = 1;

    output->dtype = CSINN_DTYPE_UINT8;
    output->layout = CSINN_LAYOUT_NHWC;
    output->is_const = 0;
    output->quant_channel = 1;

    mean->dtype = CSINN_DTYPE_UINT8;
    mean->layout = CSINN_LAYOUT_O;
    mean->is_const = 0;
    mean->quant_channel = 1;

    variance->dtype = CSINN_DTYPE_UINT8;
    variance->layout = CSINN_LAYOUT_O;
    variance->is_const = 0;
    variance->quant_channel = 1;

    gamma->dtype = CSINN_DTYPE_UINT8;
    gamma->layout = CSINN_LAYOUT_O;
    gamma->is_const = 0;
    gamma->quant_channel = 1;

    beta->dtype = CSINN_DTYPE_UINT8;
    beta->layout = CSINN_LAYOUT_O;
    beta->is_const = 0;
    beta->quant_channel = 1;

    params->base.layout = CSINN_LAYOUT_NHWC;
    params->epsilon = *(float *)&buffer[1 + input->dim_count];
    shl_quantize_multiplier(params->epsilon, &quantized_multiplier, &shift);
    params->epsilon_multiplier = quantized_multiplier;
    params->epsilon_shift = shift;

    params->base.api = CSINN_API;

    float *src_in = (float *)(buffer + 2 + input->dim_count);
    float *mean_in = (float *)(buffer + 2 + input->dim_count + size);
    float *var_in =
        (float *)(buffer + 2 + input->dim_count + size + input->dim[input->dim_count - 1]);
    float *gamma_in =
        (float *)(buffer + 2 + input->dim_count + size + 2 * input->dim[input->dim_count - 1]);
    float *beta_in =
        (float *)(buffer + 2 + input->dim_count + size + 3 * input->dim[input->dim_count - 1]);
    float *ref =
        (float *)(buffer + 2 + input->dim_count + size + 4 * input->dim[input->dim_count - 1]);
    uint8_t *input_tmp = malloc(size * sizeof(char));
    uint8_t *mean_tmp = malloc(input->dim[input->dim_count - 1] * sizeof(char));
    uint8_t *var_tmp = malloc(input->dim[input->dim_count - 1] * sizeof(char));
    uint8_t *gamma_tmp = malloc(input->dim[input->dim_count - 1] * sizeof(char));
    uint8_t *beta_tmp = malloc(input->dim[input->dim_count - 1] * sizeof(char));

    input->data = src_in;
    get_quant_info(input);

    for (int i = 0; i < size; i++) {
        input_tmp[i] = shl_ref_quantize_f32_to_u8(src_in[i], input->qinfo);
    }

    /* compute the max quantize error */
    for (int i = 0; i < size; i++) {
        float error1;
        float output_tmp = shl_ref_dequantize_u8_to_f32(src_in[i], input->qinfo);
        if (isinf(src_in[i]) || isnan(src_in[i])) {
            continue;
        } else {
            error1 = fabs(src_in[i] - output_tmp);
            if (error1 > 1e-6) {
                error1 = fabs(src_in[i] - output_tmp) / fabs(src_in[i] + 1e-9);
            }
        }
        if (error1 > error[0]) {
            error[0] = error1;
        }
    }

    mean->data = mean_in;
    get_quant_info(mean);
    for (int i = 0; i < input->dim[input->dim_count - 1]; i++) {
        mean_tmp[i] = shl_ref_quantize_f32_to_u8(mean_in[i], mean->qinfo);
    }

    /* compute the max quantize error */
    for (int i = 0; i < input->dim[input->dim_count - 1]; i++) {
        float error1;
        float output_tmp = shl_ref_dequantize_u8_to_f32(mean_in[i], mean->qinfo);
        if (isinf(mean_in[i]) || isnan(mean_in[i])) {
            continue;
        } else {
            error1 = fabs(mean_in[i] - output_tmp);
            if (error1 > 1e-6) {
                error1 = fabs(mean_in[i] - output_tmp) / fabs(mean_in[i] + 1e-9);
            }
        }
        if (error1 > error[1]) {
            error[1] = error1;
        }
    }

    variance->data = var_in;
    get_quant_info(variance);
    for (int i = 0; i < input->dim[input->dim_count - 1]; i++) {
        var_tmp[i] = shl_ref_quantize_f32_to_u8(var_in[i], variance->qinfo);
    }

    gamma->data = gamma_in;
    get_quant_info(gamma);

    for (int i = 0; i < input->dim[input->dim_count - 1]; i++) {
        gamma_tmp[i] = shl_ref_quantize_f32_to_u8(gamma_in[i], gamma->qinfo);
    }

    /* compute the max quantize error */
    for (int i = 0; i < input->dim[input->dim_count - 1]; i++) {
        float error1;
        float output_tmp = shl_ref_dequantize_u8_to_f32(mean_in[i], gamma->qinfo);
        if (isinf(mean_in[i]) || isnan(mean_in[i])) {
            continue;
        } else {
            error1 = fabs(mean_in[i] - output_tmp);
            if (error1 > 1e-6) {
                error1 = fabs(mean_in[i] - output_tmp) / fabs(mean_in[i] + 1e-9);
            }
        }
        if (error1 > error[2]) {
            error[2] = error1;
        }
    }

    max_error = (error[0] + error[1]) * fabs(max_value);

    beta->data = beta_in;
    get_quant_info(beta);
    for (int i = 0; i < input->dim[input->dim_count - 1]; i++) {
        beta_tmp[i] = shl_ref_quantize_f32_to_u8(beta_in[i], beta->qinfo);
    }

    /* compute the max quantize error */
    for (int i = 0; i < input->dim[input->dim_count - 1]; i++) {
        float error1;
        float output_tmp = shl_ref_dequantize_u8_to_f32(mean_in[i], beta->qinfo);
        if (isinf(mean_in[i]) || isnan(mean_in[i])) {
            continue;
        } else {
            error1 = fabs(mean_in[i] - output_tmp);
            if (error1 > 1e-6) {
                error1 = fabs(mean_in[i] - output_tmp) / fabs(mean_in[i] + 1e-9);
            }
        }
        if (error1 > error[3]) {
            error[3] = error1;
        }
    }
    max_error += error[3];

    output->data = ref;
    get_quant_info(output);

    input->data = input_tmp;
    mean->data = mean_tmp;
    variance->data = var_tmp;
    gamma->data = gamma_tmp;
    beta->data = beta_tmp;
    reference->data = ref;
    output->data = malloc(size * sizeof(char));
    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_batch_normalization_init(input, mean, variance, gamma, beta, output, params) ==
        CSINN_TRUE) {
        csinn_batch_normalization(input, mean, variance, gamma, beta, output, params);
    }

    result_verify_8(reference->data, output, input->data, difference, size, false);

    free(buffer);
    free(input_tmp);
    free(mean_tmp);
    free(var_tmp);
    free(gamma_tmp);
    free(beta_tmp);
    free(output->data);
    return done_testing();
}
