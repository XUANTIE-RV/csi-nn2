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

#include "csi_nn.h"
#include "math_snr.h"
#include "test_utils.h"

int main(int argc, char **argv)
{
    init_testsuite("Testing function of fullyconnected i8.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *weight = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
    struct csinn_fc_params *params = csinn_alloc_params(sizeof(struct csinn_fc_params), NULL);
    int in_size0, in_size1, out_size;
    int zp, quantized_multiplier, shift;
    float max_value, min_value, scale, scale1, scale2, scale3;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0] = buffer[0];   // batch
    input->dim[1] = buffer[1];   // in_size
    weight->dim[0] = buffer[2];  // out_size
    weight->dim[1] = buffer[1];  // in_size
    bias->dim[0] = buffer[2];
    output->dim[0] = buffer[0];
    output->dim[1] = buffer[2];
    input->dim_count = 2;
    weight->dim_count = 2;
    bias->dim_count = 1;
    output->dim_count = 2;
    in_size0 = input->dim[0] * input->dim[1];
    in_size1 = weight->dim[0] * weight->dim[1];
    out_size = output->dim[0] * output->dim[1];
    input->dtype = CSINN_DTYPE_INT8;
    input->layout = CSINN_LAYOUT_NC;
    input->is_const = 0;
    input->quant_channel = 1;

    weight->dtype = CSINN_DTYPE_INT8;
    weight->layout = CSINN_LAYOUT_OI;
    weight->is_const = 1;
    weight->quant_channel = 1;

    bias->dtype = CSINN_DTYPE_INT8;
    bias->layout = CSINN_LAYOUT_O;
    bias->is_const = 1;
    bias->quant_channel = 1;

    output->dtype = CSINN_DTYPE_INT8;
    output->layout = CSINN_LAYOUT_NC;
    output->is_const = 0;
    output->quant_channel = 1;
    params->base.api = CSINN_API;

    float *src_in = (float *)(buffer + 3);
    float *weight_in = (float *)(buffer + 3 + in_size0);
    float *bias_in = (float *)(buffer + 3 + in_size0 + in_size1);
    float *ref = (float *)(buffer + 3 + in_size0 + in_size1 + buffer[2]);

    int8_t *input_tmp = malloc(in_size0 * sizeof(char));
    int8_t *weight_tmp = malloc(in_size1 * sizeof(char));
    int32_t *bias_tmp = (int32_t *)malloc(buffer[2] * sizeof(int32_t));

    input->data = src_in;
    get_quant_info(input);
    scale1 = input->qinfo->scale;

    for (int i = 0; i < in_size0; i++) {
        input_tmp[i] = shl_ref_quantize_f32_to_i8(src_in[i], input->qinfo);
    }

    weight->data = weight_in;
    get_quant_info(weight);
    scale2 = weight->qinfo->scale;

    for (int i = 0; i < in_size1; i++) {
        weight_tmp[i] = shl_ref_quantize_f32_to_i8(weight_in[i], weight->qinfo);
    }

    scale = scale1 * scale2;
    for (int i = 0; i < buffer[2]; i++) {
        bias_tmp[i] = (int32_t)(bias_in[i] / scale);
    }

    output->data = ref;
    get_quant_info(output);
    scale3 = output->qinfo->scale;
    scale = (scale1 * scale2) / scale3;
    shl_quantize_multiplier(scale, &quantized_multiplier, &shift);
    output->qinfo->multiplier = quantized_multiplier;
    output->qinfo->shift = shift;

    input->data = input_tmp;
    weight->data = weight_tmp;
    bias->data = bias_tmp;

    reference->data = ref;
    output->data = malloc(out_size * sizeof(char));

    float difference = argc > 2 ? atof(argv[2]) : 1e-3;

    if (csinn_fullyconnected_init(input, output, weight, bias, params) == CSINN_TRUE) {
        csinn_fullyconnected(input, output, weight, bias, params);
    }

    shl_quantize_multiplier(scale3, &quantized_multiplier, &shift);
    output->qinfo->multiplier = quantized_multiplier;
    output->qinfo->shift = shift;

    result_verify_8(reference->data, output, input->data, difference, out_size, false);

    free(buffer);
    free(weight_tmp);
    free(input_tmp);
    free(output->data);
    return done_testing();
}
