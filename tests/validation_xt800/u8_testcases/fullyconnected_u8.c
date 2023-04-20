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

#include "../valid_data/fullyconnected_u8.dat"

#include "csi_nn.h"
#include "test_utils.h"

static void verify_fullyconnected_u8(float *input_data, float *weights_data, float *bias_data,
                                     float *ref_data, int32_t in_nodes, int32_t out_nodes,
                                     float difference)
{
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    int in_size, weights_size, bias_size, out_size;

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = in_nodes;
    input->dim_count = 2;
    input->dtype = CSINN_DTYPE_UINT8;
    input->layout = CSINN_LAYOUT_NC;
    input->name = "input";
    input->data = (float *)input_data;
    get_quant_info(input);
    in_size = input->dim[0] * input->dim[1];

    uint8_t *input_tmp = malloc(in_size * sizeof(char));
    for (int i = 0; i < in_size; i++) {
        input_tmp[i] = shl_ref_quantize_f32_to_u8(input_data[i], input->qinfo);
    }
    input->data = input_tmp;

    struct csinn_tensor *weights = csinn_alloc_tensor(NULL);
    weights->dim[0] = out_nodes;
    weights->dim[1] = in_nodes;
    weights->dim_count = 2;
    weights->dtype = CSINN_DTYPE_UINT8;
    weights->layout = CSINN_LAYOUT_OI;
    weights->name = "weights";
    weights->data = (float *)weights_data;
    get_quant_info(weights);
    weights_size = weights->dim[0] * weights->dim[1];

    uint8_t *weights_tmp = malloc(weights_size * sizeof(char));
    for (int i = 0; i < weights_size; i++) {
        weights_tmp[i] = shl_ref_quantize_f32_to_u8(weights_data[i], weights->qinfo);
    }
    weights->data = weights_tmp;

    struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
    bias->dim[0] = out_nodes;
    bias->dim_count = 1;
    bias->dtype = CSINN_DTYPE_INT32;
    bias->layout = CSINN_LAYOUT_O;
    bias->name = "bias";
    bias->data = (float *)bias_data;
    bias_size = bias->dim[0];

    int32_t *bias_tmp = malloc(bias_size * sizeof(int32_t));
    for (int i = 0; i < bias_size; i++) {
        bias_tmp[i] = (int32_t)(bias_data[i] / (input->qinfo->scale * weights->qinfo->scale));
    }
    bias->qinfo->scale = input->qinfo->scale * weights->qinfo->scale;
    bias->data = bias_tmp;

    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = out_nodes;
    output->dim_count = 2;
    output->dtype = CSINN_DTYPE_UINT8;
    output->layout = CSINN_LAYOUT_NC;

    output->name = "output";
    output->data = (float *)ref_data;
    get_quant_info(output);
    out_size = output->dim[0] * output->dim[1];
    output->data = malloc(out_size);

    struct csinn_fc_params *params = csinn_alloc_params(sizeof(struct csinn_fc_params), NULL);
    params->base.api = CSINN_API;
    params->base.name = "params";
    params->base.layout = CSINN_LAYOUT_NHWC;
    params->units = out_nodes;  // out_nodes

    if (csinn_fullyconnected_init(input, output, weights, bias, params) == CSINN_TRUE) {
        csinn_fullyconnected(input, output, weights, bias, params);
    }

    reference->data = (float *)ref_data;
    result_verify_8(reference->data, output, input->data, difference, out_size, false);
    free(input);
    free(weights);
    free(bias);
    free(output->data);
    free(output);
    free(reference);
    free(input_tmp);
    free(weights_tmp);
    free(bias_tmp);
}

int main(int argc, char **argv)
{
    init_testsuite("Testing function of fullyconnected(u8) for i805.\n");

    verify_fullyconnected_u8(fc_input_0, fc_weights_0, fc_bias_0, fc_output_0, 64, 32, 1.0);
    verify_fullyconnected_u8(fc_input_1, fc_weights_1, fc_bias_1, fc_output_1, 79, 63, 1.0);
}
