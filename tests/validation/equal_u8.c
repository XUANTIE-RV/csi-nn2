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
    init_testsuite("Testing function of equal u8.\n");

    struct csinn_tensor *input0 = csinn_alloc_tensor(NULL);
    struct csinn_tensor *input1 = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_diso_params *params = csinn_alloc_params(sizeof(struct csinn_diso_params), NULL);
    int in_size = 1;
    int out_size = 1;
    int zp, quantized_multiplier, shift;
    float scale, min_value, max_value;
    float error[2] = {0};
    float max_error;

    int *buffer = read_input_data_f32(argv[1]);
    input0->dim_count = input1->dim_count = buffer[0];
    output->dim_count = input0->dim_count;

    for (int i = 0; i < input0->dim_count; i++) {
        input0->dim[i] = buffer[1 + i];
        input1->dim[i] = input0->dim[i];
        output->dim[i] = input0->dim[i];
        in_size = in_size * input0->dim[i];
    }

    out_size = in_size;

    input0->dtype = CSINN_DTYPE_UINT8;
    input0->layout = CSINN_LAYOUT_NCHW;
    input0->is_const = 0;
    input0->quant_channel = 1;

    input1->dtype = CSINN_DTYPE_UINT8;
    input1->layout = CSINN_LAYOUT_NCHW;
    input1->is_const = 0;
    input1->quant_channel = 1;

    output->dtype = CSINN_DTYPE_UINT8;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;

    params->base.api = CSINN_API;

    uint8_t *input_tmp0 = malloc(in_size * sizeof(char));
    uint8_t *input_tmp1 = malloc(in_size * sizeof(char));
    float *src_in0 = (float *)(buffer + 1 + input0->dim_count);
    float *src_in1 = (float *)(buffer + 1 + input0->dim_count + in_size);
    float *ref = (float *)(buffer + 1 + input0->dim_count + 2 * in_size);

    input0->data = src_in0;
    get_quant_info(input0);

    for (int i = 0; i < in_size; i++) {
        input_tmp0[i] = shl_ref_quantize_f32_to_u8(src_in0[i], input0->qinfo);
    }
    /* compute the max quantize error */
    for (int i = 0; i < in_size; i++) {
        float error1;
        float output_tmp = shl_ref_dequantize_u8_to_f32(input_tmp0[i], input0->qinfo);
        if (src_in0[i] == INFINITY && output_tmp == INFINITY ||
            src_in0[i] == NAN && output_tmp == NAN) {
            continue;
        } else {
            error1 = fabs(src_in0[i] - output_tmp);
            if (error1 > 1e-6) {
                error1 = fabs(src_in0[i] - output_tmp) / fabs(src_in0[i] + 1e-9);
            }
        }
        if (error1 > error[0]) {
            error[0] = error1;
        }
    }

    input1->data = src_in1;
    get_quant_info(input1);

    for (int i = 0; i < in_size; i++) {
        input_tmp1[i] = shl_ref_quantize_f32_to_u8(src_in1[i], input1->qinfo);
    }

    /* compute the max quantize error */
    for (int i = 0; i < in_size; i++) {
        float error1;
        float output_tmp = shl_ref_dequantize_u8_to_f32(input_tmp1[i], input1->qinfo);
        if (src_in1[i] == INFINITY && output_tmp == INFINITY ||
            input_tmp1[i] == NAN && output_tmp == NAN) {
            continue;
        } else {
            error1 = fabs(src_in1[i] - output_tmp);
            if (error1 > 1e-6) {
                error1 = fabs(src_in1[i] - output_tmp) / fabs(src_in1[i] + 1e-9);
            }
        }
        if (error1 > error[1]) {
            error[1] = error1;
        }
    }

    max_error = (error[0] + error[1]);

    output->data = ref;
    get_quant_info(output);

    input0->data = input_tmp0;
    input1->data = input_tmp1;
    reference->data = ref;
    output->data = malloc(out_size * sizeof(char));

    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_equal_init(input0, input1, output, params) == CSINN_TRUE) {
        csinn_equal(input0, input1, output, params);
    }

    result_verify_8(reference->data, output, input0->data, difference, out_size, false);

    free(buffer);
    free(input_tmp0);
    free(input_tmp1);
    free(output->data);
    return done_testing();
}
