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

#include "csi_nn.h"
#include "test_utils.h"

int main(int argc, char **argv)
{
    init_testsuite("Testing function of shuffle_channel u8.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_shuffle_channel_params *params =
        csinn_alloc_params(sizeof(struct csinn_shuffle_channel_params), NULL);
    int in_size = 1, out_size = 1;
    int zero_point, multiplier, shift;
    float scale, min_value, max_value;
    float error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);

    input->dim[0] = buffer[0];  // batch
    input->dim[1] = buffer[1];  // height
    input->dim[2] = buffer[2];  // width
    input->dim[3] = buffer[3];  // channel
    params->group = buffer[4];

    output->dim[0] = input->dim[0];
    output->dim[1] = input->dim[1];
    output->dim[2] = input->dim[2];
    output->dim[3] = input->dim[3];

    input->dim_count = 4;
    input->dtype = CSINN_DTYPE_UINT8;
    input->layout = CSINN_LAYOUT_NHWC;
    input->is_const = 0;
    input->quant_channel = 1;
    params->base.layout = CSINN_LAYOUT_NHWC;
    params->base.api = CSINN_API;

    output->dim_count = 4;
    output->dtype = CSINN_DTYPE_UINT8;
    output->layout = CSINN_LAYOUT_NHWC;
    output->is_const = 0;
    output->quant_channel = 1;

    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    out_size =
        output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];  // out_size = in_size;

    float *src_in_data = (float *)(buffer + 5);
    float *ref_data = (float *)(buffer + 5 + in_size);

    uint8_t *input_data = (uint8_t *)malloc(in_size * sizeof(uint8_t));

    input->data = src_in_data;
    get_quant_info(input);

    for (int i = 0; i < in_size; i++) {
        input_data[i] = shl_ref_quantize_f32_to_u8(src_in_data[i], input->qinfo);
    }

    /* compute the max quantize error */
    for (int i = 0; i < in_size; i++) {
        float error1;
        float output_tmp = shl_ref_dequantize_u8_to_f32(input_data[i], input->qinfo);
        if (isinf(src_in_data[i]) && isinf(output_tmp) ||
            isnan(src_in_data[i]) && isnan(output_tmp)) {
            continue;
        } else {
            error1 = fabs(src_in_data[i] - output_tmp);
            if (error1 > 1e-6) {
                error1 = fabs(src_in_data[i] - output_tmp) / fabs(src_in_data[i] + 1e-9);
            }
        }
        if (error1 > error) {
            error = error1;
        }
    }

    output->data = ref_data;
    get_quant_info(output);

    input->data = input_data;
    reference->data = ref_data;
    output->data = (uint8_t *)malloc(out_size * sizeof(uint8_t));
    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_shuffle_channel_init(input, output, params) == CSINN_TRUE) {
        csinn_shuffle_channel(input, output, params);
    }

    result_verify_8(reference->data, output, input->data, difference, out_size, false);

    free(buffer);
    free(output->data);
    free(input_data);
    return done_testing();
}
