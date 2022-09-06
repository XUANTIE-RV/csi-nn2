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
    init_testsuite("Testing function of squeeze u8.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_squeeze_params *params =
        csinn_alloc_params(sizeof(struct csinn_squeeze_params), NULL);
    int in_size;
    int zp, quantized_multiplier, shift;
    float scale, min_value, max_value;
    float max_error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);
    int axis_len = buffer[3];
    output->dim[0] = input->dim[0] = buffer[0];  // batch
    output->dim[1] = input->dim[1] = buffer[1];  // height
    output->dim[2] = input->dim[2] = buffer[2];  // width
    input->dim[3] = 1;
    input->dim[4] = 1;
    input->dim[5] = 1;

    for (int i = 0; i < axis_len; i++) {
        params->axis[i] = buffer[4 + i];
    }

    input->dim_count = 6;
    output->dim_count = input->dim_count - axis_len;

    input->dtype = CSINN_DTYPE_UINT8;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    output->dtype = CSINN_DTYPE_UINT8;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;

    params->axis_num = axis_len;
    params->base.layout = CSINN_LAYOUT_NCHW;
    in_size = input->dim[0] * input->dim[1] * input->dim[2];
    params->base.api = CSINN_API;

    float *src_in = (float *)(buffer + 3);
    float *ref = (float *)(buffer + 3 + in_size);
    uint8_t *src_tmp = malloc(in_size * sizeof(char));

    input->data = src_in;
    get_quant_info(input);
    for (int i = 0; i < in_size; i++) {
        src_tmp[i] = shl_ref_quantize_f32_to_u8(src_in[i], input->qinfo);
    }

    /* compute the max quantize error */
    for (int i = 0; i < in_size; i++) {
        float error1;
        float output_tmp = shl_ref_dequantize_u8_to_f32(src_tmp[i], input->qinfo);
        if (isinf(src_in[i]) || isnan(src_in[i])) {
            continue;
        } else {
            error1 = fabs(src_in[i] - output_tmp);
            if (error1 > 1e-6) {
                error1 = fabs(src_in[i] - output_tmp) / fabs(src_in[i] + 1e-9);
            }
        }
        if (error1 > max_error) {
            max_error = error1;
        }
    }

    output->data = ref;
    get_quant_info(output);

    input->data = src_tmp;
    reference->data = ref;
    output->data = malloc(in_size * sizeof(char));

    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_squeeze_init(input, output, params) == CSINN_TRUE) {
        csinn_squeeze(input, output, params);
    }

    result_verify_8(reference->data, output, input->data, difference, in_size, false);

    free(buffer);
    free(src_tmp);
    free(output->data);
    return done_testing();
}
