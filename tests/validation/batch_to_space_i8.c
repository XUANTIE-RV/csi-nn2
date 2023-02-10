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

#include "csi_nn.h"
#include "test_utils.h"

int main(int argc, char **argv)
{
    init_testsuite("Testing function of batch_to_space i8.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_batch_to_space_params *params =
        csinn_alloc_params(sizeof(struct csinn_batch_to_space_params), NULL);
    int in_size = 0;
    int out_size = 0;
    float max_error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);

    input->dim[0] = buffer[0];  // in_batch
    input->dim[1] = buffer[1];  // in_channel
    input->dim[2] = buffer[2];  // in_height
    input->dim[3] = buffer[3];  // in_width
    params->block_size = buffer[4];
    params->crop_top = buffer[5];
    params->crop_bottom = buffer[6];
    params->crop_left = buffer[7];
    params->crop_right = buffer[8];

    output->dim[0] = input->dim[0] / (params->block_size * params->block_size);
    output->dim[1] = input->dim[1];
    output->dim[2] = input->dim[2] * params->block_size - params->crop_top - params->crop_bottom;
    output->dim[3] = input->dim[3] * params->block_size - params->crop_left - params->crop_right;

    input->dim_count = 4;
    output->dim_count = 4;
    input->dtype = CSINN_DTYPE_INT8;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    output->dtype = CSINN_DTYPE_INT8;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;

    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    params->base.api = CSINN_API;

    float *src_in = (float *)(buffer + 9);
    float *ref = (float *)(buffer + 9 + in_size);
    int8_t *src_tmp = malloc(in_size * sizeof(char));

    input->data = src_in;
    get_quant_info(input);

    for (int i = 0; i < in_size; i++) {
        src_tmp[i] = shl_ref_quantize_f32_to_i8(src_in[i], input->qinfo);
    }

    /* compute the max quantize error */
    for (int i = 0; i < in_size; i++) {
        float error1;
        float output_tmp = shl_ref_dequantize_i8_to_f32(src_tmp[i], input->qinfo);
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
    output->data = malloc(out_size * sizeof(char));

    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_batch_to_space_init(input, output, params) == CSINN_TRUE) {
        csinn_batch_to_space(input, output, params);
    }

    result_verify_8(reference->data, output, input->data, difference, out_size, false);

    free(buffer);
    free(src_tmp);
    free(output->data);
    return done_testing();
}