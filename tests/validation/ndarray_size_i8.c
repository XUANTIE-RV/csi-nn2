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
    init_testsuite("Testing function of ndarray size i8.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_ndarray_size_params *params;
    int in_size = 1, out_size = 1;
    int zp, quantized_multiplier, shift;
    float scale, min_value, max_value;
    float max_error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim_count = buffer[0];
    output->dim_count = 1;
    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i + 1];
        in_size *= input->dim[i];
    }
    output->dim[0] = 1;

    out_size = 1;
    output->dtype = CSINN_DTYPE_INT8;
    output->layout = CSINN_LAYOUT_N;
    output->is_const = 0;
    output->quant_channel = 1;

    input->dtype = CSINN_DTYPE_INT8;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    params->base.api = CSINN_API;
    params->base.layout = CSINN_LAYOUT_NCHW;

    float *src_in = (float *)(buffer + 1 + input->dim_count);
    float *ref = (float *)(buffer + 1 + input->dim_count + in_size);
    float difference = argc > 2 ? atof(argv[2]) : 0.9;
    int8_t *src_tmp = malloc(in_size * sizeof(char));

    input->data = src_in;
    get_quant_info(input);

    for (int i = 0; i < in_size; i++) {
        src_tmp[i] = shl_ref_quantize_f32_to_i8(src_in[i], input->qinfo);
    }

    output->data = ref;
    get_quant_info(output);
    input->data = src_tmp;
    reference->data = ref;
    output->data = malloc(out_size * sizeof(char));

    if (csinn_ndarray_size_init(input, output, params) == CSINN_TRUE) {
        csinn_ndarray_size(input, output, params);
    }

    result_verify_8(reference->data, output, input->data, difference, out_size, false);

    free(buffer);
    free(src_tmp);
    free(output->data);
    return done_testing();
}
