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
    init_testsuite("Testing function of gather i8.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *indices = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_gather_params *params =
        csinn_alloc_params(sizeof(struct csinn_gather_params), NULL);
    int in_size = 1, indices_size = 1, out_size = 1;
    int zp, quantized_multiplier, shift;
    float scale, min_value, max_value;
    float max_error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);
    int axis = buffer[0];
    input->dim_count = buffer[1];
    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i + 2];
        in_size *= input->dim[i];
    }

    indices->dim_count = buffer[2 + input->dim_count];
    for (int i = 0; i < indices->dim_count; i++) {
        indices->dim[i] = buffer[3 + input->dim_count + i];
        indices_size *= indices->dim[i];
    }

    output->dim_count = input->dim_count + indices->dim_count - 1;
    int j = 0;
    for (int i = 0; i < axis; i++) {
        output->dim[j] = input->dim[i];
        out_size *= output->dim[j];
        j++;
    }
    for (int i = 0; i < indices->dim_count; i++) {
        output->dim[j] = indices->dim[i];
        out_size *= output->dim[j];
        j++;
    }
    for (int i = axis + 1; i < input->dim_count; i++) {
        output->dim[j] = input->dim[i];
        out_size *= output->dim[j];
        j++;
    }

    input->dtype = CSINN_DTYPE_INT8;
    input->layout = CSINN_LAYOUT_NCHW;
    indices->dtype = CSINN_DTYPE_INT32;
    output->dtype = CSINN_DTYPE_INT8;
    output->layout = CSINN_LAYOUT_NCHW;
    params->base.api = CSINN_API;
    params->axis = axis;

    float *src_in = (float *)(buffer + 3 + input->dim_count + indices->dim_count);
    indices->data = (int32_t *)(buffer + 3 + input->dim_count + indices->dim_count + in_size);
    float *ref =
        (float *)(buffer + 3 + input->dim_count + indices->dim_count + in_size + indices_size);
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

    if (csinn_gather_init(input, indices, output, params) == CSINN_TRUE) {
        csinn_gather(input, indices, output, params);
    }
    result_verify_8(reference->data, output, input->data, difference, out_size, false);

    free(buffer);
    free(src_tmp);
    free(output->data);
    return done_testing();
}
