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
    init_testsuite("Testing function of unstack u8.\n");

    int in_size = 1;
    int out_size = 1;
    float max_error = 0.05f;

    int *buffer = read_input_data_f32(argv[1]);
    struct csinn_unstack_params *params =
        csinn_alloc_params(sizeof(struct csinn_unstack_params), NULL);
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    params->axis = buffer[0];
    input->dim_count = buffer[1];
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    for (int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[2 + i];
        in_size *= input->dim[i];
    }
    params->outputs_count = input->dim[params->axis];
    struct csinn_tensor *output[params->outputs_count];
    for (int i = 0; i < params->outputs_count; i++) {
        output[i] = csinn_alloc_tensor(NULL);
        output[i]->dim_count = input->dim_count - 1;
        output[i]->dtype = CSINN_DTYPE_UINT8;
        output[i]->layout = CSINN_LAYOUT_NCHW;
        output[i]->is_const = 0;
        output[i]->quant_channel = 1;
        for (int j = 0; j < input->dim_count; j++) {
            if (j < params->axis) {
                output[i]->dim[j] = input->dim[j];
            } else if (j > params->axis) {
                output[i]->dim[j - 1] = input->dim[j];
            }
        }
    }
    float *src_out[params->outputs_count];

    out_size = in_size / params->outputs_count;
    params->base.api = CSINN_API;

    input->dtype = CSINN_DTYPE_UINT8;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    float *src_in = (float *)(buffer + 2 + input->dim_count);
    float *ref = (float *)(buffer + 2 + input->dim_count + in_size);
    uint8_t *src_tmp = malloc(in_size * sizeof(char));

    input->data = src_in;
    get_quant_info(input);

    for (int i = 0; i < in_size; i++) {
        src_tmp[i] = shl_ref_quantize_f32_to_u8(src_in[i], input->qinfo);
    }

    for (int i = 0; i < params->outputs_count; i++) {
        src_out[i] = (float *)(buffer + 2 + input->dim_count + in_size + out_size * i);
    }

    for (int j = 0; j < params->outputs_count; j++) {
        output[j]->data = src_out[j];
        get_quant_info(output[j]);
        output[j]->dtype = CSINN_DTYPE_UINT8;
        output[j]->data = malloc(out_size * sizeof(char));
    }

    input->data = src_tmp;
    reference->data = ref;
    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    if (csinn_unstack_init(input, output, params) == CSINN_TRUE) {
        csinn_unstack(input, output, params);
    }

    float *ref_addr = (float *)reference->data;
    for (int i = 0; i < params->outputs_count; i++) {
        result_verify_8(ref_addr, output[i], input->data, difference, out_size, false);
        ref_addr += out_size;
    }

    free(buffer);
    for (int i = 0; i < params->outputs_count; i++) {
        free(output[i]->data);
        output[i]->data = NULL;
    }
    return done_testing();
}
