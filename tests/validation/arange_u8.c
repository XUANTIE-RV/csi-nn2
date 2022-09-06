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
    init_testsuite("Testing function of arange u8.\n");

    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    struct csinn_tensor *reference = csinn_alloc_tensor(NULL);
    struct csinn_arange_params *params =
        csinn_alloc_params(sizeof(struct csinn_arange_params), NULL);
    int out_size = 1;
    int zero_point, multiplier, shift;
    float scale, min_value, max_value;
    float error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);

    out_size = buffer[3];
    params->start = buffer[0];
    params->stop = buffer[1];
    params->step = buffer[2];
    output->dim_count = 1;
    output->dim[0] = out_size;
    output->dtype = CSINN_DTYPE_UINT8;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;
    params->base.api = CSINN_API;

    float *ref_data = (float *)(buffer + 4);

    shl_quantize_multiplier(params->start, &multiplier, &shift);
    params->start_multiplier = multiplier;
    params->start_shift = shift;

    shl_quantize_multiplier(params->stop, &multiplier, &shift);
    params->stop_multiplier = multiplier;
    params->stop_shift = shift;

    shl_quantize_multiplier(params->step, &multiplier, &shift);
    params->step_multiplier = multiplier;
    params->step_shift = shift;

    output->data = ref_data;
    get_quant_info(output);
    input->data = 0;
    reference->data = ref_data;
    output->data = (uint8_t *)malloc(out_size * sizeof(uint8_t));

    float difference = argc > 2 ? atof(argv[2]) : 1e-3;

    if (csinn_arange_init(output, params) == CSINN_TRUE) {
        csinn_arange(output, params);
    }

    result_verify_8(reference->data, output, input->data, difference, out_size, false);

    free(buffer);
    free(output->data);
    free(input->data);
    return done_testing();
}
