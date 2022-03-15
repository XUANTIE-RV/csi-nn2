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

/* CSI-NN2 version 1.12.x */

#include "test_utils.h"
#include "csi_c860.h"
#include "math_snr.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of dequantize f32.\n");

    struct csi_tensor *it = csi_alloc_tensor(NULL);
    float *input, *output, *reference;
    int in_size, zp, quantized_multiplier, shift;
    float max_value, min_value, scale;

    int *buffer = read_input_data_f32(argv[1]);
    in_size     = buffer[0];

    input      = (float *)(buffer + 1);
    reference  = malloc(in_size * sizeof(float));
    output     = malloc(in_size * sizeof(float));
    uint8_t *input_tmp = malloc(in_size * sizeof(char));

    find_min_max(input, &max_value, &min_value, in_size);
    get_scale_and_zp(max_value, min_value, &scale, &zp);
    csi_quantize_multiplier(scale, &quantized_multiplier, &shift);
    it->data = input;
    get_quant_info(it);
    for(int i = 0; i < in_size; i++) {
        input_tmp[i] = csi_ref_quantize_f32_to_u8(input[i], it->qinfo);
    }

    for(int i = 0; i < in_size; i++) {
        reference[i] = csi_ref_dequantize_u8_to_f32(input_tmp[i], it->qinfo);
    }

    csi_dequantize_f32_c860(input_tmp, output, -it->qinfo->zero_point, it->qinfo->multiplier,
                            it->qinfo->shift, in_size);

    float difference = argc > 2 ? atof(argv[2]) : 0.9;

    result_verify_f32(reference, output, input, difference, in_size, false);

    free(buffer);
    free(reference);
    free(output);
    free(input_tmp);
    return done_testing();
}
