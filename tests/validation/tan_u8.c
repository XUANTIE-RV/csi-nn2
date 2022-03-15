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
#include "csi_nn.h"
#include "math_snr.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of tan u8.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct siso_params params;
    int in_size = 1, out_size = 1;
    int zero_point, multiplier, shift;
    float scale, min_value, max_value;
    float error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim_count = buffer[0];
    output->dim_count = input->dim_count;
    for(int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i + 1];
        output->dim[i] = input->dim[i];
        in_size *= input->dim[i];
    }

    out_size = in_size;
    input->dtype = CSINN_DTYPE_UINT8;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    output->dtype = CSINN_DTYPE_UINT8;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;
    
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    float *src_in_data = (float *)(buffer + 1 + input->dim_count);
    float *ref_data = (float *)(buffer + 1 + input->dim_count + in_size);

    uint8_t *input_data = (uint8_t *)malloc(in_size * sizeof(uint8_t));

    input->data = src_in_data;
    get_quant_info(input);

    for(int i = 0; i < in_size; i++) {
        input_data[i] = csi_ref_quantize_f32_to_u8(src_in_data[i], input->qinfo);
    }

    /* compute the max quantize error */
    for(int i = 0; i < in_size; i++) {
        float error1;
        float output_tmp  = csi_ref_dequantize_u8_to_f32(input_data[i], input->qinfo);
        if(isinf(src_in_data[i]) && isinf(output_tmp) || isnan(src_in_data[i]) && isnan(output_tmp)) {
            continue;
        } else {
            error1 = fabs(src_in_data[i] - output_tmp);
            if(error1 > 1e-6) {
                error1 = fabs(src_in_data[i] - output_tmp)/fabs(src_in_data[i] + 1e-9);
            }
        }
        if(error1 > error) {
            error = error1;
        }
    }

    output->data = ref_data;
    get_quant_info(output);

    input->data = input_data;
    reference->data = ref_data;
    output->data = (uint8_t *)malloc(out_size * sizeof(uint8_t));
    // max error: 10000 for input [-1.57, 1.57]
    float difference = argc > 2 ? atof(argv[2]) : 0.9;


    if (csi_tan_init(input, output, &params) == CSINN_TRUE) {
        csi_tan(input, output, &params);
    }

    result_verify_8(reference->data, output, input->data, difference, out_size, false);

    free(buffer);
    free(output->data);
    free(input_data);
    return done_testing();
}
