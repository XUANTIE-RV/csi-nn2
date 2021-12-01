/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

/* CSI-NN2 version 1.8.x */

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of l2 normalization u8.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct l2n_params params;
    int size = 1;
    int zp, quantized_multiplier, shift;
    float max_value, min_value, scale;
    float error;

    int *buffer = read_input_data_f32(argv[1]);
    /* get the dim para */
    output->dim_count = input->dim_count = buffer[0];
    params.epsilon = *(float *)&buffer[1];
    for (int i = 0; i < input->dim_count; ++i) {
        output->dim[i] = input->dim[i] = buffer[2 + i];
    }

    for (int i = 0; i < input->dim_count; ++i) {
        size *= input->dim[i];
    }

    input->dtype = CSINN_DTYPE_UINT8;
    output->dtype = CSINN_DTYPE_UINT8;
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    float *src_in   = (float *)(buffer + 2 + input->dim_count);
    float *ref      = (float *)(buffer + 2 + input->dim_count + size);
    uint8_t *input_tmp = malloc(size * sizeof(char));

    input->data = src_in;
    get_quant_info(input);

    for(int i = 0; i < size; i++) {
        input_tmp[i] = csi_ref_quantize_f32_to_u8(src_in[i], input->qinfo);
    }

    /* compute the max quantize error */
    for(int i = 0; i < size; i++) {
        float error1;
        float output_tmp  = csi_ref_dequantize_u8_to_f32(src_in[i], input->qinfo);
        if(isinf(src_in[i]) || isnan(src_in[i])){
            continue;
        } else {
            error1 = fabs(src_in[i] -output_tmp);
            if(error1 > 1e-6) {
                error1 = fabs(src_in[i] - output_tmp)/fabs(src_in[i] + 1e-9);
            }
        }
        if(error1 > error) {
            error = error1;
        }
    }
    error = sqrt(error * fabs(max_value) * input->dim[input->dim_count - 1]);

    output->data = ref;
    get_quant_info(output);

    input->data     = input_tmp;
    reference->data = ref;
    output->data    = malloc(size * sizeof(char));
    float difference = argc > 2 ? atof(argv[2]) : error;
    printf("The max error is %.6lf.\n", error);

    if (csi_l2_normalization_init(input, output, &params) == CSINN_TRUE) {
        csi_l2_normalization(input, output, &params);
    }

    result_verify_8(reference->data, output, input->data, difference, size, false);

    free(buffer);
    free(input_tmp);
    free(output->data);
    return done_testing();
}
