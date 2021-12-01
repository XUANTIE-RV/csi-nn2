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

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"

int main(int argc, char** argv)
{
    init_testsuite("Testing function of prelu nhwc i8.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct csi_tensor *alpha_data = csi_alloc_tensor(NULL);
    struct prelu_params params;
    int in_size = 1;
    int out_size = 1;
    int zp, quantized_multiplier, shift;
    float scale, min_value, max_value;
    float max_error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);
    output->dim[0] = input->dim[0] = buffer[0];          // batch
    output->dim[1] = input->dim[1] = buffer[1];          // height
    output->dim[2] = input->dim[2] = buffer[2];          // width
    output->dim[3] = input->dim[3] = buffer[3];          // channel
    alpha_data->dim[0] = buffer[3];
    input->dim_count = 4;
    output->dim_count = 4;
    alpha_data->dim_count = 1;
    input->dtype = CSINN_DTYPE_INT8;
    alpha_data->dtype = CSINN_DTYPE_INT8;
    output->dtype = CSINN_DTYPE_INT8;
    params.base.layout = CSINN_NHWC;

    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];
    out_size = in_size;
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;


    float *src_in   = (float *)(buffer + 4);
    float *alpha_in = (float *)(buffer + 4 + in_size);
    float *ref      = (float *)(buffer + 4 + in_size + input->dim[3]);
    int8_t *src_tmp = malloc(in_size * sizeof(char));
    int8_t *alpha_tmp = malloc(input->dim[3] * sizeof(char));

    input->qinfo = get_quant_info_i8(src_in, in_size);

    for(int i = 0; i < in_size; i++) {
        src_tmp[i] = csi_ref_quantize_f32_to_i8(src_in[i], input->qinfo);
    }

    /* compute the max quantize error */
    for(int i = 0; i < in_size; i++) {
        float error1;
        float output_tmp  = csi_ref_dequantize_i8_to_f32(src_tmp[i], input->qinfo);
        if(isinf(src_in[i]) || isnan(src_in[i])){
            continue;
        } else {
            error1 = fabs(src_in[i] -output_tmp);
            if(error1 > 1e-6) {
                error1 = fabs(src_in[i] - output_tmp)/fabs(src_in[i] + 1e-9);
            }
        }
        if(error1 > max_error) {
            max_error = error1;
        }
    }

    alpha_data->qinfo = get_quant_info_i8(alpha_in, input->dim[3]);
    for(int i = 0; i < input->dim[3]; i++) {
        alpha_tmp[i] = csi_ref_quantize_f32_to_i8(alpha_in[i], alpha_data->qinfo);
    }

    /* compute the max quantize error */
    for(int i = 0; i < input->dim[3]; i++) {
        float error1;
        float output_tmp  = csi_ref_dequantize_i8_to_f32(alpha_tmp[i], alpha_data->qinfo);
        if(isinf(alpha_in[i]) || isnan(alpha_in[i])){
            continue;
        } else {
            error1 = fabs(alpha_in[i] -output_tmp);
            if(error1 > 1e-6) {
                error1 = fabs(alpha_in[i] - output_tmp)/fabs(alpha_in[i] + 1e-9);
            }
        }
        if(error1 > max_error) {
            max_error = error1;
        }
    }

    output->qinfo = get_quant_info_i8(ref, out_size);

    input->data     = src_tmp;
    alpha_data->data = alpha_tmp;
    reference->data = ref;
    output->data    = malloc(out_size * sizeof(char));

    float difference = argc > 2 ? atof(argv[2]) : max_error;


    if (csi_prelu_init(input, alpha_data, output, &params) == CSINN_TRUE) {
        csi_prelu(input, alpha_data, output, &params);
    }

    result_verify_8(reference->data, output, input->data, difference, out_size, false);

    free(buffer);
    free(src_tmp);
    free(output->data);
    return done_testing();
}
