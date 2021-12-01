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
    init_testsuite("Testing function of gather_nd u8.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *indices = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct gather_nd_params params;
    int in_size = 1, out_size = 1, indices_size = 1;
    int zp, quantized_multiplier, shift;
    float scale, min_value, max_value;
    float max_error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim_count = buffer[0];
    output->dim_count = 0;  // init output->dim_count = 0
    for(int i = 0; i < input->dim_count; i++) {
        input->dim[i] = buffer[i + 1];
        in_size *= input->dim[i];
    }
    indices->dim_count = buffer[1 + input->dim_count];
    for(int i = 0; i < indices->dim_count; i++) {
        indices->dim[i] = buffer[i + 2 + input->dim_count];
        indices_size *= indices->dim[i];
        if(i < indices->dim_count - 1) {
            output->dim_count++;
            output->dim[i] = indices->dim[i];
        }
    }

    int axis = indices->dim[indices->dim_count - 1];

    int indices_outer_size = 1;
    indices_outer_size = indices_size / indices->dim[indices->dim_count - 1];

    int input_inner_size = 1;
    for(int i = axis; i < input->dim_count; i++) {
        input_inner_size *= input->dim[i];
        output->dim[output->dim_count] = input->dim[i];
        output->dim_count++;
    }

    out_size = indices_outer_size * input_inner_size;
    input->dtype = CSINN_DTYPE_UINT8;
    output->dtype = CSINN_DTYPE_UINT8;
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    indices->data  = (int *)(buffer + 2 + input->dim_count + indices->dim_count);
    float *src_in   = (float *)(buffer + 2 + input->dim_count + indices->dim_count + indices_size);
    float *ref      = (float *)(buffer + 2 + input->dim_count + indices->dim_count + indices_size + in_size);
    uint8_t *src_tmp = malloc(in_size * sizeof(char));

    input->qinfo = get_quant_info(src_in, in_size);

    for(int i = 0; i < in_size; i++) {
        src_tmp[i] = csi_ref_quantize_f32_to_u8(src_in[i], input->qinfo);
    }

    /* compute the max quantize error */
    for(int i = 0; i < in_size; i++) {
        float error1;
        float output_tmp  = csi_ref_dequantize_u8_to_f32(src_tmp[i], input->qinfo);
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

    output->qinfo = get_quant_info(ref, out_size);

    input->data     = src_tmp;
    reference->data = ref;
    output->data    = malloc(out_size * sizeof(char));

    float difference = argc > 2 ? atof(argv[2]) : max_error;


    if (csi_gather_nd_init(input, indices, output, &params) == CSINN_TRUE) {
        csi_gather_nd(input, indices, output, &params);
    }

    result_verify_8(reference->data, output, input->data, difference, out_size, false);

    free(buffer);
    free(src_tmp);
    free(output->data);
    return done_testing();
}
