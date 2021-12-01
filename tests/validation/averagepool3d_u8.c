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
    init_testsuite("Testing function of averagepool3d u8.\n");

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct pool_params params;
    int in_size = 1;
    int out_size = 1;
    int zp, quantized_multiplier, shift;
    float scale, min_value, max_value;
    float max_error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);
    input->dim[0] = buffer[0];       //batch
    input->dim[1] = buffer[1];       //channel
    input->dim[2] = buffer[2];       //depth
    input->dim[3] = buffer[3];       //height
    input->dim[4] = buffer[4];       //width

    output->dim[0] = buffer[0];
    output->dim[1] = buffer[1];
    output->dim[2] = buffer[17];
    output->dim[3] = buffer[18];
    output->dim[4] = buffer[19];

    params.stride_depth  = buffer[5];
    params.stride_height = buffer[6];
    params.stride_width  = buffer[7];
    params.filter_depth  = buffer[8];
    params.filter_height = buffer[9];
    params.filter_width  = buffer[10];

    params.pad_left  = buffer[11];
    params.pad_right = buffer[12];
    params.pad_top   = buffer[13];
    params.pad_down  = buffer[14];
    params.pad_front = buffer[15];
    params.pad_back  = buffer[16];
    params.count_include_pad = buffer[20];
    params.base.layout = CSINN_NCDHW;

    input->dtype = CSINN_DTYPE_UINT8;
    output->dtype = CSINN_DTYPE_UINT8;
    input->dim_count = 5;
    output->dim_count = 5;

    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3] * input->dim[4];
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3] * output->dim[4];
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;

    float *src_in   = (float *)(buffer + 20);
    float *ref      = (float *)(buffer + 20 + in_size);
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

    if (csi_averagepool3d_init(input, output, &params) == CSINN_TRUE) {
        csi_averagepool3d(input, output, &params);
    }
 

    result_verify_8(reference->data, output, input->data, difference, out_size, false);

    free(buffer);
    free(src_tmp);
    free(output->data);
    return done_testing();
}
