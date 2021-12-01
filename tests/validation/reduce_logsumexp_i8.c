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
    init_testsuite("Testing function of reduce_logsumexp f32.\n");

    struct csi_tensor *input0 = csi_alloc_tensor(NULL);
    struct csi_tensor *output = csi_alloc_tensor(NULL);
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    struct reduce_params params;
    int in_size0;
    int out_size = 0;
    int zp, quantized_multiplier, shift;
    float scale, min_value, max_value;
    float max_error = 0.0f;

    int *buffer = read_input_data_f32(argv[1]);

    reference->dim[0] = input0->dim[0] = buffer[0];          // batch
    reference->dim[1] = input0->dim[1] = buffer[1];          // height
    reference->dim[2] = input0->dim[2] = buffer[2];          // width
    reference->dim[3] = input0->dim[3] = buffer[3];          // channel

    params.axis_count = 1;
    params.axis = (int *)malloc(sizeof(int) * params.axis_count);
    params.axis[0] = buffer[4];

    in_size0 = input0->dim[0] * input0->dim[1] * input0->dim[2] * input0->dim[3];
    input0->dim_count = 4;
    input0->dtype = CSINN_DTYPE_INT8;
    output->dtype = CSINN_DTYPE_INT8;
    params.base.api = CSINN_API;
    params.base.run_mode = CSINN_RM_LAYER;


    if(params.axis[0]==-1) {
        out_size = 1;
        output->dim_count = 1;
        output->dim[0] = 1;
    } else {
        out_size = in_size0/input0->dim[params.axis[0]];
        output->dim_count = 4;  // keep_dim = 1
        for(int i = 0; i < output->dim_count; i++) {
            if(params.axis[0] == i) {
                output->dim[i] = 1;
            } else {
                output->dim[i] = input0->dim[i];
            }
        }
    }

    float *src_in   = (float *)(buffer + 5);
    float *ref      = (float *)(buffer + 5 + in_size0 );
    int8_t *src_tmp = malloc(in_size0 * sizeof(char));

    input0->data = src_in;
    get_quant_info(input0);

    for(int i = 0; i < in_size0; i++) {
        src_tmp[i] = csi_ref_quantize_f32_to_i8(src_in[i], input0->qinfo);
    }

    /* compute the max quantize error */
    for(int i = 0; i < in_size0; i++) {
        float error1;
        float output_tmp  = csi_ref_dequantize_i8_to_f32(src_tmp[i], input0->qinfo);
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

    output->data = ref;
    get_quant_info(output);

    input0->data     = src_tmp;
    reference->data = ref;
    output->data    = malloc(out_size * sizeof(char));

    float difference = argc > 2 ? atof(argv[2]) : max_error;



    if (csi_reduce_logsumexp_init(input0, output, &params) == CSINN_TRUE) {
        csi_reduce_logsumexp(input0, output, &params);
    }

    result_verify_8(reference->data, output, input0->data, difference, out_size, false);

    free(buffer);
    free(src_tmp);
    free(output->data);
    return done_testing();
}
