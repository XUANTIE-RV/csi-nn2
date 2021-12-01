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

/* CSI-NN2 version 1.10.x */

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"
#include "../valid_data/basic_math_func_u8.dat"



static void verify_mul_u8(float *input_0_data,
                          float *input_1_data,
                          float *ref_data,
                          int32_t size,
                          float difference)
{
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size, out_size;

    struct csi_tensor *input0 = csi_alloc_tensor(NULL);
    input0->dim[0] = 1;
    input0->dim[1] = 1;
    input0->dim[2] = 1;
    input0->dim[3] = size;
    input0->dim_count = 4;
    input0->dtype = CSINN_DTYPE_UINT8;
    input0->layout = CSINN_LAYOUT_NHWC;
    input0->name = "input0";
    input0->data = (float *)input_0_data;
    get_quant_info(input0);
    in_size = input0->dim[0] * input0->dim[1] * input0->dim[2] * input0->dim[3];

    uint8_t *src_tmp_0 = malloc(in_size * sizeof(char));
    for(int i = 0; i < in_size; i++) {
        src_tmp_0[i] = csi_ref_quantize_f32_to_u8(input_0_data[i], input0->qinfo);
    }
    input0->data = src_tmp_0;

    struct csi_tensor *input1 = csi_alloc_tensor(NULL);
    input1->dim[0] = 1;
    input1->dim[1] = 1;
    input1->dim[2] = 1;
    input1->dim[3] = size;
    input1->dim_count = 4;
    input1->dtype = CSINN_DTYPE_UINT8;
    input1->layout = CSINN_LAYOUT_NHWC;
    input1->name = "input1";
    input1->data = (float *)input_1_data;
    get_quant_info(input1);
    in_size = input1->dim[0] * input1->dim[1] * input1->dim[2] * input1->dim[3];

    uint8_t *src_tmp_1 = malloc(in_size * sizeof(char));
    for(int i = 0; i < in_size; i++) {
        src_tmp_1[i] = csi_ref_quantize_f32_to_u8(input_1_data[i], input1->qinfo);
    }
    input1->data = src_tmp_1;

    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = 1;
    output->dim[2] = 1;
    output->dim[3] = size;
    output->dim_count = 4;
    output->dtype = CSINN_DTYPE_UINT8;
    output->layout = CSINN_LAYOUT_NHWC;
    output->name = "output";
    output->data = (float *)ref_data;
    get_quant_info(output);
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    output->data = malloc(size);

    struct diso_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_LAYER;

    if (csi_mul_init(input0, input1, output, &params) == CSINN_TRUE) {
        csi_mul(input0, input1, output, &params);
    }

    reference->data  = (float *)ref_data;
    result_verify_8(reference->data, output, input0->data, difference, out_size, false);
    free(input0);
    free(input1);
    free(output->data);
    free(output);
    free(reference);
    free(src_tmp_0);
    free(src_tmp_1);
}


int main(int argc, char** argv)
{
    init_testsuite("Testing function of elementwise mul(u8) for i805.\n");

    verify_mul_u8(mul_input0_0, mul_input1_0, mul_output_0, 64, 1.0);
    verify_mul_u8(mul_input0_1, mul_input1_1, mul_output_1, 79, 1.0);
}
