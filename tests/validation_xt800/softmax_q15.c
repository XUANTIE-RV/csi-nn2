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
#include "./valid_data/softmax_data.dat"


static void verify_softmax_q15(void *input_data,
                               void *ref_data,
                               int32_t size,
                               float difference)
{
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size, out_size;

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    input->dim[0] = size;
    input->dim_count = 1;
    input->dtype = CSINN_DTYPE_INT16;
    input->name = "input";
    in_size = input->dim[0];

    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim[0] = input->dim[0];
    output->dim_count = 1;
    output->dtype = CSINN_DTYPE_INT16;
    output->name = "output";
    out_size = output->dim[0];

    struct softmax_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_LAYER;

    input->data      = (uint16_t *)input_data;
    reference->data  = (uint16_t *)ref_data;

    if (csi_softmax_init(input, output, &params) == CSINN_TRUE) {
        csi_softmax(input, output, &params);
    }
    result_verify_q15(reference->data, output->data, input->data, difference, out_size, false);
    free(input);
    free(output);
    free(reference);
}      


int main(int argc, char** argv)
{
    init_testsuite("Testing function of softmax q15 for xt800.\n");

    verify_softmax_q15(q15_softmax_input0, q15_softmax_result0, 32, 0.0f);
    verify_softmax_q15(q15_softmax_input1, q15_softmax_result1, 32, 0.0f);
    verify_softmax_q15(q15_softmax_input2, q15_softmax_result2, 32, 0.0f);
    verify_softmax_q15(q15_softmax_input3, q15_softmax_result3, 32, 0.0f);
    verify_softmax_q15(q15_softmax_input4, q15_softmax_result4, 32, 0.0f);

    verify_softmax_q15(q15_relu_input0, q15_softmax_result5, 1024, 0.0f);
    verify_softmax_q15(q15_relu_input1, q15_softmax_result6, 1024, 0.0f);
    verify_softmax_q15(q15_relu_input2, q15_softmax_result7, 1024, 0.0f);
    verify_softmax_q15(q15_relu_input3, q15_softmax_result8, 1024, 0.0f);
    verify_softmax_q15(q15_relu_input4, q15_softmax_result9, 1024, 0.0f);

    verify_softmax_q15(q15_softmax_input0, q15_softmax_result10, 31, 0.0f);
    verify_softmax_q15(q15_softmax_input1, q15_softmax_result11, 31, 0.0f);
    verify_softmax_q15(q15_softmax_input2, q15_softmax_result12, 31, 0.0f);
    verify_softmax_q15(q15_softmax_input3, q15_softmax_result13, 31, 0.0f);
    verify_softmax_q15(q15_softmax_input4, q15_softmax_result14, 31, 0.0f);

    verify_softmax_q15(q15_relu_input0, q15_softmax_result15, 1023, 0.0f);
    verify_softmax_q15(q15_relu_input1, q15_softmax_result16, 1023, 0.0f);
    verify_softmax_q15(q15_relu_input2, q15_softmax_result17, 1023, 0.0f);
    verify_softmax_q15(q15_relu_input3, q15_softmax_result18, 1023, 0.0f);
    verify_softmax_q15(q15_relu_input4, q15_softmax_result19, 1023, 0.0f);
}
