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

#include "./valid_data/activation.dat"
#include "csi_nn.h"
#include "math_snr.h"
#include "shl_thead_rvv.h"
#include "test_utils.h"

void verify_leaky_relu(void *input_data, void *ref_data, int (*func)(), int in_c, int in_h,
                       int in_w, float alpha, enum csinn_dtype_enum dtype)
{
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = in_c;
    input->dim[2] = in_h;
    input->dim[3] = in_w;
    input->dim_count = 4;
    input->name = "input";
    int in_size = csinn_tensor_size(input);

    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = in_c;
    output->dim[2] = in_h;
    output->dim[3] = in_w;
    output->dim_count = 4;
    output->name = "output";
    int out_size = csinn_tensor_size(output);

    struct csinn_relu_params *params = csinn_alloc_params(sizeof(struct csinn_relu_params), NULL);
    params->base.name = "params";
    params->n = alpha;

    input->data = input_data;
    output->data = shl_mem_alloc(out_size * sizeof(float));

    func(input, output, params);

    evaluate_error(output->data, ref_data, out_size, dtype);

    csinn_free_tensor(input);
    shl_mem_free(output->data);
    csinn_free_tensor(output);
}

int main(int argc, char **argv)
{
    init_testsuite("Test function of leaky_relu for RVV.\n");
    verify_leaky_relu(leaky_relu_fp32_in, leaky_relu_fp32_out, shl_rvv_leaky_relu_fp32, 2, 5, 11,
                      0.2, CSINN_DTYPE_FLOAT32);
    verify_leaky_relu(leaky_relu_fp16_in, leaky_relu_fp16_out, shl_rvv_leaky_relu_fp16, 2, 5, 11,
                      0.2, CSINN_DTYPE_FLOAT16);
    // verify_leaky_relu(leaky_relu_int8_in, leaky_relu_int8_out, shl_rvv_leaky_relu_int8, 2, 5,
    //                   11, 0.2, CSINN_DTYPE_INT8);

    return done_testing();
}
