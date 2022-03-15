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

/* CSI-NN2 version 1.13.x */

#include "./valid_data/basic_math.dat"
#include "csi_nn.h"
#include "csi_thead_rvv.h"
#include "math_snr.h"
#include "test_utils.h"

void verify_add(void *input0_data, void *input1_data, void *ref_data, int (*func)(), int in_c,
                int in_h, int in_w, enum csinn_dtype_enum dtype)
{
    struct csi_tensor *input0 = csi_alloc_tensor(NULL);
    input0->dim[0] = 1;
    input0->dim[1] = in_c;
    input0->dim[2] = in_h;
    input0->dim[3] = in_w;
    input0->dim_count = 4;
    input0->name = "input0";
    int in0_size = csi_tensor_size(input0);

    struct csi_tensor *input1 = csi_alloc_tensor(NULL);
    input1->dim[0] = 1;
    input1->dim[1] = in_c;
    input1->dim[2] = in_h;
    input1->dim[3] = in_w;
    input1->dim_count = 4;
    input1->name = "input1";
    int in1_size = csi_tensor_size(input1);

    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = in_c;
    output->dim[2] = in_h;
    output->dim[3] = in_w;
    output->dim_count = 4;
    output->name = "output";
    int out_size = csi_tensor_size(output);

    struct diso_params params;
    params.base.name = "params";

    input0->data = input0_data;
    input1->data = input1_data;
    output->data = csi_mem_alloc(out_size * sizeof(float));

    func(input0, input1, output, &params);

    evaluate_error(output->data, ref_data, out_size, dtype);

    csi_free_tensor(input0);
    csi_free_tensor(input1);
    csi_mem_free(output->data);
    csi_free_tensor(output);
}

int main(int argc, char **argv)
{
    init_testsuite("Test function of add for RVV.\n");
    verify_add(add_fp32_in0, add_fp32_in1, add_fp32_out, csi_nn_rvv_add_fp32, 2, 5, 11,
               CSINN_DTYPE_FLOAT32);
    verify_add(add_fp16_in0, add_fp16_in1, add_fp16_out, csi_nn_rvv_add_fp16, 2, 5, 11,
               CSINN_DTYPE_FLOAT16);
    // verify_add(add_int8_in0, add_int8_in1, add_int8_out, csi_nn_rvv_add_int8, 2, 5, 11,
    //            CSINN_DTYPE_INT8);
    return done_testing();
}
