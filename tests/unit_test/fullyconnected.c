/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
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

#include "./valid_data/fullyconnected.dat"

#include "csi_nn.h"
#include "shl_thead_rvv.h"
#include "test_utils.h"

void verify_fc_reorder(void *weight_data, void *ref_weight, void (*reorder)(), int in_nodes,
                       int out_nodes, enum csinn_dtype_enum dtype)
{
    struct csinn_tensor *weight = csinn_alloc_tensor(NULL);
    weight->dim[0] = out_nodes;
    weight->dim[1] = in_nodes;
    weight->dim_count = 2;
    weight->name = "weight";
    int weight_size = csinn_tensor_size(weight);

    weight->data = weight_data;

    reorder(weight);
    evaluate_error(weight->data, ref_weight, weight_size, dtype);

    csinn_free_tensor(weight);
}

void verify_fc_compute(void *input_data, void *weight_data, void *bias_data, void *ref_data,
                       int (*compute)(), int in_nodes, int out_nodes, enum csinn_dtype_enum dtype)
{
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = in_nodes;
    input->dim_count = 2;
    input->name = "input";
    int in_size = csinn_tensor_size(input);

    struct csinn_tensor *weight = csinn_alloc_tensor(NULL);
    weight->dim[0] = out_nodes;
    weight->dim[1] = in_nodes;
    weight->dim_count = 2;
    weight->name = "weight";
    int weight_size = csinn_tensor_size(weight);

    struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
    bias->dim[0] = out_nodes;
    bias->dim_count = 1;
    bias->name = "bias";

    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = out_nodes;
    output->dim_count = 2;
    output->name = "output";
    int out_size = csinn_tensor_size(output);

    struct csinn_fc_params *params = csinn_alloc_params(sizeof(struct csinn_fc_params), NULL);
    params->base.name = "params";

    input->data = input_data;
    weight->data = weight_data;
    bias->data = bias_data;
    output->data = shl_mem_alloc(out_size * sizeof(float));

    compute(input, output, weight, bias, params);
    evaluate_error(output->data, ref_data, out_size, dtype);

    csinn_free_tensor(input);
    shl_mem_free(output->data);
    csinn_free_tensor(output);
    csinn_free_tensor(weight);
    csinn_free_tensor(bias);
}

int main(int argc, char **argv)
{
    init_testsuite("Test function of fullyconnected for RVV.\n");

    verify_fc_reorder(fc_fp32_weight, fc_fp32_weight_ref, shl_rvv_fc_gemv_transform_weight_fp32, 17,
                      31, CSINN_DTYPE_FLOAT32);
    verify_fc_compute(fc_fp32_in, fc_fp32_weight_ref, fc_fp32_bias, fc_fp32_out,
                      shl_rvv_fullyconnected_packn_fp32, 17, 31, CSINN_DTYPE_FLOAT32);

    verify_fc_reorder(fc_fp16_weight, fc_fp16_weight_ref, shl_rvv_fc_gemv_transform_weight_fp16, 17,
                      31, CSINN_DTYPE_FLOAT16);
    verify_fc_compute(fc_fp16_in, fc_fp16_weight_ref, fc_fp16_bias, fc_fp16_out,
                      shl_rvv_fullyconnected_packn_fp16, 17, 31, CSINN_DTYPE_FLOAT16);

    // verify_fc_reorder(fc_int8_weight, fc_int8_weight_ref,
    //                   shl_rvv_fc_gemv_transform_weight_int8,
    //                   17, 31, CSINN_DTYPE_INT8);
    // verify_fc_compute(fc_int8_in, fc_int8_weight_ref, fc_int8_bias, fc_int8_out,
    //                   shl_rvv_fullyconnected_packn_int8, 17, 31, CSINN_DTYPE_INT8);

    return done_testing();
}
