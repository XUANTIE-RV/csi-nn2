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

#include "c920/c920.h"

int shl_c920_fullyconnected_gemm_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *weights, struct csinn_tensor *bias,
                                      struct csinn_fc_params *params)
{
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp32(input);
    }

    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *weights_data = (float *)weights->data;
    float *bias_data = (float *)bias->data;
    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    const int bias_dims_count = bias->dim_count;
    int batches = 1;
    /* compute the outer size */
    for (int i = 0; i < output_dims_count - 1; i++) {
        batches *= output->dim[i];
    }
    int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    int m = batches;
    int n = output_depth;
    int k = accum_depth;

    float *input_reorder = (float *)shl_mem_alloc(m * k * sizeof(float));
    shl_c920_reorder_a_block_8xk_fp32(input_data, input_reorder, m, k, m, k);
    shl_c920_gemm_a0b1_8xpack2n_fp32(output_data, input_reorder, weights_data, bias_data, m, k, n);

    shl_mem_free(input_reorder);
    return CSINN_TRUE;
}

int shl_c920_fullyconnected_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *weights, struct csinn_tensor *bias,
                                      struct csinn_fc_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    struct csinn_session *sess = params->base.sess;
    bool binary_model_op_init = shl_c920_get_binary_model_op_init(sess);
    if (!binary_model_op_init) {
        shl_rvv_fc_gemm_reorder_weight_fp32(weights);
    }
    cb->exec = shl_c920_fullyconnected_gemm_fp32;
    return CSINN_TRUE;
}
