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

int shl_c920_fullyconnected_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *weights, struct csinn_tensor *bias,
                                      struct csinn_fc_params *params)
{
    if (input->layout >= CSINN_LAYOUT_NC1C0 && input->layout <= CSINN_LAYOUT_NC1DHWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(input);
    }

    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *weights_data = NULL;
    __fp16 *bias_data = (__fp16 *)bias->data;
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

    __fp16 *weights_fp16 = NULL;
    if (weights->is_const && weights->dtype == CSINN_DTYPE_INT8) {
        int size = csinn_tensor_size(weights);
        int8_t *weights_int8 = (int8_t *)weights->data;
        weights_fp16 = (__fp16 *)shl_mem_alloc(size * sizeof(__fp16));
        if (weights->quant_channel == 1) {
            int32_t zp = weights->qinfo->zero_point;
            float scale = weights->qinfo->scale;
            shl_rvv_dequantize_i8_to_f16(weights_int8, weights_fp16, size, zp, scale);
        } else if (weights->quant_channel == output_depth) {
            shl_rvv_fc_npack2n_dequantize_per_channel_i8_to_f16(weights, params, weights_fp16);
        } else {
            shl_debug_error("%s unsupported quant_channel: %d\n", __func__, weights->quant_channel);
        }
        weights_data = weights_fp16;
    } else if (weights->dtype == CSINN_DTYPE_FLOAT16) {
        weights_data = (__fp16 *)weights->data;
    } else {
        shl_debug_error("weights unsupport dtype: %d\n", weights->dtype);
        return CSINN_FALSE;
    }

    __fp16 *input_reorder = (__fp16 *)shl_mem_alloc(m * k * sizeof(__fp16));
    shl_c920_reorder_a_block_8xk_fp16(input_data, input_reorder, m, k, m, k);
    shl_c920_gemm_a0b1_8xpack2n_fp16(output_data, input_reorder, weights_data, bias_data, m, k, n);

    shl_mem_free(input_reorder);
    if (weights->is_const && weights->dtype == CSINN_DTYPE_INT8) {
        shl_mem_free(weights_fp16);
        return CSINN_TRUE;
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, weights);
    return CSINN_TRUE;
}

int shl_c920_fullyconnected_init_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *weights, struct csinn_tensor *bias,
                                      struct csinn_fc_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    struct csinn_session *sess = params->base.sess;
    bool binary_model_op_init = shl_c920_get_binary_model_op_init(sess);
    if (!binary_model_op_init) {
        if (weights->is_const && weights->dtype == CSINN_DTYPE_INT8) {
            shl_rvv_fc_gemm_reorder_weight_fp16_w_int8(weights);
        } else if (weights->dtype == CSINN_DTYPE_FLOAT16) {
            shl_rvv_fc_gemm_reorder_weight_fp16(weights);
        } else {
            shl_debug_error("weights unsupport dtype: %d\n", weights->dtype);
            return CSINN_FALSE;
        }
    }
    cb->exec = shl_c920_fullyconnected_gemm_fp16;
    return CSINN_TRUE;
}
