/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

#include "e907/e907.h"

static void shl_e907_fullyconnectd_int8_internel(const int8_t *input, int32_t *output,
                                                 int8_t *weight, const int32_t *bias, int in_nodes,
                                                 int out_nodes)
{
    const int xlenb = shl_rvp_get_xlenb();  // xlen in byte
    const int xlenw = xlenb >> 2;           // xlen in word
    for (int i = 0; i < out_nodes; i++) {
        int8_t *weight_ptr = weight + i * in_nodes;
        intXLEN_t acc = 0;
        int j = 0;
        for (; j + xlenb - 1 < in_nodes; j += xlenb) {
            intXLEN_t *input_8xn = (intXLEN_t *)(input + j);
            intXLEN_t *weight_8xn = (intXLEN_t *)(weight_ptr + j);
            acc = __rv__smaqa(acc, input_8xn[0], weight_8xn[0]);
        }

        int32_t res = bias[i];
#if __riscv_xlen == 64
        int32_t *tmp = (int32_t *)(&acc);
        res += tmp[0] + tmp[1];
#elif __riscv_xlen == 32
        res += acc;
#endif

        for (; j < in_nodes; j++) {
            res += input[j] * weight_ptr[j];
        }
        output[i] = res;
    }
}

int shl_e907_fullyconnected_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *weights, struct csinn_tensor *bias,
                                 struct csinn_fc_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *weight_data = (int8_t *)weights->data;
    int32_t *bias_data = (int32_t *)bias->data;

    const int output_dims_count = output->dim_count;
    const int weights_dims_count = weights->dim_count;
    int batches = 1;
    /* compute the outer size */
    for (int i = 0; i < output_dims_count - 1; i++) {
        batches *= output->dim[i];
    }
    const int output_depth = weights->dim[weights_dims_count - 2];  // output_nodes
    const int accum_depth = weights->dim[weights_dims_count - 1];   // input_nodes

    int32_t *output_tmp = (int32_t *)shl_mem_alloc(output_depth * sizeof(int32_t));
    for (int b = 0; b < batches; ++b) {
        int8_t *input_ptr = input_data + b * accum_depth;
        int8_t *weight_ptr = weight_data;
        int32_t *bias_ptr = bias_data;
        int32_t *output_ptr = output_tmp;

        shl_e907_fullyconnectd_int8_internel(input_ptr, output_ptr, weight_ptr, bias_ptr,
                                             accum_depth, output_depth);

        if (weights->quant_channel == 1) {
            shl_rvp_requantize(output_ptr, weights->qinfo->multiplier, weights->qinfo->shift,
                               output_depth);
        } else if (weights->quant_channel == output_depth) {
            // support channel quantization
            for (int c = 0; c < weights->quant_channel; c++) {
                shl_rvp_requantize(output_ptr + c, weights->qinfo[c].multiplier,
                                   weights->qinfo[c].shift, 1);
            }
        }
        shl_rvp_saturated_int8(output_ptr, output_data + b * output_depth,
                               output->qinfo->zero_point, output_depth);
    }

    if (output_tmp) {
        shl_mem_free(output_tmp);
        output_tmp = NULL;
    }

    return CSINN_TRUE;
}
