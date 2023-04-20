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

/* SHL version 2.1.x */

#include "shl_thead_rvv.h"

int shl_rvv_fullyconnected_init_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params)
{
    const int weights_dims_count = weights->dim_count;
    const int out_nodes = weights->dim[weights_dims_count - 2];
    const int in_nodes = weights->dim[weights_dims_count - 1];
    struct csinn_callback *cb = params->base.cb;

    if (params->base.quant_type != CSINN_QUANT_INT8_ASYM_W_SYM) {
        cb->exec = shl_ref_fullyconnected_quant;
        return CSINN_TRUE;
    }

    // enable fuse zeropoint to bias
    if (!params->fc_extra.fuse_zp2bias) {
        int32_t *bias_data = (int32_t *)bias->data;
        int8_t *weights_data = (int8_t *)weights->data;
        int32_t input_zp = input->qinfo->zero_point;

        if (bias_data == NULL) {
            // XXX: memory leak
            bias_data = (int32_t *)shl_mem_alloc(out_nodes * sizeof(int32_t));
            bias->data = bias_data;
        }
        for (int oc = 0; oc < out_nodes; oc++) {
            int32_t tmp = 0;
            for (int j = 0; j < in_nodes; j++) {
                tmp += weights_data[oc * in_nodes + j] * input_zp;
            }
            bias_data[oc] -= tmp;
        }
    }

    shl_rvv_fc_gemm_transform_weight_int8(weights);
    // support channel quantization
    for (int i = 0; i < weights->quant_channel; i++) {
        float real_scale = input->qinfo->scale * weights->qinfo[i].scale / output->qinfo->scale;
        shl_quantize_multiplier(real_scale, &(weights->qinfo[i].multiplier),
                                &(weights->qinfo[i].shift));
    }
    cb->exec = shl_rvv_fullyconnected_gemm_int8;

    return CSINN_TRUE;
}
