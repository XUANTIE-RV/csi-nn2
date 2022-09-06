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

#include "shl_c908.h"

int shl_c908_fullyconnected_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *weights, struct csinn_tensor *bias,
                                 struct csinn_fc_params *params)
{
    const int weights_dims_count = weights->dim_count;
    const int out_nodes = weights->dim[weights_dims_count - 2];
    const int in_nodes = weights->dim[weights_dims_count - 1];
    struct csinn_callback *cb = params->base.cb;
    if (input->dtype == CSINN_DTYPE_FLOAT32) {
        shl_rvv_fc_gemv_transform_weight_fp32(weights);
        cb->exec = shl_rvv_fullyconnected_packn_fp32;
    } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
        shl_rvv_fc_gemv_transform_weight_fp16(weights);
        cb->exec = shl_rvv_fullyconnected_packn_fp16;
    } else if (input->dtype == CSINN_DTYPE_INT8) {
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

        // support channel quantization
        for (int i = 0; i < weights->quant_channel; i++) {
            float real_scale = input->qinfo->scale * weights->qinfo[i].scale / output->qinfo->scale;
            shl_quantize_multiplier(real_scale, &(weights->qinfo[i].multiplier),
                                    &(weights->qinfo[i].shift));
        }
        if (in_nodes % 4 == 0) {
            shl_rvv_fc_gemv_transform_weight_int8_dot(weights);
            cb->exec = shl_rvv_fullyconnected_packn_int8_dot;
        } else {
            shl_rvv_fc_gemv_transform_weight_int8(weights);
            cb->exec = shl_rvv_fullyconnected_packn_int8;
        }
    } else if (input->dtype == CSINN_DTYPE_INT4) {
        // support channel quantization
        for (int i = 0; i < weights->quant_channel; i++) {
            float real_scale = input->qinfo->scale * weights->qinfo[i].scale / output->qinfo->scale;
            shl_quantize_multiplier(real_scale, &(weights->qinfo[i].multiplier),
                                    &(weights->qinfo[i].shift));
        }
        if (in_nodes % 8 == 0) {
            shl_rvv_fc_gemv_transform_weight_int4_dot(weights);
            cb->exec = shl_rvv_fullyconnected_packn_int4_dot;
        } else {
            shl_debug_warning("fc is not optimized for int4, call reference func replaced.\n");
            cb->exec = shl_ref_fullyconnected_quant;
        }
    }
    return CSINN_TRUE;
}
