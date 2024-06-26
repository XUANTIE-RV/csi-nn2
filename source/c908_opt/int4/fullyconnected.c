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

#include "c908/c908.h"

int shl_c908_fullyconnected_init_int4(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *weights, struct csinn_tensor *bias,
                                      struct csinn_fc_params *params)
{
    const int weights_dims_count = weights->dim_count;
    const int out_nodes = weights->dim[weights_dims_count - 2];
    const int in_nodes = weights->dim[weights_dims_count - 1];
    struct csinn_callback *cb = params->base.cb;
#ifdef SHL_USE_DOT_INT4
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
#endif
    return CSINN_TRUE;
}
