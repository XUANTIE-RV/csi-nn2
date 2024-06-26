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

#include "rvv/rvv.h"

int shl_rvv_fullyconnected_init_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *weights, struct csinn_tensor *bias,
                                     struct csinn_fc_params *params)
{
    const int weights_dims_count = weights->dim_count;
    const int out_nodes = weights->dim[weights_dims_count - 2];
    const int in_nodes = weights->dim[weights_dims_count - 1];
    struct csinn_callback *cb = params->base.cb;

    struct csinn_session *sess = params->base.sess;
    bool binary_model_op_init = shl_rvv_get_binary_model_op_init(sess);
    if (!binary_model_op_init) {
        shl_rvv_fc_gemm_reorder_weight_fp32(weights);
    }
    cb->exec = shl_rvv_fullyconnected_gemm_fp32;

    return CSINN_TRUE;
}
