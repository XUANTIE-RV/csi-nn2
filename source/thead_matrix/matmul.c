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

#include "rvm/rvm.h"

int shl_rvm_matmul_init_fp16(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                             struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    struct csinn_session *sess = params->base.sess;
    bool binary_model_op_init = shl_rvm_get_binary_model_op_init(sess);
    if (!params->trans_a && !params->trans_b) {
        if (mat0->dtype == CSINN_DTYPE_FLOAT16) {
            if (mat1->is_const && mat1->dtype == CSINN_DTYPE_INT8) {
                if (!binary_model_op_init) {
                    shl_rvm_matmul_reorder_weight_fp16_w_int8(mat1);
                }
                cb->exec = shl_rvm_matmul_fp16_w_int8;
            } else if (mat1->dtype == CSINN_DTYPE_FLOAT16) {
                if (mat1->is_const && !binary_model_op_init) {
                    shl_rvm_matmul_reorder_weight_fp16(mat1);
                }
                cb->exec = shl_rvm_matmul_fp16;
            }
        }
    }
    if (cb->exec == NULL) {
        shl_debug_warning(
            "matmul is not optimized to achieve under this condition, call reference func "
            "replaced.\n");
        cb->exec = shl_ref_matmul_quant;
    }
    return CSINN_TRUE;
}

int shl_rvm_matmul_init_int8(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                             struct csinn_tensor *output, struct csinn_matmul_params *params)
{
    struct csinn_callback *cb = params->base.cb;
    struct csinn_session *sess = params->base.sess;
    bool binary_model_op_init = shl_rvm_get_binary_model_op_init(sess);
    if (!params->trans_a && !params->trans_b) {
        if (mat0->dtype == CSINN_DTYPE_INT8 && mat1->dtype == CSINN_DTYPE_INT8) {
            if (mat1->is_const && !binary_model_op_init) {
                shl_rvm_matmul_reorder_weight_int8(mat1);
            }
            cb->exec = shl_rvm_matmul_int8;
        }
    }
    if (cb->exec == NULL) {
        shl_debug_warning(
            "matmul is not optimized to achieve under this condition, call reference func "
            "replaced.\n");
        cb->exec = shl_ref_matmul_quant;
    }
    return CSINN_TRUE;
}
