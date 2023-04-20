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

#include "shl_thead_rvm.h"

#define RVM_OP_PATTERN_MAX 60
static struct shl_cb_table shl_rvm_cb_table[RVM_OP_PATTERN_MAX];

void shl_rvm_reg_op(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *init, void *exec,
                    void *est)
{
    static int i = 0;
    shl_rvm_cb_table[i].shl_cb_key = op_name * CSINN_DTYPE_SIZE + dtype;
    shl_rvm_cb_table[i].shl_cb_value.init = init;
    shl_rvm_cb_table[i].shl_cb_value.exec = exec;
    shl_rvm_cb_table[i].shl_cb_value.est = est;
    i++;
}

struct csinn_callback *shl_cb_map_rvv(int op, int dtype);
struct csinn_callback *shl_cb_map_rvm(int op, int dtype)
{
    struct csinn_callback *cb = NULL;
    for (int i = 0; i < RVM_OP_PATTERN_MAX; i++) {
        if (shl_rvm_cb_table[i].shl_cb_key == (op * CSINN_DTYPE_SIZE + dtype)) {
            cb = &(shl_rvm_cb_table[i].shl_cb_value);
            break;
        }
    }
    if ((cb == NULL) || (cb->est == NULL && (cb->init == NULL || cb->exec == NULL))) {
        cb = shl_cb_map_rvv(op, dtype);
    }
    return cb;
}

void shl_target_init_rvm()
{
    shl_rvm_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_rvm_conv2d_init_fp16, NULL,
                   shl_gref_conv2d);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D, shl_rvm_conv2d_init_int8, NULL,
                   shl_gref_conv2d);

    shl_rvm_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D,
                   shl_rvm_depthwise_conv2d_init_fp16, NULL, shl_gref_depthwise_conv2d);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D, shl_rvm_depthwise_conv2d_init_int8,
                   NULL, shl_gref_depthwise_conv2d);

    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D_RELU, shl_rvm_conv2d_init_int8, NULL,
                   shl_gref_conv2d_relu);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D_RELU,
                   shl_rvm_depthwise_conv2d_init_int8, NULL, shl_gref_depthwise_conv2d_relu);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D_RELU6, shl_rvm_conv2d_init_int8, NULL,
                   shl_gref_conv2d_relu6);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D_RELU6,
                   shl_rvm_depthwise_conv2d_init_int8, NULL, shl_gref_depthwise_conv2d_relu6);

    shl_rvm_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL2D, shl_rvm_maxpool2d_init_fp16, NULL,
                   shl_gref_maxpool2d);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_MAXPOOL2D, shl_rvm_maxpool2d_init_int8, NULL,
                   shl_gref_maxpool2d);
    shl_rvm_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_AVGPOOL2D, shl_rvm_avgpool2d_init_fp16, NULL,
                   shl_gref_avgpool2d);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_AVGPOOL2D, shl_rvm_avgpool2d_init_int8, NULL,
                   shl_gref_avgpool2d);

    shl_rvm_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_AVGPOOL2D, shl_rvm_global_avgpool2d_init,
                   NULL, shl_gref_global_avgpool2d);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_GLOBAL_AVGPOOL2D, shl_rvm_global_avgpool2d_init, NULL,
                   shl_gref_global_avgpool2d);
    shl_rvm_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_MAXPOOL2D, shl_rvm_global_maxpool2d_init,
                   NULL, shl_gref_global_maxpool2d);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_GLOBAL_MAXPOOL2D, shl_rvm_global_maxpool2d_init, NULL,
                   shl_gref_global_maxpool2d);

    shl_register_op_callback(CSINN_RVM, shl_cb_map_rvm);
    shl_register_runtime_callback(CSINN_RVM, shl_gref_runtime_callback);
}
