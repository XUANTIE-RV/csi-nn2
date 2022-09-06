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

#define C908_OP_PATTERN_MAX 60
static struct csinn_callback __c908_cb_table[C908_OP_PATTERN_MAX];
static int __c908_cb_key[C908_OP_PATTERN_MAX];

void shl_c908_reg_op(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *init,
                     void *exec, void *est)
{
    static int i = 0;
    __c908_cb_key[i] = op_name * CSINN_DTYPE_SIZE + dtype;
    __c908_cb_table[i].init = init;
    __c908_cb_table[i].exec = exec;
    __c908_cb_table[i].est = est;
    i++;
}

struct csinn_callback *shl_cb_map_rvv(int op, int dtype);
struct csinn_callback *shl_cb_map_c908(int op, int dtype)
{
    struct csinn_callback *cb = NULL;
    for (int i = 0; i < C908_OP_PATTERN_MAX; i++) {
        if (__c908_cb_key[i] == (op * CSINN_DTYPE_SIZE + dtype)) {
            cb = &__c908_cb_table[i];
            break;
        }
    }
    if ((cb == NULL) || (cb->est == NULL && (cb->init == NULL || cb->exec == NULL))) {
        cb = shl_cb_map_rvv(op, dtype);
    }
    return cb;
}

void shl_target_init_c908()
{
    shl_c908_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_c908_conv2d_init_fp32, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_c908_conv2d_init_fp16, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D, shl_c908_conv2d_init_int8, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_CONV2D, shl_c908_conv2d_init_int4, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, shl_c908_conv2d_init_fp32, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, shl_c908_conv2d_init_fp16, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_GROUP_CONV2D, shl_c908_conv2d_init_int8, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_GROUP_CONV2D, shl_c908_conv2d_init_int4, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTHWISE_CONV2D,
                    shl_c908_depthwise_conv2d_init_fp32, NULL, shl_gref_depthwise_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D,
                    shl_c908_depthwise_conv2d_init_fp16, NULL, shl_gref_depthwise_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D,
                    shl_c908_depthwise_conv2d_init_int8, NULL, shl_gref_depthwise_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_DEPTHWISE_CONV2D,
                    shl_c908_depthwise_conv2d_init_int4, NULL, shl_gref_depthwise_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXPOOL2D, shl_c908_maxpool2d_init_fp32, NULL,
                    shl_gref_maxpool2d);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL2D, shl_c908_maxpool2d_init_fp16, NULL,
                    shl_gref_maxpool2d);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_MAXPOOL2D, shl_c908_maxpool2d_init_int8, NULL,
                    shl_gref_maxpool2d);
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_MAXPOOL2D, shl_c908_maxpool2d_init_int4, NULL,
                    shl_gref_maxpool2d);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_AVGPOOL2D, shl_c908_avgpool2d_init_fp32, NULL,
                    shl_gref_avgpool2d);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_AVGPOOL2D, shl_c908_avgpool2d_init_fp16, NULL,
                    shl_gref_avgpool2d);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_AVGPOOL2D, shl_c908_avgpool2d_init_int8, NULL,
                    shl_gref_avgpool2d);
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_AVGPOOL2D, shl_c908_avgpool2d_init_int4, NULL,
                    shl_gref_avgpool2d);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_FULLYCONNECTED, shl_c908_fullyconnected_init,
                    NULL, shl_gref_fullyconnected);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED, shl_c908_fullyconnected_init,
                    NULL, shl_gref_fullyconnected);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_FULLYCONNECTED, shl_c908_fullyconnected_init, NULL,
                    shl_gref_fullyconnected);
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_FULLYCONNECTED, shl_c908_fullyconnected_init, NULL,
                    shl_gref_fullyconnected);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DATA_CONVERT, shl_rvv_data_convert_init, NULL,
                    shl_gref_data_convert);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DATA_CONVERT, shl_rvv_data_convert_init, NULL,
                    shl_gref_data_convert);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DATA_CONVERT, shl_rvv_data_convert_init, NULL,
                    shl_gref_data_convert);
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_DATA_CONVERT, shl_rvv_data_convert_init, NULL,
                    shl_gref_data_convert);

    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D_RELU, shl_c908_conv2d_init_int8, NULL,
                    shl_gref_conv2d_relu);
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_CONV2D_RELU, shl_c908_conv2d_init_int4, NULL,
                    shl_gref_conv2d_relu);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D_RELU,
                    shl_c908_depthwise_conv2d_init_int8, NULL, shl_gref_depthwise_conv2d_relu);
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_DEPTHWISE_CONV2D_RELU,
                    shl_c908_depthwise_conv2d_init_int4, NULL, shl_gref_depthwise_conv2d_relu);

    shl_register_runtime_callback(CSINN_C908, NULL);

    shl_register_op_callback(CSINN_C908, shl_cb_map_c908);
    shl_register_runtime_callback(CSINN_C908, shl_gref_runtime_callback);
}
