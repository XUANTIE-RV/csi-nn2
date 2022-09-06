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

#include "shl_thead_rvv.h"

#define RVV_OP_PATTERN_MAX 80
static struct csinn_callback __rvv_cb_table[RVV_OP_PATTERN_MAX];
static int __rvv_cb_key[RVV_OP_PATTERN_MAX];

void shl_rvv_reg_op(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *init, void *exec,
                    void *est)
{
    static int i = 0;
    __rvv_cb_key[i] = op_name * CSINN_DTYPE_SIZE + dtype;
    __rvv_cb_table[i].init = init;
    __rvv_cb_table[i].exec = exec;
    __rvv_cb_table[i].est = est;
    i++;
}

struct csinn_callback *shl_cb_map_ref(int op, int dtype);
struct csinn_callback *shl_cb_map_rvv(int op, int dtype)
{
    struct csinn_callback *cb = NULL;
    for (int i = 0; i < RVV_OP_PATTERN_MAX; i++) {
        if (__rvv_cb_key[i] == (op * CSINN_DTYPE_SIZE + dtype)) {
            cb = &__rvv_cb_table[i];
            break;
        }
    }
    if ((cb == NULL) || (cb->est == NULL && (cb->init == NULL || cb->exec == NULL))) {
        cb = shl_cb_map_ref(op, dtype);
    }
    return cb;
}

void shl_target_init_rvv()
{
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_rvv_conv2d_init_fp32, NULL,
                   shl_gref_conv2d);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_rvv_conv2d_init_fp16, NULL,
                   shl_gref_conv2d);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D, shl_rvv_conv2d_init_int8, NULL,
                   shl_gref_conv2d);
    shl_rvv_reg_op(CSINN_DTYPE_INT4, CSINN_OP_CONV2D, shl_rvv_conv2d_init_int4, NULL,
                   shl_gref_conv2d);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, shl_rvv_conv2d_init_fp32, NULL,
                   shl_gref_conv2d);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, shl_rvv_conv2d_init_fp16, NULL,
                   shl_gref_conv2d);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_GROUP_CONV2D, shl_rvv_conv2d_init_int8, NULL,
                   shl_gref_conv2d);
    shl_rvv_reg_op(CSINN_DTYPE_INT4, CSINN_OP_GROUP_CONV2D, shl_rvv_conv2d_init_int4, NULL,
                   shl_gref_conv2d);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTHWISE_CONV2D,
                   shl_rvv_depthwise_conv2d_init_fp32, NULL, shl_gref_depthwise_conv2d);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D,
                   shl_rvv_depthwise_conv2d_init_fp16, NULL, shl_gref_depthwise_conv2d);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D, shl_rvv_depthwise_conv2d_init_int8,
                   NULL, shl_gref_depthwise_conv2d);
    shl_rvv_reg_op(CSINN_DTYPE_INT4, CSINN_OP_DEPTHWISE_CONV2D, shl_rvv_depthwise_conv2d_init_int4,
                   NULL, shl_gref_depthwise_conv2d);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXPOOL2D, shl_rvv_maxpool2d_init_fp32, NULL,
                   shl_gref_maxpool2d);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL2D, shl_rvv_maxpool2d_init_fp16, NULL,
                   shl_gref_maxpool2d);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_MAXPOOL2D, shl_rvv_maxpool2d_init_int8, NULL,
                   shl_gref_maxpool2d);
    shl_rvv_reg_op(CSINN_DTYPE_INT4, CSINN_OP_MAXPOOL2D, shl_rvv_maxpool2d_init_int4, NULL,
                   shl_gref_maxpool2d);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_AVGPOOL2D, shl_rvv_avgpool2d_init_fp32, NULL,
                   shl_gref_avgpool2d);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_AVGPOOL2D, shl_rvv_avgpool2d_init_fp16, NULL,
                   shl_gref_avgpool2d);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_AVGPOOL2D, shl_rvv_avgpool2d_init_int8, NULL,
                   shl_gref_avgpool2d);
    shl_rvv_reg_op(CSINN_DTYPE_INT4, CSINN_OP_AVGPOOL2D, shl_rvv_avgpool2d_init_int4, NULL,
                   shl_gref_avgpool2d);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_FULLYCONNECTED, shl_rvv_fullyconnected_init, NULL,
                   shl_gref_fullyconnected);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED, shl_rvv_fullyconnected_init, NULL,
                   shl_gref_fullyconnected);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_FULLYCONNECTED, shl_rvv_fullyconnected_init, NULL,
                   shl_gref_fullyconnected);
    shl_rvv_reg_op(CSINN_DTYPE_INT4, CSINN_OP_FULLYCONNECTED, shl_rvv_fullyconnected_init, NULL,
                   shl_gref_fullyconnected);

    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D_RELU, shl_rvv_conv2d_init_int8, NULL,
                   shl_gref_conv2d_relu);
    shl_rvv_reg_op(CSINN_DTYPE_INT4, CSINN_OP_CONV2D_RELU, shl_rvv_conv2d_init_int4, NULL,
                   shl_gref_conv2d_relu);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D_RELU,
                   shl_rvv_depthwise_conv2d_init_int8, NULL, shl_gref_depthwise_conv2d_relu);
    shl_rvv_reg_op(CSINN_DTYPE_INT4, CSINN_OP_DEPTHWISE_CONV2D_RELU,
                   shl_rvv_depthwise_conv2d_init_int4, NULL, shl_gref_depthwise_conv2d_relu);

    shl_rvv_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ADD, NULL, shl_rvv_add_fp32, shl_gref_add);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ADD, NULL, shl_rvv_add_fp16, shl_gref_add);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_ADD, NULL, shl_rvv_add_int8, shl_gref_add);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MUL, NULL, shl_rvv_mul_fp32, shl_gref_mul);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MUL, NULL, shl_rvv_mul_fp16, shl_gref_mul);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_MUL, NULL, shl_rvv_mul_int8, shl_gref_mul);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONCAT, NULL, shl_rvv_concat_fp32,
                   shl_gref_concat);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONCAT, NULL, shl_rvv_concat_fp16,
                   shl_gref_concat);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONCAT, NULL, shl_rvv_concat_int8, shl_gref_concat);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LEAKY_RELU, NULL, shl_rvv_leaky_relu_fp32,
                   shl_gref_leaky_relu);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LEAKY_RELU, NULL, shl_rvv_leaky_relu_fp16,
                   shl_gref_leaky_relu);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_LEAKY_RELU, NULL, shl_rvv_leaky_relu_int8,
                   shl_gref_leaky_relu);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU, NULL, shl_rvv_relu_fp32, shl_gref_relu);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU, NULL, shl_rvv_relu_fp16, shl_gref_relu);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_RELU, NULL, shl_rvv_relu_int8, shl_gref_relu);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU6, NULL, shl_rvv_relu6_fp32, shl_gref_relu6);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU6, NULL, shl_rvv_relu6_fp16, shl_gref_relu6);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_RELU6, NULL, shl_rvv_relu6_int8, shl_gref_relu6);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_AVGPOOL2D, shl_rvv_global_avgpool2d_init,
                   NULL, shl_gref_global_avgpool2d);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_AVGPOOL2D, shl_rvv_global_avgpool2d_init,
                   NULL, shl_gref_global_avgpool2d);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_GLOBAL_AVGPOOL2D, shl_rvv_global_avgpool2d_init, NULL,
                   shl_gref_global_avgpool2d);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SIGMOID, NULL, shl_rvv_sigmoid_fp16,
                   shl_gref_sigmoid);
    shl_rvv_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SOFTMAX, NULL, shl_rvv_softmax_fp16,
                   shl_gref_softmax);
    shl_rvv_reg_op(CSINN_DTYPE_INT8, CSINN_OP_SUM, NULL, shl_rvv_sum_stride_int8, shl_gref_sum);

    shl_register_runtime_callback(CSINN_RVV, NULL);
    shl_register_op_callback(CSINN_RVV, shl_cb_map_rvv);
    shl_register_runtime_callback(CSINN_RVV, shl_gref_runtime_callback);
}
