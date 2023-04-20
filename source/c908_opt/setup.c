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

#include "shl_c908.h"

#define C908_OP_PATTERN_MAX 60
static struct shl_cb_table shl_c908_cb_table[C908_OP_PATTERN_MAX];

void shl_c908_reg_op(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *init,
                     void *exec, void *est)
{
    static int i = 0;
    shl_c908_cb_table[i].shl_cb_key = op_name * CSINN_DTYPE_SIZE + dtype;
    shl_c908_cb_table[i].shl_cb_value.init = init;
    shl_c908_cb_table[i].shl_cb_value.exec = exec;
    shl_c908_cb_table[i].shl_cb_value.est = est;
    i++;
}

struct csinn_callback *shl_cb_map_rvv(int op, int dtype);
struct csinn_callback *shl_cb_map_c908(int op, int dtype)
{
    struct csinn_callback *cb = NULL;
    for (int i = 0; i < C908_OP_PATTERN_MAX; i++) {
        if (shl_c908_cb_table[i].shl_cb_key == (op * CSINN_DTYPE_SIZE + dtype)) {
            cb = &(shl_c908_cb_table[i].shl_cb_value);
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
#ifndef CONFIG_C908_CONVOLUTION_FP32_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_c908_conv2d_init_fp32, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, shl_c908_conv2d_init_fp32, NULL,
                    shl_gref_conv2d);
#endif
#ifndef CONFIG_C908_CONVOLUTION_FP16_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_c908_conv2d_init_fp16, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, shl_c908_conv2d_init_fp16, NULL,
                    shl_gref_conv2d);
#endif
#ifndef CONFIG_C908_CONVOLUTION_INT8_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D, shl_c908_conv2d_init_int8, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_GROUP_CONV2D, shl_c908_conv2d_init_int8, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D_RELU, shl_c908_conv2d_init_int8, NULL,
                    shl_gref_conv2d_relu);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_GROUP_CONV2D_RELU, shl_c908_conv2d_init_int8, NULL,
                    shl_gref_group_conv2d_relu);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D_RELU6, shl_c908_conv2d_init_int8, NULL,
                    shl_gref_conv2d_relu6);
#endif
#ifndef CONFIG_C908_DEPTHWISE_CONVOLUTION_FP32_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTHWISE_CONV2D,
                    shl_c908_depthwise_conv2d_init_fp32, NULL, shl_gref_depthwise_conv2d);
#endif
#ifndef CONFIG_C908_DEPTHWISE_CONVOLUTION_FP16_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D,
                    shl_c908_depthwise_conv2d_init_fp16, NULL, shl_gref_depthwise_conv2d);
#endif
#ifndef CONFIG_C908_DEPTHWISE_CONVOLUTION_INT8_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D,
                    shl_c908_depthwise_conv2d_init_int8, NULL, shl_gref_depthwise_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D_RELU,
                    shl_c908_depthwise_conv2d_init_int8, NULL, shl_gref_depthwise_conv2d_relu);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D_RELU6,
                    shl_c908_depthwise_conv2d_init_int8, NULL, shl_gref_depthwise_conv2d_relu6);
#endif
#ifndef CONFIG_C908_FULLYCONNECTED_FP32_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_FULLYCONNECTED, shl_c908_fullyconnected_init_fp32,
                    NULL, shl_gref_fullyconnected);
#endif
#ifndef CONFIG_C908_FULLYCONNECTED_FP16_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED, shl_c908_fullyconnected_init_fp16,
                    NULL, shl_gref_fullyconnected);
#endif
#ifndef CONFIG_C908_FULLYCONNECTED_INT8_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_FULLYCONNECTED, shl_c908_fullyconnected_init_int8,
                    NULL, shl_gref_fullyconnected);
#endif
    shl_c908_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DATA_CONVERT, shl_rvv_data_convert_init, NULL,
                    shl_gref_data_convert);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DATA_CONVERT, shl_rvv_data_convert_init, NULL,
                    shl_gref_data_convert);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DATA_CONVERT, shl_rvv_data_convert_init, NULL,
                    shl_gref_data_convert);

#ifdef SHL_USE_DOT_INT4
#ifndef CONFIG_C908_CONVOLUTION_INT4_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_CONV2D, shl_c908_conv2d_init_int4, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_GROUP_CONV2D, shl_c908_conv2d_init_int4, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_CONV2D_RELU, shl_c908_conv2d_init_int4, NULL,
                    shl_gref_conv2d_relu);
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_GROUP_CONV2D_RELU, shl_c908_conv2d_init_int4, NULL,
                    shl_gref_group_conv2d_relu);
#endif
#ifndef CONFIG_C908_DEPTHWISE_CONVOLUTION_INT4_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_DEPTHWISE_CONV2D,
                    shl_c908_depthwise_conv2d_init_int4, NULL, shl_gref_depthwise_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_DEPTHWISE_CONV2D_RELU,
                    shl_c908_depthwise_conv2d_init_int4, NULL, shl_gref_depthwise_conv2d_relu);
#endif
#ifndef CONFIG_C908_FULLYCONNECTED_INT4_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_FULLYCONNECTED, shl_c908_fullyconnected_init_int4,
                    NULL, shl_gref_fullyconnected);
#endif
    shl_c908_reg_op(CSINN_DTYPE_INT4, CSINN_OP_DATA_CONVERT, shl_rvv_data_convert_init, NULL,
                    shl_gref_data_convert);
#endif

    shl_register_runtime_callback(CSINN_C908, NULL);

    shl_register_op_callback(CSINN_C908, shl_cb_map_c908);
    shl_register_runtime_callback(CSINN_C908, shl_gref_runtime_callback);
}
