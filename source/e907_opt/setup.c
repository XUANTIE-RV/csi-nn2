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

/* SHL version 2.1.x */

#include "shl_e907.h"

static struct shl_cb_op_list shl_e907_cb_op_list;

int shl_e907_reg_op(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *init, void *exec)
{
    struct shl_cb_op_list *list_end = shl_cb_list_end(&shl_e907_cb_op_list);
    struct shl_cb_op_list *next = shl_mem_alloc(sizeof(struct shl_cb_op_list));
    next->cb = shl_mem_alloc(sizeof(struct csinn_callback));
    next->cb->init = init;
    next->cb->exec = exec;
    next->dtype = dtype;
    next->op_name = op_name;
    list_end->next = next;
    return CSINN_TRUE;
}

int shl_e907_reg_op_est(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *est)
{
    struct csinn_callback *cb = shl_cb_list_match(&shl_e907_cb_op_list, dtype, op_name);
    if (cb == NULL) {
        shl_debug_info("%s: cannot find e907 est\n", __func__);
    } else {
        cb->est = est;
    }

    return CSINN_TRUE;
}

struct csinn_callback *__attribute__((weak)) shl_cb_map_rvv(int op, int dtype);
struct csinn_callback *shl_cb_map_e907(int op, int dtype)
{
    struct csinn_callback *cb = shl_cb_list_match(&shl_e907_cb_op_list, dtype, op);
    if (cb == NULL) {
        cb = shl_cb_map_rvv(op, dtype);
    }
    return cb;
}

void __attribute__((weak)) shl_target_init_e907()
{
    shl_register_runtime_callback(CSINN_E907, NULL);
    shl_register_op_callback(CSINN_E907, shl_cb_map_e907);
#ifndef CONFIG_E907_OPT_CONVOLUTION_DISABLED
    shl_e907_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D, NULL, shl_e907_conv2d_int8);
#endif
#ifndef CONFIG_E907_OPT_CONCAT_DISABLED
    shl_e907_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONCAT, NULL, shl_e907_concat_int8);
#endif
#ifndef CONFIG_E907_OPT_RELU_DISABLED
    shl_e907_reg_op(CSINN_DTYPE_INT8, CSINN_OP_RELU, NULL, shl_e907_relu_int8);
#endif
#ifndef CONFIG_E907_OPT_FC_DISABLED
    shl_e907_reg_op(CSINN_DTYPE_INT8, CSINN_OP_FULLYCONNECTED, shl_e907_fullyconnected_init,
                    shl_e907_fullyconnected_int8);
#endif
#ifndef CONFIG_E907_OPT_MUL_DISABLED
    shl_e907_reg_op(CSINN_DTYPE_INT8, CSINN_OP_MUL, NULL, shl_e907_mul_int8);
#endif
#ifndef CONFIG_E907_OPT_SUM_DISABLED
    shl_e907_reg_op(CSINN_DTYPE_INT8, CSINN_OP_SUM, NULL, shl_e907_sum_int8);
#endif
#ifndef CONFIG_E907_OPT_SOFTMAX_DISABLED
    shl_e907_reg_op(CSINN_DTYPE_INT8, CSINN_OP_SOFTMAX, NULL, shl_e907_softmax_int8);
#endif

    shl_register_runtime_callback(CSINN_E907, shl_gref_runtime_callback);
#ifndef CONFIG_GRAPH_REFERENCE_CONVOLUTION_DISABLED
    shl_e907_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_CONV2D, shl_gref_conv2d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONCAT_DISABLED
    shl_e907_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_CONCAT, shl_gref_concat);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU_DISABLED
    shl_e907_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_RELU, shl_gref_relu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FULLYCONNECTED_DISABLED
    shl_e907_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_FULLYCONNECTED, shl_gref_fullyconnected);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MUL_DISABLED
    shl_e907_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_MUL, shl_gref_mul);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SUM_DISABLED
    shl_e907_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_SUM, shl_gref_sum);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SOFTMAX_DISABLED
    shl_e907_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_SOFTMAX, shl_gref_softmax);
#endif
}
