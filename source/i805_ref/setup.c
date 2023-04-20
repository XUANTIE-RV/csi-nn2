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

#include "shl_ref_i805.h"

static void *setup_cb_map()
{
    static struct csinn_callback cb_map[CSINN_OP_AND_UTILS_SIZE][2];
    memset(cb_map, 0, sizeof(struct csinn_callback) * CSINN_OP_AND_UTILS_SIZE * 2);

    /* q7 dtype */
    cb_map[CSINN_OP_AVGPOOL2D][0].init = shl_i805_ref_avgpool2d_init_q7;
    cb_map[CSINN_OP_CONV2D][0].init = shl_i805_ref_conv2d_init_q7;
    cb_map[CSINN_OP_DEPTHWISE_CONV2D][0].init = shl_i805_ref_depthwise_conv2d_init_q7;
    cb_map[CSINN_OP_MAXPOOL2D][0].init = shl_i805_ref_maxpool2d_init_q7;
    cb_map[CSINN_OP_FULLYCONNECTED][0].exec = shl_i805_ref_fullyconnected_q7;
    cb_map[CSINN_OP_RELU][0].exec = shl_i805_ref_relu_q7;
    cb_map[CSINN_OP_SIGMOID][0].exec = shl_i805_ref_sigmoid_q7;
    cb_map[CSINN_OP_SOFTMAX][0].exec = shl_i805_ref_softmax_q7;
    cb_map[CSINN_OP_TANH][0].exec = shl_i805_ref_tanh_q7;

    /* q15 dtype */
    cb_map[CSINN_OP_CONV2D][1].init = shl_i805_ref_conv2d_init_q15;
    cb_map[CSINN_OP_FULLYCONNECTED][1].exec = shl_i805_ref_fullyconnected_q15;
    cb_map[CSINN_OP_RELU][1].exec = shl_i805_ref_relu_q15;
    cb_map[CSINN_OP_SIGMOID][1].exec = shl_i805_ref_sigmoid_q15;
    cb_map[CSINN_OP_SOFTMAX][1].exec = shl_i805_ref_softmax_q15;
    cb_map[CSINN_OP_TANH][1].exec = shl_i805_ref_tanh_q15;

    return cb_map;
}

static int get_cb_map_index(int op, int dtype)
{
    switch (dtype) {
        case CSINN_DTYPE_INT8:
            return op * 2;
            break;
        case CSINN_DTYPE_INT16:
            return op * 2 + 1;
            break;
        default:
            return CSINN_UNSUPPORT_DTYPE;
    }
}

static struct csinn_callback *__cb_map_table_ref_i805;
struct csinn_callback *shl_cb_map_ref_i805(int op, int dtype)
{
    return &__cb_map_table_ref_i805[get_cb_map_index(op, dtype)];
}

void shl_target_init_ref_i805()
{
    __cb_map_table_ref_i805 = setup_cb_map();
    shl_register_runtime_callback(CSINN_REF_I805, NULL);
    shl_register_op_callback(CSINN_REF_I805, shl_cb_map_ref_i805);
}