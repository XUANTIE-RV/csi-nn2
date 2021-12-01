/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

/* CSI-NN2 version 1.10.x */

#include "csi_i805.h"

static void *setup_init_map()
{
    static void* init_map[CSINN_OP_AND_UTILS_SIZE][2];
    /* q7 dtype */
    // init_map[CSINN_OP_AVGPOOL2D][0] = csi_i805_avgpool2d_init_q7;
    init_map[CSINN_OP_ADD][0] = csi_i805_add_init_u8;
    init_map[CSINN_OP_CONV2D][0] = csi_i805_conv2d_init_u8;
    init_map[CSINN_OP_DEPTHWISE_CONV2D][0] = csi_i805_depthwise_conv2d_init_u8;
    init_map[CSINN_OP_FULLYCONNECTED][0] = csi_i805_fullyconnected_init_u8;
    init_map[CSINN_OP_MAXPOOL2D][0] = csi_i805_maxpool2d_init_q7;
    init_map[CSINN_OP_MUL][0] = csi_i805_mul_init_u8;
    init_map[CSINN_OP_RELU][0] = csi_i805_relu_init_u8;
    init_map[CSINN_OP_RELU6][0] = csi_i805_relu6_init_u8;

    /* q15 dtype */
    init_map[CSINN_OP_CONV2D][1] = csi_i805_conv2d_init_q15;

    return init_map;
}

static int get_init_map_index(int op, int dtype)
{
    switch (dtype) {
    case CSINN_DTYPE_UINT8:
        return op * 2;
        break;
    case CSINN_DTYPE_INT16:
        return op * 2 + 1;
        break;
    default:
        return CSINN_UNSUPPORT_DTYPE;
    }
}

void *csi_init_map_i805(int op, int dtype)
{
    void **init_map_table = setup_init_map();
    int idx = get_init_map_index(op, dtype);
    if (idx >= 0) {
        return init_map_table[idx];
    } else {
        return NULL;
    }
}


static void *setup_bc_map()
{
    static void* bc_map[CSINN_OP_AND_UTILS_SIZE][2];

    /* q7 dtype */
    bc_map[CSINN_OP_ADD][0] = csi_i805_add_u8;
    bc_map[CSINN_OP_AVGPOOL2D][0] = csi_ref_avgpool2d_quant;
    bc_map[CSINN_OP_CONV2D][0] = csi_i805_conv2d_u8;
    // bc_map[CSINN_OP_CONV2D][0] = csi_ref_conv2d_quant;

    bc_map[CSINN_OP_CLIP][0] =  csi_ref_clip_quant;
    bc_map[CSINN_OP_DEPTHWISE_CONV2D][0] = csi_i805_depthwise_conv2d_u8;
    bc_map[CSINN_OP_FULLYCONNECTED][0] = csi_i805_fullyconnected_u8;
    bc_map[CSINN_OP_MAXPOOL2D][0] = csi_ref_maxpool2d_quant;
    bc_map[CSINN_OP_MUL][0] = csi_i805_mul_u8;
    bc_map[CSINN_OP_RELU][0] = csi_i805_relu_u8;
    bc_map[CSINN_OP_RELU6][0] = csi_i805_relu6_u8;
    bc_map[CSINN_OP_RESHAPE][0] = csi_i805_reshape_u8;
    bc_map[CSINN_OP_SQUEEZE][0] = csi_ref_squeeze;
    bc_map[CSINN_OP_SIGMOID][0] = csi_i805_sigmoid_q7;
    bc_map[CSINN_OP_SOFTMAX][0] = csi_ref_softmax_quant;
    bc_map[CSINN_OP_TANH][0] = csi_i805_tanh_q7;

    /* q15 dtype */
    bc_map[CSINN_OP_CONV2D][1] = csi_ref_conv2d_quant;
    bc_map[CSINN_OP_FULLYCONNECTED][1] = csi_i805_fullyconnected_q15;
    bc_map[CSINN_OP_RELU][1] = csi_i805_relu_q15;
    bc_map[CSINN_OP_SIGMOID][1] = csi_i805_sigmoid_q15;
    bc_map[CSINN_OP_SOFTMAX][1] = csi_i805_softmax_q15;
    bc_map[CSINN_OP_TANH][1] = csi_i805_tanh_q15;

    return bc_map;
}

static int get_bc_map_index(int op, int dtype)
{
    switch (dtype) {
    case CSINN_DTYPE_UINT8:
        return op * 2;
        break;
    case CSINN_DTYPE_INT16:
        return op * 2 + 1;
        break;
    default:
        return CSINN_UNSUPPORT_DTYPE;
    }
}

void *__attribute__((weak)) csi_bc_map_i805(int op, int dtype)
{
    static int has_init;
    static void **bc_map_table;
    if (has_init == 0) {
        bc_map_table = setup_bc_map();
        has_init = 1;
    }
    return bc_map_table[get_bc_map_index(op, dtype)];
}
