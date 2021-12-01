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

#include "csi_e804.h"

static void *setup_init_map()
{
    static void* init_map[CSINN_OP_AND_UTILS_SIZE][2];
    /* q7 dtype */
    init_map[CSINN_OP_AVGPOOL2D][0] = csi_e804_avgpool2d_init_q7;
    init_map[CSINN_OP_CONV2D][0] = csi_e804_conv2d_init_q7;
    init_map[CSINN_OP_DEPTHWISE_CONV2D][0] = csi_e804_depthwise_conv2d_init_q7;
    init_map[CSINN_OP_MAXPOOL2D][0] = csi_e804_maxpool2d_init_q7;
    
    /* q15 dtype */
    init_map[CSINN_OP_CONV2D][1] = csi_e804_conv2d_init_q15;

    return init_map;
}

static int get_init_map_index(int op, int dtype)
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

void *csi_init_map_e804(int op, int dtype)
{
    void **init_map_table = setup_init_map();
    return init_map_table[get_init_map_index(op, dtype)];
}


static void *setup_bc_map()
{
    static void* bc_map[CSINN_OP_AND_UTILS_SIZE][2];

    /* q7 dtype */
    bc_map[CSINN_OP_AVGPOOL2D][0] = csi_ref_avgpool2d_quant;
    bc_map[CSINN_OP_CONV2D][0] = csi_ref_conv2d_quant;
    bc_map[CSINN_OP_DEPTHWISE_CONV2D][0] = csi_ref_depthwise_conv2d_quant;
    bc_map[CSINN_OP_FULLYCONNECTED][0] = csi_e804_fullyconnected_q7;
    bc_map[CSINN_OP_MAXPOOL2D][0] = csi_ref_maxpool2d_quant;
    bc_map[CSINN_OP_RELU][0] = csi_e804_relu_q7;
    bc_map[CSINN_OP_SIGMOID][0] = csi_e804_sigmoid_q7;
    bc_map[CSINN_OP_SOFTMAX][0] = csi_e804_softmax_q7;
    bc_map[CSINN_OP_TANH][0] = csi_e804_tanh_q7;

    /* q15 dtype */
    bc_map[CSINN_OP_CONV2D][1] = csi_ref_conv2d_quant;
    bc_map[CSINN_OP_FULLYCONNECTED][1] = csi_e804_fullyconnected_q15;
    bc_map[CSINN_OP_RELU][1] = csi_e804_relu_q15;
    bc_map[CSINN_OP_SIGMOID][1] = csi_e804_sigmoid_q15;
    bc_map[CSINN_OP_SOFTMAX][1] = csi_e804_softmax_q15;
    bc_map[CSINN_OP_TANH][1] = csi_e804_tanh_q15;

    return bc_map;
}

static int get_bc_map_index(int op, int dtype)
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

void *csi_bc_map_e804(int op, int dtype) 
{
    static int has_init;
    static void **bc_map_table;
    if (has_init == 0) {
        bc_map_table = setup_bc_map();
        has_init = 1;
    }
    return bc_map_table[get_bc_map_index(op, dtype)];
}
