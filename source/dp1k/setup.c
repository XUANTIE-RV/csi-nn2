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

/* CSI-NN2 version 1.8.x */

#include "csi_dp1k.h"

void csi_dp1k_set_tensor(struct csi_tensor *tensor, struct csi_session *sess) {
    csi_dp1k_input(tensor, sess);
    tensor->sess = sess;
}

void csi_dp1k_session_init(struct csi_session *sess) {
    sess->base_dtype = CSINN_DTYPE_FLOAT32; // support float only currently.
    sess->base_layout = CSINN_LAYOUT_NCHW;
    csi_dp1000_session_init(sess);
}

void csi_dp1k_session_deinit(struct csi_session *sess) {
    free(sess->td);
    sess->td = NULL;
}

void csi_dp1k_session_setup(struct csi_session *sess) {
    csi_dp1000_session_setup(sess);
}

void csi_dp1k_set_input_number(int number, struct csi_session *sess) {
    csi_dp1000_set_input_number(number, sess);
}

void csi_dp1k_set_output_number(int number, struct csi_session *sess) {
    csi_dp1000_set_output_number(number, sess);
}

void csi_dp1k_set_input(int index, struct csi_tensor *input, struct csi_session *sess) {
    csi_dp1000_set_input(index, input, sess);
}

void csi_dp1k_set_output(int index, struct csi_tensor *output, struct csi_session *sess) {
    csi_dp1000_set_output(index, output, sess);
}

static void *setup_bc_map()
{
    static void* bc_map[CSINN_OP_AND_UTILS_SIZE];

    bc_map[CSINN_OP_ADD] = csi_dp1k_add;
    bc_map[CSINN_OP_AVGPOOL2D] = csi_dp1k_avgpool2d;
    bc_map[CSINN_OP_CONCAT] = csi_dp1k_concat;
    bc_map[CSINN_OP_CONV2D] = csi_dp1k_conv2d;
    bc_map[CSINN_OP_DEPTHWISE_CONV2D] = csi_dp1k_conv2d;
    bc_map[CSINN_OP_GROUP_CONV2D] = csi_dp1k_conv2d;
    bc_map[CSINN_OP_DECONV2D] = csi_dp1k_deconv2d;
    bc_map[CSINN_OP_DEPTHWISE_DECONV2D] = csi_dp1k_deconv2d;
    bc_map[CSINN_OP_FULLYCONNECTED] = csi_dp1k_fullyconnected;
    bc_map[CSINN_OP_LEAKY_RELU] = csi_dp1k_leaky_relu;
    bc_map[CSINN_OP_MAXPOOL2D] = csi_dp1k_maxpool;
    bc_map[CSINN_OP_MUL] = csi_dp1k_mul;
    bc_map[CSINN_OP_PRELU] = csi_dp1k_prelu;
    bc_map[CSINN_OP_RELU] = csi_dp1k_relu;
    bc_map[CSINN_OP_RESHAPE] = csi_dp1k_reshape;
    bc_map[CSINN_OP_RESIZE] = csi_dp1k_resize;
    bc_map[CSINN_OP_SIGMOID] = csi_dp1k_sigmoid;
    bc_map[CSINN_OP_SOFTMAX] = csi_dp1k_softmax;
    bc_map[CSINN_OP_TRANSPOSE] = csi_dp1k_transpose;
    bc_map[CSINN_OP_STRIDED_SLICE] = csi_dp1k_strided_slice;

    bc_map[CSINN_SESSION_INIT] = csi_dp1k_session_init;
    bc_map[CSINN_SESSION_DEINIT] = csi_dp1k_session_deinit;
    bc_map[CSINN_SESSION_SETUP] = csi_dp1k_session_setup;
    bc_map[CSINN_SET_INPUT_NUMBER] = csi_dp1k_set_input_number;
    bc_map[CSINN_SET_OUTPUT_NUMBER] = csi_dp1k_set_output_number;
    bc_map[CSINN_SET_INPUT] = csi_dp1k_set_input;
    bc_map[CSINN_SET_OUTPUT] = csi_dp1k_set_output;
    bc_map[CSINN_TENSOR_ENTRY] = csi_dp1k_set_tensor;

    return bc_map;
}

static int get_bc_map_index(int op, int dtype)
{
    return op;
}

void *csi_bc_map_dp1k(int op, int dtype) {
    static int has_init;
    static void **bc_map_table;
    if (has_init == 0) {
        bc_map_table = setup_bc_map();
        has_init = 1;
    }
    return bc_map_table[get_bc_map_index(op, dtype)];
}
