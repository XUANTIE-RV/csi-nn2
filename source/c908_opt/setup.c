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

int shl_c908_set_packn_layout(struct csinn_session *sess, bool packn_layout)
{
    struct shl_gref_target_data *gref_td = sess->td;
    struct shl_c908_option *c908_option = gref_td->cpu_option;
    c908_option->base.use_packn_layout = packn_layout;
    return CSINN_TRUE;
}

struct shl_c908_option *shl_c908_get_graph_option(struct csinn_session *sess)
{
    struct shl_gref_target_data *gref_td = sess->td;
    if (gref_td) {
        return (struct shl_c908_option *)(gref_td->cpu_option);
    } else {
        return NULL;
    }
}

void shl_c908_session_init(struct csinn_session *sess)
{
    struct shl_c908_option *c908_option = shl_mem_alloc(sizeof(struct shl_c908_option));
    struct shl_ref_graph *graph = shl_mem_alloc(sizeof(struct shl_ref_graph));
    struct shl_gref_target_data *target_data = shl_mem_alloc(sizeof(struct shl_gref_target_data));
    target_data->graph = graph;
    c908_option->base.use_packn_layout = 1;  // c908 set use_packn_layout true default
    target_data->cpu_option = c908_option;
    sess->td = target_data;
    sess->base_layout = CSINN_LAYOUT_NCHW;
}

void shl_c908_session_deinit(struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    shl_mem_free(graph->input);
    shl_mem_free(graph->output);
    struct shl_c908_option *c908_option = shl_c908_get_graph_option(sess);
    if (c908_option) {
        shl_mem_free(c908_option);
    }
}

void *shl_c908_runtime_callback(int api)
{
    switch (api) {
        case CSINN_SESSION_INIT:
            return shl_c908_session_init;
            break;
        case CSINN_SESSION_DEINIT:
            return shl_c908_session_deinit;
            break;
        case CSINN_SESSION_SETUP:
        case CSINN_SESSION_RUN:
        case CSINN_UPDATE_INPUT:
        case CSINN_UPDATE_OUTPUT:
        case CSINN_SET_INPUT_NUMBER:
        case CSINN_SET_OUTPUT_NUMBER:
        case CSINN_SET_INPUT:
        case CSINN_SET_OUTPUT:
        case CSINN_GET_INPUT:
        case CSINN_GET_OUTPUT:
        case CSINN_TENSOR_ENTRY:
        case CSINN_LOAD_BG:
            return shl_gref_runtime_callback(api);
            break;
        default:
            shl_debug_info("%s: Cannot find callback\n", __func__);
            break;
    }
    return NULL;
}

void shl_target_init_c908()
{
#ifndef CONFIG_C908_CONVOLUTION_FP32_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_c908_conv2d_init_fp32, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, shl_c908_conv2d_init_fp32, NULL,
                    shl_gref_group_conv2d);
#endif
#ifndef CONFIG_C908_CONVOLUTION_FP16_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_c908_conv2d_init_fp16, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, shl_c908_conv2d_init_fp16, NULL,
                    shl_gref_group_conv2d);
#endif
#ifndef CONFIG_C908_CONVOLUTION_INT8_DISABLED
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D, shl_c908_conv2d_init_int8, NULL,
                    shl_gref_conv2d);
    shl_c908_reg_op(CSINN_DTYPE_INT8, CSINN_OP_GROUP_CONV2D, shl_c908_conv2d_init_int8, NULL,
                    shl_gref_group_conv2d);
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
                    shl_gref_group_conv2d);
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
    shl_register_runtime_callback(CSINN_C908, shl_c908_runtime_callback);
}
