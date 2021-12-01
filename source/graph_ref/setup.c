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

#include "csi_gref.h"
#include "csi_utils.h"

void csi_gref_set_output_number(int number, struct csi_session *sess)
{
    struct csi_ref_graph *graph = csi_gref_get_graph(sess);
    graph->output_num = number;
    graph->output = malloc(sizeof(struct csi_node *) * number);
}

void csi_gref_set_input_number(int number, struct csi_session *sess)
{
    struct csi_ref_graph *graph = csi_gref_get_graph(sess);
    graph->input_num = number;
    graph->input = malloc(sizeof(struct csi_node *) * number);
}

int csi_gref_get_output(int index, struct csi_tensor *output, struct csi_session *sess)
{
    struct csi_ref_graph *graph = csi_gref_get_graph(sess);
    csi_tensor_copy(output, graph->output[index]->data);
    return CSINN_TRUE;
}

int csi_gref_get_input(int index, struct csi_tensor *input, struct csi_session *sess)
{
    struct csi_ref_graph *graph = csi_gref_get_graph(sess);
    csi_tensor_copy(input, graph->input[index]->data);
    return CSINN_TRUE;
}

void csi_gref_update_input(int index, struct csi_tensor *input, struct csi_session *sess)
{
    struct csi_ref_graph *graph = csi_gref_get_graph(sess);
    struct csi_tensor *t = graph->input[index]->data;
    t->data = input->data;
}

void csi_gref_update_output(int index, struct csi_tensor *output, struct csi_session *sess)
{
    struct csi_ref_graph *graph = csi_gref_get_graph(sess);
    struct csi_tensor *t = graph->output[index]->data;
    t->data = output->data;
}

void csi_gref_session_init(struct csi_session *sess)
{
    struct csi_ref_graph *graph = calloc(sizeof(struct csi_ref_graph), 1);
    struct csi_gref_target_data *target_data = calloc(sizeof(struct csi_gref_target_data), 1);
    target_data->graph = graph;
    sess->td = target_data;
    sess->base_layout = CSINN_LAYOUT_NCHW;
}

static int call_func(void *fn, struct csi_node *node)
{
    /* base has same address with params */
    struct csi_params_base *params = node->data;
    int (*func)();
    func = fn;
    int ret = CSINN_TRUE;
    struct csi_tensor **inputs;
    struct csi_tensor **outputs;

    switch (node->type)
    {
    case CSINN_OP_ABS:
    case CSINN_OP_ACOS:
    case CSINN_OP_ACOSH:
    case CSINN_OP_ANY:
    case CSINN_OP_ARGMAX:
    case CSINN_OP_ARGMIN:
    case CSINN_OP_ASIN:
    case CSINN_OP_ASINH:
    case CSINN_OP_ATAN:
    case CSINN_OP_ATANH:
    case CSINN_OP_AVGPOOL2D:
    case CSINN_OP_AVGPOOL3D:
    case CSINN_OP_BATCH_TO_SPACE:
    case CSINN_OP_BATCH_TO_SPACE_ND:
    case CSINN_OP_BROADCOST:
    case CSINN_OP_CEIL:
    case CSINN_OP_CLIP:
    case CSINN_OP_COL2IM:
    case CSINN_OP_COS:
    case CSINN_OP_COSH:
    case CSINN_OP_CROP:
    case CSINN_OP_CUMPROD:
    case CSINN_OP_CUMSUM:
    case CSINN_OP_DEPTH_TO_SPACE:
    case CSINN_OP_ELU:
    case CSINN_OP_ERF:
    case CSINN_OP_EXP:
    case CSINN_OP_EXPAND_DIMS:
    case CSINN_OP_EXPM1:
    case CSINN_OP_FLATTEN:
    case CSINN_OP_FLOOR:
    case CSINN_OP_GLOBAL_AVGPOOL2D:
    case CSINN_OP_GLOBAL_MAXPOOL2D:
    case CSINN_OP_HARD_SIGMOID:
    case CSINN_OP_IM2COL:
    case CSINN_OP_ISNAN:
    case CSINN_OP_L2N:
    case CSINN_OP_L2POOL2D:
    case CSINN_OP_LEAKY_RELU:
    case CSINN_OP_LOG_SOFTMAX:
    case CSINN_OP_LOG:
    case CSINN_OP_LOG1P:
    case CSINN_OP_LOGICAL_NOT:
    case CSINN_OP_LRN:
    case CSINN_OP_MAX:
    case CSINN_OP_MAXPOOL2D:
    case CSINN_OP_MAXPOOL2D_LOCAT:
    case CSINN_OP_MAXPOOL3D:
    case CSINN_OP_MEAN:
    case CSINN_OP_MEAN_STRIDE:
    case CSINN_OP_MIN:
    case CSINN_OP_NDARRAY_SIZE:
    case CSINN_OP_NEGATIIVE:
    case CSINN_OP_NOT:
    case CSINN_OP_PAD:
    case CSINN_OP_PROD:
    case CSINN_OP_REDUCE_LOGSUMEXP:
    case CSINN_OP_REDUCE_MAX:
    case CSINN_OP_REDUCE_MEAN:
    case CSINN_OP_REDUCE_MIN:
    case CSINN_OP_REDUCE_PROD:
    case CSINN_OP_REDUCE_SUM:
    case CSINN_OP_RELU:
    case CSINN_OP_RELU1:
    case CSINN_OP_RELU6:
    case CSINN_OP_RELUN:
    case CSINN_OP_REORG:
    case CSINN_OP_RESHAPE:
    case CSINN_OP_RESIZE:
    case CSINN_OP_REVERSE:
    case CSINN_OP_ROUND:
    case CSINN_OP_RSQRT:
    case CSINN_OP_SHAPE:
    case CSINN_OP_SHUFFLE_CHANNEL:
    case CSINN_OP_SIGMOID:
    case CSINN_OP_SIGN:
    case CSINN_OP_SIN:
    case CSINN_OP_SINH:
    case CSINN_OP_SLICE:
    case CSINN_OP_SOFTMAX:
    case CSINN_OP_SOFTPLUS:
    case CSINN_OP_SOFTRELU:
    case CSINN_OP_SOFTSIGN:
    case CSINN_OP_SPACE_TO_BATCH:
    case CSINN_OP_SPACE_TO_BATCH_ND:
    case CSINN_OP_SPACE_TO_DEPTH:
    case CSINN_OP_SQRT:
    case CSINN_OP_SQUARE:
    case CSINN_OP_SQUEEZE:
    case CSINN_OP_STACK:
    case CSINN_OP_STRIDED_SLICE:
    case CSINN_OP_SUM:
    case CSINN_OP_TAN:
    case CSINN_OP_TANH:
    case CSINN_OP_THRESHOLD_RELU:
    case CSINN_OP_TILE:
    case CSINN_OP_TRANSPOSE:
    case CSINN_OP_TRUNC:
    case CSINN_OP_UNPOOLING:
    case CSINN_OP_UNSTACK:
    case CSINN_OP_YUV_RGB_SCALE:
        ret = func(node->in[0]->data, node->out[0]->data, params);
        break;
    case CSINN_OP_ADD:
    case CSINN_OP_AND:
    case CSINN_OP_DIV:
    case CSINN_OP_EQUANL:
    case CSINN_OP_FLOOR_DIVIDE:
    case CSINN_OP_FLOOR_MOD:
    case CSINN_OP_GATHER_ND:
    case CSINN_OP_GATHER:
    case CSINN_OP_GREATHER_EQUAL:
    case CSINN_OP_GREATHER:
    case CSINN_OP_LESS_EQUAL:
    case CSINN_OP_LESS:
    case CSINN_OP_LOGICAL_AND:
    case CSINN_OP_LOGICAL_OR:
    case CSINN_OP_LOGICAL_XOR:
    case CSINN_OP_MATMUL:
    case CSINN_OP_MAXINUM:
    case CSINN_OP_MINIMUM:
    case CSINN_OP_MOD:
    case CSINN_OP_MUL:
    case CSINN_OP_NON_MAX_SUPPRESSION:
    case CSINN_OP_NOT_EQUAL:
    case CSINN_OP_OR:
    case CSINN_OP_POWER:
    case CSINN_OP_PRELU:
    case CSINN_OP_SEQUENCE_MASK:
    case CSINN_OP_SEGMENT_MAX:
    case CSINN_OP_UNSORTED_SEGMENT_MAX:
    case CSINN_OP_SEGMENT_MEAN:
    case CSINN_OP_UNSORTED_SEGMENT_MEAN:
    case CSINN_OP_SEGMENT_MIN:
    case CSINN_OP_UNSORTED_SEGMENT_MIN:
    case CSINN_OP_SEGMENT_PROD:
    case CSINN_OP_UNSORTED_SEGMENT_PROD:
    case CSINN_OP_SEGMENT_SUM:
    case CSINN_OP_UNSORTED_SEGMENT_SUM:
    case CSINN_OP_SUB:
    case CSINN_OP_XOR:
        ret = func(node->in[0]->data, node->in[1]->data, node->out[0]->data, params);
        break;
    case CSINN_OP_CONV2D:
    case CSINN_OP_CONV2D_RELU:
    case CSINN_OP_CONV2D_RELU6:
    case CSINN_OP_CONV2D_CHANNEL:
    case CSINN_OP_CONV2D_CHANNEL_RELU:
    case CSINN_OP_CONV2D_CHANNEL_RELU6:
    case CSINN_OP_DEPTHWISE_CONV2D:
    case CSINN_OP_DEPTHWISE_CONV2D_RELU:
    case CSINN_OP_DEPTHWISE_CONV2D_RELU6:
    case CSINN_OP_DEPTHWISE_CONV2D_CHANNEL:
    case CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU:
    case CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6:
    case CSINN_OP_GROUP_CONV2D:
    case CSINN_OP_GROUP_CONV2D_RELU:
    case CSINN_OP_GROUP_CONV2D_RELU6:
    case CSINN_OP_GROUP_CONV2D_CHANNEL:
    case CSINN_OP_GROUP_CONV2D_CHANNEL_RELU:
    case CSINN_OP_CONV3D:
    case CSINN_OP_DECONV2D:
    case CSINN_OP_DEPTHWISE_DECONV2D:
    case CSINN_OP_DECONV3D:
    case CSINN_OP_FULLYCONNECTED:
        ret = func(node->in[0]->data, node->out[0]->data, node->in[1]->data, node->in[2]->data, params);
        break;
    case CSINN_OP_FSMN:
        ret = func(node->in[0]->data, node->in[1]->data, node->in[2]->data, node->in[3]->data, node->in[4]->data, node->out[0]->data, params);
        break;
    case CSINN_OP_CONCAT:
        inputs = malloc(sizeof(struct csi_tensor *) * ((struct concat_params *)params)->inputs_count);
        for (int i = 0; i < ((struct concat_params *)params)->inputs_count; i++){
            inputs[i] = node->in[i]->data;
        }
        ret = func(inputs, node->out[0]->data, params);
        free(inputs);
        break;
    case CSINN_OP_SPLIT:
        outputs = malloc(sizeof(struct csi_tensor *) * ((struct split_params *)params)->output_num);
        for (int i = 0; i < ((struct split_params *)params)->output_num; i++){
            outputs[i] = node->out[i]->data;
        }
        ret = func(node->in[0]->data, outputs, params);
        free(outputs);
        break;
    case CSINN_OP_ALL:
        CSI_DEBUG_CALL(printf("unsupported CSINN_OP_ALL\n"));
        break;
    case CSINN_OP_ARANGE:
        CSI_DEBUG_CALL(printf("unsupported CSINN_OP_ARANGE\n"));
        break;
    case CSINN_OP_BN:
        CSI_DEBUG_CALL(printf("unsupported CSINN_OP_BN\n"));
        break;
    case CSINN_OP_MIN_STRIDE:
        CSI_DEBUG_CALL(printf("unsupported CSINN_OP_MIN_STRIDE\n"));
        break;
    case CSINN_OP_ONE_HOT:
        CSI_DEBUG_CALL(printf("unsupported CSINN_OP_ONE_HOT\n"));
        break;
    case CSINN_OP_PROPOSAL:
        CSI_DEBUG_CALL(printf("unsupported CSINN_OP_PROPOSAL\n"));
        break;
    case CSINN_OP_PSROIPOOLING:
        CSI_DEBUG_CALL(printf("unsupported CSINN_OP_PSROIPOOLING\n"));
        break;
    case CSINN_OP_ROIALIGN:
        CSI_DEBUG_CALL(printf("unsupported CSINN_OP_ROIALIGN\n"));
        break;
    case CSINN_OP_ROIPOOL:
        CSI_DEBUG_CALL(printf("unsupported CSINN_OP_ROIPOOL\n"));
        break;
    case CSINN_OP_SCATTER_ND:
        CSI_DEBUG_CALL(printf("unsupported CSINN_OP_SCATTER_ND\n"));
        break;
    case CSINN_OP_SELECT:
        CSI_DEBUG_CALL(printf("unsupported CSINN_OP_SELECT\n"));
        break;
    case CSINN_OP_TOPK:
        CSI_DEBUG_CALL(printf("unsupported CSINN_OP_TOPK\n"));
        break;
    case CSINN_OP_WHERE:
        CSI_DEBUG_CALL(printf("unsupported CSINN_OP_WHERE\n"));
        break;
    default:
        CSI_DEBUG_CALL(printf("unknown op\n"));
        return CSINN_FALSE;
    }
    return ret;
}

static int init_op(struct csi_node *node)
{
    /* base has same address with params */
    struct csi_params_base *params = node->data;
    int (*func)();

    struct csi_tensor *input = node->in[0]->data;

    func = csi_init_map(params->api, node->type, input->dtype);
    if (func != NULL) {
        if (call_func(func, node) == CSINN_TRUE) {
            return CSINN_TRUE;
        } else {
            func = NULL;
        }
    }

    if (func == NULL) {
        params->bc = csi_bc_map(params->api, CSINN_RM_LAYER, node->type, input->dtype);
        return CSINN_TRUE;
    }
}

void csi_gref_session_setup(struct csi_session *sess)
{
    struct csi_ref_graph *graph = csi_gref_get_graph(sess);
    struct csi_node *n;

    for (int i = 0; i < graph->layer_index; i++) {
        n = graph->layer[i];
        for (int j = 0; j < n->in_num; j++) {
            if (n->in[j]->ref_count > 0) {
                n->in[j]->ref_count++;
            }
        }
        for (int k = 0; k < n->out_num; k++) {
            n->out[k]->ref_count++;
        }
    }

    for (int i = 0; i< graph->output_num; i++){
        graph->output[i]->ref_count++;
    }

    for (int i = 0; i < graph->layer_index; i++) {
        struct csi_node *n = graph->layer[i];
        if (n->type == CSINN_SUBGRAPH) {
            /* TODO */
        } else if (n->type >= 0 && n->type < CSINN_SESSION_INIT) {
            init_op(n);
        } else {
            return;
        }
    }
}

static int org_mem_before_run(struct csi_node *node)
{
    for (int i = 0; i < node->out_num; i++) {
        struct csi_tensor *t = node->out[i]->data;
        t->data = malloc(csi_tensor_byte_size(t));
    }
    return CSINN_TRUE;
}

static int org_mem_after_run(struct csi_node *node)
{
    for (int i = 0; i < node->in_num; i++) {
        if (node->in[i]->ref_count > 0) {
            node->in[i]->ref_count--;
            if (node->in[i]->ref_count == 0) {
                struct csi_tensor *t = node->in[i]->data;
                free(t->data);
            }
        }
    }
    for (int i = 0; i < node->out_num; i++) {
        node->out[i]->ref_count--;
    }
    return CSINN_TRUE;
}

static int run_op(struct csi_node *node)
{
    /* base has same address with params */
    struct csi_params_base *params = node->data;
    int (*func)();

    func = params->bc;

    return call_func(func, node);
}

int csi_gref_session_run(struct csi_session *sess)
{
    struct csi_ref_graph *g = csi_gref_get_graph(sess);

    for (int i = 0; i < g->layer_index; i++) {
        struct csi_node *n = g->layer[i];
        if (n->type == CSINN_SUBGRAPH) {
            /* TODO */
        } else if (n->type >= 0 && n->type < CSINN_SESSION_INIT) {
            org_mem_before_run(n);
            run_op(n);
            org_mem_after_run(n);
        } else {
            return CSINN_FALSE;
        }
    }

    return CSINN_TRUE;
}

void csi_gref_set_tensor(struct csi_tensor *input, struct csi_session *sess)
{
    struct csi_node *in = csi_node_var_alloc(input->name, input);
    input->data = in;
}

void csi_gref_set_input(int index, struct csi_tensor *input, struct csi_session *sess)
{
    struct csi_ref_graph *graph = csi_gref_get_graph(sess);
    graph->input[index] = input->data;
}

void csi_gref_set_output(int index, struct csi_tensor *output, struct csi_session *sess)
{
    struct csi_ref_graph *graph = csi_gref_get_graph(sess);
    graph->output[index] = output->data;
}

void csi_gref_session_deinit(struct csi_session *sess)
{
    struct csi_ref_graph *graph = csi_gref_get_graph(sess);
    free(graph->input);
    free(graph->output);
}

struct csi_ref_graph *csi_gref_get_graph(struct csi_session *sess)
{
    struct csi_gref_target_data *td = sess->td;
    return td->graph;
}

static void *setup_bc_map()
{
    static void* bc_map[CSINN_OP_AND_UTILS_SIZE];

    bc_map[CSINN_OP_ABS] = csi_gref_abs;
    bc_map[CSINN_OP_ACOS] = csi_gref_acos;
    bc_map[CSINN_OP_ACOSH] = csi_gref_acosh;
    bc_map[CSINN_OP_ADD] = csi_gref_add;
    bc_map[CSINN_OP_ALL] = csi_gref_all;
    bc_map[CSINN_OP_AND] = csi_gref_and;
    bc_map[CSINN_OP_ANY] = csi_gref_any;
    bc_map[CSINN_OP_ARANGE] = csi_gref_arange;
    bc_map[CSINN_OP_ARGMAX] = csi_gref_argmax;
    bc_map[CSINN_OP_ARGMIN] = csi_gref_argmin;
    bc_map[CSINN_OP_ASIN] = csi_gref_asin;
    bc_map[CSINN_OP_ASINH] = csi_gref_asinh;
    bc_map[CSINN_OP_ATAN] = csi_gref_atan;
    bc_map[CSINN_OP_ATANH] = csi_gref_atanh;
    bc_map[CSINN_OP_AVGPOOL2D] = csi_gref_avgpool;
    bc_map[CSINN_OP_AVGPOOL3D] = csi_gref_avgpool3d;
    bc_map[CSINN_OP_BN] = csi_gref_batch_normalization;
    bc_map[CSINN_OP_BATCH_TO_SPACE] = csi_gref_batch_to_space;
    bc_map[CSINN_OP_BATCH_TO_SPACE_ND] = csi_gref_batch_to_space_nd;
    bc_map[CSINN_OP_BROADCOST] = csi_gref_broadcast_to;
    bc_map[CSINN_OP_CEIL] = csi_gref_ceil;
    bc_map[CSINN_OP_CLIP] = csi_gref_clip;
    bc_map[CSINN_OP_COL2IM] = csi_gref_col2im;
    bc_map[CSINN_OP_CONCAT] = csi_gref_concat;
    bc_map[CSINN_OP_CONV2D] = csi_gref_conv2d;
    bc_map[CSINN_OP_CONV2D_RELU] = csi_gref_conv2d_relu;
    bc_map[CSINN_OP_CONV2D_RELU6] = csi_gref_conv2d_relu6;
    bc_map[CSINN_OP_DEPTHWISE_CONV2D] = csi_gref_depthwise_conv2d;
    bc_map[CSINN_OP_DEPTHWISE_CONV2D_RELU] = csi_gref_depthwise_conv2d_relu;
    bc_map[CSINN_OP_DEPTHWISE_CONV2D_RELU6] = csi_gref_depthwise_conv2d_relu6;
    bc_map[CSINN_OP_GROUP_CONV2D] = csi_gref_group_conv2d;
    bc_map[CSINN_OP_CONV3D] = csi_gref_conv3d;
    bc_map[CSINN_OP_DECONV2D] = csi_gref_deconv2d;
    bc_map[CSINN_OP_DEPTHWISE_DECONV2D] = csi_gref_depthwise_deconv2d;
    bc_map[CSINN_OP_DECONV3D] = csi_gref_deconv3d;
    bc_map[CSINN_OP_COS] = csi_gref_cos;
    bc_map[CSINN_OP_COSH] = csi_gref_cosh;
    bc_map[CSINN_OP_CUMPROD] = csi_gref_cumprod;
    bc_map[CSINN_OP_CUMSUM] = csi_gref_cumsum;
    bc_map[CSINN_OP_DEPTH_TO_SPACE] = csi_gref_depth_to_space;
    bc_map[CSINN_OP_DIV] = csi_gref_div;
    bc_map[CSINN_OP_ELU] = csi_gref_elu;
    bc_map[CSINN_OP_EQUANL] = csi_gref_equal;
    bc_map[CSINN_OP_ERF] = csi_gref_erf;
    bc_map[CSINN_OP_EXP] = csi_gref_exp;
    bc_map[CSINN_OP_EXPAND_DIMS] = csi_gref_expand_dims;
    bc_map[CSINN_OP_EXPM1] = csi_gref_expm1;
    bc_map[CSINN_OP_FLATTEN] = csi_gref_flatten;
    bc_map[CSINN_OP_FLOOR_DIVIDE] = csi_gref_floor_divide;
    bc_map[CSINN_OP_FLOOR_MOD] = csi_gref_floor_mod;
    bc_map[CSINN_OP_FLOOR] = csi_gref_floor;
    bc_map[CSINN_OP_FSMN] = csi_gref_fsmn;
    bc_map[CSINN_OP_FULLYCONNECTED] = csi_gref_fullyconnected;
    bc_map[CSINN_OP_GATHER_ND] = csi_gref_gather_nd;
    bc_map[CSINN_OP_GATHER] = csi_gref_gather;
    bc_map[CSINN_OP_GLOBAL_AVGPOOL2D] = csi_gref_global_avgpool;
    bc_map[CSINN_OP_GLOBAL_MAXPOOL2D] = csi_gref_global_maxpool;
    bc_map[CSINN_OP_GREATHER_EQUAL] = csi_gref_greater_equal;
    bc_map[CSINN_OP_GREATHER] = csi_gref_greater;
    bc_map[CSINN_OP_HARD_SIGMOID] = csi_gref_hard_sigmoid;
    bc_map[CSINN_OP_IM2COL] = csi_gref_im2col;
    bc_map[CSINN_OP_ISNAN] = csi_gref_isnan_bool;
    bc_map[CSINN_OP_L2N] = csi_gref_l2_normalization;
    bc_map[CSINN_OP_L2POOL2D] = csi_gref_l2pool;
    bc_map[CSINN_OP_LEAKY_RELU] = csi_gref_leaky_relu;
    bc_map[CSINN_OP_LESS_EQUAL] = csi_gref_less_equal;
    bc_map[CSINN_OP_LESS] = csi_gref_less;
    bc_map[CSINN_OP_LOG_SOFTMAX] = csi_gref_log_softmax;
    bc_map[CSINN_OP_LOG] = csi_gref_log;
    bc_map[CSINN_OP_LOG1P] = csi_gref_log1p;
    bc_map[CSINN_OP_LOGICAL_AND] = csi_gref_logical_and;
    bc_map[CSINN_OP_LOGICAL_NOT] = csi_gref_logical_not;
    bc_map[CSINN_OP_LOGICAL_OR] = csi_gref_logical_or;
    bc_map[CSINN_OP_LOGICAL_XOR] = csi_gref_logical_xor;
    bc_map[CSINN_OP_LRN] = csi_gref_lrn;
    bc_map[CSINN_OP_MATMUL] = csi_gref_matmul;
    bc_map[CSINN_OP_MAX] = csi_gref_max;
    bc_map[CSINN_OP_MAXINUM] = csi_gref_maximum;
    bc_map[CSINN_OP_MAXPOOL2D] = csi_gref_maxpool;
    bc_map[CSINN_OP_MAXPOOL2D_LOCAT] = csi_gref_maxpool2d_locat;
    bc_map[CSINN_OP_MAXPOOL3D] = csi_gref_maxpool3d;
    bc_map[CSINN_OP_MEAN] = csi_gref_mean;
    bc_map[CSINN_OP_MEAN_STRIDE] = csi_gref_mean;
    bc_map[CSINN_OP_MIN] = csi_gref_min;
    bc_map[CSINN_OP_MINIMUM] = csi_gref_minimum;
    bc_map[CSINN_OP_MOD] = csi_gref_mod;
    bc_map[CSINN_OP_MUL] = csi_gref_mul;
    bc_map[CSINN_OP_NDARRAY_SIZE] = csi_gref_ndarray_size;
    bc_map[CSINN_OP_NEGATIIVE] = csi_gref_negative;
    bc_map[CSINN_OP_NON_MAX_SUPPRESSION] = csi_gref_non_max_suppression;
    bc_map[CSINN_OP_NOT_EQUAL] = csi_gref_not_equal;
    bc_map[CSINN_OP_NOT] = csi_gref_not;
    bc_map[CSINN_OP_OR] = csi_gref_or;
    bc_map[CSINN_OP_PAD] = csi_gref_pad;
    bc_map[CSINN_OP_POWER] = csi_gref_power;
    bc_map[CSINN_OP_PRELU] = csi_gref_prelu;
    bc_map[CSINN_OP_PROD] = csi_gref_prod;
    bc_map[CSINN_OP_PROPOSAL] = csi_gref_proposal;
    bc_map[CSINN_OP_PSROIPOOLING] = csi_gref_psroipooling;
    bc_map[CSINN_OP_REDUCE_LOGSUMEXP] = csi_gref_reduce_logsumexp;
    bc_map[CSINN_OP_REDUCE_MAX] = csi_gref_reduce_max;
    bc_map[CSINN_OP_REDUCE_MEAN] = csi_gref_reduce_mean;
    bc_map[CSINN_OP_REDUCE_MIN] = csi_gref_reduce_min;
    bc_map[CSINN_OP_REDUCE_PROD] = csi_gref_reduce_prod;
    bc_map[CSINN_OP_REDUCE_SUM] = csi_gref_reduce_sum;
    bc_map[CSINN_OP_RELU] = csi_gref_relu;
    bc_map[CSINN_OP_RELU1] = csi_gref_relu1;
    bc_map[CSINN_OP_RELU6] = csi_gref_relu6;
    bc_map[CSINN_OP_RELUN] = csi_gref_relun;
    bc_map[CSINN_OP_RESHAPE] = csi_gref_reshape;
    bc_map[CSINN_OP_RESIZE] = csi_gref_resize;
    bc_map[CSINN_OP_REVERSE] = csi_gref_reverse;
    bc_map[CSINN_OP_ROIALIGN] = csi_gref_roi_align;
    bc_map[CSINN_OP_ROIPOOL] = csi_gref_roipool;
    bc_map[CSINN_OP_ROUND] = csi_gref_round;
    bc_map[CSINN_OP_RSQRT] = csi_gref_rsqrt;
    bc_map[CSINN_OP_SCATTER_ND] = csi_gref_scatter_nd;
    bc_map[CSINN_OP_SEGMENT_MAX] = csi_gref_segment_max;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MAX] = NULL;
    bc_map[CSINN_OP_SEGMENT_MEAN] = csi_gref_segment_mean;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MEAN] = NULL;
    bc_map[CSINN_OP_SEGMENT_MIN] = csi_gref_segment_min;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MIN] = NULL;
    bc_map[CSINN_OP_SEGMENT_PROD] = csi_gref_segment_prod;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_PROD] = NULL;
    bc_map[CSINN_OP_SEGMENT_SUM] = csi_gref_segment_sum;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_SUM] = NULL;
    bc_map[CSINN_OP_SELECT] = csi_gref_select;
    bc_map[CSINN_OP_SEQUENCE_MASK] = csi_gref_sequence_mask;
    bc_map[CSINN_OP_SHAPE] = csi_gref_shape;
    bc_map[CSINN_OP_SHUFFLE_CHANNEL] = csi_gref_shuffle_channel;
    bc_map[CSINN_OP_SIGMOID] = csi_gref_sigmoid;
    bc_map[CSINN_OP_SIGN] = csi_gref_sign;
    bc_map[CSINN_OP_SIN] = csi_gref_sin;
    bc_map[CSINN_OP_SINH] = csi_gref_sinh;
    bc_map[CSINN_OP_SLICE] = csi_gref_slice;
    bc_map[CSINN_OP_SOFTMAX] = csi_gref_softmax;
    bc_map[CSINN_OP_SOFTPLUS] = csi_gref_softplus;
    bc_map[CSINN_OP_SOFTRELU] = csi_gref_softrelu;
    bc_map[CSINN_OP_SOFTSIGN] = csi_gref_softsign;
    bc_map[CSINN_OP_SPACE_TO_BATCH] = csi_gref_space_to_batch;
    bc_map[CSINN_OP_SPACE_TO_BATCH_ND] = csi_gref_space_to_batch_nd;
    bc_map[CSINN_OP_SPACE_TO_DEPTH] = csi_gref_space_to_depth;
    bc_map[CSINN_OP_SPLIT] = csi_gref_split;
    bc_map[CSINN_OP_SQRT] = csi_gref_sqrt;
    bc_map[CSINN_OP_SQUARE] = csi_gref_square;
    bc_map[CSINN_OP_SQUEEZE] = csi_gref_squeeze;
    bc_map[CSINN_OP_STACK] = csi_gref_stack;
    bc_map[CSINN_OP_STRIDED_SLICE] = csi_gref_strided_slice;
    bc_map[CSINN_OP_SUB] = csi_gref_sub;
    bc_map[CSINN_OP_SUM] = csi_gref_sum;
    bc_map[CSINN_OP_TAN] = csi_gref_tan;
    bc_map[CSINN_OP_TANH] = csi_gref_tanh;
    bc_map[CSINN_OP_THRESHOLD_RELU] = csi_gref_threshold_relu;
    bc_map[CSINN_OP_TILE] = csi_gref_tile;
    bc_map[CSINN_OP_TOPK] = csi_gref_topk;
    bc_map[CSINN_OP_TRUNC] = csi_gref_trunc;
    bc_map[CSINN_OP_TRANSPOSE] = csi_gref_transpose;
    bc_map[CSINN_OP_TRUNC] = csi_gref_trunc;
    bc_map[CSINN_OP_UNPOOLING] = csi_gref_unpooling;
    bc_map[CSINN_OP_UNSTACK] = csi_gref_unstack;
    bc_map[CSINN_OP_WHERE] = csi_gref_where;
    bc_map[CSINN_OP_XOR] = csi_gref_xor;
    bc_map[CSINN_OP_YUV_RGB_SCALE] = csi_gref_yuv_rgb_scale;

    bc_map[CSINN_SESSION_INIT] = csi_gref_session_init;
    bc_map[CSINN_SESSION_DEINIT] = csi_gref_session_deinit;
    bc_map[CSINN_SESSION_SETUP] = csi_gref_session_setup;
    bc_map[CSINN_SESSION_RUN] = csi_gref_session_run;
    bc_map[CSINN_UPDATE_INPUT] = csi_gref_update_input;
    bc_map[CSINN_UPDATE_OUTPUT] = csi_gref_update_output;
    bc_map[CSINN_SET_INPUT_NUMBER] = csi_gref_set_input_number;
    bc_map[CSINN_SET_OUTPUT_NUMBER] = csi_gref_set_output_number;
    bc_map[CSINN_SET_INPUT] = csi_gref_set_input;
    bc_map[CSINN_SET_OUTPUT] = csi_gref_set_output;
    bc_map[CSINN_GET_INPUT] = csi_gref_get_input;
    bc_map[CSINN_GET_OUTPUT] = csi_gref_get_output;
    bc_map[CSINN_TENSOR_ENTRY] = csi_gref_set_tensor;

    return bc_map;
}

static int get_bc_map_index(int op, int dtype)
{
    return op;
}

void *csi_bc_map_gref(int op, int dtype) {
    static int has_init;
    static void **bc_map_table;
    if (has_init == 0) {
        bc_map_table = setup_bc_map();
        has_init = 1;
    }
    return bc_map_table[get_bc_map_index(op, dtype)];
}
