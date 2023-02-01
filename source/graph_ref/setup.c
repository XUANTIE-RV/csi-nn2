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

#include "shl_gref.h"

void shl_gref_set_output_number(int number, struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    graph->output_num = number;
    graph->output = shl_mem_alloc(sizeof(struct shl_node *) * number);
}

void shl_gref_set_input_number(int number, struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    graph->input_num = number;
    graph->input = shl_mem_alloc(sizeof(struct shl_node *) * number);
}

int shl_gref_get_output(int index, struct csinn_tensor *output, struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    csinn_tensor_copy(output, graph->output[index]->data);
    return CSINN_TRUE;
}

int shl_gref_get_input(int index, struct csinn_tensor *input, struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    csinn_tensor_copy(input, graph->input[index]->data);
    return CSINN_TRUE;
}

void shl_gref_update_input(int index, struct csinn_tensor *input, struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    struct csinn_tensor *t = graph->input[index]->data;
    t->data = input->data;
}

void shl_gref_update_output(int index, struct csinn_tensor *output, struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    struct csinn_tensor *t = graph->output[index]->data;
    t->data = output->data;
}

void shl_gref_session_init(struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_mem_alloc(sizeof(struct shl_ref_graph));
    struct shl_gref_target_data *target_data = shl_mem_alloc(sizeof(struct shl_gref_target_data));
    target_data->graph = graph;
    sess->td = target_data;
    sess->base_layout = CSINN_LAYOUT_NCHW;
}

static int call_layer_func(void *fn, struct shl_node *node)
{
    /* base has same address with params */
    struct csinn_params_base *params = node->data;
    int (*func)();
    func = fn;
    int ret = CSINN_TRUE;
    struct csinn_tensor **inputs;
    struct csinn_tensor **outputs;

    switch (node->type) {
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
        case CSINN_OP_DATA_CONVERT:
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
        case CSINN_OP_NEGATIVE:
        case CSINN_OP_NOT:
        case CSINN_OP_ONE_HOT:
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
        case CSINN_OP_MAXIMUM:
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
        case CSINN_OP_CONV1D:
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
        case CSINN_OP_LAYER_NORM:
        case CSINN_OP_CACHE_MATMUL:
        case CSINN_OP_CACHE_CONV1D:
            ret = func(node->in[0]->data, node->out[0]->data, node->in[1]->data, node->in[2]->data,
                       params);
            break;
        case CSINN_OP_FSMN:
            ret = func(node->in[0]->data, node->in[1]->data, node->in[2]->data, node->in[3]->data,
                       node->in[4]->data, node->out[0]->data, params);
            break;
        case CSINN_OP_CONCAT:
            inputs = shl_mem_alloc(sizeof(struct csinn_tensor *) *
                                   ((struct csinn_concat_params *)params)->inputs_count);
            for (int i = 0; i < ((struct csinn_concat_params *)params)->inputs_count; i++) {
                inputs[i] = node->in[i]->data;
            }
            ret = func(inputs, node->out[0]->data, params);
            shl_mem_free(inputs);
            break;
        case CSINN_OP_SPLIT:
            outputs = shl_mem_alloc(sizeof(struct csinn_tensor *) *
                                    ((struct csinn_split_params *)params)->output_num);
            for (int i = 0; i < ((struct csinn_split_params *)params)->output_num; i++) {
                outputs[i] = node->out[i]->data;
            }
            ret = func(node->in[0]->data, outputs, params);
            shl_mem_free(outputs);
            break;
        case CSINN_OP_WHERE:
            ret = func(node->in[0]->data, node->in[1]->data, node->in[2]->data, node->out[0]->data,
                       params);
            break;
        case CSINN_OP_ALL:
            shl_debug_error("unsupported CSINN_OP_ALL\n");
            break;
        case CSINN_OP_ARANGE:
            shl_debug_error("unsupported CSINN_OP_ARANGE\n");
            break;
        case CSINN_OP_BN:
            shl_debug_error("unsupported CSINN_OP_BN\n");
            break;
        case CSINN_OP_MIN_STRIDE:
            shl_debug_error("unsupported CSINN_OP_MIN_STRIDE\n");
            break;
        case CSINN_OP_PROPOSAL:
            shl_debug_error("unsupported CSINN_OP_PROPOSAL\n");
            break;
        case CSINN_OP_PSROIPOOLING:
            shl_debug_error("unsupported CSINN_OP_PSROIPOOLING\n");
            break;
        case CSINN_OP_ROIALIGN:
            shl_debug_error("unsupported CSINN_OP_ROIALIGN\n");
            break;
        case CSINN_OP_ROIPOOL:
            shl_debug_error("unsupported CSINN_OP_ROIPOOL\n");
            break;
        case CSINN_OP_SCATTER_ND:
            shl_debug_error("unsupported CSINN_OP_SCATTER_ND\n");
            break;
        case CSINN_OP_SELECT:
            shl_debug_error("unsupported CSINN_OP_SELECT\n");
            break;
        case CSINN_OP_TOPK:
            shl_debug_error("unsupported CSINN_OP_TOPK\n");
            break;
        default:
            shl_debug_error("unknown op\n");
            return CSINN_FALSE;
    }
    return ret;
}

static int init_op(struct shl_node *node)
{
    /* base has same address with params */
    struct csinn_params_base *params = node->data;
    int (*func)();
    struct csinn_tensor *input = node->in[0]->data;

    int org_rm = params->sess->base_run_mode;
    params->sess->base_run_mode = CSINN_RM_LAYER;
    shl_op_callback_map(params, node->type, input->dtype);
    struct csinn_callback *cb = params->cb;
    if (cb->init != NULL) {
        if (call_layer_func(cb->init, node) != CSINN_TRUE) {
            return CSINN_FALSE;
        }
    }
    params->sess->base_run_mode = org_rm;

    return CSINN_TRUE;
}

void shl_gref_session_setup(struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    struct shl_node *n;

    struct shl_gref_target_data *td = sess->td;
    for (int i = 0; i < graph->layer_index; i++) {
        struct csinn_params_base *curr_params = graph->layer[i]->data;
        if (curr_params->quant_type != CSINN_QUANT_UNSET &&
            curr_params->quant_type != sess->base_quant_type) {
            td->is_hybrid_quantization_type = 1;
            break;
        }
    }

    struct shl_ref_graph *ggraph;
#ifdef GRAPH_REFERENCE_SUBGRAPH
    ggraph = shl_subgraph_establish(graph);
#else
    ggraph = graph;
    /* update subgraph_idx */
    for (int i = 0; i < ggraph->layer_index; i++) {
        ggraph->layer[i]->subgraph_idx = i;
        if (ggraph->layer[i]->type == CSINN_SUBGRAPH) {
            struct shl_ref_graph *s_subgraph = ggraph->layer[i]->data;
            for (int j = 0; j < s_subgraph->layer_index; j++) {
                s_subgraph->layer[j]->subgraph_idx = i;
            }
        }
    }
#endif

    for (int i = 0; i < ggraph->layer_index; i++) {
        struct shl_node *n = ggraph->layer[i];
        if (n->type == CSINN_SUBGRAPH) {
#ifdef GRAPH_REFERENCE_SUBGRAPH
            shl_subgraph_setup(n);
#endif
        } else if (n->type >= 0 && n->type < CSINN_OP_SIZE) {
            init_op(n);
        } else {
            shl_debug_error("Unknown layer\n");
            return;
        }
    }

    for (int i = 0; i < ggraph->layer_index; i++) {
        n = ggraph->layer[i];
        for (int j = 0; j < n->in_num; j++) {
            if (n->in[j]->ref_count_init > 0) {
                n->in[j]->ref_count_init++;
            }
        }
        if (n->type != CSINN_SUBGRAPH) {
            for (int k = 0; k < n->out_num; k++) {
                n->out[k]->ref_count_init++;
            }
        } else {
            struct shl_ref_graph *sgraph = n->data;
            for (int k = 0; k < sgraph->output_num; k++) {
                sgraph->output[k]->ref_count_init++;
            }
        }
    }

    for (int i = 0; i < ggraph->output_num; i++) {
        ggraph->output[i]->ref_count_init++;
    }

    // struct shl_gref_target_data *td = sess->td;
    td->graph = ggraph;
}

static void node_ref_reset(struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    struct shl_node *n;

    for (int i = 0; i < graph->layer_index; i++) {
        n = graph->layer[i];
        if (n->type == CSINN_SUBGRAPH) {
            struct shl_ref_graph *sgraph = n->data;
            for (int k = 0; k < sgraph->output_num; k++) {
                sgraph->output[k]->ref_count = sgraph->output[k]->ref_count_init;
            }
        } else {
            for (int k = 0; k < n->out_num; k++) {
                if (n->out[k] != NULL) {
                    n->out[k]->ref_count = n->out[k]->ref_count_init;
                }
            }
        }
    }
}

static int op_run_init(struct shl_node *node)
{
    for (int i = 0; i < node->out_num; i++) {
        struct csinn_tensor *t = node->out[i]->data;
        t->data = shl_mem_alloc(csinn_tensor_byte_size(t));
    }
    return CSINN_TRUE;
}

static int op_run_deinit(struct shl_node *node, struct shl_ref_graph *graph)
{
    for (int i = 0; i < node->in_num; i++) {
        if (node->in[i]->ref_count > 0) {
            node->in[i]->ref_count--;
            if (node->in[i]->ref_count == 0) {
                if (node->in[i]->in &&
                    graph->layer[node->in[i]->in[0]->subgraph_idx]->type == CSINN_SUBGRAPH) {
                    /* nothing */
                } else {
                    struct csinn_tensor *t = node->in[i]->data;
                    shl_mem_free(t->data);
                }
            }
        }
    }
    for (int i = 0; i < node->out_num; i++) {
        node->out[i]->ref_count--;
    }
    return CSINN_TRUE;
}

static int op_run(struct shl_node *node)
{
    /* base has same address with params */
    struct csinn_params_base *params = node->data;
    int (*func)();
    struct csinn_callback *cb = params->cb;
    func = cb->exec;
    return call_layer_func(func, node);
}

int shl_gref_session_run(struct csinn_session *sess)
{
    struct shl_ref_graph *g = shl_gref_get_graph(sess);
    uint64_t time_acc = 0;
    node_ref_reset(sess);
    for (int i = 0; i < g->layer_index; i++) {
        struct shl_node *n = g->layer[i];
        if (n->type == CSINN_SUBGRAPH) {
#ifdef GRAPH_REFERENCE_SUBGRAPH
            shl_subgraph_run_init(n);
            shl_subgraph_run(n);
            shl_subgraph_run_deinit(n, g);
#endif
        } else if (n->type >= 0 && n->type < CSINN_OP_SIZE) {
            op_run_init(n);
#ifdef SHL_LAYER_BENCHMARK
            uint64_t start_time = shl_get_timespec();
            op_run(n);
            uint64_t end_time = shl_get_timespec();
            shl_benchmark_layer(n, start_time, end_time, i);
            time_acc += end_time - start_time;
#else
            op_run(n);
#endif
            op_run_deinit(n, g);
        } else {
            return CSINN_FALSE;
        }
    }
#ifdef SHL_LAYER_BENCHMARK
    shl_debug_info("[layer-benchmark]: network exec time = %f\n", time_acc / 1000000.0f);
#endif
    return CSINN_TRUE;
}

void shl_gref_set_tensor(struct csinn_tensor *input, struct csinn_session *sess)
{
    struct shl_node *in = shl_node_var_alloc(input->name, input);
    input->data = in;
}

void shl_gref_set_input(int index, struct csinn_tensor *input, struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    graph->input[index] = input->data;
}

void shl_gref_set_output(int index, struct csinn_tensor *output, struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    /* FIXME: const output's data is real value, not node */
    if (output->is_const) {
        struct shl_node *const_output_node = shl_node_const_var_alloc(output->name, output);
        graph->output[index] = const_output_node;
    } else {
        graph->output[index] = output->data;
    }
}

void shl_gref_session_deinit(struct csinn_session *sess)
{
#ifdef GRAPH_REFERENCE_SUBGRAPH
    struct shl_ref_graph *g = shl_gref_get_graph(sess);

    for (int i = 0; i < g->layer_index; i++) {
        struct shl_node *n = g->layer[i];
        if (n->type == CSINN_SUBGRAPH) {
            shl_subgraph_deinit(n);
        }
    }
#endif
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    shl_mem_free(graph->input);
    shl_mem_free(graph->output);
}

struct shl_ref_graph *shl_gref_get_graph(struct csinn_session *sess)
{
    struct shl_gref_target_data *td = sess->td;
    return td->graph;
}

static void *setup_cb_map()
{
    static struct csinn_callback cb_map[CSINN_OP_AND_UTILS_SIZE];
    memset(cb_map, 0, sizeof(struct csinn_callback) * CSINN_OP_AND_UTILS_SIZE);

#ifndef CONFIG_GRAPH_REFERENCE_ABS_DISABLED
    cb_map[CSINN_OP_ABS].est = shl_gref_abs;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ACOS_DISABLED
    cb_map[CSINN_OP_ACOS].est = shl_gref_acos;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ACOSH_DISABLED
    cb_map[CSINN_OP_ACOSH].est = shl_gref_acosh;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ADD_DISABLED
    cb_map[CSINN_OP_ADD].est = shl_gref_add;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ALL_DISABLED
    cb_map[CSINN_OP_ALL].est = shl_gref_all;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_AND_DISABLED
    cb_map[CSINN_OP_AND].est = shl_gref_and;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ANY_DISABLED
    cb_map[CSINN_OP_ANY].est = shl_gref_any;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ARANGE_DISABLED
    cb_map[CSINN_OP_ARANGE].est = shl_gref_arange;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ARGMAX_DISABLED
    cb_map[CSINN_OP_ARGMAX].est = shl_gref_argmax;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ARGMIN_DISABLED
    cb_map[CSINN_OP_ARGMIN].est = shl_gref_argmin;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ASIN_DISABLED
    cb_map[CSINN_OP_ASIN].est = shl_gref_asin;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ASINH_DISABLED
    cb_map[CSINN_OP_ASINH].est = shl_gref_asinh;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ATAN_DISABLED
    cb_map[CSINN_OP_ATAN].est = shl_gref_atan;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ATANH_DISABLED
    cb_map[CSINN_OP_ATANH].est = shl_gref_atanh;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_AVERAGEPOOL_DISABLED
    cb_map[CSINN_OP_AVGPOOL2D].est = shl_gref_avgpool2d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_AVERAGEPOOL3D_DISABLED
    cb_map[CSINN_OP_AVGPOOL3D].est = shl_gref_avgpool3d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_BATCH_NORMALIZATION_DISABLED
    cb_map[CSINN_OP_BN].est = shl_gref_batch_normalization;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_BATCH_TO_SPACE_DISABLED
    cb_map[CSINN_OP_BATCH_TO_SPACE].est = shl_gref_batch_to_space;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_BATCH_TO_SPACE_ND_DISABLED
    cb_map[CSINN_OP_BATCH_TO_SPACE_ND].est = shl_gref_batch_to_space_nd;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_BROADCAST_TO_DISABLED
    cb_map[CSINN_OP_BROADCOST].est = shl_gref_broadcast_to;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CACHE_MATMUL_DISABLED
    cb_map[CSINN_OP_CACHE_MATMUL].est = shl_gref_cache_matmul;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CACHE_CONV1D_DISABLED
    cb_map[CSINN_OP_CACHE_CONV1D].est = shl_gref_cache_conv1d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CEIL_DISABLED
    cb_map[CSINN_OP_CEIL].est = shl_gref_ceil;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CLIP_DISABLED
    cb_map[CSINN_OP_CLIP].est = shl_gref_clip;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_COL2IM_DISABLED
    cb_map[CSINN_OP_COL2IM].est = shl_gref_col2im;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONCAT_DISABLED
    cb_map[CSINN_OP_CONCAT].est = shl_gref_concat;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONVOLUTION1D_DISABLED
    cb_map[CSINN_OP_CONV1D].est = shl_gref_conv1d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONVOLUTION_DISABLED
    cb_map[CSINN_OP_CONV2D].est = shl_gref_conv2d;
    cb_map[CSINN_OP_CONV2D_RELU].est = shl_gref_conv2d_relu;
    cb_map[CSINN_OP_CONV2D_RELU6].est = shl_gref_conv2d_relu6;
    cb_map[CSINN_OP_DEPTHWISE_CONV2D].est = shl_gref_depthwise_conv2d;
    cb_map[CSINN_OP_DEPTHWISE_CONV2D_RELU].est = shl_gref_depthwise_conv2d_relu;
    cb_map[CSINN_OP_DEPTHWISE_CONV2D_RELU6].est = shl_gref_depthwise_conv2d_relu6;
    cb_map[CSINN_OP_GROUP_CONV2D].est = shl_gref_group_conv2d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_DATA_CONVERT_DISABLED
    cb_map[CSINN_OP_DATA_CONVERT].est = shl_gref_data_convert;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONVOLUTION3D_DISABLED
    cb_map[CSINN_OP_CONV3D].est = shl_gref_conv3d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_DECONVOLUTION_DISABLED
    cb_map[CSINN_OP_DECONV2D].est = shl_gref_deconv2d;
    cb_map[CSINN_OP_DEPTHWISE_DECONV2D].est = shl_gref_depthwise_deconv2d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_DECONVOLUTION3D_DISABLED
    cb_map[CSINN_OP_DECONV3D].est = shl_gref_deconv3d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_COS_DISABLED
    cb_map[CSINN_OP_COS].est = shl_gref_cos;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_COSH_DISABLED
    cb_map[CSINN_OP_COSH].est = shl_gref_cosh;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CUMPROD_DISABLED
    cb_map[CSINN_OP_CUMPROD].est = shl_gref_cumprod;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CUMSUM_DISABLED
    cb_map[CSINN_OP_CUMSUM].est = shl_gref_cumsum;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_DEPTH_TO_SPACE_DISABLED
    cb_map[CSINN_OP_DEPTH_TO_SPACE].est = shl_gref_depth_to_space;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_DIV_DISABLED
    cb_map[CSINN_OP_DIV].est = shl_gref_div;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ELU_DISABLED
    cb_map[CSINN_OP_ELU].est = shl_gref_elu;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_EQUAL_DISABLED
    cb_map[CSINN_OP_EQUANL].est = shl_gref_equal;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ERF_DISABLED
    cb_map[CSINN_OP_ERF].est = shl_gref_erf;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_EXP_DISABLED
    cb_map[CSINN_OP_EXP].est = shl_gref_exp;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_EXPAND_DIMS_DISABLED
    cb_map[CSINN_OP_EXPAND_DIMS].est = shl_gref_expand_dims;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_EXPM1_DISABLED
    cb_map[CSINN_OP_EXPM1].est = shl_gref_expm1;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FLATTEN_DISABLED
    cb_map[CSINN_OP_FLATTEN].est = shl_gref_flatten;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FLOOR_DIVIDE_DISABLED
    cb_map[CSINN_OP_FLOOR_DIVIDE].est = shl_gref_floor_divide;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FLOOR_MOD_DISABLED
    cb_map[CSINN_OP_FLOOR_MOD].est = shl_gref_floor_mod;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FLOOR_DISABLED
    cb_map[CSINN_OP_FLOOR].est = shl_gref_floor;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FSMN_DISABLED
    cb_map[CSINN_OP_FSMN].est = shl_gref_fsmn;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FULLYCONNECTED_DISABLED
    cb_map[CSINN_OP_FULLYCONNECTED].est = shl_gref_fullyconnected;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GATHER_ND_DISABLED
    cb_map[CSINN_OP_GATHER_ND].est = shl_gref_gather_nd;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GATHER_DISABLED
    cb_map[CSINN_OP_GATHER].est = shl_gref_gather;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GLOBAL_AVERAGEPOOL_DISABLED
    cb_map[CSINN_OP_GLOBAL_AVGPOOL2D].est = shl_gref_global_avgpool2d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GLOBAL_MAXPOOL_DISABLED
    cb_map[CSINN_OP_GLOBAL_MAXPOOL2D].est = shl_gref_global_maxpool2d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GREATER_EQUAL_DISABLED
    cb_map[CSINN_OP_GREATHER_EQUAL].est = shl_gref_greater_equal;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GREATER_DISABLED
    cb_map[CSINN_OP_GREATHER].est = shl_gref_greater;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_HARD_SIGMOID_DISABLED
    cb_map[CSINN_OP_HARD_SIGMOID].est = shl_gref_hard_sigmoid;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_IM2COL_DISABLED
    cb_map[CSINN_OP_IM2COL].est = shl_gref_im2col;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ISNAN_DISABLED
    cb_map[CSINN_OP_ISNAN].est = shl_gref_isnan_bool;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LAYER_NORMAL_DISABLED
    cb_map[CSINN_OP_LAYER_NORM].est = shl_gref_layer_norm;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_L2_NORMALIZATION_DISABLED
    cb_map[CSINN_OP_L2N].est = shl_gref_l2_normalization;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_L2POOL_DISABLED
    cb_map[CSINN_OP_L2POOL2D].est = shl_gref_l2pool;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LEAKY_RELU_DISABLED
    cb_map[CSINN_OP_LEAKY_RELU].est = shl_gref_leaky_relu;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LESS_EQUAL_DISABLED
    cb_map[CSINN_OP_LESS_EQUAL].est = shl_gref_less_equal;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LESS_DISABLED
    cb_map[CSINN_OP_LESS].est = shl_gref_less;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LOG_SOFTMAX_DISABLED
    cb_map[CSINN_OP_LOG_SOFTMAX].est = shl_gref_log_softmax;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LOG_DISABLED
    cb_map[CSINN_OP_LOG].est = shl_gref_log;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LOG1P_DISABLED
    cb_map[CSINN_OP_LOG1P].est = shl_gref_log1p;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LOGICAL_AND_DISABLED
    cb_map[CSINN_OP_LOGICAL_AND].est = shl_gref_logical_and;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LOGICAL_NOT_DISABLED
    cb_map[CSINN_OP_LOGICAL_NOT].est = shl_gref_logical_not;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LOGICAL_OR_DISABLED
    cb_map[CSINN_OP_LOGICAL_OR].est = shl_gref_logical_or;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LOGICAL_XOR_DISABLED
    cb_map[CSINN_OP_LOGICAL_XOR].est = shl_gref_logical_xor;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LRN_DISABLED
    cb_map[CSINN_OP_LRN].est = shl_gref_lrn;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MATMUL_DISABLED
    cb_map[CSINN_OP_MATMUL].est = shl_gref_matmul;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MAX_DISABLED
    cb_map[CSINN_OP_MAX].est = shl_gref_max;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MAXIMUM_DISABLED
    cb_map[CSINN_OP_MAXIMUM].est = shl_gref_maximum;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MAXPOOL_DISABLED
    cb_map[CSINN_OP_MAXPOOL2D].est = shl_gref_maxpool2d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MAXPOOL2D_LOCAT_DISABLED
    cb_map[CSINN_OP_MAXPOOL2D_LOCAT].est = shl_gref_maxpool2d_locat;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MAXPOOL3D_DISABLED
    cb_map[CSINN_OP_MAXPOOL3D].est = shl_gref_maxpool3d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MEAN_DISABLED
    cb_map[CSINN_OP_MEAN].est = shl_gref_mean;
    cb_map[CSINN_OP_MEAN_STRIDE].est = shl_gref_mean;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MIN_DISABLED
    cb_map[CSINN_OP_MIN].est = shl_gref_min;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MINIMUM_DISABLED
    cb_map[CSINN_OP_MINIMUM].est = shl_gref_minimum;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MOD_DISABLED
    cb_map[CSINN_OP_MOD].est = shl_gref_mod;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MUL_DISABLED
    cb_map[CSINN_OP_MUL].est = shl_gref_mul;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_NDARRAY_SIZE_DISABLED
    cb_map[CSINN_OP_NDARRAY_SIZE].est = shl_gref_ndarray_size;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_NEGATIVE_DISABLED
    cb_map[CSINN_OP_NEGATIVE].est = shl_gref_negative;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_NON_MAX_SUPPRESSION_DISABLED
    cb_map[CSINN_OP_NON_MAX_SUPPRESSION].est = shl_gref_non_max_suppression;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_NOT_EQUAL_DISABLED
    cb_map[CSINN_OP_NOT_EQUAL].est = shl_gref_not_equal;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_NOT_DISABLED
    cb_map[CSINN_OP_NOT].est = shl_gref_not;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_OR_DISABLED
    cb_map[CSINN_OP_OR].est = shl_gref_or;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_PAD_DISABLED
    cb_map[CSINN_OP_PAD].est = shl_gref_pad;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_POWER_DISABLED
    cb_map[CSINN_OP_POWER].est = shl_gref_power;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_PRELU_DISABLED
    cb_map[CSINN_OP_PRELU].est = shl_gref_prelu;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_PROD_DISABLED
    cb_map[CSINN_OP_PROD].est = shl_gref_prod;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_PROPOSAL_DISABLED
    cb_map[CSINN_OP_PROPOSAL].est = shl_gref_proposal;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_PSROIPOOLING_DISABLED
    cb_map[CSINN_OP_PSROIPOOLING].est = shl_gref_psroipooling;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_REDUCE_LOGSUMEXP_DISABLED
    cb_map[CSINN_OP_REDUCE_LOGSUMEXP].est = shl_gref_reduce_logsumexp;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_REDUCE_MAX_DISABLED
    cb_map[CSINN_OP_REDUCE_MAX].est = shl_gref_reduce_max;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_REDUCE_MEAN_DISABLED
    cb_map[CSINN_OP_REDUCE_MEAN].est = shl_gref_reduce_mean;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_REDUCE_MIN_DISABLED
    cb_map[CSINN_OP_REDUCE_MIN].est = shl_gref_reduce_min;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_REDUCE_PROD_DISABLED
    cb_map[CSINN_OP_REDUCE_PROD].est = shl_gref_reduce_prod;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_REDUCE_SUM_DISABLED
    cb_map[CSINN_OP_REDUCE_SUM].est = shl_gref_reduce_sum;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU_DISABLED
    cb_map[CSINN_OP_RELU].est = shl_gref_relu;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU1_DISABLED
    cb_map[CSINN_OP_RELU1].est = shl_gref_relu1;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU6_DISABLED
    cb_map[CSINN_OP_RELU6].est = shl_gref_relu6;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELUN_DISABLED
    cb_map[CSINN_OP_RELUN].est = shl_gref_relun;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RESHAPE_DISABLED
    cb_map[CSINN_OP_RESHAPE].est = shl_gref_reshape;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RESIZE_DISABLED
    cb_map[CSINN_OP_RESIZE].est = shl_gref_resize;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_REVERSE_DISABLED
    cb_map[CSINN_OP_REVERSE].est = shl_gref_reverse;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ROIALIGN_DISABLED
    cb_map[CSINN_OP_ROIALIGN].est = shl_gref_roi_align;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ROIPOOL_DISABLED
    cb_map[CSINN_OP_ROIPOOL].est = shl_gref_roipool;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ROUND_DISABLED
    cb_map[CSINN_OP_ROUND].est = shl_gref_round;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RSQRT_DISABLED
    cb_map[CSINN_OP_RSQRT].est = shl_gref_rsqrt;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SCATTER_DISABLED
    cb_map[CSINN_OP_SCATTER_ND].est = shl_gref_scatter_nd;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SEGMENT_MAX_DISABLED
    cb_map[CSINN_OP_SEGMENT_MAX].est = shl_gref_segment_max;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SEGMENT_MEAN_DISABLED
    cb_map[CSINN_OP_SEGMENT_MEAN].est = shl_gref_segment_mean;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SEGMENT_MIN_DISABLED
    cb_map[CSINN_OP_SEGMENT_MIN].est = shl_gref_segment_min;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SEGMENT_PROD_DISABLED
    cb_map[CSINN_OP_SEGMENT_PROD].est = shl_gref_segment_prod;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SEGMENT_SUM_DISABLED
    cb_map[CSINN_OP_SEGMENT_SUM].est = shl_gref_segment_sum;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SELECT_DISABLED
    cb_map[CSINN_OP_SELECT].est = shl_gref_select;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SEQUENCE_MASK_DISABLED
    cb_map[CSINN_OP_SEQUENCE_MASK].est = shl_gref_sequence_mask;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SHAPE_DISABLED
    cb_map[CSINN_OP_SHAPE].est = shl_gref_shape;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SHUFFLE_CHANNEL_DISABLED
    cb_map[CSINN_OP_SHUFFLE_CHANNEL].est = shl_gref_shuffle_channel;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SIGMOID_DISABLED
    cb_map[CSINN_OP_SIGMOID].est = shl_gref_sigmoid;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SIGN_DISABLED
    cb_map[CSINN_OP_SIGN].est = shl_gref_sign;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SIN_DISABLED
    cb_map[CSINN_OP_SIN].est = shl_gref_sin;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SINH_DISABLED
    cb_map[CSINN_OP_SINH].est = shl_gref_sinh;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SLICE_DISABLED
    cb_map[CSINN_OP_SLICE].est = shl_gref_slice;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SOFTMAX_DISABLED
    cb_map[CSINN_OP_SOFTMAX].est = shl_gref_softmax;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SOFTPLUS_DISABLED
    cb_map[CSINN_OP_SOFTPLUS].est = shl_gref_softplus;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SOFTRELU_DISABLED
    cb_map[CSINN_OP_SOFTRELU].est = shl_gref_softrelu;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SOFTSIGN_DISABLED
    cb_map[CSINN_OP_SOFTSIGN].est = shl_gref_softsign;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SPACE_TO_BATCH_DISABLED
    cb_map[CSINN_OP_SPACE_TO_BATCH].est = shl_gref_space_to_batch;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SPACE_TO_BATCH_ND_DISABLED
    cb_map[CSINN_OP_SPACE_TO_BATCH_ND].est = shl_gref_space_to_batch_nd;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SPACE_TO_DEPTH_DISABLED
    cb_map[CSINN_OP_SPACE_TO_DEPTH].est = shl_gref_space_to_depth;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SPLIT_DISABLED
    cb_map[CSINN_OP_SPLIT].est = shl_gref_split;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SQRT_DISABLED
    cb_map[CSINN_OP_SQRT].est = shl_gref_sqrt;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SQUARE_DISABLED
    cb_map[CSINN_OP_SQUARE].est = shl_gref_square;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SQUEEZE_DISABLED
    cb_map[CSINN_OP_SQUEEZE].est = shl_gref_squeeze;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_STACK_DISABLED
    cb_map[CSINN_OP_STACK].est = shl_gref_stack;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_STRIDED_SLICE_DISABLED
    cb_map[CSINN_OP_STRIDED_SLICE].est = shl_gref_strided_slice;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SUB_DISABLED
    cb_map[CSINN_OP_SUB].est = shl_gref_sub;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SUM_DISABLED
    cb_map[CSINN_OP_SUM].est = shl_gref_sum;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_TAN_DISABLED
    cb_map[CSINN_OP_TAN].est = shl_gref_tan;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_TANH_DISABLED
    cb_map[CSINN_OP_TANH].est = shl_gref_tanh;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_THRESHOLD_RELU_DISABLED
    cb_map[CSINN_OP_THRESHOLD_RELU].est = shl_gref_threshold_relu;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_TILE_DISABLED
    cb_map[CSINN_OP_TILE].est = shl_gref_tile;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_TOPK_DISABLED
    cb_map[CSINN_OP_TOPK].est = shl_gref_topk;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_TRUNC_DISABLED
    cb_map[CSINN_OP_TRUNC].est = shl_gref_trunc;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_TRANSPOSE_DISABLED
    cb_map[CSINN_OP_TRANSPOSE].est = shl_gref_transpose;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_UNPOOLING_DISABLED
    cb_map[CSINN_OP_UNPOOLING].est = shl_gref_unpooling;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_UNSTACK_DISABLED
    cb_map[CSINN_OP_UNSTACK].est = shl_gref_unstack;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_WHERE_DISABLED
    cb_map[CSINN_OP_WHERE].est = shl_gref_where;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_XOR_DISABLED
    cb_map[CSINN_OP_XOR].est = shl_gref_xor;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_YUV_RGB_SCALE_DISABLED
    cb_map[CSINN_OP_YUV_RGB_SCALE].est = shl_gref_yuv_rgb_scale;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ONE_HOT_DISABLED
    cb_map[CSINN_OP_ONE_HOT].est = shl_gref_one_hot;
#endif

    return cb_map;
}

static int get_cb_map_index(int op, int dtype) { return op; }
static struct csinn_callback *__cb_map_table_gref;

struct csinn_callback *shl_cb_map_gref(int op, int dtype)
{
    return &__cb_map_table_gref[get_cb_map_index(op, dtype)];
}

void *shl_gref_runtime_callback(int api)
{
    switch (api) {
        case CSINN_SESSION_INIT:
            return shl_gref_session_init;
            break;
        case CSINN_SESSION_DEINIT:
            return shl_gref_session_deinit;
            break;
        case CSINN_SESSION_SETUP:
            return shl_gref_session_setup;
            break;
        case CSINN_SESSION_RUN:
            return shl_gref_session_run;
            break;
        case CSINN_UPDATE_INPUT:
            return shl_gref_update_input;
            break;
        case CSINN_UPDATE_OUTPUT:
            return shl_gref_update_output;
            break;
        case CSINN_SET_INPUT_NUMBER:
            return shl_gref_set_input_number;
            break;
        case CSINN_SET_OUTPUT_NUMBER:
            return shl_gref_set_output_number;
            break;
        case CSINN_SET_INPUT:
            return shl_gref_set_input;
            break;
        case CSINN_SET_OUTPUT:
            return shl_gref_set_output;
            break;
        case CSINN_GET_INPUT:
            return shl_gref_get_input;
            break;
        case CSINN_GET_OUTPUT:
            return shl_gref_get_output;
            break;
        case CSINN_TENSOR_ENTRY:
            return shl_gref_set_tensor;
            break;
        default:
            shl_debug_info("%s: Cannot find callback\n", __func__);
            break;
    }
    return NULL;
}

void shl_target_init_gref()
{
    __cb_map_table_gref = setup_cb_map();
    shl_register_runtime_callback(CSINN_GREF, shl_gref_runtime_callback);
    shl_register_op_callback(CSINN_GREF, shl_cb_map_gref);
}
