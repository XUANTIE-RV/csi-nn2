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
        case CSINN_OP_ONE_HOT:
            shl_debug_error("unsupported CSINN_OP_ONE_HOT\n");
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
        case CSINN_OP_WHERE:
            shl_debug_error("unsupported CSINN_OP_WHERE\n");
            break;
        default:
            shl_debug_error("unknown op\n");
            return CSINN_FALSE;
    }
    return ret;
}

void shl_gref_reset_graph_visit(struct shl_ref_graph *graph)
{
    for (int i = 0; i < graph->layer_index; i++) {
        if (graph->layer[i]->type == CSINN_SUBGRAPH) {
            graph->layer[i]->visited = 0;
            struct shl_ref_graph *s_subgraph = graph->layer[i]->data;
            for (int j = 0; j < s_subgraph->layer_index; j++) {
                s_subgraph->layer[j]->visited = 0;
            }
        } else {
            graph->layer[i]->visited = 0;
        }
    }
}

/*
 * transform graph as gloal graph and sub graph
 */
static struct shl_ref_graph *transform_graph(struct shl_ref_graph *ograph)
{
    struct shl_ref_graph *ggraph = shl_mem_alloc(sizeof(struct shl_ref_graph));
    ggraph->input = ograph->input;
    ggraph->output = ograph->output;
    ggraph->input_num = ograph->input_num;
    ggraph->output_num = ograph->output_num;
    for (int i = 0; i < ograph->layer_index; i++) {
        struct shl_node *n = ograph->layer[i];
        struct csinn_params_base *params = n->data;

        if (params->sess->base_api != params->api) {
            shl_subgraph_alloc(n, ograph, ggraph);
        } else {
            shl_gref_graph_insert(n, ggraph);
        }
    }
    return ggraph;
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

void shl_subgraph_fvisit_create(struct shl_ref_graph *graph, struct shl_node *node)
{
    shl_gref_graph_insert(node, graph);
}

/*
 * transform graph as gloal graph and sub graph
 */
static struct shl_ref_graph *convert_graph(struct shl_ref_graph *ograph)
{
    if (shl_debug_get_level() <= SHL_DEBUG_LEVEL_INFO) {
        shl_debug_info("\nOriginal graph:\n");
        shl_gref_post_dfs(ograph, shl_subgraph_fvisit_print);
        shl_gref_reset_graph_visit(ograph);
    }

    struct shl_ref_graph *subgraph = shl_subgraph_generate(ograph);
    shl_gref_reset_graph_visit(subgraph);

    shl_debug_info("\nGenerated subgraph:\n");
    for (int i = 0; i < subgraph->layer_index; i++) {
        if (subgraph->layer[i]->type == CSINN_SUBGRAPH) {
            struct shl_ref_graph *s_subgraph = subgraph->layer[i]->data;
            if (s_subgraph->layer_size == 0) continue;
            shl_gref_update_input_output(subgraph, i);
            if (shl_debug_get_level() <= SHL_DEBUG_LEVEL_INFO) {
                shl_debug_info("----  subgraph_%d:  ----\n", i);
                shl_gref_reset_graph_visit(s_subgraph);
                shl_gref_post_dfs(s_subgraph, shl_subgraph_fvisit_print);
                shl_gref_reset_graph_visit(s_subgraph);
                shl_debug_info("----subgraph_%d end.----\n", i);
            }

            struct shl_ref_graph *new_sgraph = shl_mem_alloc(sizeof(struct shl_ref_graph));
            new_sgraph->input = s_subgraph->input;
            new_sgraph->output = s_subgraph->output;
            new_sgraph->input_num = s_subgraph->input_num;
            new_sgraph->output_num = s_subgraph->output_num;
            shl_gref_post_dfs(new_sgraph, shl_subgraph_fvisit_create);
            subgraph->layer[i]->data = new_sgraph;

            shl_gref_reset_graph_visit(s_subgraph);
        } else {
            shl_debug_info("%s\n", subgraph->layer[i]->name);
        }
    }

    shl_gref_reset_graph_visit(subgraph);
    struct shl_ref_graph *ggraph = shl_subgraph_rebuild(subgraph);

    struct shl_ref_graph *sorted_graph = shl_subgraph_topology_sort(ggraph);
    shl_debug_info("\nsorted subgraph:\n");
    for (int i = 0; i < sorted_graph->layer_index; i++) {
        if (sorted_graph->layer[i]->type == CSINN_SUBGRAPH) {
            struct shl_ref_graph *s_subgraph = sorted_graph->layer[i]->data;
            if (s_subgraph->layer_size == 0) continue;
            if (shl_debug_get_level() <= SHL_DEBUG_LEVEL_INFO) {
                shl_debug_info("----  subgraph_%d:  ----\n", i);
                shl_gref_reset_graph_visit(s_subgraph);
                shl_gref_post_dfs(s_subgraph, shl_subgraph_fvisit_print);
                shl_gref_reset_graph_visit(s_subgraph);
                shl_debug_info("----subgraph_%d end.----\n", i);
            }
            shl_gref_reset_graph_visit(s_subgraph);
        } else {
            shl_debug_info("%s\n", sorted_graph->layer[i]->name);
        }
    }

    return sorted_graph;
}

void shl_gref_session_setup(struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    struct shl_node *n;

    for (int i = 0; i < graph->layer_index; i++) {
        n = graph->layer[i];
        for (int j = 0; j < n->in_num; j++) {
            if (n->in[j]->ref_count_init > 0) {
                n->in[j]->ref_count_init++;
            }
        }
        for (int k = 0; k < n->out_num; k++) {
            n->out[k]->ref_count_init++;
        }
    }

    for (int i = 0; i < graph->output_num; i++) {
        graph->output[i]->ref_count_init++;
    }

    struct shl_ref_graph *ggraph = convert_graph(graph);

    for (int i = 0; i < ggraph->layer_index; i++) {
        struct shl_node *n = ggraph->layer[i];
        if (n->type == CSINN_SUBGRAPH) {
            shl_subgraph_setup(n);
        } else if (n->type >= 0 && n->type < CSINN_OP_SIZE) {
            init_op(n);
        } else {
            shl_debug_error("Unknown layer\n");
            return;
        }
    }
    struct shl_gref_target_data *td = sess->td;
    td->graph = ggraph;
}

static void node_ref_reset(struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    struct shl_node *n;

    for (int i = 0; i < graph->layer_index; i++) {
        n = graph->layer[i];
        for (int k = 0; k < n->out_num; k++) {
            if (n->out[k] != NULL) {
                n->out[k]->ref_count = n->out[k]->ref_count_init;
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

static int op_run_deinit(struct shl_node *node)
{
    for (int i = 0; i < node->in_num; i++) {
        if (node->in[i]->ref_count > 0) {
            node->in[i]->ref_count--;
            if (node->in[i]->ref_count == 0) {
                struct csinn_tensor *t = node->in[i]->data;
                shl_mem_free(t->data);
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
            shl_subgraph_run_init(n);
            shl_subgraph_run(n);
            shl_subgraph_run_deinit(n);
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
            op_run_deinit(n);
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
    struct shl_ref_graph *g = shl_gref_get_graph(sess);

    for (int i = 0; i < g->layer_index; i++) {
        struct shl_node *n = g->layer[i];
        if (n->type == CSINN_SUBGRAPH) {
            shl_subgraph_deinit(n);
        }
    }
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    shl_mem_free(graph->input);
    shl_mem_free(graph->output);
}

struct shl_ref_graph *shl_gref_get_graph(struct csinn_session *sess)
{
    struct shl_gref_target_data *td = sess->td;
    return td->graph;
}

int shl_gref_is_root_node(struct shl_ref_graph *graph, struct shl_node *node)
{
    int is_root = 1;
    for (int i = 0; i < node->in_num; i++) {
        struct csinn_tensor *in_tensor = node->in[i]->data;
        if (in_tensor->is_const) continue;
        int find_res = 0;
        for (int j = 0; j < graph->input_num; j++) {
            if (node->in[i] == graph->input[j]) {
                find_res = 1;
                break;
            }
        }
        if (find_res == 0) {
            is_root = 0;
            break;
        }
    }
    return is_root;
}

void shl_gref_post_dfs(struct shl_ref_graph *graph,
                       void (*fvisit)(struct shl_ref_graph *, struct shl_node *))
{
    int stack_size = 32;
    struct shl_node **node_stack = shl_mem_alloc(sizeof(struct shl_node *) * stack_size);
    int *input_idx_stack = shl_mem_alloc(sizeof(int) * stack_size);
    int stack_top = -1;

    struct shl_node *curr_node;
    for (int i = 0; i < graph->output_num; i++) {
        struct csinn_tensor *ot = graph->output[i]->data;
        if (ot->is_const) continue;
        curr_node = graph->output[i]->in[0];
        if (curr_node->visited == 0) {
            ++stack_top;
            if (stack_top >= stack_size) {
                stack_size += 32;
                node_stack = shl_mem_realloc(node_stack, sizeof(struct shl_node *) * stack_size);
                input_idx_stack = shl_mem_realloc(input_idx_stack, sizeof(int) * stack_size);
            }
            node_stack[stack_top] = curr_node;
            input_idx_stack[stack_top] = 0;
            curr_node->visited = 1;
        }
        while (stack_top != -1) {
            curr_node = node_stack[stack_top];
            if (input_idx_stack[stack_top] == shl_node_get_non_const_in_number(curr_node)) {
                fvisit(graph, curr_node);
                --stack_top;
            } else {
                struct shl_node *next_node = NULL;
                if (shl_node_find(graph->input, graph->input_num,
                                  curr_node->in[input_idx_stack[stack_top]]) == -1) {
                    next_node = curr_node->in[input_idx_stack[stack_top]]->in[0];
                    if (next_node && next_node->type == CSINN_SUBGRAPH_RETURN) {
                        next_node = graph->layer[next_node->subgraph_idx];
                    }
                }
                input_idx_stack[stack_top] += 1;
                if (next_node && next_node->visited == 0) {
                    ++stack_top;
                    if (stack_top >= stack_size) {
                        stack_size += 32;
                        node_stack =
                            shl_mem_realloc(node_stack, sizeof(struct shl_node *) * stack_size);
                        input_idx_stack =
                            shl_mem_realloc(input_idx_stack, sizeof(int) * stack_size);
                    }
                    node_stack[stack_top] = next_node;
                    input_idx_stack[stack_top] = 0;
                    next_node->visited = 1;
                }
            }
        }
    }

    shl_mem_free(node_stack);
    shl_mem_free(input_idx_stack);
}

void shl_gref_update_input_output(struct shl_ref_graph *ograph, int index)
{
    if (ograph->layer[index]->type != CSINN_SUBGRAPH) {
        return;
    }
    struct shl_ref_graph *graph = ograph->layer[index]->data;
    if (graph->layer_size == 0) return;

    /* update inputs */
    graph->input = NULL;
    graph->input_num = 0;
    struct shl_node **tensor_node_set = NULL;
    int set_num = 0;
    for (int i = 0; i < graph->layer_index; i++) {
        for (int j = 0; j < shl_node_get_non_const_in_number(graph->layer[i]); j++) {
            struct shl_node *in_tensor_node = graph->layer[i]->in[j];
            if (shl_node_find(graph->layer, graph->layer_index, in_tensor_node->in[0]) == -1 &&
                shl_node_find(tensor_node_set, set_num, in_tensor_node) == -1) {
                graph->input = shl_mem_realloc(graph->input,
                                               sizeof(struct shl_node *) * (graph->input_num + 1));
                graph->input[graph->input_num] = in_tensor_node;
                graph->input_num++;

                // tensor_node_set[set_num] = in_tensor_node;
                tensor_node_set =
                    shl_mem_realloc(tensor_node_set, sizeof(struct shl_node *) * (set_num + 1));
                tensor_node_set[set_num] = in_tensor_node;
                set_num++;
            }
        }
    }
    shl_mem_free(tensor_node_set);

    /* update outputs */
    graph->output = NULL;
    graph->output_num = 0;
    for (int i = 0; i < graph->layer_index; i++) {
        for (int j = 0; j < graph->layer[i]->out_num; j++) {
            struct shl_node *out_tensor_node = graph->layer[i]->out[j];

            int find_res_inside = 0;
            for (int k = 0; k < graph->layer_index; k++) {
                if (k == i) continue;
                if (shl_node_find(graph->layer[k]->in, graph->layer[k]->in_num, out_tensor_node) >
                    -1) {
                    find_res_inside = 1;
                    break;
                }
            }

            int find_res_outside = 0;
            for (int s_idx = 0; s_idx < ograph->layer_index; s_idx++) {
                if (s_idx == index) continue;
                if (ograph->layer[s_idx]->type != CSINN_SUBGRAPH) {
                    if (shl_node_find(ograph->layer[s_idx]->in, ograph->layer[s_idx]->in_num,
                                      out_tensor_node) > -1) {
                        find_res_outside = 1;
                        break;
                    }
                } else {
                    struct shl_ref_graph *outside_sgraph = ograph->layer[s_idx]->data;
                    if (outside_sgraph->layer_size == 0) continue;

                    for (int inner_idx = 0; inner_idx < outside_sgraph->layer_index; inner_idx++) {
                        if (shl_node_find(outside_sgraph->layer[inner_idx]->in,
                                          outside_sgraph->layer[inner_idx]->in_num,
                                          out_tensor_node) > -1) {
                            find_res_outside = 1;
                            break;
                        }
                    }
                    if (find_res_outside) {
                        break;
                    }
                }
            }

            if (!find_res_inside || find_res_outside) {
                graph->output = shl_mem_realloc(
                    graph->output, sizeof(struct shl_node *) * (graph->output_num + 1));
                graph->output[graph->output_num] = out_tensor_node;
                graph->output_num++;
            }
        }
    }
}

static void *setup_cb_map()
{
    static struct csinn_callback cb_map[CSINN_OP_AND_UTILS_SIZE];
    memset(cb_map, 0, sizeof(struct csinn_callback) * CSINN_OP_AND_UTILS_SIZE);

    cb_map[CSINN_OP_ABS].est = shl_gref_abs;
    cb_map[CSINN_OP_ACOS].est = shl_gref_acos;
    cb_map[CSINN_OP_ACOSH].est = shl_gref_acosh;
    cb_map[CSINN_OP_ADD].est = shl_gref_add;
    cb_map[CSINN_OP_ALL].est = shl_gref_all;
    cb_map[CSINN_OP_AND].est = shl_gref_and;
    cb_map[CSINN_OP_ANY].est = shl_gref_any;
    cb_map[CSINN_OP_ARANGE].est = shl_gref_arange;
    cb_map[CSINN_OP_ARGMAX].est = shl_gref_argmax;
    cb_map[CSINN_OP_ARGMIN].est = shl_gref_argmin;
    cb_map[CSINN_OP_ASIN].est = shl_gref_asin;
    cb_map[CSINN_OP_ASINH].est = shl_gref_asinh;
    cb_map[CSINN_OP_ATAN].est = shl_gref_atan;
    cb_map[CSINN_OP_ATANH].est = shl_gref_atanh;
    cb_map[CSINN_OP_AVGPOOL2D].est = shl_gref_avgpool2d;
    cb_map[CSINN_OP_AVGPOOL3D].est = shl_gref_avgpool3d;
    cb_map[CSINN_OP_BN].est = shl_gref_batch_normalization;
    cb_map[CSINN_OP_BATCH_TO_SPACE].est = shl_gref_batch_to_space;
    cb_map[CSINN_OP_BATCH_TO_SPACE_ND].est = shl_gref_batch_to_space_nd;
    cb_map[CSINN_OP_BROADCOST].est = shl_gref_broadcast_to;
    cb_map[CSINN_OP_CACHE_MATMUL].est = shl_gref_cache_matmul;
    cb_map[CSINN_OP_CACHE_CONV1D].est = shl_gref_cache_conv1d;
    cb_map[CSINN_OP_CEIL].est = shl_gref_ceil;
    cb_map[CSINN_OP_CLIP].est = shl_gref_clip;
    cb_map[CSINN_OP_COL2IM].est = shl_gref_col2im;
    cb_map[CSINN_OP_CONCAT].est = shl_gref_concat;
    cb_map[CSINN_OP_CONV1D].est = shl_gref_conv1d;
    cb_map[CSINN_OP_CONV2D].est = shl_gref_conv2d;
    cb_map[CSINN_OP_CONV2D_RELU].est = shl_gref_conv2d_relu;
    cb_map[CSINN_OP_CONV2D_RELU6].est = shl_gref_conv2d_relu6;
    cb_map[CSINN_OP_DATA_CONVERT].est = shl_gref_data_convert;
    cb_map[CSINN_OP_DEPTHWISE_CONV2D].est = shl_gref_depthwise_conv2d;
    cb_map[CSINN_OP_DEPTHWISE_CONV2D_RELU].est = shl_gref_depthwise_conv2d_relu;
    cb_map[CSINN_OP_DEPTHWISE_CONV2D_RELU6].est = shl_gref_depthwise_conv2d_relu6;
    cb_map[CSINN_OP_GROUP_CONV2D].est = shl_gref_group_conv2d;
    cb_map[CSINN_OP_CONV3D].est = shl_gref_conv3d;
    cb_map[CSINN_OP_DECONV2D].est = shl_gref_deconv2d;
    cb_map[CSINN_OP_DEPTHWISE_DECONV2D].est = shl_gref_depthwise_deconv2d;
    cb_map[CSINN_OP_DECONV3D].est = shl_gref_deconv3d;
    cb_map[CSINN_OP_COS].est = shl_gref_cos;
    cb_map[CSINN_OP_COSH].est = shl_gref_cosh;
    cb_map[CSINN_OP_CUMPROD].est = shl_gref_cumprod;
    cb_map[CSINN_OP_CUMSUM].est = shl_gref_cumsum;
    cb_map[CSINN_OP_DEPTH_TO_SPACE].est = shl_gref_depth_to_space;
    cb_map[CSINN_OP_DIV].est = shl_gref_div;
    cb_map[CSINN_OP_ELU].est = shl_gref_elu;
    cb_map[CSINN_OP_EQUANL].est = shl_gref_equal;
    cb_map[CSINN_OP_ERF].est = shl_gref_erf;
    cb_map[CSINN_OP_EXP].est = shl_gref_exp;
    cb_map[CSINN_OP_EXPAND_DIMS].est = shl_gref_expand_dims;
    cb_map[CSINN_OP_EXPM1].est = shl_gref_expm1;
    cb_map[CSINN_OP_FLATTEN].est = shl_gref_flatten;
    cb_map[CSINN_OP_FLOOR_DIVIDE].est = shl_gref_floor_divide;
    cb_map[CSINN_OP_FLOOR_MOD].est = shl_gref_floor_mod;
    cb_map[CSINN_OP_FLOOR].est = shl_gref_floor;
    cb_map[CSINN_OP_FSMN].est = shl_gref_fsmn;
    cb_map[CSINN_OP_FULLYCONNECTED].est = shl_gref_fullyconnected;
    cb_map[CSINN_OP_GATHER_ND].est = shl_gref_gather_nd;
    cb_map[CSINN_OP_GATHER].est = shl_gref_gather;
    cb_map[CSINN_OP_GLOBAL_AVGPOOL2D].est = shl_gref_global_avgpool2d;
    cb_map[CSINN_OP_GLOBAL_MAXPOOL2D].est = shl_gref_global_maxpool2d;
    cb_map[CSINN_OP_GREATHER_EQUAL].est = shl_gref_greater_equal;
    cb_map[CSINN_OP_GREATHER].est = shl_gref_greater;
    cb_map[CSINN_OP_HARD_SIGMOID].est = shl_gref_hard_sigmoid;
    cb_map[CSINN_OP_IM2COL].est = shl_gref_im2col;
    cb_map[CSINN_OP_ISNAN].est = shl_gref_isnan_bool;
    cb_map[CSINN_OP_LAYER_NORM].est = shl_gref_layer_norm;
    cb_map[CSINN_OP_L2N].est = shl_gref_l2_normalization;
    cb_map[CSINN_OP_L2POOL2D].est = shl_gref_l2pool;
    cb_map[CSINN_OP_LEAKY_RELU].est = shl_gref_leaky_relu;
    cb_map[CSINN_OP_LESS_EQUAL].est = shl_gref_less_equal;
    cb_map[CSINN_OP_LESS].est = shl_gref_less;
    cb_map[CSINN_OP_LOG_SOFTMAX].est = shl_gref_log_softmax;
    cb_map[CSINN_OP_LOG].est = shl_gref_log;
    cb_map[CSINN_OP_LOG1P].est = shl_gref_log1p;
    cb_map[CSINN_OP_LOGICAL_AND].est = shl_gref_logical_and;
    cb_map[CSINN_OP_LOGICAL_NOT].est = shl_gref_logical_not;
    cb_map[CSINN_OP_LOGICAL_OR].est = shl_gref_logical_or;
    cb_map[CSINN_OP_LOGICAL_XOR].est = shl_gref_logical_xor;
    cb_map[CSINN_OP_LRN].est = shl_gref_lrn;
    cb_map[CSINN_OP_MATMUL].est = shl_gref_matmul;
    cb_map[CSINN_OP_MAX].est = shl_gref_max;
    cb_map[CSINN_OP_MAXIMUM].est = shl_gref_maximum;
    cb_map[CSINN_OP_MAXPOOL2D].est = shl_gref_maxpool2d;
    cb_map[CSINN_OP_MAXPOOL2D_LOCAT].est = shl_gref_maxpool2d_locat;
    cb_map[CSINN_OP_MAXPOOL3D].est = shl_gref_maxpool3d;
    cb_map[CSINN_OP_MEAN].est = shl_gref_mean;
    cb_map[CSINN_OP_MEAN_STRIDE].est = shl_gref_mean;
    cb_map[CSINN_OP_MIN].est = shl_gref_min;
    cb_map[CSINN_OP_MINIMUM].est = shl_gref_minimum;
    cb_map[CSINN_OP_MOD].est = shl_gref_mod;
    cb_map[CSINN_OP_MUL].est = shl_gref_mul;
    cb_map[CSINN_OP_NDARRAY_SIZE].est = shl_gref_ndarray_size;
    cb_map[CSINN_OP_NEGATIIVE].est = shl_gref_negative;
    cb_map[CSINN_OP_NON_MAX_SUPPRESSION].est = shl_gref_non_max_suppression;
    cb_map[CSINN_OP_NOT_EQUAL].est = shl_gref_not_equal;
    cb_map[CSINN_OP_NOT].est = shl_gref_not;
    cb_map[CSINN_OP_OR].est = shl_gref_or;
    cb_map[CSINN_OP_PAD].est = shl_gref_pad;
    cb_map[CSINN_OP_POWER].est = shl_gref_power;
    cb_map[CSINN_OP_PRELU].est = shl_gref_prelu;
    cb_map[CSINN_OP_PROD].est = shl_gref_prod;
    cb_map[CSINN_OP_PROPOSAL].est = shl_gref_proposal;
    cb_map[CSINN_OP_PSROIPOOLING].est = shl_gref_psroipooling;
    cb_map[CSINN_OP_REDUCE_LOGSUMEXP].est = shl_gref_reduce_logsumexp;
    cb_map[CSINN_OP_REDUCE_MAX].est = shl_gref_reduce_max;
    cb_map[CSINN_OP_REDUCE_MEAN].est = shl_gref_reduce_mean;
    cb_map[CSINN_OP_REDUCE_MIN].est = shl_gref_reduce_min;
    cb_map[CSINN_OP_REDUCE_PROD].est = shl_gref_reduce_prod;
    cb_map[CSINN_OP_REDUCE_SUM].est = shl_gref_reduce_sum;
    cb_map[CSINN_OP_RELU].est = shl_gref_relu;
    cb_map[CSINN_OP_RELU1].est = shl_gref_relu1;
    cb_map[CSINN_OP_RELU6].est = shl_gref_relu6;
    cb_map[CSINN_OP_RELUN].est = shl_gref_relun;
    cb_map[CSINN_OP_RESHAPE].est = shl_gref_reshape;
    cb_map[CSINN_OP_RESIZE].est = shl_gref_resize;
    cb_map[CSINN_OP_REVERSE].est = shl_gref_reverse;
    cb_map[CSINN_OP_ROIALIGN].est = shl_gref_roi_align;
    cb_map[CSINN_OP_ROIPOOL].est = shl_gref_roipool;
    cb_map[CSINN_OP_ROUND].est = shl_gref_round;
    cb_map[CSINN_OP_RSQRT].est = shl_gref_rsqrt;
    cb_map[CSINN_OP_SCATTER_ND].est = shl_gref_scatter_nd;
    cb_map[CSINN_OP_SEGMENT_MAX].est = shl_gref_segment_max;
    cb_map[CSINN_OP_SEGMENT_MEAN].est = shl_gref_segment_mean;
    cb_map[CSINN_OP_SEGMENT_MIN].est = shl_gref_segment_min;
    cb_map[CSINN_OP_SEGMENT_PROD].est = shl_gref_segment_prod;
    cb_map[CSINN_OP_SEGMENT_SUM].est = shl_gref_segment_sum;
    cb_map[CSINN_OP_SELECT].est = shl_gref_select;
    cb_map[CSINN_OP_SEQUENCE_MASK].est = shl_gref_sequence_mask;
    cb_map[CSINN_OP_SHAPE].est = shl_gref_shape;
    cb_map[CSINN_OP_SHUFFLE_CHANNEL].est = shl_gref_shuffle_channel;
    cb_map[CSINN_OP_SIGMOID].est = shl_gref_sigmoid;
    cb_map[CSINN_OP_SIGN].est = shl_gref_sign;
    cb_map[CSINN_OP_SIN].est = shl_gref_sin;
    cb_map[CSINN_OP_SINH].est = shl_gref_sinh;
    cb_map[CSINN_OP_SLICE].est = shl_gref_slice;
    cb_map[CSINN_OP_SOFTMAX].est = shl_gref_softmax;
    cb_map[CSINN_OP_SOFTPLUS].est = shl_gref_softplus;
    cb_map[CSINN_OP_SOFTRELU].est = shl_gref_softrelu;
    cb_map[CSINN_OP_SOFTSIGN].est = shl_gref_softsign;
    cb_map[CSINN_OP_SPACE_TO_BATCH].est = shl_gref_space_to_batch;
    cb_map[CSINN_OP_SPACE_TO_BATCH_ND].est = shl_gref_space_to_batch_nd;
    cb_map[CSINN_OP_SPACE_TO_DEPTH].est = shl_gref_space_to_depth;
    cb_map[CSINN_OP_SPLIT].est = shl_gref_split;
    cb_map[CSINN_OP_SQRT].est = shl_gref_sqrt;
    cb_map[CSINN_OP_SQUARE].est = shl_gref_square;
    cb_map[CSINN_OP_SQUEEZE].est = shl_gref_squeeze;
    cb_map[CSINN_OP_STACK].est = shl_gref_stack;
    cb_map[CSINN_OP_STRIDED_SLICE].est = shl_gref_strided_slice;
    cb_map[CSINN_OP_SUB].est = shl_gref_sub;
    cb_map[CSINN_OP_SUM].est = shl_gref_sum;
    cb_map[CSINN_OP_TAN].est = shl_gref_tan;
    cb_map[CSINN_OP_TANH].est = shl_gref_tanh;
    cb_map[CSINN_OP_THRESHOLD_RELU].est = shl_gref_threshold_relu;
    cb_map[CSINN_OP_TILE].est = shl_gref_tile;
    cb_map[CSINN_OP_TOPK].est = shl_gref_topk;
    cb_map[CSINN_OP_TRUNC].est = shl_gref_trunc;
    cb_map[CSINN_OP_TRANSPOSE].est = shl_gref_transpose;
    cb_map[CSINN_OP_UNPOOLING].est = shl_gref_unpooling;
    cb_map[CSINN_OP_UNSTACK].est = shl_gref_unstack;
    cb_map[CSINN_OP_WHERE].est = shl_gref_where;
    cb_map[CSINN_OP_XOR].est = shl_gref_xor;
    cb_map[CSINN_OP_YUV_RGB_SCALE].est = shl_gref_yuv_rgb_scale;

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
