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

#include "rvm/rvm.h"

#define RVM_OP_PATTERN_MAX 60
static struct shl_cb_table shl_rvm_cb_table[RVM_OP_PATTERN_MAX];

void shl_rvm_reg_op(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *init, void *exec,
                    void *est)
{
    static int i = 0;
    if (i >= RVM_OP_PATTERN_MAX) {
        shl_debug_error("RVM callback length is greater than RVM_OP_PATTERN_MAX!\n");
    }
    shl_rvm_cb_table[i].shl_cb_key = op_name * CSINN_DTYPE_SIZE + dtype;
    shl_rvm_cb_table[i].shl_cb_value.init = init;
    shl_rvm_cb_table[i].shl_cb_value.exec = exec;
    shl_rvm_cb_table[i].shl_cb_value.est = est;
    i++;
}

struct csinn_callback *shl_cb_map_rvv(int op, int dtype);
struct csinn_callback *shl_cb_map_rvm(int op, int dtype)
{
    struct csinn_callback *cb = NULL;
    for (int i = 0; i < RVM_OP_PATTERN_MAX; i++) {
        if (shl_rvm_cb_table[i].shl_cb_key == (op * CSINN_DTYPE_SIZE + dtype)) {
            cb = &(shl_rvm_cb_table[i].shl_cb_value);
            break;
        }
    }
    if ((cb == NULL) || (cb->est == NULL && (cb->init == NULL || cb->exec == NULL))) {
        cb = shl_cb_map_rvv(op, dtype);
    }
    return cb;
}

struct shl_rvm_option *shl_rvm_get_graph_option(struct csinn_session *sess)
{
    struct shl_gref_target_data *gref_td = sess->td;
    if (gref_td) {
        return (struct shl_rvm_option *)(gref_td->cpu_option);
    } else {
        return NULL;
    }
}

void shl_rvm_session_init(struct csinn_session *sess)
{
    struct shl_rvm_option *rvm_option = shl_mem_alloc(sizeof(struct shl_rvm_option));
    struct shl_ref_graph *graph = shl_mem_alloc(sizeof(struct shl_ref_graph));
    struct shl_gref_target_data *target_data = shl_mem_alloc(sizeof(struct shl_gref_target_data));
    target_data->graph = graph;
    target_data->cpu_option = rvm_option;
    sess->td = target_data;
    shl_rvm_set_binary_model_op_init(sess, false);
    sess->base_layout = CSINN_LAYOUT_NHWC;
}

void shl_rvm_session_deinit(struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    shl_mem_free(graph->input);
    shl_mem_free(graph->output);
    shl_mem_free(graph->layer);
    struct shl_rvm_option *rvm_option = shl_rvm_get_graph_option(sess);
    if (rvm_option) {
        shl_mem_free(rvm_option);
    }
    shl_mem_free(graph);
    shl_mem_free(sess->td);
    shl_mem_free(sess->input);
    shl_mem_free(sess->output);
}

static int pre_init(struct shl_node *node)
{
    /* base has same address with params */
    struct csinn_params_base *params = node->data;

    int (*func)();

    int org_rm = params->sess->base_run_mode;
    params->sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_callback *cb = shl_gref_best_callback(node);

    params->sess->base_run_mode = org_rm;

    return CSINN_TRUE;
}

static int init_op(struct shl_node *node)
{
    /* base has same address with params */
    struct csinn_params_base *params = node->data;
    struct csinn_callback *cb = params->cb;

    if (cb->init != NULL) {
        if (shl_gref_call_layer_func(cb->init, node) != CSINN_TRUE) {
            return CSINN_FALSE;
        }
    }

    return CSINN_TRUE;
}

static void sess_op_init(struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);

    // pre init, find best callback
    for (int i = 0; i < graph->layer_index; i++) {
        struct shl_node *n = graph->layer[i];
        if (n->type >= 0 && n->type < CSINN_OP_SIZE) {
            pre_init(n);
        } else {
            shl_debug_error("Unknown layer\n");
            return;
        }
    }

    // call init
    for (int i = 0; i < graph->layer_index; i++) {
        struct shl_node *n = graph->layer[i];
        if (n->type >= 0 && n->type < CSINN_OP_SIZE) {
            init_op(n);
        } else {
            shl_debug_error("Unknown layer\n");
            return;
        }
    }
}

void shl_rvm_session_setup(struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    struct shl_node *n;
    FILE *b;
    char *path;
    int bm_offset = 8192;
    struct shl_binary_model_section_info *sinfo;
    bool save_binary_model = false;

    if (sess->model.save_mode == CSINN_SAVE_AND_RUN || sess->model.save_mode == CSINN_SAVE_ONLY) {
        if (sess->base_dtype == CSINN_DTYPE_INT8 || sess->base_dtype == CSINN_DTYPE_FLOAT16) {
            save_binary_model = true;
        } else {
            shl_debug_warning("Unsupport to save this dtype binary model yet\n");
        }
    }

    struct shl_ref_graph *ggraph = graph;

    sess_op_init(sess);

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
        }
    }

    for (int i = 0; i < ggraph->output_num; i++) {
        ggraph->output[i]->ref_count_init++;
    }

    if (save_binary_model) {
        if (sess->model.bm_path == NULL) {
            path = "shl.hhb.bm";
        } else {
            path = sess->model.bm_path;
        }
        b = fopen(path, "wb");
        shl_dump_bm_header(b);

        /* TODO: start from more */
        bm_offset = 8192;
        fseek(b, bm_offset, SEEK_SET);
        sinfo = shl_mem_alloc(sizeof(struct shl_binary_model_section_info));

        /* only dump top(global) graph, unsupport subgraph */
        fseek(b, bm_offset, SEEK_SET);
        int ggraph_size = shl_dump_bm_graph_struct_section(b, ggraph);
        sinfo->sections[0].graph_offset = bm_offset / 4096;
        sinfo->sections[0].graph_size = ggraph_size;
        bm_offset = shl_gref_size_align(bm_offset + ggraph_size, 4096);

        fseek(b, bm_offset, SEEK_SET);
        int info_size = shl_dump_bm_graph_info_section(b, sess);
        sinfo->sections[0].info_offset = bm_offset / 4096;
        sinfo->sections[0].info_size = info_size;
        bm_offset = shl_gref_size_align(bm_offset + info_size, 4096);

        /* save section info */
        sinfo->section_num = 2;
        fseek(b, 4096, SEEK_SET);
        shl_dump_bm_section_info(b, sinfo);
        fclose(b);
    }
}

/* use tensor name to match same */
static void merge_output(struct shl_ref_graph *graph, struct csinn_session *sess)
{
    /* match graph output */
    for (int i = 0; i < graph->output_num; i++) {
        struct shl_node *gnode = graph->output[i];
        char *sname = gnode->name;
        for (int j = 0; j < graph->layer_index; j++) {
            struct shl_node *node = graph->layer[j];
            for (int m = 0; m < node->out_num; m++) {
                if (strcmp(node->name, sname) == 0) {
                    /* TODO: free graph output node */
                    graph->output[i] = node->out[m];
                    break;
                }
            }
        }
    }

    for (int i = 0; i < sess->input_num; i++) {
        /* TODO: free sess output node */
        sess->input[i] = graph->input[i]->data;
    }

    for (int i = 0; i < sess->output_num; i++) {
        /* TODO: free sess output node */
        sess->output[i] = graph->output[i]->data;
    }
}

static void graph_match_session(struct shl_ref_graph *graph, struct csinn_session *sess)
{
    struct shl_gref_target_data *td = sess->td;
    td->graph = graph;

    for (int i = 0; i < graph->layer_index; i++) {
        struct shl_node *n = graph->layer[i];
        /* fix op callback, skip subgraph */
        if (n->type < CSINN_OP_SIZE) {
            struct csinn_params_base *base = n->data;
            base->sess = sess;
            struct csinn_tensor *input = n->in[0]->data;

            int org_rm = base->sess->base_run_mode;
            base->sess->base_run_mode = CSINN_RM_LAYER;
            shl_op_callback_map(base, n->type, input->dtype);
            base->sess->base_run_mode = org_rm;
        }
    }
}

int shl_rvm_load_binary_model(struct csinn_session *sess)
{
    char *bm_base = sess->model.bm_addr;
    struct shl_binary_model_section_info *sinfo =
        (struct shl_binary_model_section_info *)(bm_base + 4096);
    struct shl_ref_graph *ggraph = shl_mem_alloc(sizeof(struct shl_ref_graph));
    shl_bm_graph_struct_load(
        ggraph, (struct shl_ref_graph *)(bm_base + sinfo->sections[0].graph_offset * 4096));
    graph_match_session(ggraph, sess);
    merge_output(ggraph, sess);
    shl_rvm_set_binary_model_op_init(sess, true);
    sess_op_init(sess);

    return CSINN_TRUE;
}

void *shl_rvm_runtime_callback(int api)
{
    switch (api) {
        case CSINN_SESSION_INIT:
            return shl_rvm_session_init;
            break;
        case CSINN_SESSION_DEINIT:
            return shl_rvm_session_deinit;
            break;
        case CSINN_SESSION_SETUP:
            return shl_rvm_session_setup;
            break;
        case CSINN_LOAD_BG:
            return shl_rvm_load_binary_model;
            break;
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
            return shl_gref_runtime_callback(api);
            break;
        default:
            shl_debug_info("%s: Cannot find callback\n", __func__);
            break;
    }
    return NULL;
}

void shl_target_init_rvm()
{
    shl_rvm_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_rvm_conv2d_init_fp16, NULL,
                   shl_gref_conv2d);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D, shl_rvm_conv2d_init_int8, NULL,
                   shl_gref_conv2d);

    shl_rvm_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D,
                   shl_rvm_depthwise_conv2d_init_fp16, NULL, shl_gref_depthwise_conv2d);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D, shl_rvm_depthwise_conv2d_init_int8,
                   NULL, shl_gref_depthwise_conv2d);

    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D_RELU, shl_rvm_conv2d_init_int8, NULL,
                   shl_gref_conv2d_relu);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D_RELU,
                   shl_rvm_depthwise_conv2d_init_int8, NULL, shl_gref_depthwise_conv2d_relu);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_CONV2D_RELU6, shl_rvm_conv2d_init_int8, NULL,
                   shl_gref_conv2d_relu6);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D_RELU6,
                   shl_rvm_depthwise_conv2d_init_int8, NULL, shl_gref_depthwise_conv2d_relu6);

    shl_rvm_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED, shl_rvm_fullyconnected_init_fp16,
                   NULL, shl_gref_fullyconnected);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_FULLYCONNECTED, shl_rvm_fullyconnected_init_int8,
                   NULL, shl_gref_fullyconnected);

    shl_rvm_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MATMUL, shl_rvm_matmul_init_fp16, NULL,
                   shl_gref_matmul);
    shl_rvm_reg_op(CSINN_DTYPE_INT8, CSINN_OP_MATMUL, shl_rvm_matmul_init_int8, NULL,
                   shl_gref_matmul);

    shl_register_op_callback(CSINN_RVM, shl_cb_map_rvm);
    shl_register_runtime_callback(CSINN_RVM, shl_rvm_runtime_callback);
}
