/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

#include "c906/c906.h"
#include "c906/cap.h"
#include "c906/perf.h"

static struct shl_cb_op_list shl_c906_cb_op_list;

int shl_c906_reg_op(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *init, void *exec)
{
    struct shl_cb_op_list *list_end = shl_cb_list_end(&shl_c906_cb_op_list);
    struct shl_cb_op_list *next = shl_mem_alloc(sizeof(struct shl_cb_op_list));
    next->cb = shl_mem_alloc(sizeof(struct csinn_callback));
    next->cb->init = init;
    next->cb->exec = exec;
    next->dtype = dtype;
    next->op_name = op_name;
    list_end->next = next;
    return CSINN_TRUE;
}

int shl_c906_reg_op_est(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *est)
{
    struct csinn_callback *cb = shl_cb_list_match(&shl_c906_cb_op_list, dtype, op_name);
    if (cb == NULL) {
        shl_debug_info("%s: cannot find c906 est\n", __func__);
    } else {
        cb->est = est;
    }

    return CSINN_TRUE;
}

int shl_c906_reg_op_cap(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *caps)
{
    struct csinn_callback *cb = shl_cb_list_match(&shl_c906_cb_op_list, dtype, op_name);
    if (cb == NULL) {
        shl_debug_info("%s: cannot find c906 caps\n", __func__);
    } else {
        cb->caps = caps;
    }

    return CSINN_TRUE;
}

int shl_c906_reg_op_perf(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *perf)
{
    struct csinn_callback *cb = shl_cb_list_match(&shl_c906_cb_op_list, dtype, op_name);
    if (cb == NULL) {
        shl_debug_info("%s: cannot find c906 perf\n", __func__);
    } else {
        cb->perf = perf;
    }

    return CSINN_TRUE;
}

struct csinn_callback *__attribute__((weak)) shl_cb_map_rvv(int op, int dtype);
struct csinn_callback *shl_cb_map_c906(int op, int dtype)
{
    struct csinn_callback *cb = shl_cb_list_match(&shl_c906_cb_op_list, dtype, op);
    if (cb == NULL) {
        cb = shl_cb_map_rvv(op, dtype);
    }
    return cb;
}

int shl_c906_set_packn_layout(struct csinn_session *sess, bool packn_layout)
{
    struct shl_gref_target_data *gref_td = sess->td;
    struct shl_c906_option *c906_option = gref_td->cpu_option;
    c906_option->base.use_packn_layout = packn_layout;
    return CSINN_TRUE;
}

struct shl_c906_option *shl_c906_get_graph_option(struct csinn_session *sess)
{
    struct shl_gref_target_data *gref_td = sess->td;
    if (gref_td) {
        return (struct shl_c906_option *)(gref_td->cpu_option);
    } else {
        return NULL;
    }
}

void shl_c906_session_init(struct csinn_session *sess)
{
    struct shl_c906_option *c906_option = shl_mem_alloc(sizeof(struct shl_c906_option));
    struct shl_ref_graph *graph = shl_mem_alloc(sizeof(struct shl_ref_graph));
    struct shl_gref_target_data *target_data = shl_mem_alloc(sizeof(struct shl_gref_target_data));
    target_data->graph = graph;
    c906_option->base.use_packn_layout = 0;  // c906 set use_packn_layout false default
    target_data->cpu_option = c906_option;
    sess->td = target_data;
    shl_c906_set_binary_model_op_init(sess, false);
    sess->base_layout = CSINN_LAYOUT_NCHW;
}

void shl_c906_session_deinit(struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    shl_mem_free(graph->input);
    shl_mem_free(graph->output);
    shl_mem_free(graph->layer);
    struct shl_c906_option *c906_option = shl_c906_get_graph_option(sess);
    if (c906_option) {
        shl_mem_free(c906_option);
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

    // different layout
    bool use_packn = false;
    for (int i = 0; i < graph->layer_index; i++) {
        struct csinn_params_base *curr_params = graph->layer[i]->data;
        if (curr_params->api == CSINN_TVMGEN) {
            use_packn = false;
            break;
        }
    }
    shl_c906_set_packn_layout(sess, use_packn);

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

void shl_c906_session_setup(struct csinn_session *sess)
{
    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    struct shl_node *n;
    FILE *b;
    char *path;
    int bm_offset = 8192;
    struct shl_binary_model_section_info *sinfo;
    bool save_binary_model = false;

    if (sess->model.save_mode == CSINN_SAVE_AND_RUN || sess->model.save_mode == CSINN_SAVE_ONLY) {
        if (sess->base_dtype == CSINN_DTYPE_FLOAT16 || sess->base_dtype == CSINN_DTYPE_FLOAT32) {
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

int shl_c906_load_binary_model(struct csinn_session *sess)
{
    char *bm_base = sess->model.bm_addr;
    struct shl_binary_model_section_info *sinfo =
        (struct shl_binary_model_section_info *)(bm_base + 4096);
    struct shl_ref_graph *ggraph = shl_mem_alloc(sizeof(struct shl_ref_graph));
    shl_bm_graph_struct_load(
        ggraph, (struct shl_ref_graph *)(bm_base + sinfo->sections[0].graph_offset * 4096));
    graph_match_session(ggraph, sess);
    merge_output(ggraph, sess);
    shl_c906_set_binary_model_op_init(sess, true);
    sess_op_init(sess);

    return CSINN_TRUE;
}

void *shl_c906_runtime_callback(int api)
{
    switch (api) {
        case CSINN_SESSION_INIT:
            return shl_c906_session_init;
            break;
        case CSINN_SESSION_DEINIT:
            return shl_c906_session_deinit;
            break;
        case CSINN_SESSION_SETUP:
            return shl_c906_session_setup;
            break;
        case CSINN_LOAD_BG:
            return shl_c906_load_binary_model;
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

void __attribute__((weak)) shl_target_init_c906()
{
    shl_register_runtime_callback(CSINN_C906, NULL);
    shl_register_op_callback(CSINN_C906, shl_cb_map_c906);

#ifndef CONFIG_C906_CONVOLUTION_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_c906_conv2d_init_fp16, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, shl_c906_conv2d_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_CONVOLUTION_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_c906_conv2d_init_fp32, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, shl_c906_conv2d_init_fp32, NULL);
#endif
#ifndef CONFIG_C906_MAXPOOL_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL2D, shl_c906_maxpool2d_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_MAXPOOL_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXPOOL2D, shl_c906_maxpool2d_init_fp32, NULL);
#endif
#ifndef CONFIG_C906_AVERAGEPOOL_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_AVGPOOL2D, shl_c906_avgpool2d_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_AVERAGEPOOL_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_AVGPOOL2D, shl_c906_avgpool2d_init_fp32, NULL);
#endif
#ifndef CONFIG_C906_DEPTHWISE_CONVOLUTION_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D,
                    shl_c906_depthwise_conv2d_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_DEPTHWISE_CONVOLUTION_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTHWISE_CONV2D,
                    shl_c906_depthwise_conv2d_init_fp32, NULL);
#endif
#ifndef CONFIG_C906_DEPTHWISE_CONVOLUTION1D_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV1D,
                    shl_c906_depthwise_conv1d_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_FULLYCONNECTED_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED, shl_c906_fullyconnected_init_fp16,
                    NULL);
#endif
#ifndef CONFIG_C906_DIV_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DIV, shl_c906_div_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_DIV_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DIV, shl_c906_div_init_fp32, NULL);
#endif
#ifndef CONFIG_C906_ABS_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ABS, NULL, shl_c906_abs_fp16);
#endif
#ifndef CONFIG_C906_ADD_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ADD, NULL, shl_c906_add_fp16);
#endif
#ifndef CONFIG_C906_CACHE_CONV1D_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CACHE_MATMUL, shl_c906_cache_matmul_init,
                    shl_c906_cache_matmul_fp16);
#endif
#ifndef CONFIG_C906_CACHE_CONV1D_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CACHE_CONV1D, shl_c906_cache_conv1d_init,
                    shl_c906_cache_conv1d_fp16);
#endif
#ifndef CONFIG_C906_CLIP_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CLIP, NULL, shl_c906_clip_fp16);
#endif
#ifndef CONFIG_C906_CONCAT_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONCAT, NULL, shl_c906_concat_fp16);
#endif
#ifndef CONFIG_C906_GLOBAL_AVERAGEPOOL_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_AVGPOOL2D, NULL,
                    shl_c906_global_avgpool2d_fp16);
#endif
#ifndef CONFIG_C906_GLOBAL_MAXPOOL_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_MAXPOOL2D, NULL,
                    shl_c906_global_maxpool2d_fp16);
#endif
#ifndef CONFIG_C906_LEAKY_RELU_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LEAKY_RELU, NULL, shl_c906_leaky_relu_fp16);
#endif
#ifndef CONFIG_C906_LRN_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LRN, NULL, shl_c906_lrn_fp16);
#endif
#ifndef CONFIG_C906_MATMUL_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MATMUL, shl_c906_matmul_init_fp16, NULL);
#endif
#ifndef CONFIG_C906_MINIMUM_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MINIMUM, NULL, shl_c906_minimum_fp16);
#endif
#ifndef CONFIG_C906_MUL_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MUL, NULL, shl_c906_mul_fp16);
#endif
#ifndef CONFIG_C906_PRELU_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_PRELU, NULL, shl_c906_prelu_fp16);
#endif
#ifndef CONFIG_C906_RELU_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU, NULL, shl_c906_relu_fp16);
#endif
#ifndef CONFIG_C906_RELU1_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU1, NULL, shl_c906_relu1_fp16);
#endif
#ifndef CONFIG_C906_RELU6_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU6, NULL, shl_c906_relu6_fp16);
#endif
#ifndef CONFIG_C906_RESHAPE_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RESHAPE, NULL, shl_c906_reshape_fp16);
#endif
#ifndef CONFIG_C906_SPLIT_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SPLIT, NULL, shl_c906_split_fp16);
#endif
#ifndef CONFIG_C906_SUN_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SUB, NULL, shl_c906_sub_fp16);
#endif
#ifndef CONFIG_C906_REDUCE_SUM_FP16_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_REDUCE_SUM, NULL, shl_c906_reduce_sum_fp16);
#endif
#ifndef CONFIG_C906_ABS_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ABS, NULL, shl_c906_abs_f32);
#endif
#ifndef CONFIG_C906_ADD_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ADD, NULL, shl_c906_add_f32);
#endif
#ifndef CONFIG_C906_CLIP_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CLIP, NULL, shl_c906_clip_f32);
#endif
#ifndef CONFIG_C906_CONCAT_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONCAT, NULL, shl_c906_concat_f32);
#endif
#ifndef CONFIG_C906_GLOBAL_AVERAGEPOOL_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_AVGPOOL2D, NULL,
                    shl_c906_global_avgpool2d_f32);
#endif
#ifndef CONFIG_C906_GLOBAL_MAXPOOL_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_MAXPOOL2D, NULL,
                    shl_c906_global_maxpool2d_f32);
#endif
#ifndef CONFIG_C906_LEAKY_RELU_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LEAKY_RELU, NULL, shl_c906_leaky_relu_f32);
#endif
#ifndef CONFIG_C906_MINIMUM_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MINIMUM, NULL, shl_c906_minimum_f32);
#endif
#ifndef CONFIG_C906_MUL_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MUL, NULL, shl_c906_mul_f32);
#endif
#ifndef CONFIG_C906_PRELU_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_PRELU, NULL, shl_c906_prelu_f32);
#endif
#ifndef CONFIG_C906_RELU_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU, NULL, shl_c906_relu_f32);
#endif
#ifndef CONFIG_C906_RELU1_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU1, NULL, shl_c906_relu1_f32);
#endif
#ifndef CONFIG_C906_RELU6_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU6, NULL, shl_c906_relu6_f32);
#endif
#ifndef CONFIG_C906_SPLIT_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SPLIT, NULL, shl_c906_split_f32);
#endif
#ifndef CONFIG_C906_SUB_FP32_DISABLED
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SUB, NULL, shl_c906_sub_f32);
#endif

#ifdef SHL_BUILD_GREF
    shl_register_runtime_callback(CSINN_C906, shl_c906_runtime_callback);
#ifndef CONFIG_GRAPH_REFERENCE_CONVOLUTION_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_gref_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_gref_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_CONV2D, shl_gref_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_CONV2D_RELU, shl_gref_conv2d_relu);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, shl_gref_group_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, shl_gref_group_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D, shl_gref_depthwise_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTHWISE_CONV2D, shl_gref_depthwise_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D, shl_gref_depthwise_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_INT8, CSINN_OP_DEPTHWISE_CONV2D_RELU,
                        shl_gref_depthwise_conv2d_relu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONVOLUTION1D_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV1D, shl_gref_conv1d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV1D, shl_gref_conv1d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV1D, shl_gref_depthwise_conv1d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MAXPOOL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL2D, shl_gref_maxpool2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXPOOL2D, shl_gref_maxpool2d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_AVERAGEPOOL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_AVGPOOL2D, shl_gref_avgpool2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_AVGPOOL2D, shl_gref_avgpool2d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FULLYCONNECTED_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED, shl_gref_fullyconnected);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_DIV_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_DIV, shl_gref_div);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_DIV, shl_gref_div);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ABS_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_ABS, shl_gref_abs);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ADD_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_ADD, shl_gref_add);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CACHE_MATMUL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CACHE_MATMUL, shl_gref_cache_matmul);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CACHE_CONV1D_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CACHE_CONV1D, shl_gref_cache_conv1d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CLIP_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CLIP, shl_gref_clip);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONCAT_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CONCAT, shl_gref_concat);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GLOBAL_AVERAGEPOOL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_AVGPOOL2D, shl_gref_global_avgpool2d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GLOBAL_MAXPOOL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_MAXPOOL2D, shl_gref_global_maxpool2d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LEAKY_RELU_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_LEAKY_RELU, shl_gref_leaky_relu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LRN_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_LRN, shl_gref_lrn);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MATMUL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_MATMUL, shl_gref_matmul);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MINIMUM_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_MINIMUM, shl_gref_minimum);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MUL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_MUL, shl_gref_mul);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_PRELU_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_PRELU, shl_gref_prelu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU, shl_gref_relu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU1_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU1, shl_gref_relu1);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU6_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU6, shl_gref_relu6);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RESHAPE_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_RESHAPE, shl_gref_reshape);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SPLIT_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_SPLIT, shl_gref_split);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SUB_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_SUB, shl_gref_sub);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SUM_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_REDUCE_SUM, shl_gref_reduce_sum);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ABS_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_ABS, shl_gref_abs);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ADD_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_ADD, shl_gref_add);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CLIP_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_CLIP, shl_gref_clip);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONCAT_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_CONCAT, shl_gref_concat);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GLOBAL_AVERAGEPOOL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_AVGPOOL2D, shl_gref_global_avgpool2d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GLOBAL_MAXPOOL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_MAXPOOL2D, shl_gref_global_maxpool2d);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LEAKY_RELU_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_LEAKY_RELU, shl_gref_leaky_relu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MINIMUM_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_MINIMUM, shl_gref_minimum);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MUL_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_MUL, shl_gref_mul);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_PRELU_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_PRELU, shl_gref_prelu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU, shl_gref_relu);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU1_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU1, shl_gref_relu1);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU6_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU6, shl_gref_relu6);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SPLIT_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_SPLIT, shl_gref_split);
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SUB_DISABLED
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_SUB, shl_gref_sub);
#endif
#endif
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_c906_conv2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, shl_c906_conv2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTHWISE_CONV2D,
                        shl_c906_depthwise_conv2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV1D, shl_c906_conv1d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXPOOL2D, shl_c906_maxpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_AVGPOOL2D, shl_c906_avgpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_DIV, shl_c906_div_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_ABS, shl_c906_abs_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_ADD, shl_c906_add_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_CLIP, shl_c906_clip_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_CONCAT, shl_c906_concat_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_AVGPOOL2D,
                        shl_c906_global_avgpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_MAXPOOL2D,
                        shl_c906_global_maxpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_LEAKY_RELU, shl_c906_leaky_relu_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_MINIMUM, shl_c906_minimum_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_MUL, shl_c906_mul_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_PRELU, shl_c906_prelu_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU, shl_c906_relu_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU1, shl_c906_relu1_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU6, shl_c906_relu6_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_SPLIT, shl_c906_split_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT32, CSINN_OP_SUB, shl_c906_sub_cap);

    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_c906_conv2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, shl_c906_conv2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D,
                        shl_c906_depthwise_conv2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED, shl_c906_fullyconnected_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV1D, shl_c906_conv1d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV1D,
                        shl_c906_depthwise_conv1d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL2D, shl_c906_maxpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_AVGPOOL2D, shl_c906_avgpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_DIV, shl_c906_div_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_ABS, shl_c906_abs_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_ADD, shl_c906_add_cap);
    /* skip cache_matmul and cache_conv1d */
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_CLIP, shl_c906_clip_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_CONCAT, shl_c906_concat_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_AVGPOOL2D,
                        shl_c906_global_avgpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_MAXPOOL2D,
                        shl_c906_global_maxpool2d_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_LEAKY_RELU, shl_c906_leaky_relu_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_LRN, shl_c906_lrn_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_MATMUL, shl_c906_matmul_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_MINIMUM, shl_c906_minimum_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_MUL, shl_c906_mul_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_PRELU, shl_c906_prelu_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU, shl_c906_relu_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU1, shl_c906_relu1_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU6, shl_c906_relu6_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_RESHAPE, shl_c906_reshape_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_SPLIT, shl_c906_split_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_SUB, shl_c906_sub_cap);
    shl_c906_reg_op_cap(CSINN_DTYPE_FLOAT16, CSINN_OP_REDUCE_SUM, shl_c906_reduce_sum_cap);

    /* register perf functions */
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_c906_conv2d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, shl_c906_conv2d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTHWISE_CONV2D,
                         shl_c906_depthwise_conv2d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV1D, shl_c906_conv1d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXPOOL2D, shl_c906_maxpool2d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_AVGPOOL2D, shl_c906_avgpool2d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_DIV, shl_c906_div_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_ABS, shl_c906_abs_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_ADD, shl_c906_add_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_CLIP, shl_c906_clip_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_CONCAT, shl_c906_concat_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_AVGPOOL2D,
                         shl_c906_global_avgpool2d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_MAXPOOL2D,
                         shl_c906_global_maxpool2d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_LEAKY_RELU, shl_c906_leaky_relu_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_MINIMUM, shl_c906_minimum_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_MUL, shl_c906_mul_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_PRELU, shl_c906_prelu_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU, shl_c906_relu_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU1, shl_c906_relu1_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU6, shl_c906_relu6_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_SPLIT, shl_c906_split_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT32, CSINN_OP_SUB, shl_c906_sub_perf);

    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_c906_conv2d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, shl_c906_conv2d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D,
                         shl_c906_depthwise_conv2d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED,
                         shl_c906_fullyconnected_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV1D, shl_c906_conv1d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV1D,
                         shl_c906_depthwise_conv1d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL2D, shl_c906_maxpool2d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_AVGPOOL2D, shl_c906_avgpool2d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_DIV, shl_c906_div_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_ABS, shl_c906_abs_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_ADD, shl_c906_add_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_CLIP, shl_c906_clip_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_CONCAT, shl_c906_concat_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_AVGPOOL2D,
                         shl_c906_global_avgpool2d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_MAXPOOL2D,
                         shl_c906_global_maxpool2d_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_LEAKY_RELU, shl_c906_leaky_relu_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_LRN, shl_c906_lrn_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_MATMUL, shl_c906_matmul_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_MINIMUM, shl_c906_minimum_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_MUL, shl_c906_mul_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_PRELU, shl_c906_prelu_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU, shl_c906_relu_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU1, shl_c906_relu1_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU6, shl_c906_relu6_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_RESHAPE, shl_c906_reshape_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_SPLIT, shl_c906_split_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_SUB, shl_c906_sub_perf);
    shl_c906_reg_op_perf(CSINN_DTYPE_FLOAT16, CSINN_OP_REDUCE_SUM, shl_c906_reduce_sum_perf);
}
