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

#include "csi_gref.h"
#include "csi_utils.h"
#include "csi_gref.h"

int csi_gref_get_output_number(struct csi_session *sess)
{
    return sess->output_num;
}

int csi_gref_get_input_number(struct csi_session *sess)
{
    return sess->input_num;
}

void csi_gref_set_output_number(int number, struct csi_session *sess)
{
    struct csi_ref_graph *graph = csi_gref_get_graph(sess);
    sess->output_num = number;
    sess->output = calloc(sess->output_num, sizeof(struct csi_tensor *));
    graph->output_num = number;
}

void csi_gref_set_input_number(int number, struct csi_session *sess)
{
    struct csi_ref_graph *graph = csi_gref_get_graph(sess);
    sess->input_num = number;
    sess->input = calloc(sess->input_num, sizeof(struct csi_tensor *));
    graph->input_num = number;
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
    sess->base_layout = CSINN_NCHW;
}

static int call_func(void *fn, struct csi_node *node)
{
    /* base has same address with params */
    struct csi_params_base *params = node->data;
    int (*func)();
    func = fn;
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
    case CSINN_OP_CONCAT:
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
    case CSINN_OP_SPLIT:
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
        func(node->in[0]->data, node->out[0]->data, params);
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
        func(node->in[0]->data, node->in[1]->data, node->out[0]->data, params);
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
        func(node->in[0]->data, node->out[0]->data, node->in[1]->data, node->in[2]->data, params);
        break;
    case CSINN_OP_ALL:
    case CSINN_OP_ARANGE:
    case CSINN_OP_BN:
    case CSINN_OP_MEAN_STRIDE:
    case CSINN_OP_MIN_STRIDE:
    case CSINN_OP_ONE_HOT:
    case CSINN_OP_PROPOSAL:
    case CSINN_OP_PSROIPOOLING:
    case CSINN_OP_ROIALIGN:
    case CSINN_OP_ROIPOOL:
    case CSINN_OP_SCATTER_ND:
    case CSINN_OP_SELECT:
    case CSINN_OP_TOPK:
    case CSINN_OP_WHERE:
        CSI_DEBUG_CALL(printf("unsupported op\n"));
        break;
    default:
        CSI_DEBUG_CALL(printf("unknown op\n"));
        return CSINN_FALSE;
    }
}

static int init_op(struct csi_node *node)
{
    /* base has same address with params */
    struct csi_params_base *params = node->data;
    int (*func)();

    struct csi_tensor *input = node->in[0]->data;

    func = csi_init_map(params->api, node->type, input->dtype);
    if (func != NULL) {
        return call_func(func, node);
    } else {
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
            n->in[j]->ref_count++;
        }
        for (int k = 0; k < n->out_num; k++) {
            n->out[k]->ref_count++;
            /* avoid free output */
            if (i == graph->layer_index - 1) {
                n->out[k]->ref_count++;
            }
        }
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
        node->ref_count--;
        if (node->ref_count == 0) {
            struct csi_tensor *t = node->out[i]->data;
            free(t->data);
        }
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
}

struct csi_ref_graph *csi_gref_get_graph(struct csi_session *sess)
{
    struct csi_gref_target_data *td = sess->td;
    return td->graph;
}

void* csi_bc_map_table_gref[CSINN_OP_AND_UTILS_SIZE][1] = {
    {csi_gref_abs}, /* CSINN_OP_ABS */
    {csi_gref_acos}, /* CSINN_OP_ACOS */
    {csi_gref_acosh}, /* CSINN_OP_ACOSH */
    {csi_gref_add}, /* CSINN_OP_ADD */
    {csi_gref_all}, /* CSINN_OP_ALL */
    {csi_gref_and}, /* CSINN_OP_AND */
    {csi_gref_any}, /* CSINN_OP_ANY */
    {csi_gref_arange}, /* CSINN_OP_ARANGE */
    {csi_gref_argmax}, /* CSINN_OP_ARGMAX */
    {csi_gref_argmin}, /* CSINN_OP_ARGMIN */
    {csi_gref_asin}, /* CSINN_OP_ASIN */
    {csi_gref_asinh}, /* CSINN_OP_ASINH */
    {csi_gref_atan}, /* CSINN_OP_ATAN */
    {csi_gref_atanh}, /* CSINN_OP_ATANH */
    {csi_gref_avgpool}, /* CSINN_OP_AVGPOOL2D */
    {csi_gref_avgpool3d}, /* CSINN_OP_AVGPOOL3D */
    {csi_gref_batch_normalization}, /* CSINN_OP_BN */
    {csi_gref_batch_to_space}, /* CSINN_OP_BATCH_TO_SPACE */
    {csi_gref_batch_to_space_nd}, /* CSINN_OP_BATCH_TO_SPACE_ND */
    {csi_gref_broadcast_to}, /* CSINN_OP_BROADCOST */
    {csi_gref_ceil}, /* CSINN_OP_CEIL */
    {csi_gref_clip}, /* CSINN_OP_CLIP */
    {csi_gref_col2im}, /* CSINN_OP_COL2IM */
    {csi_gref_concat}, /* CSINN_OP_CONCAT */
    {csi_gref_conv2d}, /* CSINN_OP_CONV2D */
    {csi_gref_conv2d_relu}, /* CSINN_OP_CONV2D_RELU */
    {csi_gref_conv2d_relu6}, /* CSINN_OP_CONV2D_RELU6 */
    {NULL}, /* CSINN_OP_CONV2D_CHANNEL */
    {NULL}, /* CSINN_OP_CONV2D_CHANNEL_RELU */
    {NULL}, /* CSINN_OP_CONV2D_CHANNEL_RELU6 */
    {csi_gref_depthwise_conv2d}, /* CSINN_OP_DEPTHWISE_CONV2D */
    {csi_gref_depthwise_conv2d_relu}, /* CSINN_OP_DEPTHWISE_CONV2D_RELU */
    {csi_gref_depthwise_conv2d_relu6}, /* CSINN_OP_DEPTHWISE_CONV2D_RELU6 */
    {NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL */
    {NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU */
    {NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6 */
    {csi_gref_group_conv2d}, /* CSINN_OP_GROUP_CONV2D */
    {NULL}, /* CSINN_OP_GROUP_CONV2D_RELU */
    {NULL}, /* CSINN_OP_GROUP_CONV2D_RELU6 */
    {NULL}, /* CSINN_OP_GROUP_CONV2D_CHANNEL */
    {NULL}, /* CSINN_OP_GROUP_CONV2D_CHANNEL_RELU */
    {csi_gref_conv3d}, /* CSINN_OP_CONV3D */
    {csi_gref_cos}, /* CSINN_OP_COS */
    {csi_gref_cosh}, /* CSINN_OP_COSH */
    {csi_gref_crop}, /* CSINN_OP_CROP */
    {csi_gref_cumprod}, /* CSINN_OP_CUMPROD */
    {csi_gref_cumsum}, /* CSINN_OP_CUMSUM */
    {csi_gref_deconv2d}, /* CSINN_OP_DECONV2D */
    {csi_gref_depthwise_deconv2d}, /* CSINN_OP_DEPTHWISE_DECONV2D */
    {csi_gref_deconv3d}, /* CSINN_OP_DECONV3D */
    {csi_gref_depth_to_space}, /* CSINN_OP_DEPTH_TO_SPACE */
    {csi_gref_div}, /* CSINN_OP_DIV */
    {csi_gref_elu}, /* CSINN_OP_ELU */
    {csi_gref_equal}, /* CSINN_OP_EQUANL */
    {csi_gref_erf}, /* CSINN_OP_ERF */
    {csi_gref_exp}, /* CSINN_OP_EXP */
    {csi_gref_expand_dims}, /* CSINN_OP_EXPAND_DIMS */
    {csi_gref_expm1}, /* CSINN_OP_EXPM1 */
    {csi_gref_flatten}, /* CSINN_OP_FLATTEN */
    {csi_gref_floor_divide}, /* CSINN_OP_FLOOR_DIVIDE */
    {csi_gref_floor_mod}, /* CSINN_OP_FLOOR_MOD */
    {csi_gref_floor}, /* CSINN_OP_FLOOR */
    {csi_gref_fullyconnected}, /* CSINN_OP_FULLYCONNECTED */
    {csi_gref_gather_nd}, /* CSINN_OP_GATHER_ND */
    {csi_gref_gather}, /* CSINN_OP_GATHER */
    {csi_gref_global_avgpool}, /* CSINN_OP_GLOBAL_AVGPOOL2D */
    {csi_gref_global_maxpool}, /* CSINN_OP_GLOBAL_MAXPOOL2D */
    {csi_gref_greater_equal}, /* CSINN_OP_GREATHER_EQUAL */
    {csi_gref_greater}, /* CSINN_OP_GREATHER */
    {csi_gref_hard_sigmoid}, /* CSINN_OP_HARD_SIGMOID */
    {csi_gref_im2col}, /* CSINN_OP_IM2COL */
    {csi_gref_isnan_bool}, /* CSINN_OP_ISNAN */
    {csi_gref_l2_normalization}, /* CSINN_OP_L2N */
    {csi_gref_l2pool}, /* CSINN_OP_L2POOL2D */
    {csi_gref_leaky_relu}, /* CSINN_OP_LEAKY_RELU */
    {csi_gref_less_equal}, /* CSINN_OP_LESS_EQUAL */
    {csi_gref_less_equal}, /* CSINN_OP_LESS */
    {csi_gref_log_softmax}, /* CSINN_OP_LOG_SOFTMAX */
    {csi_gref_log}, /* CSINN_OP_LOG */
    {csi_gref_log1p}, /* CSINN_OP_LOG1P */
    {csi_gref_logical_and}, /* CSINN_OP_LOGICAL_AND */
    {csi_gref_logical_not}, /* CSINN_OP_LOGICAL_NOT */
    {csi_gref_logical_or}, /* CSINN_OP_LOGICAL_OR */
    {csi_gref_logical_xor}, /* CSINN_OP_LOGICAL_XOR */
    {csi_gref_lrn},  /* CSINN_OP_LRN */
    {csi_gref_matmul}, /* CSINN_OP_MATMUL */
    {csi_gref_max}, /* CSINN_OP_MAX */
    {csi_gref_maximum}, /* CSINN_OP_MAXINUM */
    {csi_gref_maxpool}, /* CSINN_OP_MAXPOOL2D */
    {csi_gref_maxpool2d_locat}, /* CSINN_OP_MAXPOOL2D_LOCAT */
    {csi_gref_maxpool3d}, /* CSINN_OP_MAXPOOL3D */
    {csi_gref_mean}, /* CSINN_OP_MEAN */
    {NULL}, /* CSINN_OP_MEAN_STRIDE */
    {csi_gref_min}, /* CSINN_OP_MIN */
    {NULL}, /* CSINN_OP_MIN_STRIDE */
    {csi_gref_minimum}, /* CSINN_OP_MINIMUM */
    {csi_gref_mod}, /* CSINN_OP_MOD */
    {csi_gref_mul}, /* CSINN_OP_MUL */
    {csi_gref_ndarray_size}, /* CSINN_OP_NDARRAY_SIZE */
    {csi_gref_negative}, /* CSINN_OP_NEGATIIVE */
    {csi_gref_non_max_suppression}, /* CSINN_OP_NON_MAX_SUPPRESSION */
    {csi_gref_not_equal}, /* CSINN_OP_NOT_EQUAL */
    {csi_gref_not}, /* CSINN_OP_NOT */
    {NULL}, /* CSINN_OP_ONE_HOT */
    {csi_gref_or}, /* CSINN_OP_OR */
    {csi_gref_pad}, /* CSINN_OP_PAD */
    {csi_gref_power}, /* CSINN_OP_POWER */
    {csi_gref_prelu}, /* CSINN_OP_PRELU */
    {csi_gref_prod}, /* CSINN_OP_PROD */
    {csi_gref_proposal}, /* CSINN_OP_PROPOSAL */
    {csi_gref_psroipooling}, /* CSINN_OP_PSROIPOOLING */
    {csi_gref_reduce_logsumexp}, /* CSINN_OP_REDUCE_LOGSUMEXP */
    {csi_gref_reduce_max}, /* CSINN_OP_REDUCE_MAX */
    {csi_gref_reduce_mean}, /* CSINN_OP_REDUCE_MEAN */
    {csi_gref_reduce_min}, /* CSINN_OP_REDUCE_MIN */
    {csi_gref_reduce_prod}, /* CSINN_OP_REDUCE_PROD */
    {csi_gref_reduce_sum}, /* CSINN_OP_REDUCE_SUM */
    {csi_gref_relu}, /* CSINN_OP_RELU */
    {csi_gref_relu1}, /* CSINN_OP_RELU1 */
    {csi_gref_relu6}, /* CSINN_OP_RELU6 */
    {csi_gref_relun}, /* CSINN_OP_RELUN */
    {csi_gref_reorg}, /* CSINN_OP_REORG */
    {csi_gref_reshape}, /* CSINN_OP_RESHAPE */
    {csi_gref_resize}, /* CSINN_OP_RESIZE */
    {csi_gref_reverse}, /* CSINN_OP_REVERSE */
    {csi_gref_roi_align}, /* CSINN_OP_ROIALIGN */
    {csi_gref_roipool}, /* CSINN_OP_ROIPOOL */
    {csi_gref_round}, /* CSINN_OP_ROUND */
    {csi_gref_rsqrt}, /* CSINN_OP_RSQRT */
    {csi_gref_scatter_nd}, /* CSINN_OP_SCATTER_ND */
    {csi_gref_segment_max}, /* CSINN_OP_SEGMENT_MAX */
    {NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MAX */
    {csi_gref_segment_mean}, /* CSINN_OP_SEGMENT_MEAN */
    {NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MEAN */
    {csi_gref_segment_min}, /* CSINN_OP_SEGMENT_MIN */
    {NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MIN */
    {csi_gref_segment_prod}, /* CSINN_OP_SEGMENT_PROD */
    {NULL}, /* CSINN_OP_UNSORTED_SEGMENT_PROD */
    {csi_gref_segment_sum}, /* CSINN_OP_SEGMENT_SUM */
    {NULL}, /* CSINN_OP_UNSORTED_SEGMENT_SUM */
    {csi_gref_select}, /* CSINN_OP_SELECT */
    {csi_gref_sequence_mask}, /* CSINN_OP_SEQUENCE_MASK */
    {csi_gref_shape}, /* CSINN_OP_SHAPE */
    {csi_gref_shuffle_channel}, /* CSINN_OP_SHUFFLE_CHANNEL */
    {csi_gref_sigmoid}, /* CSINN_OP_SIGMOID */
    {csi_gref_sign}, /* CSINN_OP_SIGN */
    {csi_gref_sin}, /* CSINN_OP_SIN */
    {csi_gref_sinh}, /* CSINN_OP_SINH */
    {csi_gref_slice}, /* CSINN_OP_SLICE */
    {csi_gref_softmax}, /* CSINN_OP_SOFTMAX */
    {csi_gref_softplus}, /* CSINN_OP_SOFTPLUS */
    {csi_gref_softrelu}, /* CSINN_OP_SOFTRELU */
    {csi_gref_softsign}, /* CSINN_OP_SOFTSIGN */
    {csi_gref_space_to_batch}, /* CSINN_OP_SPACE_TO_BATCH */
    {csi_gref_space_to_batch_nd}, /* CSINN_OP_SPACE_TO_BATCH_ND */
    {csi_gref_space_to_depth}, /* CSINN_OP_SPACE_TO_DEPTH */
    {csi_gref_split}, /* CSINN_OP_SPLIT */
    {csi_gref_sqrt}, /* CSINN_OP_SQRT */
    {csi_gref_square}, /* CSINN_OP_SQUARE */
    {csi_gref_squeeze}, /* CSINN_OP_SQUEEZE */
    {csi_gref_stack}, /* CSINN_OP_STACK */
    {csi_gref_strided_slice}, /* CSINN_OP_STRIDED_SLICE */
    {csi_gref_sub}, /* CSINN_OP_SUB */
    {csi_gref_sum}, /* CSINN_OP_SUM */
    {csi_gref_tan}, /* CSINN_OP_TAN */
    {csi_gref_tanh}, /* CSINN_OP_TANH */
    {csi_gref_threshold_relu}, /* CSINN_OP_THRESHOLD_RELU */
    {csi_gref_tile}, /* CSINN_OP_TILE */
    {csi_gref_topk}, /* CSINN_OP_TOPK */
    {csi_gref_transpose}, /* CSINN_OP_TRANSPOSE */
    {csi_gref_trunc}, /* CSINN_OP_TRUNC */
    {csi_gref_unpooling}, /* CSINN_OP_UNPOOLING */
    {csi_gref_unstack}, /* CSINN_OP_UNSTACK */
    {csi_gref_where}, /* CSINN_OP_WHERE */
    {csi_gref_xor}, /* CSINN_OP_XOR */
    {csi_gref_yuv_rgb_scale}, /* CSINN_OP_YUV_RGB_SCALE */

    /* utils functions */
    {csi_gref_session_init},
    {csi_gref_session_deinit},
    {csi_gref_session_setup},
    {csi_gref_session_run},
    {csi_gref_update_input},
    {csi_gref_update_output},
    {csi_gref_set_input_number},
    {csi_gref_set_output_number},
    {csi_gref_get_input_number},
    {csi_gref_get_output_number},
    {csi_gref_set_input},
    {csi_gref_set_output},
    {csi_gref_get_input},
    {csi_gref_get_output},
    {csi_gref_set_tensor},
};

void *csi_bc_map_gref(int op, int dtype)
{
    return csi_bc_map_table_gref[op][0];
}
