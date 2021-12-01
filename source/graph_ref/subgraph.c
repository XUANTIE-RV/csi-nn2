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

#include "csi_gref.h"
#include "csi_utils.h"

void csi_subgraph_alloc(struct csi_node *node, struct csi_ref_graph *ograph, struct csi_ref_graph *ggraph)
{
    struct csi_ref_graph *sgraph = csi_mem_alloc(sizeof(struct csi_ref_graph));
    sgraph->input_num = 1;
    sgraph->output_num = 1;
    sgraph->input = csi_mem_alloc(sgraph->input_num * sizeof(struct csi_node *));
    sgraph->output = csi_mem_alloc(sgraph->output_num * sizeof(struct csi_node *));
    csi_gref_graph_insert(node, sgraph);
    int node_input_num = 0;
    for (int i = 0; i < node->in_num; i++) {
        struct csi_tensor *node_in = node->in[i]->data;
        if (!node_in->is_const) {
            node_input_num++;
        }
    }

    struct csi_node *sg_in = csi_node_alloc(CSINN_SUBGRAPH, "graph_in", node_input_num, node_input_num, sgraph);
    for (int i = 0; i < node_input_num; i++) {
        sg_in->in[i] = node->in[i];
        struct csi_tensor *sg_in_tensor = csi_alloc_tensor(NULL);
        csi_tensor_copy(sg_in_tensor, node->in[i]->data);
        struct csi_node *sg_in_node = csi_node_var_alloc("graph_in_tensor", sg_in_tensor);
        node->in[i] = sg_in_node;
        csi_gref_graph_insert(sg_in, ggraph);
    }
    sgraph->input[0] = node->in[0];
    sgraph->output[0] = node->out[0];
    struct csi_node *sg_out = csi_node_alloc(CSINN_SUBGRAPH_RETURN, "graph_out", node->out_num, node->out_num, ggraph->layer[ggraph->layer_index]);
    for (int i = 0; i < node->out_num; i++) {
        sg_out->out[i] = node->out[i];
        struct csi_tensor *sg_out_tensor = csi_alloc_tensor(NULL);
        csi_tensor_copy(sg_out_tensor, node->out[i]->data);
        struct csi_node *sg_out_node = csi_node_var_alloc("graph_out_tensor", sg_out_tensor);
        node->out[i] = sg_out_node;
        sg_out->in[i] = sg_out_node;
        csi_gref_graph_insert(sg_out, sgraph);
    }
}

int csi_subgraph_init(struct csi_node *n)
{
    struct csi_ref_graph *sgraph = n->data;
    struct csi_node *node = sgraph->layer[0];
    struct csi_params_base *params = node->data;
    struct csi_session *sub_sess = csi_alloc_session();
    sub_sess->base_api = CSINN_LIGHT;
    sub_sess->base_dtype = CSINN_DTYPE_FLOAT32;
    sub_sess->debug_level = CSI_DEBUG_LEVEL_INFO;
    csi_session_init(sub_sess);

    params->sess = sub_sess;
    int (*func)();
    struct csi_tensor *input0, *output, *kernel, *bias;
    input0 = node->in[0]->data;
    input0->sess = sub_sess;
    func = csi_bc_map(params->api, CSINN_RM_LAYER, node->type, input0->dtype);

    int ret = CSINN_TRUE;

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
        csi_set_input_number(1, sub_sess);
        csi_set_output_number(1, sub_sess);
        csi_set_tensor_entry(input0, sub_sess);
        csi_set_input(0, input0, sub_sess);
        output = node->out[0]->data;
        output->sess = sub_sess;
        ret = func(input0, output, params);
        csi_set_output(0, output, sub_sess);
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
        csi_set_input_number(1, sub_sess);
        csi_set_output_number(1, sub_sess);
        csi_set_tensor_entry(input0, sub_sess);
        csi_set_input(0, input0, sub_sess);
        output = node->out[0]->data;
        output->sess = sub_sess;
        kernel = node->in[1]->data;
        kernel->sess = sub_sess;
        bias = node->in[2]->data;
        bias->sess = sub_sess;
        ret = func(input0, output, kernel, bias, params);
        csi_set_output(0, output, sub_sess);
        break;
    default:
        CSI_DEBUG_CALL(printf("unknown op1\n"));
        return CSINN_FALSE;
    }
    csi_session_setup(sub_sess);

    return ret;
}

int csi_subgraph_deinit(struct csi_node *n)
{
    struct csi_ref_graph *sgraph = n->data;
    struct csi_node *node = sgraph->layer[0];
    struct csi_params_base *params = node->data;
    csi_session_deinit(params->sess);
    return 0;
}

static int csi_subgraph_entry(struct csi_node *n)
{
    struct csi_ref_graph *sgraph = n->data;

    for (int i = 0; i < n->in_num; i++) {
        struct csi_tensor *tsrc = n->in[i]->data;
        struct csi_tensor *tdst = sgraph->input[i]->data;
        // if (tdst->data == NULL) {
            tdst->data = tsrc->data;
        // } else if (tdst->data != tsrc->data) {
        //     memcpy(tdst->data, tsrc->data, csi_tensor_byte_size(tsrc));
        // }
    }
    for (int i = 0; i < sgraph->output_num; i++) {
        struct csi_tensor *out = sgraph->output[i]->data;
        out->data = NULL;
    }
    return CSINN_TRUE;
}

static int csi_subgraph_return(struct csi_ref_graph *graph, struct csi_node *ret_node)
{
    for (int i = 0; i < graph->output_num; i++) {
        struct csi_tensor *tsrc = ret_node->out[i]->data;
        struct csi_tensor *tdst = graph->output[i]->data;
        // if (tdst->data == NULL) {
            tdst->data = tsrc->data;
        // } else if (tdst->data != tsrc->data) {
        //     memcpy(tdst->data, tsrc->data, csi_tensor_byte_size(tsrc));
        // }
    }
    return CSINN_TRUE;
}

int csi_subgraph_run_init(struct csi_node *n)
{
    csi_subgraph_entry(n);
}

int csi_subgraph_run_deinit(struct csi_node *n)
{

}

int csi_subgraph_run(struct csi_node *n)
{
    struct csi_ref_graph *sgraph = n->data;
    struct csi_node *node = sgraph->layer[0];
    struct csi_params_base *params = node->data;
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
        csi_update_input(0, node->in[0]->data, params->sess);
        csi_session_run(params->sess);
        csi_get_output(0, node->out[0]->data, params->sess);
        break;
    default:
        CSI_DEBUG_CALL(printf("unknown op2\n"));
        return CSINN_FALSE;
    }

    /* CSINN_SUBGRAPH_RETURN */
    csi_subgraph_return(sgraph, node);
    return ret;
}
