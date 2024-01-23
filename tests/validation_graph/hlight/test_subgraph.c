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

#include "csi_nn.h"
#include "shl_gref.h"

struct csinn_session *create_session_base(int input_num, int output_num)
{
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_CPU_GRAPH;
    sess->base_quant_type = CSINN_QUANT_INT8_ASYM;
    sess->base_api = CSINN_REF;
    sess->base_dtype = CSINN_DTYPE_INT8;
    // sess->debug_level = CSINN_DEBUG_LEVEL_INFO;
    csinn_session_init(sess);
    csinn_set_input_number(input_num, sess);
    csinn_set_output_number(output_num, sess);

    return sess;
}

struct csinn_tensor *create_tensor_base(char *name, struct csinn_session *sess, int *shape,
                                        int shape_len)
{
    struct csinn_tensor *tensor = csinn_alloc_tensor(sess);
    tensor->name = name;
    tensor->layout = CSINN_LAYOUT_NCHW;
    for (int i = 0; i < shape_len; i++) {
        tensor->dim[i] = shape[i];
    }
    tensor->dim_count = shape_len;
    tensor->qinfo = (struct csinn_quant_info *)malloc(sizeof(struct csinn_quant_info));
    tensor->qinfo->max = 1.0;
    tensor->qinfo->min = 0;
    tensor->quant_channel = 1;

    return tensor;
}

struct shl_ref_graph *convert_graph2subgraph(struct shl_ref_graph *ograph)
{
    if (shl_debug_get_level() <= CSINN_DEBUG_LEVEL_INFO) {
        shl_debug_info("\nOriginal graph:\n");
        shl_gref_post_dfs(ograph, shl_subgraph_fvisit_print);
        shl_gref_reset_graph_visit(ograph);
    }

    struct shl_ref_graph *subgraph = shl_subgraph_generate(ograph);

    shl_debug_info("\nGenerated subgraph:\n");
    for (int i = 0; i < subgraph->layer_index; i++) {
        if (subgraph->layer[i]->type == CSINN_SUBGRAPH) {
            struct shl_ref_graph *s_subgraph = subgraph->layer[i]->data;
            if (s_subgraph->layer_size == 0) continue;
            shl_gref_update_input_output(subgraph, i);
            if (shl_debug_get_level() <= CSINN_DEBUG_LEVEL_INFO) {
                shl_debug_info("\n----  subgraph_%d:  ----\n", i);
                shl_gref_reset_graph_visit(s_subgraph);
                shl_gref_post_dfs(s_subgraph, shl_subgraph_fvisit_print);
                shl_gref_reset_graph_visit(s_subgraph);
                shl_debug_info("----subgraph_%d end.----\n\n", i);
            }
        } else {
            shl_debug_info("%s\n", subgraph->layer[i]->name);
        }
    }

    struct shl_ref_graph *ggraph = shl_subgraph_rebuild(subgraph);

    return ggraph;
}

/** Normal structure
 *  input -> relu ->relu -> softmax(cpu) -> relu
 *
 * Results:
 *  subgraph 1: relu->relu
 *  subgraph 2: softmax
 *  subgraph 3: relu
 */
void test_model1()
{
    printf("Start to test test_model1:\n");
    struct csinn_session *sess = create_session_base(1, 1);

    int input_shape[] = {1, 3, 32, 32};
    struct csinn_tensor *input = create_tensor_base("input", sess, input_shape, 4);
    struct csinn_relu_params *params_1 = csinn_alloc_params(sizeof(struct csinn_relu_params), sess);
    params_1->base.layout = CSINN_LAYOUT_NCHW;
    params_1->base.name = "params_1";
    params_1->base.api = CSINN_TH1520;
    struct csinn_tensor *relu1_out = create_tensor_base("relu1_out", sess, input_shape, 4);
    csinn_relu_init(input, relu1_out, params_1);

    struct csinn_relu_params *params_2 = csinn_alloc_params(sizeof(struct csinn_relu_params), sess);
    params_2->base.layout = CSINN_LAYOUT_NCHW;
    params_2->base.name = "params_2";
    params_2->base.api = CSINN_TH1520;
    struct csinn_tensor *relu2_out = create_tensor_base("relu2_out", sess, input_shape, 4);
    csinn_relu_init(relu1_out, relu2_out, params_2);

    struct csinn_softmax_params *params_3 =
        csinn_alloc_params(sizeof(struct csinn_softmax_params), sess);
    params_3->axis = 1;
    params_3->base.layout = CSINN_LAYOUT_NCHW;
    params_3->base.name = "params_3";
    struct csinn_tensor *softmax_out = create_tensor_base("softmax_out", sess, input_shape, 4);
    csinn_softmax_init(relu2_out, softmax_out, params_3);

    struct csinn_relu_params *params_4 = csinn_alloc_params(sizeof(struct csinn_relu_params), sess);
    params_4->base.layout = CSINN_LAYOUT_NCHW;
    params_4->base.name = "params_4";
    params_4->base.api = CSINN_TH1520;
    struct csinn_tensor *relu3_out = create_tensor_base("relu3_out", sess, input_shape, 4);
    csinn_relu_init(softmax_out, relu3_out, params_4);

    csinn_set_tensor_entry(input, sess);
    csinn_set_input(0, input, sess);

    csinn_relu(input, relu1_out, params_1);
    csinn_relu(relu1_out, relu2_out, params_2);
    csinn_softmax(relu2_out, softmax_out, params_3);
    csinn_relu(softmax_out, relu3_out, params_4);

    csinn_set_output(0, relu3_out, sess);

    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    struct shl_ref_graph *ggraph = convert_graph2subgraph(graph);

    // check results
    int fail = 0;
    if (ggraph->layer_index != 3) {
        printf("Actual subgraph number: %d, Reference: 3\n", ggraph->layer_index);
        fail = 1;
    }
    if (ggraph->layer[0]->type != CSINN_SUBGRAPH) {
        printf("0-th layer's type is %d, should be CSINN_SUBGRAPH(197)\n", ggraph->layer[0]->type);
        fail = 1;
    }
    if (ggraph->layer[1]->type != CSINN_OP_SOFTMAX) {
        printf("1-th layer's type is %d, should be CSINN_OP_SOFTMAX(153)\n",
               ggraph->layer[1]->type);
        fail = 1;
    }
    if (ggraph->layer[2]->type != CSINN_SUBGRAPH) {
        printf("2-th layer's type is %d, should be CSINN_SUBGRAPH(197)\n", ggraph->layer[2]->type);
        fail = 1;
    }
    if (fail) {
        printf("Test test_model1 fails.\n");
    } else {
        printf("Test test_model1 succeed.\n");
    }
}

/** Multi-branch structure:
 *          input
 *            |
 *           relu
 *          /    \
 *        relu  softmax(cpu)
 *          \    /
 *            add
 *             |
 *           output
 *
 * Results:
 *  subgraph 1: relu relu
 *  subgraph 2: softmax
 *  subgraph 3: add
 */
void test_model2()
{
    printf("Start to test test_model2:\n");
    struct csinn_session *sess = create_session_base(1, 1);

    int input_shape[] = {1, 3, 32, 32};
    struct csinn_tensor *input = create_tensor_base("input", sess, input_shape, 4);
    struct csinn_relu_params *params_1 = csinn_alloc_params(sizeof(struct csinn_relu_params), sess);
    params_1->base.layout = CSINN_LAYOUT_NCHW;
    params_1->base.name = "params_1";
    params_1->base.api = CSINN_TH1520;
    struct csinn_tensor *relu1_out = create_tensor_base("relu1_out", sess, input_shape, 4);
    csinn_relu_init(input, relu1_out, params_1);

    struct csinn_relu_params *params_2 = csinn_alloc_params(sizeof(struct csinn_relu_params), sess);
    params_2->base.layout = CSINN_LAYOUT_NCHW;
    params_2->base.name = "params_2";
    params_2->base.api = CSINN_TH1520;
    struct csinn_tensor *relu2_out = create_tensor_base("relu2_out", sess, input_shape, 4);
    csinn_relu_init(relu1_out, relu2_out, params_2);

    struct csinn_softmax_params *params_3 =
        csinn_alloc_params(sizeof(struct csinn_softmax_params), sess);
    params_3->axis = 1;
    params_3->base.layout = CSINN_LAYOUT_NCHW;
    params_3->base.name = "params_3";
    struct csinn_tensor *softmax_out = create_tensor_base("softmax_out", sess, input_shape, 4);
    csinn_softmax_init(relu1_out, softmax_out, params_3);

    struct csinn_diso_params *params_4 = csinn_alloc_params(sizeof(struct csinn_diso_params), sess);
    params_4->base.name = "params_4";
    params_4->base.api = CSINN_TH1520;
    struct csinn_tensor *add_out = create_tensor_base("add_out", sess, input_shape, 4);
    csinn_add_init(relu2_out, softmax_out, add_out, params_4);

    csinn_set_tensor_entry(input, sess);
    csinn_set_input(0, input, sess);

    csinn_relu(input, relu1_out, params_1);
    csinn_relu(relu1_out, relu2_out, params_2);
    csinn_softmax(relu2_out, softmax_out, params_3);
    csinn_add(relu2_out, softmax_out, add_out, params_4);

    csinn_set_output(0, add_out, sess);

    struct shl_ref_graph *graph = shl_gref_get_graph(sess);
    struct shl_ref_graph *ggraph = convert_graph2subgraph(graph);

    // check results
    int fail = 0;
    if (ggraph->layer_index != 3) {
        printf("Actual subgraph number: %d, Reference: 3\n", ggraph->layer_index);
        fail = 1;
    }
    if (ggraph->layer[0]->type != CSINN_SUBGRAPH) {
        printf("0-th layer's type is %d, should be CSINN_SUBGRAPH(197)\n", ggraph->layer[0]->type);
        fail = 1;
    }
    if (ggraph->layer[1]->type != CSINN_OP_SOFTMAX) {
        printf("1-th layer's type is %d, should be CSINN_OP_SOFTMAX(153)\n",
               ggraph->layer[1]->type);
        fail = 1;
    }
    if (ggraph->layer[2]->type != CSINN_SUBGRAPH) {
        printf("2-th layer's type is %d, should be CSINN_SUBGRAPH(197)\n", ggraph->layer[2]->type);
        fail = 1;
    }
    if (fail) {
        printf("Test test_model2 fails.\n");
    } else {
        printf("Test test_model2 succeed.\n");
    }
}

void test_subgraph()
{
    test_model1();
    test_model2();
}

int main(int argc, char **argv)
{
    printf("Testing function of subgraph fusion.\n");

    test_subgraph();

    return 0;
}