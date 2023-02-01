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

int shl_gref_graph_insert(struct shl_node *node, struct shl_ref_graph *graph)
{
    if (graph->layer_size == 0 || graph->layer_index == graph->layer_size - 1) {
        graph->layer_size += 128;
        graph->layer = shl_mem_realloc(graph->layer, graph->layer_size * sizeof(struct shl_node *));
    }
    graph->layer[graph->layer_index] = node;
    graph->layer_index++;
    return CSINN_TRUE;
}

int shl_gref_siso_op(struct csinn_tensor *input, struct csinn_tensor *output, int op, void *params)
{
    struct csinn_params_base *ptr = params;
    struct shl_node *layer = shl_node_alloc(op, ptr->name, 1, 1, params);
    struct shl_node *in0 = (struct shl_node *)input->data;
    struct shl_node *out = shl_node_var_alloc(output->name, output);
    shl_node_add_in(layer, in0, 0);
    shl_node_add_out(layer, out, 0);
    output->data = out;
    struct shl_ref_graph *graph = shl_gref_get_graph(input->sess);
    shl_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}

int shl_gref_diso_op(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, int op, void *params)
{
    struct csinn_params_base *ptr = params;
    struct shl_node *layer = shl_node_alloc(op, ptr->name, 2, 1, params);
    struct shl_node *in0;
    struct shl_node *in1;
    if (input0->is_const) {
        in0 = shl_node_const_var_alloc(input0->name, input0);
    } else {
        in0 = (struct shl_node *)input0->data;
    }
    if (input1->is_const) {
        in1 = shl_node_const_var_alloc(input1->name, input1);
    } else {
        in1 = (struct shl_node *)input1->data;
    }
    struct shl_node *out = shl_node_var_alloc(output->name, output);
    shl_node_add_in(layer, in0, 0);
    shl_node_add_in(layer, in1, 1);
    shl_node_add_out(layer, out, 0);
    output->data = out;
    struct shl_ref_graph *graph = shl_gref_get_graph(input0->sess);
    shl_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}

/* single input double const single output */
int shl_gref_sidcso_op(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *const0, struct csinn_tensor *const1, int op,
                       void *params)
{
    struct csinn_params_base *ptr = params;
    struct shl_node *layer = shl_node_alloc(op, ptr->name, 3, 1, params);
    struct shl_node *in0 = (struct shl_node *)input->data;
    struct shl_node *in1 = shl_node_const_var_alloc(const0->name, const0);
    struct shl_node *in2 = shl_node_const_var_alloc(const1->name, const1);
    struct shl_node *out = shl_node_var_alloc(output->name, output);
    shl_node_add_in(layer, in0, 0);
    shl_node_add_in(layer, in1, 1);
    shl_node_add_in(layer, in2, 2);
    shl_node_add_out(layer, out, 0);
    output->data = out;
    struct shl_ref_graph *graph = shl_gref_get_graph(input->sess);
    shl_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}
