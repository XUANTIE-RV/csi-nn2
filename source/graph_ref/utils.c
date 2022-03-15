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

/* CSI-NN2 version 1.12.x */

#include "csi_gref.h"

int csi_gref_graph_insert(struct csi_node *node, struct csi_ref_graph *graph)
{
    if (graph->layer_size == 0 || graph->layer_index == graph->layer_size - 1) {
        graph->layer_size += 128;
        graph->layer = csi_mem_realloc(graph->layer, graph->layer_size * sizeof(struct csi_node *));
    }
    graph->layer[graph->layer_index] = node;
    graph->layer_index++;
    return CSINN_TRUE;
}

int csi_gref_siso_op(struct csi_tensor *input,
                     struct csi_tensor *output,
                     int op,
                     void *params)
{
    struct csi_params_base *ptr = params;
    struct csi_node *layer = csi_node_alloc(op, ptr->name, 1, 1, params);
    struct csi_node *in0 = (struct csi_node *)input->data;
    struct csi_node *out = csi_node_var_alloc(output->name, output);
    csi_node_add_in(layer, in0, 0);
    csi_node_add_out(layer, out, 0);
    output->data = out;
    struct csi_ref_graph *graph = csi_gref_get_graph(input->sess);
    csi_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}

int csi_gref_diso_op(struct csi_tensor *input0,
                     struct csi_tensor *input1,
                     struct csi_tensor *output,
                     int op,
                     void *params)
{
    struct csi_params_base *ptr = params;
    struct csi_node *layer = csi_node_alloc(op, ptr->name, 2, 1, params);
    struct csi_node *in0 = (struct csi_node *)input0->data;
    struct csi_node *in1;
    if (input1->is_const) {
        in1 = csi_node_const_var_alloc(input1->name, input1);
    } else {
        in1 = (struct csi_node *)input1->data;
    }
    struct csi_node *out = csi_node_var_alloc(output->name, output);
    csi_node_add_in(layer, in0, 0);
    csi_node_add_in(layer, in1, 1);
    csi_node_add_out(layer, out, 0);
    output->data = out;
    struct csi_ref_graph *graph = csi_gref_get_graph(input0->sess);
    csi_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}


/* single input double const single output */
int csi_gref_sidcso_op(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct csi_tensor *const0,
                       struct csi_tensor *const1,
                       int op,
                       void *params)
{
    struct csi_params_base *ptr = params;
    struct csi_node *layer = csi_node_alloc(op, ptr->name, 3, 1, params);
    struct csi_node *in0 = (struct csi_node *)input->data;
    struct csi_node *in1 = csi_node_const_var_alloc(const0->name, const0);
    struct csi_node *in2 = csi_node_const_var_alloc(const1->name, const1);
    struct csi_node *out = csi_node_var_alloc(output->name, output);
    csi_node_add_in(layer, in0, 0);
    csi_node_add_in(layer, in1, 1);
    csi_node_add_in(layer, in2, 2);
    csi_node_add_out(layer, out, 0);
    output->data = out;
    struct csi_ref_graph *graph = csi_gref_get_graph(input->sess);
    csi_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}

