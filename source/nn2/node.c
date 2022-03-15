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

#include "csi_nn.h"
#include "csi_node.h"

struct csi_node *csi_node_alloc(int node_type, char *name, int in_num, int out_num, void *data)
{
    struct csi_node *ret = csi_mem_alloc(sizeof(struct csi_node));

    ret->type = node_type;
    ret->name = name;
    ret->data = data;
    ret->in_num = in_num;
    ret->out_num = out_num;
    if (in_num != 0) {
        ret->in = csi_mem_alloc(in_num * sizeof(struct csi_node *));
    }
    if (out_num != 0) {
        ret->out = csi_mem_alloc(out_num * sizeof(struct csi_node *));
    }
    ret->subgraph_idx = -1;

    return ret;
}

struct csi_node *csi_node_var_alloc(char *name, void *data)
{
    return csi_node_alloc(CSINN_TENSOR, name, 1, 1, data);
}

struct csi_node *csi_node_const_var_alloc(char *name, void *data)
{
    return csi_node_alloc(CSINN_TENSOR, name, 0, 1, data);
}

int csi_node_free(struct csi_node *node)
{
    csi_mem_free(node->in);
    csi_mem_free(node->out);
    csi_mem_free(node);
    return CSINN_TRUE;
}

int csi_node_add_in(struct csi_node *node, struct csi_node *in, int index)
{
    node->in[index] = in;
    return CSINN_TRUE;
}

int csi_node_add_out(struct csi_node *node, struct csi_node *out, int index)
{
    node->out[index] = out;

    if (out->type == CSINN_TENSOR && out->in_num == 1) {
        out->in[0] = node;
    }
    return CSINN_TRUE;
}

int csi_node_get_in_number(struct csi_node *node)
{
    return node->in_num;
}

int csi_node_get_out_number(struct csi_node *node)
{
    return node->out_num;
}

int csi_node_get_non_const_in_number(struct csi_node *node)
{
    int in_num = csi_node_get_in_number(node);
    int const_in_num = 0;
    for (int i = 0; i < in_num; i++) {
        struct csi_tensor *data = node->in[i]->data;
        if (data->is_const) {
            const_in_num ++;
        }
    }
    return (in_num - const_in_num);
}

struct csi_node *csi_node_get_in(struct csi_node *node, int index)
{
    return node->in[index];
}

struct csi_node *csi_node_get_out(struct csi_node *node, int index)
{
    return node->out[index];
}

int csi_node_restrict_map_insert(int value, struct csi_node *node)
{
    node->restricted_map =
        csi_mem_realloc(node->restricted_map, (node->restricted_map_num + 1) * sizeof(int));
    node->restricted_map[node->restricted_map_num] = value;
    node->restricted_map_num++;
    return CSINN_TRUE;
}

int csi_node_find(struct csi_node **list, int len, struct csi_node *node)
{
    int res = -1;
    if (!list || len < 1) {
        return res;
    }
    for (int i = 0; i < len; i++) {
        if (list[i] == node) {
            res = i;
            break;
        }
    }
    return res;
}
