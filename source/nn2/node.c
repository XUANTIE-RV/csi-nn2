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

#include "shl_memory.h"
#include "shl_node.h"
#include "shl_utils.h"

struct shl_node *shl_node_alloc(int node_type, char *name, int in_num, int out_num, void *data)
{
    struct shl_node *ret = shl_mem_alloc(sizeof(struct shl_node));

    ret->type = node_type;
    ret->name = name;
    ret->data = data;
    ret->in_num = in_num;
    ret->out_num = out_num;
    if (in_num != 0) {
        ret->in = shl_mem_alloc(in_num * sizeof(struct shl_node *));
    }
    if (out_num != 0) {
        ret->out = shl_mem_alloc(out_num * sizeof(struct shl_node *));
    }
    ret->subgraph_idx = -1;

    return ret;
}

struct shl_node *shl_node_var_alloc(char *name, void *data)
{
    return shl_node_alloc(CSINN_TENSOR, name, 1, 1, data);
}

struct shl_node *shl_node_const_var_alloc(char *name, void *data)
{
    return shl_node_alloc(CSINN_TENSOR, name, 0, 1, data);
}

int shl_node_free(struct shl_node *node)
{
    shl_mem_free(node->in);
    shl_mem_free(node->out);
    shl_mem_free(node);
    return CSINN_TRUE;
}

int shl_node_add_in(struct shl_node *node, struct shl_node *in, int index)
{
    node->in[index] = in;
    if (in->type == CSINN_TENSOR) {
        if (in->out_num == 1 && !in->out[0]) {
            in->out[0] = node;
        } else {
            in->out = shl_mem_realloc(in->out, (in->out_num + 1) * sizeof(struct shl_node *),
                                      in->out_num * sizeof(struct shl_node *));
            in->out[in->out_num] = node;
            in->out_num++;
        }
    }
    return CSINN_TRUE;
}

int shl_node_add_out(struct shl_node *node, struct shl_node *out, int index)
{
    node->out[index] = out;

    if (out->type == CSINN_TENSOR && out->in_num == 1) {
        out->in[0] = node;
    }
    return CSINN_TRUE;
}

int shl_node_get_in_number(struct shl_node *node) { return node->in_num; }

int shl_node_get_out_number(struct shl_node *node) { return node->out_num; }

int shl_node_get_non_const_in_number(struct shl_node *node)
{
    int in_num = shl_node_get_in_number(node);
    int const_in_num = 0;
    for (int i = 0; i < in_num; i++) {
        struct csinn_tensor *data = node->in[i]->data;
        if (data->is_const) {
            const_in_num++;
        }
    }
    return (in_num - const_in_num);
}

struct shl_node *shl_node_get_in(struct shl_node *node, int index) { return node->in[index]; }

struct shl_node *shl_node_get_out(struct shl_node *node, int index) { return node->out[index]; }

int shl_node_restrict_map_insert(int value, struct shl_node *node)
{
    node->restricted_map =
        shl_mem_realloc(node->restricted_map, (node->restricted_map_num + 1) * sizeof(int),
                        node->restricted_map_num * sizeof(int));
    node->restricted_map[node->restricted_map_num] = value;
    node->restricted_map_num++;
    return CSINN_TRUE;
}

int shl_node_find(struct shl_node **list, int len, struct shl_node *node)
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
