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

#include "csi_node.h"

struct csi_node *csi_node_alloc(int node_type, char *name, int in_num, int out_num, void *data)
{
    struct csi_node *ret = calloc(sizeof(struct csi_node), 1);

    ret->type = node_type;
    ret->name = name;
    ret->data = data;
    ret->in_num = in_num;
    ret->out_num = out_num;
    ret->in = calloc(sizeof(struct csi_node*), in_num);
    ret->out = calloc(sizeof(struct csi_node*), out_num);

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
    free(node->in);
    free(node->out);
    free(node);
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

struct csi_node *csi_node_get_in(struct csi_node *node, int index)
{
    return node->in[index];
}

struct csi_node *csi_node_get_out(struct csi_node *node, int index)
{
    return node->out[index];
}
