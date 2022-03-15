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

#ifndef INCLUDE_CSI_NODE_H_
#define INCLUDE_CSI_NODE_H_

struct csi_node {
    int type;
    struct csi_node **in;
    struct csi_node **out;
    int subgraph_idx;
    int in_num;
    int out_num;
    char *name;
    void *data;
    int ref_count;
    int ref_count_init;
    int visited;
    int *restricted_map;
    int restricted_map_num;
};

/* node */
struct csi_node *csi_node_alloc(int node_type, char *name, int in_num, int out_num, void *data);
struct csi_node *csi_node_var_alloc(char *name, void *data);
struct csi_node *csi_node_const_var_alloc(char *name, void *data);
int csi_node_free(struct csi_node *node);
int csi_node_add_in(struct csi_node *node, struct csi_node *in, int index);
int csi_node_add_out(struct csi_node *node, struct csi_node *out, int index);
int csi_node_get_in_number(struct csi_node *node);
int csi_node_get_out_number(struct csi_node *node);
int csi_node_get_non_const_in_number(struct csi_node *node);
struct csi_node *csi_node_get_in(struct csi_node *node, int index);
struct csi_node *csi_node_get_out(struct csi_node *node, int index);
int csi_node_restrict_map_insert(int value, struct csi_node *node);
int csi_node_find(struct csi_node **list, int len, struct csi_node *node);

#endif  // INCLUDE_CSI_NODE_H_
