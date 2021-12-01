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

/* CSI-NN2 version 1.8.x */

#include "csi_gref.h"

int csi_gref_split(struct csi_tensor *input,
                   struct csi_tensor **output,
                   struct split_params *params)
{
    struct csi_node *layer = csi_node_alloc(CSINN_OP_SPLIT, params->base.name, 1, params->output_num, params);

    struct csi_node *in_tensor = (struct csi_node *)(input->data);
    csi_node_add_in(layer, in_tensor, 0);

    for (int i = 0; i< params->output_num; i++){
        struct csi_node *out = csi_node_var_alloc(output[i]->name, output[i]);
        csi_node_add_out(layer, out, i);
        output[i]->data = out;
    }
    struct csi_ref_graph *graph = csi_gref_get_graph(input->sess);
    csi_gref_graph_insert(layer, graph);
    return CSINN_FALSE;
}
