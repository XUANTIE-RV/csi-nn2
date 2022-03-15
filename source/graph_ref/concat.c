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

int csi_gref_concat(struct csi_tensor **input,
                    struct csi_tensor *output,
                    struct concat_params *params)
{
    struct csi_node *layer = csi_node_alloc(CSINN_OP_CONCAT, params->base.name, params->inputs_count, 1, params);

    for (int i =0; i < params->inputs_count; i++){
        struct csi_node *in_tensor = (struct csi_node *)(input[i]->data);
        if (input[i]->is_const) {
            in_tensor = csi_node_const_var_alloc(input[i]->name, input[i]);
        } else {
            in_tensor = (struct csi_node *)(input[i]->data);
        }
        csi_node_add_in(layer, in_tensor, i);
    }

    struct csi_node *out = csi_node_var_alloc(output->name, output);
    csi_node_add_out(layer, out, 0);
    output->data = out;
    struct csi_ref_graph *graph = csi_gref_get_graph(input[0]->sess);
    csi_gref_graph_insert(layer, graph);

    return CSINN_TRUE;
}

