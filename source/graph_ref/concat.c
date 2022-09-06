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

/* CSI-NN2 version 2.0.x */

#include "shl_gref.h"

int shl_gref_concat(struct csinn_tensor **input, struct csinn_tensor *output,
                    struct csinn_concat_params *params)
{
    struct shl_node *layer =
        shl_node_alloc(CSINN_OP_CONCAT, params->base.name, params->inputs_count, 1, params);

    for (int i = 0; i < params->inputs_count; i++) {
        struct shl_node *in_tensor = (struct shl_node *)(input[i]->data);
        if (input[i]->is_const) {
            in_tensor = shl_node_const_var_alloc(input[i]->name, input[i]);
        } else {
            in_tensor = (struct shl_node *)(input[i]->data);
        }
        shl_node_add_in(layer, in_tensor, i);
    }

    struct shl_node *out = shl_node_var_alloc(output->name, output);
    shl_node_add_out(layer, out, 0);
    output->data = out;
    struct shl_ref_graph *graph = shl_gref_get_graph(input[0]->sess);
    shl_gref_graph_insert(layer, graph);

    return CSINN_TRUE;
}
