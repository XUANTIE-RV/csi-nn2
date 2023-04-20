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

/* SHL version 2.1.x */

#include "shl_gref.h"

int shl_gref_split(struct csinn_tensor *input, struct csinn_tensor **output,
                   struct csinn_split_params *params)
{
    struct shl_node *layer =
        shl_node_alloc(CSINN_OP_SPLIT, params->base.name, 1, params->output_num, params);

    struct shl_node *in_tensor = (struct shl_node *)(input->data);
    shl_node_add_in(layer, in_tensor, 0);

    for (int i = 0; i < params->output_num; i++) {
        struct shl_node *out = shl_node_var_alloc(output[i]->name, output[i]);
        shl_node_add_out(layer, out, i);
        output[i]->data = out;
    }
    struct shl_ref_graph *graph = shl_gref_get_graph(input->sess);
    shl_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}

/* TODO: support onnx split, both refer and opt implementations */
int shl_gref_split_infer_shape(struct csinn_tensor *input, struct csinn_tensor **output,
                               struct csinn_split_params *params)
{
    int32_t axis = params->axis;
    int32_t out_num = params->output_num;
    int32_t *split_index = params->split_index;

    for (int i = 0; i < out_num; i++) {
        output[i]->dim_count = input->dim_count;
        for (int j = 0; j < input->dim_count; j++) {
            output[i]->dim[j] = input->dim[j];
        }
    }
    if (split_index != NULL) {
        // recover output->dim[axis]
        output[0]->dim[axis] = split_index[0];
        for (int i = 1; i < out_num - 1; i++) {
            output[i]->dim[axis] = split_index[i] - split_index[i - 1];
        }
        output[out_num - 1]->dim[axis] = input->dim[axis] - split_index[out_num - 2];
    } else {
        int avg_dim = (input->dim[axis] + out_num - 1) / out_num;
        for (int i = 0; i < out_num - 1; i++) {
            output[i]->dim[axis] = avg_dim;
        }
        output[out_num - 1]->dim[axis] = input->dim[axis] - avg_dim * (out_num - 1);
    }
    return CSINN_TRUE;
}
