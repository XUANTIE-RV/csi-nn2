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

int shl_gref_where_softmax(struct csinn_tensor *condition, struct csinn_tensor *y,
                           struct csinn_tensor *output, struct csinn_where_softmax_params *params)
{
    struct csinn_params_base *ptr = (struct csinn_params_base *)params;
    struct shl_node *layer = shl_node_alloc(CSINN_OP_WHERE_SOFTMAX, ptr->name, 2, 1, params);
    struct shl_node *in0 = (struct shl_node *)condition->data;
    struct shl_node *in1;
    if (y->is_const) {
        in1 = shl_node_const_var_alloc(y->name, y);
    } else {
        in1 = (struct shl_node *)y->data;
    }
    struct shl_node *out = shl_node_var_alloc(output->name, output);
    shl_node_add_in(layer, in0, 0);
    shl_node_add_in(layer, in1, 1);
    shl_node_add_out(layer, out, 0);
    output->data = out;
    struct shl_ref_graph *graph = shl_gref_get_graph(condition->sess);
    shl_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}

int shl_gref_where_softmax_infer_shape(struct csinn_tensor *condition, struct csinn_tensor *y,
                                       struct csinn_tensor *output,
                                       struct csinn_where_softmax_params *params)
{
    int shape_rank = 0;
    shape_rank = condition->dim_count > shape_rank ? condition->dim_count : shape_rank;
    shape_rank = y->dim_count > shape_rank ? y->dim_count : shape_rank;
    output->dim_count = shape_rank;
    for (int i = 0; i < shape_rank; i++) {
        int out_dim = 0;
        int c_idx = condition->dim_count - 1 - i;
        if (c_idx >= 0 && condition->dim[c_idx] > out_dim) {
            out_dim = condition->dim[c_idx];
        }
        int y_idx = y->dim_count - 1 - i;
        if (y_idx >= 0 && y->dim[y_idx] > out_dim) {
            out_dim = y->dim[y_idx];
        }
        output->dim[shape_rank - 1 - i] = out_dim;
    }
    return CSINN_TRUE;
}
