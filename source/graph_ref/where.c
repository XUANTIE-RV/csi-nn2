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

#include "shl_gref.h"

int shl_gref_where(struct csinn_tensor *condition, struct csinn_tensor *x, struct csinn_tensor *y,
                   struct csinn_tensor *output, struct csinn_where_params *params)
{
    struct csinn_params_base *ptr = (struct csinn_params_base *)params;
    struct shl_node *layer = shl_node_alloc(CSINN_OP_WHERE, ptr->name, 3, 1, params);
    struct shl_node *in0 = (struct shl_node *)condition->data;
    struct shl_node *in1;
    struct shl_node *in2;
    if (x->is_const) {
        in1 = shl_node_const_var_alloc(x->name, x);
    } else {
        in1 = (struct shl_node *)x->data;
    }
    if (y->is_const) {
        in2 = shl_node_const_var_alloc(y->name, y);
    } else {
        in2 = (struct shl_node *)y->data;
    }
    struct shl_node *out = shl_node_var_alloc(output->name, output);
    shl_node_add_in(layer, in0, 0);
    shl_node_add_in(layer, in1, 1);
    shl_node_add_in(layer, in2, 2);
    shl_node_add_out(layer, out, 0);
    output->data = out;
    struct shl_ref_graph *graph = shl_gref_get_graph(condition->sess);
    shl_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}

int shl_gref_where_infer_shape(struct csinn_tensor *condition, struct csinn_tensor *x,
                               struct csinn_tensor *y, struct csinn_tensor *output,
                               struct csinn_where_params *params)
{
    shl_tensor_try_nc1xc0_to_ndarray_shape(condition);
    shl_tensor_try_nc1xc0_to_ndarray_shape(x);
    shl_tensor_try_nc1xc0_to_ndarray_shape(y);

    if (x->data == NULL || y->data == NULL) {
        // Return the indices of non-zero elements
        int c_size = 1;
        for (int i = 0; i < condition->dim_count; i++) {
            c_size *= condition->dim[i];
        }
        uint8_t *c_data = (uint8_t *)condition->data;
        int nonzero_count = 0;
        for (int i = 0; i < c_size; i++) {
            if (c_data[i] != 0) {
                nonzero_count++;
            }
        }
        output->dim_count = 2;
        output->dim[0] = nonzero_count;
        output->dim[1] = condition->dim_count;
    } else {
        // Multiplex x and y
        int shape_rank = 0;
        shape_rank = condition->dim_count > shape_rank ? condition->dim_count : shape_rank;
        shape_rank = x->dim_count > shape_rank ? x->dim_count : shape_rank;
        shape_rank = y->dim_count > shape_rank ? y->dim_count : shape_rank;
        output->dim_count = shape_rank;
        for (int i = 0; i < shape_rank; i++) {
            int out_dim = 0;
            int c_idx = condition->dim_count - 1 - i;
            if (c_idx >= 0 && condition->dim[c_idx] > out_dim) {
                out_dim = condition->dim[c_idx];
            }
            int x_idx = x->dim_count - 1 - i;
            if (x_idx >= 0 && x->dim[x_idx] > out_dim) {
                out_dim = x->dim[x_idx];
            }
            int y_idx = y->dim_count - 1 - i;
            if (y_idx >= 0 && y->dim[y_idx] > out_dim) {
                out_dim = y->dim[y_idx];
            }
            output->dim[shape_rank - 1 - i] = out_dim;
        }
    }
    return CSINN_TRUE;
}
