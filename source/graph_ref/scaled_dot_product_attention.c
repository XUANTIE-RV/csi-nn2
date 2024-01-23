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

#include "shl_gref.h"

int shl_gref_scaled_dot_product_attention(struct csinn_tensor *query, struct csinn_tensor *key,
                                          struct csinn_tensor *value, struct csinn_tensor *output,
                                          struct csinn_scale_dot_attention_params *params)
{
    struct csinn_params_base *ptr = (struct csinn_params_base *)params;
    struct shl_node *layer =
        shl_node_alloc(CSINN_OP_SCALED_DOT_PRODUCT_ATTENTION, ptr->name, 3, 1, params);
    struct shl_node *in0 = (struct shl_node *)query->data;
    struct shl_node *in1 = (struct shl_node *)key->data;
    struct shl_node *in2 = (struct shl_node *)value->data;
    struct shl_node *out = shl_node_var_alloc(output->name, output);
    shl_node_add_in(layer, in0, 0);
    shl_node_add_in(layer, in1, 1);
    shl_node_add_in(layer, in2, 2);
    shl_node_add_out(layer, out, 0);
    output->data = out;
    struct shl_ref_graph *graph = shl_gref_get_graph(query->sess);
    shl_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}

int shl_gref_scaled_dot_product_attention_infer_shape(
    struct csinn_tensor *query, struct csinn_tensor *key, struct csinn_tensor *value,
    struct csinn_tensor *output, struct csinn_scale_dot_attention_params *params)
{
    shl_tensor_try_nc1xc0_to_ndarray_shape(query);
    output->layout = query->layout;
    output->dim_count = query->dim_count;
    for (int i = 0; i < query->dim_count; i++) {
        output->dim[i] = query->dim[i];
    }
    return CSINN_TRUE;
}
