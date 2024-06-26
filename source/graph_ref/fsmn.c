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

int shl_gref_fsmn(struct csinn_tensor *frame, struct csinn_tensor *l_filter,
                  struct csinn_tensor *r_filter, struct csinn_tensor *frame_sequence,
                  struct csinn_tensor *frame_counter, struct csinn_tensor *output,
                  struct csinn_fsmn_params *params)
{
    struct csinn_params_base *ptr = (void *)params;
    struct shl_node *layer = shl_node_alloc(CSINN_OP_FSMN, ptr->name, 5, 1, params);
    struct shl_node *in0 = (struct shl_node *)frame->data;
    struct shl_node *in1 = shl_node_const_var_alloc(l_filter->name, l_filter);
    struct shl_node *in2 = shl_node_const_var_alloc(r_filter->name, r_filter);
    struct shl_node *in3 = shl_node_const_var_alloc(frame_sequence->name, frame_sequence);
    struct shl_node *in4 = shl_node_const_var_alloc(frame_counter->name, frame_counter);
    struct shl_node *out = shl_node_var_alloc(output->name, output);
    shl_node_add_in(layer, in0, 0);
    shl_node_add_in(layer, in1, 1);
    shl_node_add_in(layer, in2, 2);
    shl_node_add_in(layer, in3, 3);
    shl_node_add_in(layer, in4, 4);
    shl_node_add_out(layer, out, 0);
    output->data = out;
    struct shl_ref_graph *graph = shl_gref_get_graph(frame->sess);
    shl_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}

int shl_gref_fsmn_infer_shape(struct csinn_tensor *frame, struct csinn_tensor *l_filter,
                              struct csinn_tensor *r_filter, struct csinn_tensor *frame_sequence,
                              struct csinn_tensor *frame_counter, struct csinn_tensor *output,
                              struct csinn_fsmn_params *params)
{
    shl_debug_error("shl_gref_all_infer_shape unsupport\n");
    return CSINN_FALSE;
}
