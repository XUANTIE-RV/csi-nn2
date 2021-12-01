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

int csi_gref_fsmn(struct csi_tensor *frame,
                  struct csi_tensor *l_filter,
                  struct csi_tensor *r_filter,
                  struct csi_tensor *frame_sequence,
                  struct csi_tensor *frame_counter,
                  struct csi_tensor *output,
                  struct fsmn_params *params)
{
    struct csi_params_base *ptr = (void *)params;
    struct csi_node *layer = csi_node_alloc(CSINN_OP_FSMN, ptr->name, 5, 1, params);
    struct csi_node *in0 = (struct csi_node *)frame->data;
    struct csi_node *in1 = csi_node_const_var_alloc(l_filter->name, l_filter);
    struct csi_node *in2 = csi_node_const_var_alloc(r_filter->name, r_filter);
    struct csi_node *in3 = csi_node_const_var_alloc(frame_sequence->name, frame_sequence);
    struct csi_node *in4 = csi_node_const_var_alloc(frame_counter->name, frame_counter);
    struct csi_node *out = csi_node_var_alloc(output->name, output);
    csi_node_add_in(layer, in0, 0);
    csi_node_add_in(layer, in1, 1);
    csi_node_add_in(layer, in2, 2);
    csi_node_add_in(layer, in3, 3);
    csi_node_add_in(layer, in4, 4);
    csi_node_add_out(layer, out, 0);
    output->data = out;
    struct csi_ref_graph *graph = csi_gref_get_graph(frame->sess);
    csi_gref_graph_insert(layer, graph);
    return CSINN_TRUE;
}
