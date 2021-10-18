/*
 * Copyright (C) 2016-2020 C-SKY Limited. All rights reserved.
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

#include "csi_ovx.h"

int csi_ovx_split(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct split_params *params)
{
    vsi_nn_node_t *node;
    vsi_nn_node_id_t node_id;
    vsi_nn_tensor_id_t input_id;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_id_t output_id;
    struct __target_data *td = input->t_private;
    vsi_nn_graph_t *graph = td->graph;
    uint32_t input_num = 1;
    uint32_t split_count = params->output_num;
    int i = 0;
    uint32_t *slices = (uint32_t*)malloc(sizeof(uint32_t)*split_count);
    for (i = 0; i < split_count; i++) {
        if (i == 0) {
            slices[i] = params->split_index[i];
        } else if (i == split_count - 1) {
            slices[i] = input->dim[params->axis] - params->split_index[i-1];
        } else {
            slices[i] = params->split_index[i] - params->split_index[i-1];
        }
    }
    node = vsi_nn_AddNode(graph, VSI_NN_OP_SPLIT, input_num, params->output_num, &node_id);
    node->nn_param.split.axis = input->dim_count - 1 - params->axis;
    node->nn_param.split.slices_num = split_count;
    node->nn_param.split.slices = slices;

    attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;

    /* input */
    node->input.tensors[0] = (vsi_nn_tensor_id_t)input->data;

    /* output */
    for (i = 0; i < split_count; i++) {
        attr.dtype.scale = output[i].scale;
        attr.dtype.zero_point = output[i].zero_point;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
        memset(attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        attr.vtl = TRUE;
        attr.is_const = FALSE;
        attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
        output_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL);
        node->output.tensors[i] = output_id;
        output[i].data = (void *)output_id;
        output[i].t_private = td;
    }
}
