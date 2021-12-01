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


#include "csi_ovx.h"
#include "vsi_nn_pub.h"

int csi_ovx_batch_to_space(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct batch_to_space_params *params)
{
    vsi_nn_node_t *node;
    vsi_nn_node_id_t node_id;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_id_t output_id;
    vsi_nn_graph_t *graph = csi_ovx_get_graph(input->sess);
    output->sess = input->sess;
    uint32_t input_num = 1;
    uint32_t output_num = 1;
    node = vsi_nn_AddNode(graph, VSI_NN_OP_BATCH2SPACE, input_num, output_num, &node_id);

    /* FIXME */
	node->nn_param.batch2space.block_size_num = 2;
    int32_t *block_size = (int32_t *)malloc(node->nn_param.batch2space.block_size_num * sizeof(int32_t));
    block_size[0] = params->block_size;
    block_size[1] = params->block_size;
    node->nn_param.batch2space.block_size = block_size;
    node->nn_param.batch2space.crop[0] = params->crop_left;
    node->nn_param.batch2space.crop[1] = params->crop_right;
    node->nn_param.batch2space.crop[2] = params->crop_top;
    node->nn_param.batch2space.crop[3] = params->crop_bottom;

    attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;

    /* input */
    node->input.tensors[0] = (vsi_nn_tensor_id_t)input->data;

    /* output */
    attr.dtype.scale = output->qinfo->scale;
    attr.dtype.zero_point = output->qinfo->zero_point;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    memset(attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = TRUE;
    attr.is_const = FALSE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    output_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL);
    node->output.tensors[0] = output_id;
    output->data = (void *)output_id;
}

