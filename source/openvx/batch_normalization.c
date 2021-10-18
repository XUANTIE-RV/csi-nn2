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

int csi_ovx_batch_normalization(struct csi_tensor *input,
                                struct csi_tensor *mean,
                                struct csi_tensor *variance,
                                struct csi_tensor *gamma,
                                struct csi_tensor *beta,
                                struct csi_tensor *output,
                                struct bn_params *params)
{
    vsi_nn_node_t *node;
    vsi_nn_node_id_t node_id;
    vsi_nn_tensor_id_t input_id;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_id_t output_id;
    struct __target_data *td = input->t_private;
    output->t_private = td;
    vsi_nn_graph_t *graph = td->graph;
    uint32_t input_num = 5;
    uint32_t output_num = 1;
    node = vsi_nn_AddNode(graph, VSI_NN_OP_BATCH_NORM, input_num, output_num, &node_id);

    node->nn_param.batch_norm.eps = params->epsilon;
    attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;

    /* input */
    node->input.tensors[0] = (vsi_nn_tensor_id_t)input->data;

    /* mean */
    attr.size[0] = mean->dim[0];
    attr.dim_num = 1;
    attr.dtype.scale = mean->scale;
    attr.dtype.zero_point = mean->zero_point;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    attr.vtl = FALSE;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    input_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, mean->data);
    node->input.tensors[1] = input_id;

    /* variance */
    attr.size[0] = variance->dim[0];
    attr.dim_num = 1;
    attr.dtype.scale = variance->scale;
    attr.dtype.zero_point = variance->zero_point;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    attr.vtl = FALSE;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    input_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, variance->data);
    node->input.tensors[1] = input_id;

    /* gamma */
    attr.size[0] = gamma->dim[0];
    attr.dim_num = 1;
    attr.dtype.scale = gamma->scale;
    attr.dtype.zero_point = gamma->zero_point;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    attr.vtl = FALSE;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    input_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, gamma->data);
    node->input.tensors[1] = input_id;

    /* beta */
    attr.size[0] = beta->dim[0];
    attr.dim_num = 1;
    attr.dtype.scale = beta->scale;
    attr.dtype.zero_point = beta->zero_point;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    attr.vtl = FALSE;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    input_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, beta->data);
    node->input.tensors[1] = input_id;

    /* output */
    attr.dtype.scale = output->scale;
    attr.dtype.zero_point = output->zero_point;
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
