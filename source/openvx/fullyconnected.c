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


int csi_ovx_fullyconnected(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct csi_tensor *weights,
                           struct csi_tensor *bias,
                           struct fc_params *params)
{
    vsi_nn_node_t *node;
    vsi_nn_node_id_t node_id;
    vsi_nn_tensor_id_t input_id;
    vsi_nn_tensor_id_t output_id;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_graph_t *graph = csi_ovx_get_graph(input->sess);
    output->sess = input->sess;
    uint32_t input_num = 3;
    uint32_t output_num = 1;
    node = vsi_nn_AddNode(graph, VSI_NN_OP_FCL, input_num, output_num, &node_id);
    node->nn_param.fcl.weights = weights->dim[0];
    node->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    node->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;

    /* input */
    node->input.tensors[0] = (vsi_nn_tensor_id_t)input->data;

    /* weights */
    attr.size[0] = weights->dim[1];
    attr.size[1] = weights->dim[0];
    attr.dim_num = 2;
    attr.dtype.scale = weights->qinfo->scale;
    attr.dtype.zero_point = weights->qinfo->zero_point;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    attr.vtl = FALSE;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    input_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, weights->data);
    node->input.tensors[1] = input_id;

    /* bias */
    if (bias->dim_count == 0) {
        bias->dim[0] = weights->dim[0];
        int32_t *bias_data = (int32_t*)malloc(sizeof(int32_t) * bias->dim[0]);
        memset(bias_data, 0, sizeof(int32_t) * bias->dim[0]);
    }
    attr.size[0] = bias->dim[0];
    attr.dim_num = 1;
    attr.dtype.scale = bias->qinfo->scale;
    attr.dtype.zero_point = bias->qinfo->zero_point;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    input_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, bias->data);
    node->input.tensors[2] = input_id;

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

int csi_ovx_fullyconnected_relu(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *weights,
                                struct csi_tensor *bias,
                                struct fc_params *params)
{
    vsi_nn_node_t *node;
    vsi_nn_node_id_t node_id;
    vsi_nn_tensor_id_t input_id;
    vsi_nn_tensor_id_t output_id;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_graph_t *graph = csi_ovx_get_graph(input->sess);
    output->sess = input->sess;
    uint32_t input_num = 3;
    uint32_t output_num = 1;
    node = vsi_nn_AddNode(graph, VSI_NN_OP_FCL, input_num, output_num, &node_id);
    node->nn_param.fcl.weights = weights->dim[0];
    node->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    node->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;

    /* input */
    node->input.tensors[0] = (vsi_nn_tensor_id_t)input->data;

    /* weights */
    attr.size[0] = weights->dim[1];
    attr.size[1] = weights->dim[0];
    attr.dim_num = 2;
    attr.dtype.scale = weights->qinfo->scale;
    attr.dtype.zero_point = weights->qinfo->zero_point;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    attr.vtl = FALSE;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    input_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, weights->data);
    node->input.tensors[1] = input_id;

    /* bias */
    attr.size[0] = bias->dim[0];
    attr.dim_num = 1;
    attr.dtype.scale = bias->qinfo->scale;
    attr.dtype.zero_point = bias->qinfo->zero_point;
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    input_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, bias->data);
    node->input.tensors[2] = input_id;

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
