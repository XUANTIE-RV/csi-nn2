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

#include "csi_nn.h"
#include "csi_ovx.h"
#include "csi_utils.h"

int csi_ovx_proposal(struct csi_tensor *cls_prob,
                     struct csi_tensor *bbox_pred,
                     struct csi_tensor *im_info,
                     struct csi_tensor *output,
                     struct proposal_params *params)
{
    vsi_nn_node_t *node;
    vsi_nn_node_id_t node_id;
    vsi_nn_tensor_id_t input_id;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_id_t output_id;
    struct __target_data *td = cls_prob->t_private;
    output->t_private = td;
    vsi_nn_graph_t *graph = td->graph;
    uint32_t input_num = 4;
    uint32_t output_num = 2;
    node = vsi_nn_AddNode(graph, VSI_NN_OP_PROPOSAL, input_num, output_num,
                          &node_id);

    node->nn_param.proposal.anchor.ratio = params->ratios;
    node->nn_param.proposal.anchor.ratio_num = params->ratios_num;
    node->nn_param.proposal.anchor.scale = params->scales;
    node->nn_param.proposal.anchor.scale_num = params->scales_num;
    node->nn_param.proposal.anchor.base_size = 16;
    float *im_info_data = im_info->data;
    node->nn_param.proposal.im_info.size[0] = im_info_data[1];
    node->nn_param.proposal.im_info.size[1] = im_info_data[0];
    node->nn_param.proposal.im_info.scale[0] = im_info_data[2];
    node->nn_param.proposal.im_info.scale[1] = im_info_data[2];
    node->nn_param.proposal.feat_stride = (uint32_t)params->feature_stride;
    node->nn_param.proposal.pre_nms_topn = (uint32_t)params->rpn_pre_nms_top_n;
    node->nn_param.proposal.post_nms_topn = (uint32_t)params->rpn_post_nms_top_n;
    node->nn_param.proposal.min_size = (uint32_t)params->rpn_min_size;
    node->nn_param.proposal.nms_thresh = params->threshold;

    attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;

    /* input */
    node->input.tensors[0] = (vsi_nn_tensor_id_t)cls_prob->data;
    node->input.tensors[1] = (vsi_nn_tensor_id_t)bbox_pred->data;

    /* output0 */
    attr.dtype.scale = output->scale;
    attr.dtype.zero_point = output->zero_point;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    memset(attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = FALSE;
    attr.is_const = FALSE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    output_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL);
    node->output.tensors[0] = output_id;
    output[0].data = (void *)output_id;

    /* output1 */
    attr.dtype.scale = 1;
    attr.dtype.zero_point = 0;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    memset(attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = TRUE;
    attr.is_const = FALSE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    output_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL);
    node->output.tensors[1] = output_id;
    output[1].data = (void *)output_id;
}
