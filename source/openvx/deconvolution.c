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
#include "csi_utils.h"
#include "csi_ovx.h"

int csi_ovx_deconv2d(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct csi_tensor *kernel,
                     struct csi_tensor *bias,
                     struct conv2d_params *params)
{
    vsi_nn_node_t *node;
    vsi_nn_node_id_t node_id;
    vsi_nn_tensor_id_t input_id;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_id_t output_id;
    struct __target_data *td = input->t_private;
    output->t_private = td;
    vsi_nn_graph_t *graph = td->graph;
    uint32_t input_num = 3;
    uint32_t output_num = 1;

    node = vsi_nn_AddNode(graph, VSI_NN_OP_DECONVOLUTION, input_num, output_num, &node_id);
    node->nn_param.deconv.ksize[0] = kernel->dim[3];
    node->nn_param.deconv.ksize[1] = kernel->dim[2];
    node->nn_param.deconv.weights = output->dim[1];
    node->nn_param.deconv.stride[0] = params->stride_width;
    node->nn_param.deconv.stride[1] = params->stride_height;
    node->nn_param.deconv.pad[0] = params->pad_left;
    node->nn_param.deconv.pad[1] = params->pad_right;
    node->nn_param.deconv.pad[2] = params->pad_top;
    node->nn_param.deconv.pad[3] = params->pad_down;
    node->nn_param.deconv.group = 1;
    // node->nn_param.deconv.dilation[0] = dilation_width;
    // node->nn_param.deconv.dilation[1] = dilation_height;
    node->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    node->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;

    /* input */
    node->input.tensors[0] = (vsi_nn_tensor_id_t)input->data;

    /* kernel */
    attr.size[0] = kernel->dim[3];
    attr.size[1] = kernel->dim[2];
    attr.size[2] = kernel->dim[1];
    attr.size[3] = kernel->dim[0];
    attr.dim_num = 4;
    attr.dtype.scale = kernel->scale;
    attr.dtype.zero_point = kernel->zero_point;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    attr.vtl = FALSE;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    input_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, kernel->data);
    node->input.tensors[1] = input_id;

    /* bias */
    if (bias == NULL || bias->dim_count == 0) {
        node->input.tensors[2] = VSI_NN_TENSOR_ID_NA;
    } else {
        attr.size[0] = bias->dim[0];
        attr.dim_num = 1;
        attr.dtype.scale = bias->scale;
        attr.dtype.zero_point = bias->zero_point;
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        input_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, bias->data);
        node->input.tensors[2] = input_id;
    }

    /* output */
    attr.dtype.scale = output->scale;
    attr.dtype.zero_point = output->zero_point;
    memset(attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = TRUE;
    attr.is_const = FALSE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    output_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL);
    node->output.tensors[0] = output_id;
    output->data = (void *)output_id;
}

void csi_depthwise_deconv2d_f32(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct csi_tensor *kernel,
                      struct csi_tensor *bias,
                      int32_t stride_height,
                      int32_t stride_width,
                      int32_t pad_top,
                      int32_t pad_left,
                      int32_t pad_down,
                      int32_t pad_right,
                      int32_t dilation_height,
                      int32_t dilation_width)
{
    /* unsupport */
    assert(0);
}

void csi_depthwise_deconv2d_u8(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct csi_tensor *kernel,
                     struct csi_tensor *bias,
                     int32_t stride_height,
                     int32_t stride_width,
                     int32_t pad_top,
                     int32_t pad_left,
                     int32_t pad_down,
                     int32_t pad_right,
                     int32_t dilation_height,
                     int32_t dilation_width)
{
    vsi_nn_node_t *node;
    vsi_nn_node_id_t node_id;
    vsi_nn_tensor_id_t input_id;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_id_t output_id;
    struct __target_data *td = input->t_private;
    output->t_private = td;
    vsi_nn_graph_t *graph = td->graph;
    uint32_t input_num = 3;
    uint32_t output_num = 1;

    node = vsi_nn_AddNode(graph, VSI_NN_OP_DECONVOLUTION, input_num, output_num, &node_id);
    node->nn_param.deconv.ksize[0] = kernel->dim[3];
    node->nn_param.deconv.ksize[1] = kernel->dim[2];
    node->nn_param.deconv.weights = output->dim[1];
    node->nn_param.deconv.stride[0] = stride_width;
    node->nn_param.deconv.stride[1] = stride_height;
    node->nn_param.deconv.pad[0] = pad_left;
    node->nn_param.deconv.pad[1] = pad_right;
    node->nn_param.deconv.pad[2] = pad_top;
    node->nn_param.deconv.pad[3] = pad_down;
    node->nn_param.deconv.group = output->dim[1];
    // node->nn_param.deconv.dilation[0] = dilation_width;
    // node->nn_param.deconv.dilation[1] = dilation_height;
    node->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    node->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;

    /* input */
    node->input.tensors[0] = (vsi_nn_tensor_id_t)input->data;

    /* kernel */
    attr.size[0] = kernel->dim[3];
    attr.size[1] = kernel->dim[2];
    attr.size[2] = kernel->dim[1];
    attr.size[3] = kernel->dim[0];
    attr.dim_num = 4;
    attr.dtype.scale = kernel->scale;
    attr.dtype.zero_point = kernel->zero_point;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    attr.vtl = FALSE;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    input_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, kernel->data);
    node->input.tensors[1] = input_id;

    /* bias */
    if (bias == NULL || bias->dim_count == 0) {
        node->input.tensors[2] = VSI_NN_TENSOR_ID_NA;
    } else {
        attr.size[0] = bias->dim[0];
        attr.dim_num = 1;
        attr.dtype.scale = bias->scale;
        attr.dtype.zero_point = bias->zero_point;
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        input_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, bias->data);
        node->input.tensors[2] = input_id;
    }

    /* output */
    attr.dtype.scale = output->scale;
    attr.dtype.zero_point = output->zero_point;
    memset(attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = TRUE;
    attr.is_const = FALSE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    output_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL);
    node->output.tensors[0] = output_id;
    output->data = (void *)output_id;
}