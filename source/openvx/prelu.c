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


int csi_ovx_prelu(struct csi_tensor *input,
                  struct csi_tensor *alpha,
                  struct csi_tensor *output,
                  struct prelu_params *params)
{
    vsi_nn_node_t *node;
    vsi_nn_node_id_t node_id;
    vsi_nn_tensor_id_t input_id;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_id_t output_id;
    vsi_nn_graph_t *graph = csi_ovx_get_graph(input->sess);
    output->sess = input->sess;
    uint32_t input_num = 2;
    uint32_t output_num = 1;
    node = vsi_nn_AddNode(graph, VSI_NN_OP_PRELU, input_num, output_num, &node_id);
    node->nn_param.prelu.axis = input->dim_count - 1 - params->axis;
    assert(node->nn_param.prelu.axis == 2);
    assert(input->dim_count == 4);

    attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;

    /* input */
    node->input.tensors[0] = (vsi_nn_tensor_id_t)input->data;

    /* alpha */
    attr.dim_num = input->dim_count;
    memset(attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.size[0] = 1;
    attr.size[1] = 1;
    attr.size[2] = alpha->dim[0];
    attr.size[3] = 1;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.vtl = FALSE;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;

    float alpha_f;
    uint16_t *alpha_u16 = (uint16_t*)malloc(sizeof(uint16_t)*alpha->dim[0]);
    for (int i = 0; i < alpha->dim[0]; i++) {
        alpha_f = (((uint8_t*)(alpha->data))[i] - alpha->zero_point) * alpha->scale;
        alpha_u16[i] = vsi_nn_Fp32ToFp16(alpha_f);
    }
    uint8_t *alpha_u8 = (uint8_t*)malloc(sizeof(uint16_t)*alpha->dim[0]);
    memcpy(alpha_u8, alpha_u16, (sizeof(uint16_t)*alpha->dim[0]));

    input_id = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, (uint8_t *)alpha_u16);
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
