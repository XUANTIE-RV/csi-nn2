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
#include "csi_internal_c906.h"


void* csi_bc_map_table_c906[CSINN_OP_SIZE][2] = {
    {csi_abs_u8_c906, csi_abs_f32_c906}, /* CSINN_OP_ABS */
    {NULL, NULL}, /* CSINN_OP_ACOS */
    {NULL, NULL}, /* CSINN_OP_ACOSH */
    {csi_add_u8_c906, csi_add_f32_c906}, /* CSINN_OP_ADD */
    {NULL, NULL}, /* CSINN_OP_ALL */
    {NULL, NULL}, /* CSINN_OP_AND */
    {NULL, NULL}, /* CSINN_OP_ANY */
    {NULL, NULL}, /* CSINN_OP_ARANGE */
    {NULL, NULL}, /* CSINN_OP_ARGMAX */
    {NULL, NULL}, /* CSINN_OP_ARGMIN */
    {NULL, NULL}, /* CSINN_OP_ASIN */
    {NULL, NULL}, /* CSINN_OP_ASINH */
    {NULL, NULL}, /* CSINN_OP_ATAN */
    {NULL, NULL}, /* CSINN_OP_ATANH */
    {NULL, NULL}, /* CSINN_OP_AVGPOOL2D */
    {NULL, NULL}, /* CSINN_OP_AVGPOOL3D */
    {NULL, NULL}, /* CSINN_OP_BN */
    {NULL, NULL}, /* CSINN_OP_BATCH_TO_SPACE */
    {csi_broadcast_to_u8_c906, csi_broadcast_to_f32_c906}, /* CSINN_OP_BROADCOST */
    {NULL, NULL}, /* CSINN_OP_CEIL */
    {csi_clip_u8_c906, csi_clip_f32_c906}, /* CSINN_OP_CLIP */
    {NULL, NULL}, /* CSINN_OP_COL2IM */
    {NULL, NULL}, /* CSINN_OP_CONCAT */
    {NULL, NULL}, /* CSINN_OP_CONV2D */
    {NULL, NULL}, /* CSINN_OP_CONV2D_RELU */
    {NULL, NULL}, /* CSINN_OP_CONV2D_RELU6 */
    {NULL, NULL}, /* CSINN_OP_CONV2D_CHANNEL */
    {NULL, NULL}, /* CSINN_OP_CONV2D_CHANNEL_RELU */
    {NULL, NULL}, /* CSINN_OP_CONV2D_CHANNEL_RELU6 */
    {NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D */
    {NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_RELU */
    {NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_RELU6 */
    {NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL */
    {NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU */
    {NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6 */
    {NULL, NULL}, /* CSINN_OP_GROUP_CONV2D */
    {NULL, NULL}, /* CSINN_OP_GROUP_CONV2D_RELU */
    {NULL, NULL}, /* CSINN_OP_GROUP_CONV2D_CHANNEL */
    {NULL, NULL}, /* CSINN_OP_GROUP_CONV2D_CHANNEL_RELU */
    {NULL, NULL}, /* CSINN_OP_CONV3D */
    {NULL, NULL}, /* CSINN_OP_COS */
    {NULL, NULL}, /* CSINN_OP_COSH */
    {NULL, NULL}, /* CSINN_OP_CUMPROD */
    {NULL, NULL}, /* CSINN_OP_CUMSUM */
    {NULL, NULL}, /* CSINN_OP_DECONV2D */
    {NULL, NULL}, /* CSINN_OP_DEPTHWISE_DECONV2D */
    {NULL, NULL}, /* CSINN_OP_DECONV3D */
    {NULL, NULL}, /* CSINN_OP_DEPTH_TO_SPACE */
    {NULL, NULL}, /* CSINN_OP_DIV */
    {NULL, NULL}, /* CSINN_OP_ELU */
    {NULL, NULL}, /* CSINN_OP_EQUANL */
    {NULL, NULL}, /* CSINN_OP_ERF */
    {NULL, NULL}, /* CSINN_OP_EXP */
    {NULL, NULL}, /* CSINN_OP_EXPAND_DIMS */
    {NULL, NULL}, /* CSINN_OP_EXPM1 */
    {NULL, NULL}, /* CSINN_OP_FLATTEN */
    {NULL, NULL}, /* CSINN_OP_FLOOR_DIVIDE */
    {NULL, NULL}, /* CSINN_OP_FLOOR_MOD */
    {NULL, NULL}, /* CSINN_OP_FLOOR */
    {csi_fullyconnected_u8_c906, csi_fullyconnected_f32_c906}, /* CSINN_OP_FULLYCONNECTED */
    {NULL, NULL}, /* CSINN_OP_GATHER_ND */
    {NULL, NULL}, /* CSINN_OP_GATHER */
    {NULL, NULL}, /* CSINN_OP_GLOBAL_AVGPOOL2D */
    {NULL, NULL}, /* CSINN_OP_GLOBAL_MAXPOOL2D */
    {NULL, NULL}, /* CSINN_OP_GREATHER_EQUAL */
    {NULL, NULL}, /* CSINN_OP_GREATHER */
    {NULL, NULL}, /* CSINN_OP_HARD_SIGMOID */
    {NULL, NULL}, /* CSINN_OP_IM2COL */
    {NULL, NULL}, /* CSINN_OP_ISNAN */
    {NULL, NULL}, /* CSINN_OP_L2N */
    {NULL, NULL}, /* CSINN_OP_L2POOL2D */
    {csi_leaky_relu_u8_c906, csi_leaky_relu_f32_c906}, /* CSINN_OP_LEAKY_RELU */
    {NULL, NULL}, /* CSINN_OP_LESS_EQUAL */
    {NULL, NULL}, /* CSINN_OP_LESS */
    {NULL, NULL}, /* CSINN_OP_LOG_SOFTMAX */
    {NULL, NULL}, /* CSINN_OP_LOG */
    {NULL, NULL}, /* CSINN_OP_LOG1P */
    {NULL, NULL}, /* CSINN_OP_LOGICAL_AND */
    {NULL, NULL}, /* CSINN_OP_LOGICAL_NOT */
    {NULL, NULL}, /* CSINN_OP_LOGICAL_OR */
    {NULL, NULL}, /* CSINN_OP_LOGICAL_XOR */
    {NULL, NULL}, /* CSINN_OP_LRN */
    {NULL, NULL}, /* CSINN_OP_MATMUL */
    {NULL, NULL}, /* CSINN_OP_MAX */
    {NULL, NULL}, /* CSINN_OP_MAXINUM */
    {NULL, NULL}, /* CSINN_OP_MAXPOOL2D */
    {NULL, NULL}, /* CSINN_OP_MAXPOOL2D_LOCAT */
    {NULL, NULL}, /* CSINN_OP_MAXPOOL3D */
    {NULL, NULL}, /* CSINN_OP_MEAN */
    {NULL, NULL}, /* CSINN_OP_MEAN_STRIDE */
    {NULL, NULL}, /* CSINN_OP_MIN */
    {NULL, NULL}, /* CSINN_OP_MIN_STRIDE */
    {NULL, NULL}, /* CSINN_OP_MINIMUM */
    {NULL, NULL}, /* CSINN_OP_MOD */
    {NULL, NULL}, /* CSINN_OP_MUL */
    {NULL, NULL}, /* CSINN_OP_NDARRAY_SIZE */
    {NULL, NULL}, /* CSINN_OP_NEGATIIVE */
    {NULL, NULL}, /* CSINN_OP_NON_MAX_SUPPRESSION */
    {NULL, NULL}, /* CSINN_OP_NOT_EQUAL */
    {NULL, NULL}, /* CSINN_OP_NOT */
    {NULL, NULL}, /* CSINN_OP_ONE_HOT */
    {NULL, NULL}, /* CSINN_OP_OR */
    {NULL, NULL}, /* CSINN_OP_PAD */
    {NULL, NULL}, /* CSINN_OP_POWER */
    {csi_prelu_u8_c906, csi_prelu_f32_c906}, /* CSINN_OP_PRELU */
    {NULL, NULL}, /* CSINN_OP_PROD */
    {NULL, NULL}, /* CSINN_OP_PROPOSAL */
    {NULL, NULL}, /* CSINN_OP_PSROIPOOLING */
    {NULL, NULL}, /* CSINN_OP_REDUCE_LOGSUMEXP */
    {NULL, NULL}, /* CSINN_OP_REDUCE_MAX */
    {NULL, NULL}, /* CSINN_OP_REDUCE_MEAN */
    {NULL, NULL}, /* CSINN_OP_REDUCE_MIN */
    {NULL, NULL}, /* CSINN_OP_REDUCE_PROD */
    {NULL, NULL}, /* CSINN_OP_REDUCE_SUM */
    {csi_relu_u8_c906, csi_relu_f32_c906}, /* CSINN_OP_RELU */
    {csi_relu1_u8_c906, csi_relu1_f32_c906}, /* CSINN_OP_RELU1 */
    {csi_relu6_u8_c906, csi_relu6_f32_c906}, /* CSINN_OP_RELU6 */
    {NULL, NULL}, /* CSINN_OP_RELUN */
    {NULL, NULL}, /* CSINN_OP_REORG */
    {NULL, NULL}, /* CSINN_OP_RESHAPE */
    {NULL, NULL}, /* CSINN_OP_RESIZE */
    {NULL, NULL}, /* CSINN_OP_REVERSE */
    {NULL, NULL}, /* CSINN_OP_ROIALIGN */
    {NULL, NULL}, /* CSINN_OP_ROIPOOL */
    {NULL, NULL}, /* CSINN_OP_ROUND */
    {NULL, NULL}, /* CSINN_OP_RSQRT */
    {NULL, NULL}, /* CSINN_OP_SEGMENT_MAX */
    {NULL, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MAX */
    {NULL, NULL}, /* CSINN_OP_SEGMENT_MEAN */
    {NULL, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MEAN */
    {NULL, NULL}, /* CSINN_OP_SEGMENT_MIN */
    {NULL, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MIN */
    {NULL, NULL}, /* CSINN_OP_SEGMENT_PROD */
    {NULL, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_PROD */
    {NULL, NULL}, /* CSINN_OP_SEGMENT_SUM */
    {NULL, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_SUM */
    {NULL, NULL}, /* CSINN_OP_SELECT */
    {NULL, NULL}, /* CSINN_OP_SEQUENCE_MASK */
    {NULL, NULL}, /* CSINN_OP_SHAPE */
    {NULL, NULL}, /* CSINN_OP_SHUFFLE_CHANNEL */
    {NULL, NULL}, /* CSINN_OP_SIGMOID */
    {NULL, NULL}, /* CSINN_OP_SIGN */
    {NULL, NULL}, /* CSINN_OP_SIN */
    {NULL, NULL}, /* CSINN_OP_SINH */
    {NULL, NULL}, /* CSINN_OP_SLICE */
    {NULL, NULL}, /* CSINN_OP_SOFTMAX */
    {NULL, NULL}, /* CSINN_OP_SOFTPLUS */
    {NULL, NULL}, /* CSINN_OP_SOFTRELU */
    {NULL, NULL}, /* CSINN_OP_SOFTSIGN */
    {NULL, NULL}, /* CSINN_OP_SPACE_TO_BATCH */
    {NULL, NULL}, /* CSINN_OP_SPACE_TO_DEPTH */
    {NULL, NULL}, /* CSINN_OP_SPLIT */
    {NULL, NULL}, /* CSINN_OP_SQRT */
    {NULL, NULL}, /* CSINN_OP_SQUARE */
    {NULL, NULL}, /* CSINN_OP_SQUEEZE */
    {NULL, NULL}, /* CSINN_OP_STACK */
    {NULL, NULL}, /* CSINN_OP_STRIDED_SLICE */
    {NULL, NULL}, /* CSINN_OP_SUB */
    {NULL, NULL}, /* CSINN_OP_SUM */
    {NULL, NULL}, /* CSINN_OP_TAN */
    {NULL, NULL}, /* CSINN_OP_TANH */
    {NULL, NULL}, /* CSINN_OP_THRESHOLD_RELU */
    {NULL, NULL}, /* CSINN_OP_TILE */
    {NULL, NULL}, /* CSINN_OP_TOPK */
    {NULL, NULL}, /* CSINN_OP_TRANSPOSE */
    {NULL, NULL}, /* CSINN_OP_TRUNC */
    {NULL, NULL}, /* CSINN_OP_UNPOOLING */
    {NULL, NULL}, /* CSINN_OP_UNSTACK */
    {NULL, NULL}, /* CSINN_OP_WHERE */
    {NULL, NULL}, /* CSINN_OP_XOR */
    {NULL, NULL}, /* CSINN_OP_YUV_RGB_SCALE */
};

void *csi_bc_map_c906(int op, int dtype)
{
    int dt;
    switch (dtype) {
        case CSINN_DTYPE_UINT8:
            dt = 0;
            break;
        case CSINN_DTYPE_FLOAT32:
            dt = 1;
            break;
        default:
            return NULL;
    }
    return csi_bc_map_table_c906[op][dt];
}