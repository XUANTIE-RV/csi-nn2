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

#include "csi_dp1k.h"

void csi_dp1k_set_tensor(struct csi_tensor *tensor, struct csi_session *sess) {
    csi_dp1k_input(tensor, sess);
    tensor->sess = sess;
}

void csi_dp1k_session_init(struct csi_session *sess) {
    sess->base_dtype = CSINN_DTYPE_FLOAT32; // support float only currently.
    sess->base_layout = CSINN_NCHW;
    csi_dp1000_session_init(sess);
}

void csi_dp1k_session_deinit(struct csi_session *sess) {
    free(sess->td);
    sess->td = NULL;
}

void csi_dp1k_session_setup(struct csi_session *sess) {
    csi_dp1000_session_setup(sess);
}

void csi_dp1k_set_input_number(int number, struct csi_session *sess) {
    sess->input_num = number;
    sess->input = calloc(1, sizeof(struct csi_tensor *) * number);
    csi_dp1000_set_input_number(number, sess);
}

void csi_dp1k_set_output_number(int number, struct csi_session *sess) {
    sess->output_num = number;
    sess->output = calloc(1, sizeof(struct csi_tensor *) * number);
    csi_dp1000_set_output_number(number, sess);
}

int csi_dp1k_get_input_number(struct csi_session *sess) {
    return sess->input_num;
}

int csi_dp1k_get_output_number(struct csi_session *sess) {
    return sess->output_num;
}

void csi_dp1k_set_input(int index, struct csi_tensor *input, struct csi_session *sess) {
    sess->input[index] = input;
    csi_dp1000_set_input(index, input, sess);
}

void csi_dp1k_set_output(int index, struct csi_tensor *output, struct csi_session *sess) {
    sess->output[index] = output;
    csi_dp1000_set_output(index, output, sess);
}

void* csi_bc_map_table_dp1k[CSINN_OP_AND_UTILS_SIZE][1] = {
    {NULL},                         /* CSINN_OP_ABS */
    {NULL},                         /* CSINN_OP_ACOS */
    {NULL},                         /* CSINN_OP_ACOSH */
    {csi_dp1k_add},                 /* CSINN_OP_ADD */
    {NULL},                         /* CSINN_OP_ALL */
    {NULL},                         /* CSINN_OP_AND */
    {NULL},                         /* CSINN_OP_ANY */
    {NULL},                         /* CSINN_OP_ARANGE */
    {NULL},                         /* CSINN_OP_ARGMAX */
    {NULL},                         /* CSINN_OP_ARGMIN */
    {NULL},                         /* CSINN_OP_ASIN */
    {NULL},                         /* CSINN_OP_ASINH */
    {NULL},                         /* CSINN_OP_ATAN */
    {NULL},                         /* CSINN_OP_ATANH */
    {csi_dp1k_avgpool2d},           /* CSINN_OP_AVGPOOL2D */
    {NULL},                         /* CSINN_OP_AVGPOOL3D */
    {NULL},                         /* CSINN_OP_BN */
    {NULL},                         /* CSINN_OP_BATCH_TO_SPACE */
    {NULL},                         /* CSINN_OP_BATCH_TO_SPACE_ND */
    {NULL},                         /* CSINN_OP_BROADCOST */
    {NULL},                         /* CSINN_OP_CEIL */
    {NULL},                         /* CSINN_OP_CLIP */
    {NULL},                         /* CSINN_OP_COL2IM */
    {csi_dp1k_concat},              /* CSINN_OP_CONCAT */
    {csi_dp1k_conv2d},              /* CSINN_OP_CONV2D */
    {NULL},                         /* CSINN_OP_CONV2D_RELU */
    {NULL},                         /* CSINN_OP_CONV2D_RELU6 */
    {NULL},                         /* CSINN_OP_CONV2D_CHANNEL */
    {NULL},                         /* CSINN_OP_CONV2D_CHANNEL_RELU */
    {NULL},                         /* CSINN_OP_CONV2D_CHANNEL_RELU6 */
    {csi_dp1k_conv2d},              /* CSINN_OP_DEPTHWISE_CONV2D */
    {NULL},                         /* CSINN_OP_DEPTHWISE_CONV2D_RELU */
    {NULL},                         /* CSINN_OP_DEPTHWISE_CONV2D_RELU6 */
    {NULL},                         /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL */
    {NULL},                         /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU */
    {NULL},                         /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6 */
    {csi_dp1k_conv2d},              /* CSINN_OP_GROUP_CONV2D */
    {NULL},                         /* CSINN_OP_GROUP_CONV2D_RELU */
    {NULL},                         /* CSINN_OP_GROUP_CONV2D_RELU6 */
    {NULL},                         /* CSINN_OP_GROUP_CONV2D_CHANNEL */
    {NULL},                         /* CSINN_OP_GROUP_CONV2D_CHANNEL_RELU */
    {NULL},                         /* CSINN_OP_CONV3D */
    {NULL},                         /* CSINN_OP_COS */
    {NULL},                         /* CSINN_OP_COSH */
    {NULL},                         /* CSINN_OP_CROP */
    {NULL},                         /* CSINN_OP_CUMPROD */
    {NULL},                         /* CSINN_OP_CUMSUM */
    {csi_dp1k_deconv2d},            /* CSINN_OP_DECONV2D */
    {csi_dp1k_deconv2d},            /* CSINN_OP_DEPTHWISE_DECONV2D */
    {NULL},                         /* CSINN_OP_DECONV3D */
    {NULL},                         /* CSINN_OP_DEPTH_TO_SPACE */
    {NULL},                         /* CSINN_OP_DIV */
    {NULL},                         /* CSINN_OP_ELU */
    {NULL},                         /* CSINN_OP_EQUANL */
    {NULL},                         /* CSINN_OP_ERF */
    {NULL},                         /* CSINN_OP_EXP */
    {NULL},                         /* CSINN_OP_EXPAND_DIMS */
    {NULL},                         /* CSINN_OP_EXPM1 */
    {NULL},                         /* CSINN_OP_FLATTEN */
    {NULL},                         /* CSINN_OP_FLOOR_DIVIDE */
    {NULL},                         /* CSINN_OP_FLOOR_MOD */
    {NULL},                         /* CSINN_OP_FLOOR */
    {csi_dp1k_fullyconnected},      /* CSINN_OP_FULLYCONNECTED */
    {NULL},                         /* CSINN_OP_GATHER_ND */
    {NULL},                         /* CSINN_OP_GATHER */
    {NULL},                         /* CSINN_OP_GLOBAL_AVGPOOL2D */
    {NULL},                         /* CSINN_OP_GLOBAL_MAXPOOL2D */
    {NULL},                         /* CSINN_OP_GREATHER_EQUAL */
    {NULL},                         /* CSINN_OP_GREATHER */
    {NULL},                         /* CSINN_OP_HARD_SIGMOID */
    {NULL},                         /* CSINN_OP_IM2COL */
    {NULL},                         /* CSINN_OP_ISNAN */
    {NULL},                         /* CSINN_OP_L2N */
    {NULL},                         /* CSINN_OP_L2POOL2D */
    {csi_dp1k_leaky_relu},          /* CSINN_OP_LEAKY_RELU */
    {NULL},                         /* CSINN_OP_LESS_EQUAL */
    {NULL},                         /* CSINN_OP_LESS */
    {NULL},                         /* CSINN_OP_LOG_SOFTMAX */
    {NULL},                         /* CSINN_OP_LOG */
    {NULL},                         /* CSINN_OP_LOG1P */
    {NULL},                         /* CSINN_OP_LOGICAL_AND */
    {NULL},                         /* CSINN_OP_LOGICAL_NOT */
    {NULL},                         /* CSINN_OP_LOGICAL_OR */
    {NULL},                         /* CSINN_OP_LOGICAL_XOR */
    {NULL},                         /* CSINN_OP_LRN */
    {NULL},                         /* CSINN_OP_MATMUL */
    {NULL},                         /* CSINN_OP_MAX */
    {NULL},                         /* CSINN_OP_MAXINUM */
    {csi_dp1k_maxpool},             /* CSINN_OP_MAXPOOL2D */
    {NULL},                         /* CSINN_OP_MAXPOOL2D_LOCAT */
    {NULL},                         /* CSINN_OP_MAXPOOL3D */
    {NULL},                         /* CSINN_OP_MEAN */
    {NULL},                         /* CSINN_OP_MEAN_STRIDE */
    {NULL},                         /* CSINN_OP_MIN */
    {NULL},                         /* CSINN_OP_MIN_STRIDE */
    {NULL},                         /* CSINN_OP_MINIMUM */
    {NULL},                         /* CSINN_OP_MOD */
    {csi_dp1k_mul},                 /* CSINN_OP_MUL */
    {NULL},                         /* CSINN_OP_NDARRAY_SIZE */
    {NULL},                         /* CSINN_OP_NEGATIIVE */
    {NULL},                         /* CSINN_OP_NON_MAX_SUPPRESSION */
    {NULL},                         /* CSINN_OP_NOT_EQUAL */
    {NULL},                         /* CSINN_OP_NOT */
    {NULL},                         /* CSINN_OP_ONE_HOT */
    {NULL},                         /* CSINN_OP_OR */
    {NULL},                         /* CSINN_OP_PAD */
    {NULL},                         /* CSINN_OP_POWER */
    {csi_dp1k_prelu},               /* CSINN_OP_PRELU */
    {NULL},                         /* CSINN_OP_PROD */
    {NULL},                         /* CSINN_OP_PROPOSAL */
    {NULL},                         /* CSINN_OP_PSROIPOOLING */
    {NULL},                         /* CSINN_OP_REDUCE_LOGSUMEXP */
    {NULL},                         /* CSINN_OP_REDUCE_MAX */
    {NULL},                         /* CSINN_OP_REDUCE_MEAN */
    {NULL},                         /* CSINN_OP_REDUCE_MIN */
    {NULL},                         /* CSINN_OP_REDUCE_PROD */
    {NULL},                         /* CSINN_OP_REDUCE_SUM */
    {csi_dp1k_relu},                /* CSINN_OP_RELU */
    {NULL},                         /* CSINN_OP_RELU1 */
    {NULL},                         /* CSINN_OP_RELU6 */
    {NULL},                         /* CSINN_OP_RELUN */
    {NULL},                         /* CSINN_OP_REORG */
    {csi_dp1k_reshape},             /* CSINN_OP_RESHAPE */
    {csi_dp1k_resize},              /* CSINN_OP_RESIZE */
    {NULL},                         /* CSINN_OP_REVERSE */
    {NULL},                         /* CSINN_OP_ROIALIGN */
    {NULL},                         /* CSINN_OP_ROIPOOL */
    {NULL},                         /* CSINN_OP_ROUND */
    {NULL},                         /* CSINN_OP_RSQRT */
    {NULL},                         /* CSINN_OP_SCATTER_ND */
    {NULL},                         /* CSINN_OP_SEGMENT_MAX */
    {NULL},                         /* CSINN_OP_UNSORTED_SEGMENT_MAX */
    {NULL},                         /* CSINN_OP_SEGMENT_MEAN */
    {NULL},                         /* CSINN_OP_UNSORTED_SEGMENT_MEAN */
    {NULL},                         /* CSINN_OP_SEGMENT_MIN */
    {NULL},                         /* CSINN_OP_UNSORTED_SEGMENT_MIN */
    {NULL},                         /* CSINN_OP_SEGMENT_PROD */
    {NULL},                         /* CSINN_OP_UNSORTED_SEGMENT_PROD */
    {NULL},                         /* CSINN_OP_SEGMENT_SUM */
    {NULL},                         /* CSINN_OP_UNSORTED_SEGMENT_SUM */
    {NULL},                         /* CSINN_OP_SELECT */
    {NULL},                         /* CSINN_OP_SEQUENCE_MASK */
    {NULL},                         /* CSINN_OP_SHAPE */
    {NULL},                         /* CSINN_OP_SHUFFLE_CHANNEL */
    {csi_dp1k_sigmoid},             /* CSINN_OP_SIGMOID */
    {NULL},                         /* CSINN_OP_SIGN */
    {NULL},                         /* CSINN_OP_SIN */
    {NULL},                         /* CSINN_OP_SINH */
    {NULL},                         /* CSINN_OP_SLICE */
    {csi_dp1k_softmax},             /* CSINN_OP_SOFTMAX */
    {NULL},                         /* CSINN_OP_SOFTPLUS */
    {NULL},                         /* CSINN_OP_SOFTRELU */
    {NULL},                         /* CSINN_OP_SOFTSIGN */
    {NULL},                         /* CSINN_OP_SPACE_TO_BATCH */
    {NULL},                         /* CSINN_OP_SPACE_TO_BATCH_ND */
    {NULL},                         /* CSINN_OP_SPACE_TO_DEPTH */
    {NULL},                         /* CSINN_OP_SPLIT */
    {NULL},                         /* CSINN_OP_SQRT */
    {NULL},                         /* CSINN_OP_SQUARE */
    {NULL},                         /* CSINN_OP_SQUEEZE */
    {NULL},                         /* CSINN_OP_STACK */
    {NULL},                         /* CSINN_OP_STRIDED_SLICE */
    {NULL},                         /* CSINN_OP_SUB */
    {NULL},                         /* CSINN_OP_SUM */
    {NULL},                         /* CSINN_OP_TAN */
    {NULL},                         /* CSINN_OP_TANH */
    {NULL},                         /* CSINN_OP_THRESHOLD_RELU */
    {NULL},                         /* CSINN_OP_TILE */
    {NULL},                         /* CSINN_OP_TOPK */
    {csi_dp1k_transpose},           /* CSINN_OP_TRANSPOSE */
    {NULL},                         /* CSINN_OP_TRUNC */
    {NULL},                         /* CSINN_OP_UNPOOLING */
    {NULL},                         /* CSINN_OP_UNSTACK */
    {NULL},                         /* CSINN_OP_WHERE */
    {NULL},                         /* CSINN_OP_XOR */
    {NULL},                         /* CSINN_OP_YUV_RGB_SCALE */

    /* utils functions */
    {csi_dp1k_session_init},      /* CSINN_SESSION_INIT */
    {csi_dp1k_session_deinit},    /* CSINN_SESSION_DEINIT */
    {csi_dp1k_session_setup},     /* CSINN_SESSION_SETUP */
    {NULL},                       /* CSINN_SESSION_RUN */
    {NULL},                       /* CSINN_UPDATE_INPUT */
    {NULL},                       /* CSINN_UPDATE_OUTPUT */
    {csi_dp1k_set_input_number},  /* CSINN_SET_INPUT_NUMBER */
    {csi_dp1k_set_output_number}, /* CSINN_SET_OUTPUT_NUMBER */
    {csi_dp1k_get_input_number},  /* CSINN_GET_INPUT_NUMBER */
    {csi_dp1k_get_output_number}, /* CSINN_GET_OUTPUT_NUMBER */
    {csi_dp1k_set_input},         /* CSINN_SET_INPUT */
    {csi_dp1k_set_output},        /* CSINN_SET_OUTPUT */
    {NULL},                       /* CSINN_GET_INPUT */
    {NULL},                       /* CSINN_GET_OUTPUT */

    /* graph */
    {csi_dp1k_set_tensor},        /* CSINN_TENSOR */
};

void *csi_bc_map_dp1k(int op, int dtype) {
    return (op < CSINN_OP_AND_UTILS_SIZE) ? csi_bc_map_table_dp1k[op][0] : NULL;
}
