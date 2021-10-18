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

void csi_ovx_show_top5(int index, struct csi_session *sess)
{
    uint32_t i, sz, stride;
    float *buffer = NULL;
    uint8_t *tensor_data = NULL;
    uint32_t class[5];
    float prob[5];

    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    vsi_nn_tensor_t *tensor = vsi_nn_GetTensor(graph, graph->output.tensors[index]);

    sz = 1;
    for(i = 0; i < tensor->attr.dim_num; i++) {
        sz *= tensor->attr.size[i];
    }

    stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
    tensor_data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, tensor);
    buffer = (float *)malloc(sizeof(float) * sz);

    for (i = 0; i < sz; i++) {
        vsi_nn_DtypeToFloat32(&tensor_data[stride * i], &buffer[i], &tensor->attr.dtype);
    }

#ifdef DEBUG_TEST
    csi_statistical_mean_std(buffer, sz);
#endif

    csi_get_top5(buffer, sz, prob, class);

    printf(" --- Top ---\n");
    for(i = 0; i< 5; i++) {
        printf("%3d: %8.6f\n", class[i], prob[i]);
    }
    if (tensor_data) {
        vsi_nn_Free(tensor_data);
    }
    if (buffer) {
        free(buffer);
    }
}

void csi_ovx_save_output(int index, const char *filename, struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    vsi_nn_tensor_t *tensor;

    tensor = vsi_nn_GetTensor(graph, graph->output.tensors[index]);
    vsi_nn_SaveTensorToTextByFp32(graph, tensor, filename, NULL);
}

int csi_ovx_get_output_number(struct csi_session *sess)
{
    return sess->output_num;
}

int csi_ovx_get_input_number(struct csi_session *sess)
{
    return sess->input_num;
}

void csi_ovx_set_output_number(int number, struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    sess->output_num = number;
    vsi_nn_SetGraphOutputs(graph, NULL, number);
}

void csi_ovx_set_input_number(int number, struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    sess->input_num = number;
    vsi_nn_SetGraphInputs(graph, NULL, number);
}

static int csi_ovx_get_tensor_internal(struct csi_tensor *ret, vsi_nn_tensor_t *tensor,
                                       vsi_nn_graph_t *graph)
{
    if (ret->data == NULL) {
        ret->data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, tensor);
        ret->dim_count = tensor->attr.dim_num;
        for (int i = 0; i < ret->dim_count; i++) {
            ret->dim[i] = tensor->attr.size[ret->dim_count - 1 - i];
        }

        ret->scale = tensor->attr.dtype.scale;
        ret->zero_point = tensor->attr.dtype.zero_point;
        if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_UINT8) {
            ret->dtype = CSINN_DTYPE_UINT8;
        } else if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32) {
            ret->dtype = CSINN_DTYPE_FLOAT32;
        }
    } else {
        if (ret->dim_count != tensor->attr.dim_num) {
            return CSINN_FALSE;
        }
        int size = 1;
        for (int i = 0; i < ret->dim_count; i++) {
            size *= tensor->attr.size[ret->dim_count - 1 - i];
        }

        uint8_t *data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, tensor);
        if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_UINT8) {
            memcpy(ret->data, data, size);
        } else if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32) {
            memcpy(ret->data, data, size * 4);
        }
    }
    return CSINN_TRUE;
}

int csi_ovx_get_output(int index, struct csi_tensor *output, struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    vsi_nn_tensor_t *tensor = vsi_nn_GetTensor(graph, graph->output.tensors[index]);

    return csi_ovx_get_tensor_internal(output, tensor, graph);
}

int csi_ovx_get_tensor(int index, struct csi_tensor *ret, struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    vsi_nn_tensor_t *tensor = vsi_nn_GetTensor(graph, index);

    return csi_ovx_get_tensor_internal(ret, tensor, graph);
}

int csi_ovx_get_input(int index, struct csi_tensor *input, struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    vsi_nn_tensor_t *tensor = vsi_nn_GetTensor(graph, graph->input.tensors[index]);

    return csi_ovx_get_tensor_internal(input, tensor, graph);
}

void csi_ovx_set_tensor(struct csi_tensor *tensor, struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_id_t ret;

    uint8_t *input_data;
    uint32_t sz = 1;
    uint32_t stride = 1;
    int i = 0;

    for (i = 0; i < tensor->dim_count; i++) {
        attr.size[i] = tensor->dim[tensor->dim_count - 1 - i];
    }
    attr.dim_num = tensor->dim_count;
    attr.dtype.scale = tensor->scale;
    attr.dtype.zero_point = tensor->zero_point;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    attr.vtl = FALSE;
    attr.is_const = FALSE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;

    for (i = 0; i < 4; i++) {
        sz *= tensor->dim[i];
    }
    stride = vsi_nn_TypeGetBytes(attr.dtype.vx_type);
    input_data = (uint8_t *)malloc(stride * sz * sizeof(uint8_t));

    ret = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, input_data);
    tensor->data = (void *)ret;
    tensor->sess = sess;
}

void csi_ovx_set_const_tensor(struct csi_tensor *tensor, struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_id_t ret;

    for (int i = 0; i < tensor->dim_count; i++) {
        attr.size[i] = tensor->dim[tensor->dim_count - 1 - i];
    }
    attr.dim_num = tensor->dim_count;
    attr.vtl = FALSE;
    attr.is_const = TRUE;
    if (tensor->dtype == CSINN_DTYPE_UINT8) {
        attr.dtype.scale = tensor->scale;
        attr.dtype.zero_point = tensor->zero_point;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
        attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    } else {
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    }

    ret = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, tensor->data);
    tensor->data = (void *)ret;
    tensor->sess = sess;
}

uint8_t *csi_ovx_input_f32_to_u8(uint32_t index, float *data, struct csi_session *sess)
{
    vsi_nn_tensor_t *tensor;
    vsi_status status = VSI_FAILURE;
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    tensor = vsi_nn_GetTensor(graph, graph->input.tensors[index]);
    uint32_t i = 0;
    uint8_t *tensorData;
    uint32_t sz = 1;
    uint32_t stride = 1;
    tensorData = NULL;
    sz = vsi_nn_GetElementNum(tensor);
    stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
    tensorData = (uint8_t *)malloc(stride * sz * sizeof(uint8_t));
    memset(tensorData, 0, stride * sz * sizeof(uint8_t));

    for (i = 0; i < sz; i++) {
        vsi_nn_Float32ToDtype(data[i], &tensorData[stride * i], &tensor->attr.dtype);
    }
    return tensorData;
}

void csi_ovx_update_input(uint32_t idx, struct csi_tensor *input, struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    vsi_nn_tensor_t *tensor = vsi_nn_GetTensor(graph, graph->input.tensors[idx]);

    /* Copy the Pre-processed data to input tensor */
    vsi_nn_CopyDataToTensor(graph, tensor, input->data);
}

void csi_ovx_session_init(struct csi_session *sess)
{
    vsi_nn_graph_t *graph;
    vsi_nn_context_t ctx;
    struct csi_ovx_target_data *target_data = calloc(sizeof(struct csi_ovx_target_data), 1);
    ctx = vsi_nn_CreateContext();
#define VNN_VERSION_MAJOR 1
#define VNN_VERSION_MINOR 1
#define VNN_VERSION_PATCH 12
#define NET_TOTAL_TENSOR_NUM 0
#define NET_NODE_NUM 0
    graph = vsi_nn_CreateGraph(ctx, NET_TOTAL_TENSOR_NUM, NET_NODE_NUM);
    vsi_nn_SetGraphVersion(graph, VNN_VERSION_MAJOR, VNN_VERSION_MINOR, VNN_VERSION_PATCH);
    target_data->graph = graph;
    sess->td = target_data;
    sess->base_dtype = CSINN_DTYPE_UINT8;
    sess->base_layout = CSINN_NCHW;
}

void csi_ovx_session_setup(struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    vsi_nn_SetupGraph(graph, FALSE);
    vsi_nn_VerifyGraph(graph);
}

void csi_ovx_session_run(struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    uint64_t start_time, end_time;
    start_time = csi_get_timespec();
    vsi_nn_RunGraph(graph);
    end_time = csi_get_timespec();
    printf("Run graph execution time: %.5fms, FPS=%.2f\n", ((float)(end_time-start_time))/1000000,
                    1000000000.0/((float)(end_time-start_time)));
}

void csi_ovx_set_input(int index, struct csi_tensor *input, struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    graph->input.tensors[index] = (vsi_nn_tensor_id_t)input->data;
}

void csi_ovx_set_output(int index, struct csi_tensor *output, struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    vsi_nn_tensor_t *tensor;
    graph->output.tensors[index] = (vsi_nn_tensor_id_t)output->data;

    tensor = vsi_nn_GetTensor(graph, graph->output.tensors[index]);
    tensor->attr.vtl = FALSE;
}

void csi_ovx_session_deinit(struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);

    vsi_nn_context_t ctx;
    if (graph) {
        ctx = graph->ctx;
        vsi_nn_ReleaseGraph(&graph);
        vsi_nn_ReleaseContext(&ctx);
    }
    free(sess->td);
}

void *csi_ovx_get_graph(struct csi_session *sess)
{
    struct csi_ovx_target_data *td = sess->td;
    return td->graph;
}

void* csi_bc_map_table_ovx[CSINN_OP_SIZE][1] = {
    {csi_ovx_abs}, /* CSINN_OP_ABS */
    {NULL}, /* CSINN_OP_ACOS */
    {NULL}, /* CSINN_OP_ACOSH */
    {csi_ovx_add}, /* CSINN_OP_ADD */
    {NULL}, /* CSINN_OP_ALL */
    {csi_ovx_and}, /* CSINN_OP_AND */
    {NULL}, /* CSINN_OP_ANY */
    {NULL}, /* CSINN_OP_ARANGE */
    {csi_ovx_argmax}, /* CSINN_OP_ARGMAX */
    {csi_ovx_argmin}, /* CSINN_OP_ARGMIN */
    {NULL}, /* CSINN_OP_ASIN */
    {NULL}, /* CSINN_OP_ASINH */
    {NULL}, /* CSINN_OP_ATAN */
    {NULL}, /* CSINN_OP_ATANH */
    {csi_ovx_averagepool}, /* CSINN_OP_AVGPOOL2D */
    {NULL}, /* CSINN_OP_AVGPOOL3D */
    {csi_ovx_batch_normalization}, /* CSINN_OP_BN */
    {csi_ovx_batch_to_space}, /* CSINN_OP_BATCH_TO_SPACE */
    {NULL}, /* CSINN_OP_BROADCOST */
    {NULL}, /* CSINN_OP_CEIL */
    {NULL}, /* CSINN_OP_CLIP */
    {NULL}, /* CSINN_OP_COL2IM */
    {csi_ovx_concat}, /* CSINN_OP_CONCAT */
    {csi_ovx_conv2d}, /* CSINN_OP_CONV2D */
    {NULL}, /* CSINN_OP_CONV2D_RELU */
    {NULL}, /* CSINN_OP_CONV2D_RELU6 */
    {NULL}, /* CSINN_OP_CONV2D_CHANNEL */
    {NULL}, /* CSINN_OP_CONV2D_CHANNEL_RELU */
    {NULL}, /* CSINN_OP_CONV2D_CHANNEL_RELU6 */
    {csi_ovx_depthwise_conv2d}, /* CSINN_OP_DEPTHWISE_CONV2D */
    {NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_RELU */
    {NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_RELU6 */
    {NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL */
    {NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU */
    {NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6 */
    {csi_ovx_group_conv2d}, /* CSINN_OP_GROUP_CONV2D */
    {NULL}, /* CSINN_OP_GROUP_CONV2D_RELU */
    {NULL}, /* CSINN_OP_GROUP_CONV2D_CHANNEL */
    {NULL}, /* CSINN_OP_GROUP_CONV2D_CHANNEL_RELU */
    {NULL}, /* CSINN_OP_CONV3D */
    {NULL}, /* CSINN_OP_COS */
    {NULL}, /* CSINN_OP_COSH */
    {NULL}, /* CSINN_OP_CUMPROD */
    {NULL}, /* CSINN_OP_CUMSUM */
    {csi_ovx_deconv2d}, /* CSINN_OP_DECONV2D */
    {csi_ovx_depthwise_deconv2d}, /* CSINN_OP_DEPTHWISE_DECONV2D */
    {NULL}, /* CSINN_OP_DECONV3D */
    {csi_ovx_depth_to_space}, /* CSINN_OP_DEPTH_TO_SPACE */
    {csi_ovx_div}, /* CSINN_OP_DIV */
    {csi_ovx_elu}, /* CSINN_OP_ELU */
    {csi_ovx_equal}, /* CSINN_OP_EQUANL */
    {NULL}, /* CSINN_OP_ERF */
    {csi_ovx_exp}, /* CSINN_OP_EXP */
    {csi_ovx_expand_dims_u8}, /* CSINN_OP_EXPAND_DIMS */
    {NULL}, /* CSINN_OP_EXPM1 */
    {csi_ovx_flatten}, /* CSINN_OP_FLATTEN */
    {csi_ovx_floor_divide}, /* CSINN_OP_FLOOR_DIVIDE */
    {NULL}, /* CSINN_OP_FLOOR_MOD */
    {csi_ovx_floor}, /* CSINN_OP_FLOOR */
    {csi_ovx_fullyconnected}, /* CSINN_OP_FULLYCONNECTED */
    {NULL}, /* CSINN_OP_GATHER_ND */
    {NULL}, /* CSINN_OP_GATHER */
    {csi_ovx_global_averagepool}, /* CSINN_OP_GLOBAL_AVGPOOL2D */
    {csi_ovx_global_maxpool}, /* CSINN_OP_GLOBAL_MAXPOOL2D */
    {csi_ovx_greater_equal}, /* CSINN_OP_GREATHER_EQUAL */
    {csi_ovx_greater}, /* CSINN_OP_GREATHER */
    {NULL}, /* CSINN_OP_HARD_SIGMOID */
    {NULL}, /* CSINN_OP_IM2COL */
    {NULL}, /* CSINN_OP_ISNAN */
    {csi_ovx_l2_normalization}, /* CSINN_OP_L2N */
    {csi_ovx_l2pool}, /* CSINN_OP_L2POOL2D */
    {csi_ovx_leaky_relu}, /* CSINN_OP_LEAKY_RELU */
    {csi_ovx_less_equal}, /* CSINN_OP_LESS_EQUAL */
    {csi_ovx_less}, /* CSINN_OP_LESS */
    {NULL}, /* CSINN_OP_LOG_SOFTMAX */
    {NULL}, /* CSINN_OP_LOG */
    {NULL}, /* CSINN_OP_LOG1P */
    {NULL}, /* CSINN_OP_LOGICAL_AND */
    {NULL}, /* CSINN_OP_LOGICAL_NOT */
    {NULL}, /* CSINN_OP_LOGICAL_OR */
    {NULL}, /* CSINN_OP_LOGICAL_XOR */
    {csi_ovx_lrn},  /* CSINN_OP_LRN */
    {csi_ovx_matmul}, /* CSINN_OP_MATMUL */
    {csi_ovx_max}, /* CSINN_OP_MAX */
    {csi_ovx_maximum}, /* CSINN_OP_MAXINUM */
    {csi_ovx_maxpool}, /* CSINN_OP_MAXPOOL2D */
    {csi_ovx_maxpool2d_locat}, /* CSINN_OP_MAXPOOL2D_LOCAT */
    {NULL}, /* CSINN_OP_MAXPOOL3D */
    {csi_ovx_mean}, /* CSINN_OP_MEAN */
    {csi_ovx_mean}, /* CSINN_OP_MEAN_STRIDE */
    {csi_ovx_min}, /* CSINN_OP_MIN */
    {NULL}, /* CSINN_OP_MIN_STRIDE */
    {csi_ovx_minimum}, /* CSINN_OP_MINIMUM */
    {NULL}, /* CSINN_OP_MOD */
    {csi_ovx_mul}, /* CSINN_OP_MUL */
    {NULL}, /* CSINN_OP_NDARRAY_SIZE */
    {csi_ovx_negative}, /* CSINN_OP_NEGATIIVE */
    {NULL}, /* CSINN_OP_NON_MAX_SUPPRESSION */
    {csi_ovx_not_equal}, /* CSINN_OP_NOT_EQUAL */
    {NULL}, /* CSINN_OP_NOT */
    {NULL}, /* CSINN_OP_ONE_HOT */
    {csi_ovx_or}, /* CSINN_OP_OR */
    {csi_ovx_pad}, /* CSINN_OP_PAD */
    {csi_ovx_power}, /* CSINN_OP_POWER */
    {csi_ovx_prelu}, /* CSINN_OP_PRELU */
    {csi_ovx_prod}, /* CSINN_OP_PROD */
    {csi_ovx_proposal}, /* CSINN_OP_PROPOSAL */
    {csi_ovx_psroipooling}, /* CSINN_OP_PSROIPOOLING */
    {NULL}, /* CSINN_OP_REDUCE_LOGSUMEXP */
    {NULL}, /* CSINN_OP_REDUCE_MAX */
    {NULL}, /* CSINN_OP_REDUCE_MEAN */
    {NULL}, /* CSINN_OP_REDUCE_MIN */
    {NULL}, /* CSINN_OP_REDUCE_PROD */
    {NULL}, /* CSINN_OP_REDUCE_SUM */
    {csi_ovx_relu}, /* CSINN_OP_RELU */
    {csi_ovx_relu1}, /* CSINN_OP_RELU1 */
    {csi_ovx_relu6}, /* CSINN_OP_RELU6 */
    {csi_ovx_relun}, /* CSINN_OP_RELUN */
    {csi_ovx_reorg}, /* CSINN_OP_REORG */
    {csi_ovx_reshape}, /* CSINN_OP_RESHAPE */
    {csi_ovx_resize}, /* CSINN_OP_RESIZE */
    {csi_ovx_reverse}, /* CSINN_OP_REVERSE */
    {NULL}, /* CSINN_OP_ROIALIGN */
    {NULL}, /* CSINN_OP_ROIPOOL */
    {NULL}, /* CSINN_OP_ROUND */
    {csi_ovx_rsqrt}, /* CSINN_OP_RSQRT */
    {NULL}, /* CSINN_OP_SEGMENT_MAX */
    {NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MAX */
    {NULL}, /* CSINN_OP_SEGMENT_MEAN */
    {NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MEAN */
    {NULL}, /* CSINN_OP_SEGMENT_MIN */
    {NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MIN */
    {NULL}, /* CSINN_OP_SEGMENT_PROD */
    {NULL}, /* CSINN_OP_UNSORTED_SEGMENT_PROD */
    {NULL}, /* CSINN_OP_SEGMENT_SUM */
    {NULL}, /* CSINN_OP_UNSORTED_SEGMENT_SUM */
    {csi_ovx_select}, /* CSINN_OP_SELECT */
    {NULL}, /* CSINN_OP_SEQUENCE_MASK */
    {NULL}, /* CSINN_OP_SHAPE */
    {NULL}, /* CSINN_OP_SHUFFLE_CHANNEL */
    {csi_ovx_sigmoid}, /* CSINN_OP_SIGMOID */
    {NULL}, /* CSINN_OP_SIGN */
    {NULL}, /* CSINN_OP_SIN */
    {NULL}, /* CSINN_OP_SINH */
    {csi_ovx_slice}, /* CSINN_OP_SLICE */
    {csi_ovx_softmax}, /* CSINN_OP_SOFTMAX */
    {csi_ovx_softplus}, /* CSINN_OP_SOFTPLUS */
    {NULL}, /* CSINN_OP_SOFTRELU */
    {NULL}, /* CSINN_OP_SOFTSIGN */
    {csi_ovx_space_to_batch}, /* CSINN_OP_SPACE_TO_BATCH */
    {csi_ovx_space_to_depth}, /* CSINN_OP_SPACE_TO_DEPTH */
    {csi_ovx_split}, /* CSINN_OP_SPLIT */
    {csi_ovx_sqrt}, /* CSINN_OP_SQRT */
    {csi_ovx_square}, /* CSINN_OP_SQUARE */
    {csi_ovx_squeeze}, /* CSINN_OP_SQUEEZE */
    {csi_ovx_stack}, /* CSINN_OP_STACK */
    {NULL}, /* CSINN_OP_STRIDED_SLICE */
    {csi_ovx_sub}, /* CSINN_OP_SUB */
    {csi_ovx_sum}, /* CSINN_OP_SUM */
    {NULL}, /* CSINN_OP_TAN */
    {csi_ovx_tanh}, /* CSINN_OP_TANH */
    {NULL}, /* CSINN_OP_THRESHOLD_RELU */
    {csi_ovx_tile}, /* CSINN_OP_TILE */
    {NULL}, /* CSINN_OP_TOPK */
    {csi_ovx_transpose}, /* CSINN_OP_TRANSPOSE */
    {NULL}, /* CSINN_OP_TRUNC */
    {csi_ovx_unpooling}, /* CSINN_OP_UNPOOLING */
    {csi_ovx_unstack}, /* CSINN_OP_UNSTACK */
    {NULL}, /* CSINN_OP_WHERE */
    {NULL}, /* CSINN_OP_XOR */
    {NULL}, /* CSINN_OP_YUV_RGB_SCALE */

    /* utils functions */
    {csi_ovx_session_init},
    {csi_ovx_session_deinit},
    {csi_ovx_session_setup},
    {csi_ovx_session_run},
    {csi_ovx_update_input},
    {csi_ovx_set_input_number},
    {csi_ovx_set_output_number},
    {csi_ovx_get_input_number},
    {csi_ovx_get_output_number},
    {csi_ovx_set_input},
    {csi_ovx_set_output},
    {csi_ovx_get_input},
    {csi_ovx_get_output},
};

void *csi_bc_map_ovx(int op, int dtype)
{
    int dt;
    switch (dtype) {
        case CSINN_DTYPE_UINT8:
            dt = 0;
            break;
        default:
            return NULL;
    }

    return csi_bc_map_table_ovx[op][dt];
}

void csi_ovx_nbg(struct csi_tensor **input, struct csi_tensor **output,
                 uint32_t inputs_count, uint32_t outputs_count, const char *url)
{
    vsi_nn_node_t *node;
    vsi_nn_graph_t *graph = csi_ovx_get_graph(input[0]->sess);

    uint32_t input_num = inputs_count;
    uint32_t output_num = outputs_count;
    node = vsi_nn_AddNode(graph, VSI_NN_OP_NBG, input_num, output_num, NULL);
    node->nn_param.nbg.type = VSI_NN_NBG_FILE;
    node->nn_param.nbg.url = url;

    /* input */
    for (uint32_t i = 0; i < input_num; i++) {
        node->input.tensors[i] = (vsi_nn_tensor_id_t)input[i]->data;
    }

    /* output */
    for (uint32_t i = 0; i < output_num; i++) {
        node->output.tensors[i] = (vsi_nn_tensor_id_t)output[i]->data;
    }
}

