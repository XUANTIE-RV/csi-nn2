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

#include "csi_nn.h"
#include "csi_utils.h"
#include "csi_ovx.h"
#include "vsi_nn_pub.h"

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

#ifdef CSI_DEBUG
    //csi_statistical_mean_std(buffer, sz);
#endif

    csi_get_top5(buffer, sz, prob, class);

    printf(" --- Top ---\n");
    sz = sz > 5? 5 : sz;
    for(i = 0; i< sz; i++) {
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

void csi_ovx_set_output_number(int number, struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    vsi_nn_SetGraphOutputs(graph, NULL, number);
}

void csi_ovx_set_input_number(int number, struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
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
        if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_UINT8) {
            ret->qinfo->scale = tensor->attr.dtype.scale;
            ret->qinfo->zero_point = tensor->attr.dtype.zero_point;
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
        free(data);
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

int csi_ovx_set_tensor(struct csi_tensor *tensor, struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_id_t ret;

    for (int i = 0; i < tensor->dim_count; i++) {
        attr.size[i] = tensor->dim[tensor->dim_count - 1 - i];
    }
    attr.dim_num = tensor->dim_count;
    attr.vtl = FALSE;
    if (tensor->dtype == CSINN_DTYPE_UINT8) {
        attr.dtype.scale = tensor->qinfo->scale;
        attr.dtype.zero_point = tensor->qinfo->zero_point;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
        attr.dtype.vx_type = VSI_NN_TYPE_UINT8;
    } else if (tensor->dtype == CSINN_DTYPE_FLOAT32) {
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    } else if (tensor->dtype == CSINN_DTYPE_INT32) {
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    } else if(tensor->dtype == CSINN_DTYPE_UINT16) {
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_UINT16;
    } else {
        csi_debug_error("Unsupport for dtype: %d\n", tensor->dtype);
        return CSINN_UNSUPPORT_DTYPE;
    }
    if (tensor->is_const == 1) {
        attr.is_const = TRUE;
        ret = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, tensor->data);
    } else {
        attr.is_const = FALSE;
        ret = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL);
    }

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
    if (sess->model_name && sess->model_save != CSINN_RUN_ONLY) {
        // create general graph
        const char value[] = "1";
        setenv("VIV_VX_ENABLE_SAVE_NETWORK_BINARY", value, 1);
        setenv("VIV_VX_SAVE_NETWORK_BINARY_PATH", sess->model_name, 1);
    }
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
    sess->base_layout = CSINN_LAYOUT_NCHW;
}

void csi_ovx_session_setup(struct csi_session *sess)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    vsi_nn_SetupGraph(graph, FALSE);
    vsi_nn_VerifyGraph(graph);
}

void csi_ovx_session_run(struct csi_session *sess)
{
    uint64_t start_time, end_time;
    start_time = csi_get_timespec();
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    vsi_nn_RunGraph(graph);
    end_time = csi_get_timespec();
    printf("Run graph execution time: %.5fms, FPS=%.2f\n", ((float)(end_time-start_time))/1000000,
                    1000000000.0/((float)(end_time-start_time)));
    for (int idx = 0; idx < sess->input_num; idx++){
        sess->input[idx]->data = NULL;
    }
    for (int idx = 0; idx < sess->output_num; idx++){
        sess->output[idx]->data = NULL;
    }
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

void csi_ovx_nbg(const char *url, struct csi_session *sess)
{
    vsi_nn_node_t *node;
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);

    uint32_t input_num = sess->input_num;
    uint32_t output_num = sess->output_num;
    node = vsi_nn_AddNode(graph, VSI_NN_OP_NBG, input_num, output_num, NULL);
    node->nn_param.nbg.type = VSI_NN_NBG_FILE;
    node->nn_param.nbg.url = url;

    /* input */
    for (uint32_t i = 0; i < input_num; i++) {
        node->input.tensors[i] = (vsi_nn_tensor_id_t)sess->input[i]->data;
    }

    /* output */
    for (uint32_t i = 0; i < output_num; i++) {
        node->output.tensors[i] = (vsi_nn_tensor_id_t)sess->output[i]->data;
    }

    csi_ovx_session_setup(sess);
}

void csi_ovx_set_graph_attribute(struct csi_session *sess, int device_index)
{
    vsi_nn_graph_t *graph = csi_ovx_get_graph(sess);
    csi_debug_info("set device :%d\n", device_index);
    vxSetGraphAttribute(graph->g, VX_GRAPH_DEVICE_INDEX_VIV, &device_index, sizeof(device_index));
    csi_debug_info("verify...\n");
}

int csi_ovx_get_device_number()
{
    int deviceCount;
    vsi_nn_context_t context = vsi_nn_CreateContext();
    vxQueryContext(context->c, VX_CONTEXT_DEVICE_COUNT_VIV, &deviceCount, sizeof(deviceCount));
    vsi_nn_ReleaseContext(&context);
    return deviceCount;
}

static void *setup_bc_map()
{
    static void* bc_map[CSINN_OP_AND_UTILS_SIZE];

    bc_map[CSINN_OP_ABS] = csi_ovx_abs;
    bc_map[CSINN_OP_ADD] = csi_ovx_add;
    bc_map[CSINN_OP_AND] = csi_ovx_and;
    bc_map[CSINN_OP_ARGMAX] = csi_ovx_argmax;
    bc_map[CSINN_OP_ARGMIN] = csi_ovx_argmin;
    bc_map[CSINN_OP_AVGPOOL2D] = csi_ovx_averagepool;
    bc_map[CSINN_OP_BN] = csi_ovx_batch_normalization;
    bc_map[CSINN_OP_BATCH_TO_SPACE] = csi_ovx_batch_to_space;
    bc_map[CSINN_OP_CLIP] = csi_ovx_clip;
    bc_map[CSINN_OP_CONCAT] = csi_ovx_concat;
    bc_map[CSINN_OP_CONV2D] = csi_ovx_conv2d;
    bc_map[CSINN_OP_DEPTHWISE_CONV2D] = csi_ovx_depthwise_conv2d;
    bc_map[CSINN_OP_GROUP_CONV2D] = csi_ovx_group_conv2d;
    bc_map[CSINN_OP_CROP] = csi_ovx_crop;
    bc_map[CSINN_OP_DECONV2D] = csi_ovx_deconv2d;
    bc_map[CSINN_OP_DEPTHWISE_DECONV2D] = csi_ovx_depthwise_deconv2d;
    bc_map[CSINN_OP_DEPTH_TO_SPACE] = csi_ovx_depth_to_space;
    bc_map[CSINN_OP_DIV] = csi_ovx_div;
    bc_map[CSINN_OP_ELU] = csi_ovx_elu;
    bc_map[CSINN_OP_EQUANL] = csi_ovx_equal;
    bc_map[CSINN_OP_EXP] = csi_ovx_exp;
    bc_map[CSINN_OP_EXPAND_DIMS] = csi_ovx_expand_dims_u8;
    bc_map[CSINN_OP_FLATTEN] = csi_ovx_flatten;
    bc_map[CSINN_OP_FLOOR_DIVIDE] = csi_ovx_floor_divide;
    bc_map[CSINN_OP_FLOOR] = csi_ovx_floor;
    bc_map[CSINN_OP_FULLYCONNECTED] = csi_ovx_fullyconnected;
    bc_map[CSINN_OP_GATHER_ND] = csi_ovx_gather_nd;
    bc_map[CSINN_OP_GATHER] = csi_ovx_gather;
    bc_map[CSINN_OP_GLOBAL_AVGPOOL2D] = csi_ovx_global_averagepool;
    bc_map[CSINN_OP_GLOBAL_MAXPOOL2D] = csi_ovx_global_maxpool;
    bc_map[CSINN_OP_GREATHER_EQUAL] = csi_ovx_greater_equal;
    bc_map[CSINN_OP_GREATHER] = csi_ovx_greater;
    bc_map[CSINN_OP_L2N] = csi_ovx_l2_normalization;
    bc_map[CSINN_OP_L2POOL2D] = csi_ovx_l2pool;
    bc_map[CSINN_OP_LEAKY_RELU] = csi_ovx_leaky_relu;
    bc_map[CSINN_OP_LESS_EQUAL] = csi_ovx_less_equal;
    bc_map[CSINN_OP_LESS] = csi_ovx_less;
    bc_map[CSINN_OP_LOG] = csi_ovx_log;
    bc_map[CSINN_OP_LOG_SOFTMAX] = csi_ovx_log_softmax;
    bc_map[CSINN_OP_LRN] = csi_ovx_lrn;
    bc_map[CSINN_OP_MATMUL] = csi_ovx_matmul;
    bc_map[CSINN_OP_MAX] = csi_ovx_max;
    bc_map[CSINN_OP_MAXINUM] = csi_ovx_maximum;
    bc_map[CSINN_OP_MAXPOOL2D] = csi_ovx_maxpool;
    bc_map[CSINN_OP_MAXPOOL2D_LOCAT] = csi_ovx_maxpool2d_locat;
    bc_map[CSINN_OP_MEAN] = csi_ovx_mean;
    bc_map[CSINN_OP_MEAN_STRIDE] = csi_ovx_mean;
    bc_map[CSINN_OP_MIN] = csi_ovx_min;
    bc_map[CSINN_OP_MINIMUM] = csi_ovx_minimum;
    bc_map[CSINN_OP_MIN_STRIDE] = csi_ovx_min;
    bc_map[CSINN_OP_MUL] = csi_ovx_mul;
    bc_map[CSINN_OP_NEGATIIVE] = csi_ovx_negative;
    bc_map[CSINN_OP_NOT_EQUAL] = csi_ovx_not_equal;
    bc_map[CSINN_OP_OR] = csi_ovx_or;
    bc_map[CSINN_OP_PAD] = csi_ovx_pad;
    bc_map[CSINN_OP_POWER] = csi_ovx_power;
    bc_map[CSINN_OP_PRELU] = csi_ovx_prelu;
    bc_map[CSINN_OP_PROD] = csi_ovx_prod;
    bc_map[CSINN_OP_PROPOSAL] = csi_ovx_proposal;
    bc_map[CSINN_OP_PSROIPOOLING] = csi_ovx_psroipooling;
    bc_map[CSINN_OP_RELU] = csi_ovx_relu;
    bc_map[CSINN_OP_RELU1] = csi_ovx_relu1;
    bc_map[CSINN_OP_RELU6] = csi_ovx_relu6;
    bc_map[CSINN_OP_RELUN] = csi_ovx_relun;
    bc_map[CSINN_OP_REORG] = csi_ovx_reorg;
    bc_map[CSINN_OP_RESHAPE] = csi_ovx_reshape;
    bc_map[CSINN_OP_RESIZE] = csi_ovx_resize;
    bc_map[CSINN_OP_REVERSE] = csi_ovx_reverse;
    bc_map[CSINN_OP_ROIPOOL] = csi_ovx_roipool;
    bc_map[CSINN_OP_RSQRT] = csi_ovx_rsqrt;
    bc_map[CSINN_OP_SELECT] = csi_ovx_select;
    bc_map[CSINN_OP_SHUFFLE_CHANNEL] = csi_ovx_shuffle_channel;
    bc_map[CSINN_OP_SIGMOID] = csi_ovx_sigmoid;
    bc_map[CSINN_OP_SIN] = csi_ovx_sin;
    bc_map[CSINN_OP_SLICE] = csi_ovx_slice;
    bc_map[CSINN_OP_SOFTMAX] = csi_ovx_softmax;
    bc_map[CSINN_OP_SOFTPLUS] = csi_ovx_softplus;
    bc_map[CSINN_OP_SOFTRELU] = csi_ovx_softrelu;
    bc_map[CSINN_OP_SPACE_TO_BATCH] = csi_ovx_space_to_batch;
    bc_map[CSINN_OP_SPACE_TO_DEPTH] = csi_ovx_space_to_depth;
    bc_map[CSINN_OP_SPLIT] = csi_ovx_split;
    bc_map[CSINN_OP_SQRT] = csi_ovx_sqrt;
    bc_map[CSINN_OP_SQUARE] = csi_ovx_square;
    bc_map[CSINN_OP_SQUEEZE] = csi_ovx_squeeze;
    bc_map[CSINN_OP_STACK] = csi_ovx_stack;
    bc_map[CSINN_OP_STRIDED_SLICE] = csi_ovx_strided_slice;
    bc_map[CSINN_OP_SUB] = csi_ovx_sub;
    bc_map[CSINN_OP_SUM] = csi_ovx_sum;
    bc_map[CSINN_OP_TANH] = csi_ovx_tanh;
    bc_map[CSINN_OP_TILE] = csi_ovx_tile;
    bc_map[CSINN_OP_TOPK] = csi_ovx_topk;
    bc_map[CSINN_OP_TRANSPOSE] = csi_ovx_transpose;
    bc_map[CSINN_OP_UNPOOLING] = csi_ovx_unpooling;
    bc_map[CSINN_OP_UNSTACK] = csi_ovx_unstack;

    /* utils functions */
    bc_map[CSINN_SESSION_INIT] = csi_ovx_session_init;
    bc_map[CSINN_SESSION_DEINIT] = csi_ovx_session_deinit;
    bc_map[CSINN_SESSION_SETUP] = csi_ovx_session_setup;
    bc_map[CSINN_SESSION_RUN] = csi_ovx_session_run;
    bc_map[CSINN_UPDATE_INPUT] = csi_ovx_update_input;
    bc_map[CSINN_SET_INPUT_NUMBER] = csi_ovx_set_input_number;
    bc_map[CSINN_SET_OUTPUT_NUMBER] = csi_ovx_set_output_number;
    bc_map[CSINN_SET_INPUT] = csi_ovx_set_input;
    bc_map[CSINN_SET_OUTPUT] = csi_ovx_set_output;
    bc_map[CSINN_GET_INPUT] = csi_ovx_get_input;
    bc_map[CSINN_GET_OUTPUT] = csi_ovx_get_output;
    bc_map[CSINN_TENSOR_ENTRY] = csi_ovx_set_tensor;
    bc_map[CSINN_LOAD_BG] = csi_ovx_nbg;

    return bc_map;
}

static int get_bc_map_index(int op, int dtype)
{
    return op;
}

void *csi_bc_map_ovx(int op, int dtype) {
    static int has_init;
    static void **bc_map_table;
    if (has_init == 0) {
        bc_map_table = setup_bc_map();
        has_init = 1;
    }
    return bc_map_table[get_bc_map_index(op, dtype)];
}
