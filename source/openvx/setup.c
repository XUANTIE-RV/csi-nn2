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

#include <time.h>
#include "csi_nn.h"
#include "csi_utils.h"
#include "csi_ovx.h"

#define NET_TOTAL_TENSOR_NUM 0 /* useless */
#define NET_NODE_NUM 0 /* useless */
#define BILLION    1000000000

static void get_statistical_data(float *data, int sz)
{
    int i = 0;
    float max_value = data[0];
    float min_value = data[0];
    double std = 0.0;
    double sum = 0.0;
    for (i = 0; i < sz; i++)
    {
        sum += data[i];
        if (data[i] > max_value)
        {
            max_value = data[i];
        }
        if (data[i] < min_value)
        {
            min_value = data[i];
        }
    }
    double mean = sum / sz;
    sum = 0.0;
    for (i = 0; i < sz; i++)
    {
        sum += ((data[i]-mean) * (data[i]-mean));
    }
    std = sum / sz;
    printf("The max_value of output: %lf\n", max_value);
    printf("The min_value of output: %lf\n", min_value);
    printf("The mean_value of output: %lf\n", mean);
    printf("The std_value of output: %lf\n", std);
}

static vsi_bool get_top
    (
    float *pfProb,
    float *pfMaxProb,
    uint32_t *pMaxClass,
    uint32_t outputCount,
    uint32_t topNum
    )
{
    uint32_t i, j, k;

    #define MAX_TOP_NUM 20
    if (topNum > MAX_TOP_NUM) return FALSE;

    memset(pfMaxProb, 0xfe, sizeof(float) * topNum);
    memset(pMaxClass, 0xff, sizeof(float) * topNum);

    for (j = 0; j < topNum; j++)
    {
        for (i=0; i<outputCount; i++)
        {
            for (k=0; k < topNum; k ++)
            {
                if(i == pMaxClass[k])
                    break;
            }

            if (k != topNum)
                continue;

            if (pfProb[i] > *(pfMaxProb+j))
            {
                *(pfMaxProb+j) = pfProb[i];
                *(pMaxClass+j) = i;
            }
        }
    }

    return TRUE;
}

uint64_t csi_get_perf_count()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)((uint64_t)ts.tv_nsec + (uint64_t)ts.tv_sec * BILLION);
}

void csi_nn_show_top5(void *td, int index)
{
    uint32_t i,sz,stride;
    float *buffer = NULL;
    uint8_t *tensor_data = NULL;
    uint32_t MaxClass[5];
    float fMaxProb[5];
    uint32_t topk = 5;

    vsi_nn_graph_t *graph = ((struct __target_data *)td)->graph;
    vsi_nn_tensor_t *tensor = vsi_nn_GetTensor(graph, graph->output.tensors[index]);

    sz = 1;
    for(i = 0; i < tensor->attr.dim_num; i++) {
        sz *= tensor->attr.size[i];
    }

    if(topk > sz)
        topk = sz;

    stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
    tensor_data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, tensor);
    buffer = (float *)malloc(sizeof(float) * sz);

    for(i = 0; i < sz; i++) {
        vsi_nn_DtypeToFloat32(&tensor_data[stride * i], &buffer[i], &tensor->attr.dtype);
    }

#ifdef DEBUG_TEST
    get_statistical_data(buffer, sz);
#endif

    if (!get_top(buffer, fMaxProb, MaxClass, sz, topk))
    {
        printf("Fail to show result.\n");
        exit(-1);
    }

    printf(" --- Top%d ---\n", topk);
    for(i = 0; i< topk; i++) {
        printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
    }
    if(tensor_data)vsi_nn_Free(tensor_data);
    if(buffer)free(buffer);
}

void csi_nn_save_output(void *td, int index, const char *filename)
{
    vsi_nn_graph_t *graph = ((struct __target_data *)td)->graph;
    vsi_nn_tensor_t *tensor;

    tensor = vsi_nn_GetTensor(graph, graph->output.tensors[index]);
    vsi_nn_SaveTensorToTextByFp32(graph, tensor, filename, NULL);
}

int csi_nn_get_output_number(void *td)
{
    return ((struct __target_data *)td)->output_num;
}

int csi_nn_get_input_number(void *td)
{
    return ((struct __target_data *)td)->input_num;
}

struct csi_tensor *csi_nn_get_output(void *td, int index)
{
    struct csi_tensor *ret = malloc(sizeof(struct csi_tensor));
    vsi_nn_graph_t *graph = ((struct __target_data *)td)->graph;
    vsi_nn_tensor_t *tensor = vsi_nn_GetTensor(graph, graph->output.tensors[index]);

    ret->dim_count = tensor->attr.dim_num;
    for (int i = 0; i < ret->dim_count; i++) {
        ret->dim[i] = tensor->attr.size[ret->dim_count - 1 - i];
    }

    ret->data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, tensor);
    ret->scale = tensor->attr.dtype.scale;
    ret->zero_point = tensor->attr.dtype.zero_point;
    if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_UINT8) {
        ret->dtype = CSINN_DTYPE_UINT8;
    } else if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32) {
        ret->dtype = CSINN_DTYPE_FLOAT32;
    }

    return ret;
}

struct csi_tensor *csi_nn_ovx_get_tensor(void *td, int index)
{
    struct csi_tensor *ret = malloc(sizeof(struct csi_tensor));
    vsi_nn_graph_t *graph = ((struct __target_data *)td)->graph;
    vsi_nn_tensor_t *tensor = vsi_nn_GetTensor(graph, index);

    ret->dim_count = tensor->attr.dim_num;
    for (int i = 0; i < ret->dim_count; i++) {
        ret->dim[i] = tensor->attr.size[ret->dim_count - 1 - i];
    }

    ret->data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, tensor);
    ret->scale = tensor->attr.dtype.scale;
    ret->zero_point = tensor->attr.dtype.zero_point;
    if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_UINT8) {
        ret->dtype = CSINN_DTYPE_UINT8;
    } else if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32) {
        ret->dtype = CSINN_DTYPE_FLOAT32;
    }

    return ret;
}

struct csi_tensor *csi_nn_get_input(void *td, int index)
{
    struct csi_tensor *ret = malloc(sizeof(struct csi_tensor));
    vsi_nn_graph_t *graph = ((struct __target_data *)td)->graph;
    vsi_nn_tensor_t *tensor = vsi_nn_GetTensor(graph, graph->input.tensors[index]);

    ret->dim_count = tensor->attr.dim_num;
    for (int i = 0; i < ret->dim_count; i++) {
        ret->dim[i] = tensor->attr.size[ret->dim_count - 1 - i];
    }

    ret->data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, tensor);
    ret->scale = tensor->attr.dtype.scale;
    ret->zero_point = tensor->attr.dtype.zero_point;
    if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_UINT8) {
        ret->dtype = CSINN_DTYPE_UINT8;
    } else if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32) {
        ret->dtype = CSINN_DTYPE_FLOAT32;
    }

    return ret;
}

static void update_graph(vsi_nn_graph_t *graph, struct __target_data *target_data)
{
    vsi_nn_node_t *node;
    vsi_nn_node_t *prev_node;
    int i, j;
    int32_t prev_node_index;
    int32_t output_index;

    // change the scales of reshape layer
    const char *name = "";
    vsi_nn_tensor_t *curr_input_tensor;
    vsi_nn_tensor_t *curr_output_tensor;

    vsi_nn_tensor_t *proposal_im_info_tensor;
    vsi_nn_tensor_t *proposal_anchor_tensor;
    for (i = 1; i < target_data->layer_num; i++) {
        node = vsi_nn_GetNode(graph, i);
        name = vsi_nn_OpGetName(node->op);
        if (strcmp(name, "RESHAPE") == 0 ||
            strcmp(name, "PERMUTE") == 0) {
            curr_input_tensor = vsi_nn_GetTensor(graph, node->input.tensors[0]);
            curr_output_tensor = vsi_nn_GetTensor(graph, node->output.tensors[0]);
            curr_output_tensor->attr.dtype.scale = curr_input_tensor->attr.dtype.scale;
            curr_output_tensor->attr.dtype.zero_point = curr_input_tensor->attr.dtype.zero_point;
        }
        if (strcmp(name, "PROPOSAL") == 0) {
            printf("current op: proposal\n");
            proposal_im_info_tensor = vsi_nn_GetTensor(graph, node->input.tensors[2]);
            proposal_anchor_tensor = vsi_nn_GetTensor(graph, node->input.tensors[3]);

            proposal_im_info_tensor = NULL;
            proposal_anchor_tensor = NULL;
        }
    }
}

void quantize_input(struct csi_tensor *input,
                    struct csi_tensor *output)
{
    float *input_data = input->data;
    uint8_t *output_data = output->data;
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }

    for (int i = 0; i < size; i++) {
        uint8_t input_val = round(input_data[i] * output->scale) + output->offset;
        output_data[i] = input_val;
    }
}

int csi_nn_create_tensor(struct csi_tensor *input,
                         struct csi_tensor *output,
                         void *td)
{
    vsi_nn_graph_t *graph = ((struct __target_data *)td)->graph;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_id_t ret;
    vsi_status status;

    uint8_t *inputData;
    uint32_t sz = 1;
    uint32_t stride = 1;
    int i = 0;

    for (i = 0; i < input->dim_count; i++) {
        attr.size[i] = input->dim[input->dim_count - 1 - i];
    }
    attr.dim_num = input->dim_count;
    attr.dtype.scale = output->scale;
    attr.dtype.zero_point = output->zero_point;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    attr.vtl = FALSE;
    attr.is_const = FALSE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;

    for (i = 0; i < 4; i++) {
        sz *= input->dim[i];
    }
    stride = vsi_nn_TypeGetBytes(attr.dtype.vx_type);
    inputData = (uint8_t *)malloc(stride * sz * sizeof(uint8_t));

    ret = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, inputData);
    output->data = (void *)ret;
    output->t_private = td;
    return (int)ret;
}

int csi_nn_ovx_create_const(struct csi_tensor *input, void *td)
{
    vsi_nn_graph_t *graph = ((struct __target_data *)td)->graph;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_id_t ret;
    vsi_status status;

    for (int i = 0; i < input->dim_count; i++) {
        attr.size[i] = input->dim[input->dim_count - 1 - i];
    }
    attr.dim_num = input->dim_count;
    attr.dtype.scale = input->scale;
    attr.dtype.zero_point = input->zero_point;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    attr.vtl = FALSE;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;

    ret = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, input->data);
    input->data = (void *)ret;
    input->t_private = td;
    return (int)ret;
}

uint8_t *csi_nn_input_f32_to_u8(uint32_t index, float *data, void *td)
{
    vsi_nn_tensor_t *tensor;
    vsi_status status = VSI_FAILURE;
    vsi_nn_graph_t *graph = ((struct __target_data *)td)->graph;
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

static void _handle_multiple_inputs(vsi_nn_graph_t *graph, uint32_t idx,
                                    uint8_t *input_data)
{
    vsi_nn_tensor_t *tensor;
    vsi_status status = VSI_FAILURE;
    tensor = NULL;
    tensor = vsi_nn_GetTensor( graph, graph->input.tensors[idx] );

    /* Copy the Pre-processed data to input tensor */
    status = vsi_nn_CopyDataToTensor(graph, tensor, input_data);
}

void csi_nn_update_input(uint32_t idx, uint8_t *data, void *td) {
    vsi_nn_graph_t *graph = ((struct __target_data *)td)->graph;
    _handle_multiple_inputs(graph, idx, data);
}

void *csi_nn_presetup(int input, int output)
{
    vsi_nn_graph_t *graph;
    vsi_nn_context_t ctx;
    struct __target_data *target_data = malloc(sizeof(struct __target_data));
    target_data->input_num = input;
    target_data->output_num = output;
    int32_t input_num = input;
    int32_t output_num = output;
    ctx = vsi_nn_CreateContext();
#define VNN_VERSION_MAJOR 1
#define VNN_VERSION_MINOR 1
#define VNN_VERSION_PATCH 12
    graph = vsi_nn_CreateGraph(ctx, NET_TOTAL_TENSOR_NUM, NET_NODE_NUM);
    vsi_nn_SetGraphVersion(graph, VNN_VERSION_MAJOR, VNN_VERSION_MINOR, VNN_VERSION_PATCH);
    vsi_nn_SetGraphInputs(graph, NULL, input_num);
    vsi_nn_SetGraphOutputs(graph, NULL, output_num);
    target_data->graph = graph;
    return target_data;
}

void csi_nn_init(struct csi_tensor *input,
                 struct csi_tensor *output)
{
}

void csi_nn_setup(void *td)
{
    vsi_nn_graph_t *graph = ((struct __target_data *)td)->graph;
//    update_graph(graph, td);
    vsi_nn_SetupGraph(graph, FALSE);
    vsi_nn_VerifyGraph(graph);
}

void csi_nn_run(void *td)
{
    vsi_nn_graph_t *graph = ((struct __target_data *)td)->graph;
    uint64_t start_time, end_time;
    start_time = csi_get_perf_count();
    vsi_nn_RunGraph(graph);
    end_time = csi_get_perf_count();
    printf("Run graph execution time: %.5fms, FPS=%.2f\n", ((float)(end_time-start_time))/1000000,
                    1000000000.0/((float)(end_time-start_time)));
}

void csi_nn_postprocess(void* td)
{
}

void csi_nn_deinit(struct csi_tensor *input,
                   struct csi_tensor *output)
{
}

void csi_nn_set_ovx_input(int index, int input, struct __target_data *td)
{
    vsi_nn_graph_t *graph = td->graph;
    graph->input.tensors[index] = input;
}

void csi_nn_set_ovx_output(int index, struct csi_tensor *output, struct __target_data *td)
{
    vsi_nn_graph_t *graph = td->graph;
    vsi_nn_tensor_t *tensor;
    graph->output.tensors[index] = (vsi_nn_tensor_id_t)output->data;

    tensor = vsi_nn_GetTensor(graph, graph->output.tensors[index]);
    tensor->attr.vtl = FALSE;
}

void csi_ovx_free(struct __target_data *td) {
    vsi_nn_graph_t *graph = td->graph;

    vsi_nn_context_t ctx;
    if (graph) {
        ctx = graph->ctx;
        vsi_nn_ReleaseGraph(&graph);

        vsi_nn_ReleaseContext(&ctx);
    }
}
