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
#include <time.h>

void csi_statistical_mean_std(float *data, int sz)
{
    int i = 0;
    float max_value = data[0];
    float min_value = data[0];
    double std = 0.0;
    double sum = 0.0;
    for (i = 0; i < sz; i++) {
        sum += data[i];
        if (data[i] > max_value) {
            max_value = data[i];
        }
        if (data[i] < min_value) {
            min_value = data[i];
        }
    }
    double mean = sum / sz;
    sum = 0.0;
    for (i = 0; i < sz; i++) {
        sum += ((data[i] - mean) * (data[i] - mean));
    }
    std = sum / sz;
    printf("The max_value of output: %lf\n", max_value);
    printf("The min_value of output: %lf\n", min_value);
    printf("The mean_value of output: %lf\n", mean);
    printf("The std_value of output: %lf\n", std);
}

void csi_get_top5(float *buf, uint32_t size, float *prob, uint32_t *class)
{
    uint32_t i, j, k;

    memset(prob, 0xfe, sizeof(float) * 5);
    memset(class, 0xff, sizeof(uint32_t) * 5);

    for (j = 0; j < 5; j++) {
        for (i = 0; i < size; i++) {
            for (k = 0; k < 5; k++) {
                if (i == class[k]) {
                    break;
                }
            }

            if (k != 5) {
                continue;
            }

            if (buf[i] > prob[j]) {
                prob[j] = buf[i];
                class[j] = i;
            }
        }
    }
}

void csi_show_top5(struct csi_tensor *output, struct csi_session *sess)
{
    uint32_t i, size;
    uint32_t class[5];
    float prob[5];

    if (output->data == NULL) {
        return;
    }

    size = 1;
    for (i = 0; i < output->dim_count; i++) {
        size *= output->dim[i];
    }

// #ifdef CSI_DEBUG
    csi_statistical_mean_std(output->data, size);
// #endif

    csi_get_top5(output->data, size, prob, class);

    printf(" ============ top5: ===========\n");
    size = size > 5? 5:size;
    for(i = 0; i< size; i++) {
        printf("%3d: %8.6f\n", class[i], prob[i]);
    }
}

int csi_tensor_size(struct csi_tensor *tensor)
{
    if (tensor->dim_count == 0) {
        return 0;
    }
    int size = 1;
    for (int i = 0; i < tensor->dim_count; i++) {
        size *= tensor->dim[i];
    }
    return size;
}

int csi_tensor_byte_size(struct csi_tensor *tensor)
{
    int size = csi_tensor_size(tensor);
    switch (tensor->dtype)
    {
    case CSINN_DTYPE_INT16:
    case CSINN_DTYPE_UINT16:
    case CSINN_DTYPE_FLOAT16:
        size *= 2;
        break;
    case CSINN_DTYPE_INT32:
    case CSINN_DTYPE_UINT32:
    case CSINN_DTYPE_FLOAT32:
        size *= 4;
        break;
    case CSINN_DTYPE_FLOAT64:
        size *= 8;
        break;
    default:
        break;
    }
    return size;
}

struct csi_tensor *csi_alloc_tensor(struct csi_session *session)
{
    struct csi_tensor *ret = calloc(1, sizeof(struct csi_tensor));
    if (session != NULL) {
        ret->dtype = session->base_dtype;
        ret->layout = session->base_layout;
        ret->sess = session;
    }
    ret->quant_channel = 1;
    ret->qinfo = calloc(1, sizeof(struct csi_quant_info));
    return ret;
}

void csi_realloc_quant_info(struct csi_tensor *tensor, int quant_info_num)
{
    tensor->quant_channel = quant_info_num;
    tensor->qinfo  = realloc(tensor->qinfo, quant_info_num * sizeof(struct csi_quant_info));
}

void csi_tensor_copy(struct csi_tensor *dest, struct csi_tensor *src)
{
    dest->data = src->data;
    dest->dtype = src->dtype;
    memcpy(dest->dim, src->dim, MAX_DIM * 4);
    dest->dim_count = src->dim_count;
    dest->name = src->name;
    dest->layout = src->layout;
    csi_realloc_quant_info(dest, src->quant_channel);
    memcpy(dest->qinfo, src->qinfo, sizeof(struct csi_quant_info) * src->quant_channel);
    dest->sess = src->sess;
    dest->is_const = src->is_const;
}

void csi_free_tensor(struct csi_tensor *tensor)
{
    free(tensor->qinfo);
    free(tensor);
}

void *csi_alloc_params(int params_size, struct csi_session *session)
{
    struct csi_params_base *params = calloc(1, params_size);
    if (session != NULL) {
        params->api = session->base_api;
        params->layout = session->base_layout;
        params->run_mode = session->base_run_mode;
    }
    return params;
}

void csi_free_params(void *params)
{
    free(params);
}

static float csi_uint8_to_float_base(uint8_t i, struct csi_tensor *t, int index)
{
    return ((float)i - t->qinfo[index].zero_point) * t->qinfo[index].scale;
}

static float csi_int8_to_float_base(int8_t i, struct csi_tensor *t, int index)
{
    return ((float)i - t->qinfo[index].zero_point) * t->qinfo[index].scale;
}

static float csi_int32_to_float_base(int32_t i, struct csi_tensor *t, int index)
{
    return (float)i  * t->qinfo[index].scale;
}

static uint8_t csi_float_to_uint8_base(float i, struct csi_tensor *t, int index)
{
    float ret = round(i / t->qinfo[index].scale) + t->qinfo[index].zero_point;
    if (ret > 255) {
        return 255;
    } else if (ret < 0) {
        return 0;
    } else {
        return ret;
    }
}

static int8_t csi_float_to_int8_base(float i, struct csi_tensor *t, int index)
{
    float ret = round(i / t->qinfo[index].scale) + t->qinfo[index].zero_point;
    if (ret > 127) {
        return 127;
    } else if (ret < -127) {
        return -127;
    } else {
        return ret;
    }
}

static void csi_uint8_to_float(struct csi_tensor *dest, struct csi_tensor *src,
                                 int n, int inner_size)
{
    uint8_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = csi_uint8_to_float_base(src_data[index], src, i);
        }
    }
}

static void csi_float_to_uint8(struct csi_tensor *dest, struct csi_tensor *src,
                                 int n, int inner_size)
{
    float *src_data = src->data;
    uint8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = csi_float_to_uint8_base(src_data[index], dest, i);
        }
    }
}

static void csi_int8_to_float(struct csi_tensor *dest, struct csi_tensor *src,
                                 int n, int inner_size)
{
    int8_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = csi_int8_to_float_base(src_data[index], src, i);
        }
    }
}

static void csi_float_to_int8(struct csi_tensor *dest, struct csi_tensor *src,
                                 int n, int inner_size)
{
    float *src_data = src->data;
    int8_t *dest_data = dest->data;
    int32_t q_size = dest->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = csi_float_to_int8_base(src_data[index], dest, i);
        }
    }
}

static void csi_int32_to_float(struct csi_tensor *dest, struct csi_tensor *src,
                                 int n, int inner_size)
{
    int32_t *src_data = src->data;
    float *dest_data = dest->data;
    int32_t q_size = src->quant_channel;
    for (int i = 0; i < q_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            int index = n * q_size * inner_size + i * inner_size + j;
            dest_data[index] = csi_int32_to_float_base(src_data[index], src, i);
        }
    }
}



int csi_tensor_data_convert_weight(struct csi_tensor *dest, struct csi_tensor *src){
    int size = csi_tensor_size(src);
    int inner_size = src->quant_channel == 0 ? size : size / src->quant_channel;
    if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_UINT8) {
        csi_uint8_to_float(dest, src, 0, inner_size);
    } else if (dest->dtype == CSINN_DTYPE_UINT8 && src->dtype == CSINN_DTYPE_FLOAT32) {
        csi_float_to_uint8(dest, src, 0, inner_size);
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT8) {
        csi_int8_to_float(dest, src, 0, inner_size);
    } else if (dest->dtype == CSINN_DTYPE_INT8 && src->dtype == CSINN_DTYPE_FLOAT32) {
        csi_float_to_int8(dest, src, 0, inner_size);
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT32) {
        csi_int32_to_float(dest, src, 0, inner_size);
    } else {
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}

int csi_tensor_data_convert_activation(struct csi_tensor *dest, struct csi_tensor *src){
    int size = csi_tensor_size(src);
    int32_t q_size = src->quant_channel !=0 ? src->quant_channel : dest->quant_channel;
    int inner_size = size / q_size / src->dim[0];
    if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_UINT8) {
        for (int n = 0; n < src->dim[0]; n++){
            csi_uint8_to_float(dest, src, n, inner_size);
        }
    } else if (dest->dtype == CSINN_DTYPE_UINT8 && src->dtype == CSINN_DTYPE_FLOAT32) {
        for (int n = 0; n < src->dim[0]; n++){
            csi_float_to_uint8(dest, src, n, inner_size);
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT8) {
        for (int n = 0; n < src->dim[0]; n++){
            csi_int8_to_float(dest, src, n, inner_size);
        }
    } else if (dest->dtype == CSINN_DTYPE_INT8 && src->dtype == CSINN_DTYPE_FLOAT32) {
        for (int n = 0; n < src->dim[0]; n++){
            csi_float_to_int8(dest, src, n, inner_size);
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_INT32) {
        for (int n = 0; n < src->dim[0]; n++){
            csi_int32_to_float(dest, src, n, inner_size);
        }
    } else if (dest->dtype == CSINN_DTYPE_FLOAT32 && src->dtype == CSINN_DTYPE_FLOAT32) {
        memcpy(dest->data, src->data, csi_tensor_size(src) * 4);
    } else {
        return CSINN_FALSE;
    }
    return CSINN_TRUE;
}


int csi_tensor_data_convert(struct csi_tensor *dest, struct csi_tensor *src)
{
    if (src->layout != dest->layout) return CSINN_FALSE;

    switch (src->layout)
    {
    case CSINN_LAYOUT_NULL:
        return CSINN_TRUE;
    case CSINN_LAYOUT_N:
    case CSINN_LAYOUT_NC:
    case CSINN_LAYOUT_NCW:
    case CSINN_LAYOUT_NCHW:
    case CSINN_LAYOUT_NCDHW:
        return csi_tensor_data_convert_activation(dest, src);
    case CSINN_LAYOUT_O:
    case CSINN_LAYOUT_OI:
    case CSINN_LAYOUT_OIW:
    case CSINN_LAYOUT_OIHW:
    case CSINN_LAYOUT_OIDHW:
        return csi_tensor_data_convert_weight(dest, src);
    case CSINN_LAYOUT_NHWC:
    case CSINN_LAYOUT_NDHWC:
    default:
        return CSINN_FALSE;
    }
}

#define BILLION    1000000000
uint64_t csi_get_timespec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)((uint64_t)ts.tv_nsec + (uint64_t)ts.tv_sec * BILLION);
}

