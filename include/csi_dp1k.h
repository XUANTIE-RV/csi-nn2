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

/* CSI-NN2 version 1.10.x */

#ifndef _CSI_NN_DP1K_H
#define _CSI_NN_DP1K_H
#include "csi_nn.h"
#include "csi_utils.h"

int csi_dp1k_add(
    struct csi_tensor *input0,
    struct csi_tensor *input1,
    struct csi_tensor *output,
    struct diso_params *params);

int csi_dp1k_avgpool2d(
    struct csi_tensor *input,
    struct csi_tensor *output,
    struct pool_params *params);

int csi_dp1k_concat(
    struct csi_tensor **input,
    struct csi_tensor *output,
    struct concat_params *params);

int csi_dp1k_conv2d(
    struct csi_tensor *input,
    struct csi_tensor *output,
    struct csi_tensor *kernel,
    struct csi_tensor *bias,
    struct conv2d_params *params);

int csi_dp1k_deconv2d(
    struct csi_tensor *input,
    struct csi_tensor *output,
    struct csi_tensor *kernel,
    struct csi_tensor *bias,
    struct conv2d_params *params);

int csi_dp1k_fullyconnected(
    struct csi_tensor *input,
    struct csi_tensor *output,
    struct csi_tensor *weights,
    struct csi_tensor *bias,
    struct fc_params *params);

int csi_dp1k_leaky_relu(
    struct csi_tensor *input,
    struct csi_tensor *output,
    struct relu_params *params);

int csi_dp1k_maxpool2d(
    struct csi_tensor *input,
    struct csi_tensor *output,
    struct pool_params *params);

int csi_dp1k_prelu(
    struct csi_tensor *input,
    struct csi_tensor *alpha,
    struct csi_tensor *output,
    struct prelu_params *params);

int csi_dp1k_mul(
    struct csi_tensor *input0,
    struct csi_tensor *input1,
    struct csi_tensor *output,
    struct diso_params *params);

int csi_dp1k_relu(
    struct csi_tensor *input,
    struct csi_tensor *output,
    struct relu_params *params);

int csi_dp1k_reshape(
    struct csi_tensor *input,
    struct csi_tensor *output,
    struct reshape_params *params);

int csi_dp1k_resize(
    struct csi_tensor *input,
    struct csi_tensor *output,
    struct resize_params *params);

int csi_dp1k_sigmoid(
    struct csi_tensor *input,
    struct csi_tensor *output,
    struct sigmoid_params *params);

int csi_dp1k_softmax(
    struct csi_tensor *input,
    struct csi_tensor *output,
    struct softmax_params *params);

int csi_dp1k_transpose(
    struct csi_tensor *input,
    struct csi_tensor *output,
    struct transpose_params *params);

int csi_dp1k_strided_slice(
    struct csi_tensor *input,
    struct csi_tensor *output,
    struct strided_slice_params *params);

void csi_dp1k_input(struct csi_tensor *tensor, struct csi_session *sess);
void csi_dp1000_session_init(struct csi_session *sess);
void csi_dp1000_session_setup(struct csi_session *sess);
void csi_dp1000_set_input_number(int number, struct csi_session *sess);
void csi_dp1000_set_output_number(int number, struct csi_session *sess);
void csi_dp1000_set_input(int index, struct csi_tensor *input, struct csi_session *sess);
void csi_dp1000_set_output(int index, struct csi_tensor *output, struct csi_session *sess);

typedef struct _csi_dp1000_target_data {
    char* nb_model_path;
} csi_dp1000_target_data;

struct csi_quant_info_dp1k
{
    int32_t zero_point;
    float scale;
    int32_t multiplier;
    int32_t shift;
    float min;
    float max;
};

#define MAX_DIM 8
struct csi_tensor_dp1k
{
    void *data;
    int32_t dtype;
    int32_t dim[MAX_DIM];
    int32_t dim_count;
    char *name;
    int32_t layout;
    int32_t quant_channel;
    struct csi_quant_info_dp1k *qinfo;
    struct csi_session_dp1k *sess;
} __attribute__((packed));

#define CSINN_MAX_INPUT 4
#define CSINN_MAX_OUTPUT 8
struct csi_session_dp1k {
    int32_t base_dtype;
    int32_t base_layout;
    int32_t base_api;
    int32_t input_num;
    int32_t output_num;
    struct csi_tensor_dp1k *input[CSINN_MAX_INPUT];
    struct csi_tensor_dp1k *output[CSINN_MAX_OUTPUT];
    void *td;
};

struct ScaleZp_dp1k
{
    float scale;
    int32_t zero_point;
};

struct conv2d_params_dp1k
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t group;
    int32_t stride_height;
    int32_t stride_width;
    int32_t pad_top;
    int32_t pad_left;
    int32_t pad_down;
    int32_t pad_right;
    int32_t dilation_height;
    int32_t dilation_width;
    char *name;
    struct ScaleZp_dp1k *scale_zp;
    struct
    {
        struct csi_tensor *kernel_tm;
        int32_t conv_mode;
    } conv_extra;
};

struct fc_params_dp1k
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    char *name;
    int32_t units;
};

struct pool_params_dp1k
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    char *name;
    int32_t pool_type;
    int32_t filter_height;
    int32_t filter_width;
    int32_t filter_depth;
    int32_t stride_height;
    int32_t stride_width;
    int32_t stride_depth;
    int32_t pad_top;
    int32_t pad_left;
    int32_t pad_down;
    int32_t pad_right;
    int32_t pad_front;
    int32_t pad_back;
};

struct sigmoid_params_dp1k
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    char *name;
};

struct relu_params_dp1k
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    char *name;

    /* n / alpha / threshold */
    float n;
    int32_t n_multiplier;
    int32_t n_shift;
};

struct prelu_params_dp1k
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    char *name;
    int32_t axis;
};

struct softmax_params_dp1k
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    char *name;
    int32_t axis;
};

struct diso_params_dp1k
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    char *name;
};

struct transpose_params_dp1k
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    char *name;
    int32_t *permute;
};

struct concat_params_dp1k
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    char *name;
    int32_t inputs_count;
    int32_t axis;
};

struct resize_params_dp1k
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    char *name;
    int32_t resize_mode;
    bool align_corners;
};

struct reshape_params_dp1k
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    char *name;
};

struct strided_slice_params_dp1k
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    char *name;
    int32_t *begin;
    int32_t *end;
    int32_t *stride;
    int32_t slice_count;
};

extern int csi_dp1000_add(
    struct csi_tensor_dp1k *input0,
    struct csi_tensor_dp1k *input1,
    struct csi_tensor_dp1k *output,
    struct diso_params_dp1k *params);

extern int csi_dp1000_avgpool2d(
    struct csi_tensor_dp1k *input,
    struct csi_tensor_dp1k *output,
    struct pool_params_dp1k *params);

extern int csi_dp1000_concat(
    struct csi_tensor_dp1k **input,
    struct csi_tensor_dp1k *output,
    struct concat_params_dp1k *params);

extern int csi_dp1000_conv2d(
    struct csi_tensor_dp1k *input,
    struct csi_tensor_dp1k *output,
    struct csi_tensor_dp1k *kernel,
    struct csi_tensor_dp1k *bias,
    struct conv2d_params_dp1k *params);

extern int csi_dp1000_deconv2d(
    struct csi_tensor_dp1k *input,
    struct csi_tensor_dp1k *output,
    struct csi_tensor_dp1k *kernel,
    struct csi_tensor_dp1k *bias,
    struct conv2d_params_dp1k *params);

extern int csi_dp1000_fullyconnected(
    struct csi_tensor_dp1k *input,
    struct csi_tensor_dp1k *output,
    struct csi_tensor_dp1k *weights,
    struct csi_tensor_dp1k *bias,
    struct fc_params_dp1k *params);

extern int csi_dp1000_leaky_relu(
    struct csi_tensor_dp1k *input,
    struct csi_tensor_dp1k *output,
    struct relu_params_dp1k *params);

extern int csi_dp1000_maxpool2d(
    struct csi_tensor_dp1k *input,
    struct csi_tensor_dp1k *output,
    struct pool_params_dp1k *params);

extern int csi_dp1000_prelu(
    struct csi_tensor_dp1k *input,
    struct csi_tensor_dp1k *alpha,
    struct csi_tensor_dp1k *output,
    struct prelu_params_dp1k *params);

extern int csi_dp1000_mul(
    struct csi_tensor_dp1k *input0,
    struct csi_tensor_dp1k *input1,
    struct csi_tensor_dp1k *output,
    struct diso_params_dp1k *params);

extern int csi_dp1000_relu(
    struct csi_tensor_dp1k *input,
    struct csi_tensor_dp1k *output,
    struct relu_params_dp1k *params);

extern int csi_dp1000_reshape(
    struct csi_tensor_dp1k *input,
    struct csi_tensor_dp1k *output,
    struct reshape_params_dp1k *params);

extern int csi_dp1000_resize(
    struct csi_tensor_dp1k *input,
    struct csi_tensor_dp1k *output,
    struct resize_params_dp1k *params);

extern int csi_dp1000_sigmoid(
    struct csi_tensor_dp1k *input,
    struct csi_tensor_dp1k *output,
    struct sigmoid_params_dp1k *params);

extern int csi_dp1000_softmax(
    struct csi_tensor_dp1k *input,
    struct csi_tensor_dp1k *output,
    struct softmax_params_dp1k *params);

extern int csi_dp1000_transpose(
    struct csi_tensor_dp1k *input,
    struct csi_tensor_dp1k *output,
    struct transpose_params_dp1k *params);

extern int csi_dp1000_strided_slice(
    struct csi_tensor_dp1k *input,
    struct csi_tensor_dp1k *output,
    struct strided_slice_params_dp1k *params);

extern void csi_dp1000_input(struct csi_tensor_dp1k *input);

extern void csi_dp1000_model_setup(struct csi_session_dp1k *sess);

#endif
