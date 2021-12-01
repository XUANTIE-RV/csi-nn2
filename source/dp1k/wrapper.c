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

static struct csi_session_dp1k *csi_session_dp1k_convert(struct csi_session *sess)
{
    struct csi_session_dp1k *ret = calloc(1, sizeof(struct csi_session_dp1k));
    ret->base_api = sess->base_api;
    ret->base_dtype = sess->base_dtype;
    ret->base_layout = sess->base_layout;
    ret->input_num = sess->input_num;
    ret->output_num = sess->output_num;
    return ret;
}

static struct csi_tensor_dp1k *csi_tensor_dp1k_convert_const(struct csi_tensor *tensor)
{
    struct csi_tensor_dp1k *ret = calloc(1, sizeof(struct csi_tensor_dp1k));
    ret->data = tensor->data;
    ret->dtype = tensor->dtype;
    ret->dim[0] = tensor->dim[0];
    ret->dim[1] = tensor->dim[1];
    ret->dim[2] = tensor->dim[2];
    ret->dim[3] = tensor->dim[3];
    ret->dim[4] = tensor->dim[4];
    ret->dim[5] = tensor->dim[5];
    ret->dim[6] = tensor->dim[6];
    ret->dim[7] = tensor->dim[7];
    ret->dim_count = tensor->dim_count;
    ret->name = tensor->name;
    ret->layout = tensor->layout;
    ret->quant_channel = tensor->quant_channel;
    ret->qinfo = (struct csi_quant_info_dp1k *)tensor->qinfo;
    ret->sess = tensor->sess->td;

    return ret;
}

static struct csi_tensor_dp1k *csi_tensor_dp1k_convert(struct csi_tensor *tensor)
{
    if (tensor->data != NULL && tensor->is_const != 1) {
        return tensor->data;
    }
    struct csi_tensor_dp1k *ret = calloc(1, sizeof(struct csi_tensor_dp1k));
    ret->data = tensor->data;
    ret->dtype = tensor->dtype;
    ret->dim[0] = tensor->dim[0];
    ret->dim[1] = tensor->dim[1];
    ret->dim[2] = tensor->dim[2];
    ret->dim[3] = tensor->dim[3];
    ret->dim[4] = tensor->dim[4];
    ret->dim[5] = tensor->dim[5];
    ret->dim[6] = tensor->dim[6];
    ret->dim[7] = tensor->dim[7];
    ret->dim_count = tensor->dim_count;
    ret->name = tensor->name;
    ret->layout = tensor->layout;
    ret->quant_channel = tensor->quant_channel;
    ret->qinfo = (struct csi_quant_info_dp1k *)tensor->qinfo;
    ret->sess = tensor->sess->td;
    tensor->data = ret;

    return ret;
}

void csi_dp1k_input(struct csi_tensor *tensor, struct csi_session *sess) {
    struct csi_tensor_dp1k *dptensor = csi_tensor_dp1k_convert(tensor);
    csi_dp1000_input(dptensor);
    dptensor->sess = sess->td;
}

void csi_dp1000_session_init(struct csi_session *sess) {
    csi_dp1000_target_data *target_data = calloc(sizeof(csi_dp1000_target_data), 1);
    target_data->nb_model_path = "./nb_model";
    struct csi_session_dp1k *dpsess = csi_session_dp1k_convert(sess);
    dpsess->td = target_data;
    sess->td = dpsess;
}

void csi_dp1000_session_setup(struct csi_session *sess) {
    struct csi_session_dp1k *dpsess = sess->td;
    csi_dp1000_model_setup(dpsess);
}

void csi_dp1000_set_input_number(int number, struct csi_session *sess) {
    struct csi_session_dp1k *dpsess = sess->td;
    dpsess->input_num = number;
}

void csi_dp1000_set_output_number(int number, struct csi_session *sess) {
    struct csi_session_dp1k *dpsess = sess->td;
    dpsess->output_num = number;
}

void csi_dp1000_set_input(int index, struct csi_tensor *input, struct csi_session *sess) {
    struct csi_session_dp1k *dpsess = sess->td;
    struct csi_tensor_dp1k *dptensor = csi_tensor_dp1k_convert(input);
    dpsess->input[index] = dptensor;
}

void csi_dp1000_set_output(int index, struct csi_tensor *output, struct csi_session *sess) {
    struct csi_session_dp1k *dpsess = sess->td;
    struct csi_tensor_dp1k *dptensor = csi_tensor_dp1k_convert(output);
    dpsess->output[index] = dptensor;
}

int csi_dp1k_add(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params)
{
    struct csi_tensor_dp1k *dp_input0 = csi_tensor_dp1k_convert(input0);
    struct csi_tensor_dp1k *dp_input1 = csi_tensor_dp1k_convert(input1);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct diso_params_dp1k *dpp = calloc(1, sizeof(struct diso_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    csi_dp1000_add(dp_input0, dp_input1, dp_output, dpp);
    return CSINN_TRUE;
}

int csi_dp1k_avgpool2d(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct pool_params *params)
{
    struct csi_tensor_dp1k *dp_input = csi_tensor_dp1k_convert(input);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct pool_params_dp1k *dpp = calloc(1, sizeof(struct pool_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    dpp->filter_depth = params->filter_depth;
    dpp->filter_height = params->filter_height;
    dpp->filter_width = params->filter_width;
    dpp->pad_back = params->pad_back;
    dpp->pad_down = params->pad_down;
    dpp->pad_front = params->pad_front;
    dpp->pad_left = params->pad_left;
    dpp->pad_right = params->pad_right;
    dpp->pad_top = params->pad_top;
    dpp->pool_type = params->pool_type;
    dpp->stride_depth = params->stride_depth;
    dpp->stride_height = params->stride_height;
    dpp->stride_width = params->stride_width;
    csi_dp1000_avgpool2d(dp_input, dp_output, dpp);
    return CSINN_TRUE;
}

int csi_dp1k_concat(struct csi_tensor **input,
                    struct csi_tensor *output,
                    struct concat_params *params)
{
    struct csi_tensor_dp1k **dp_input;
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct concat_params_dp1k *dpp = calloc(1, sizeof(struct concat_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    dpp->axis = params->axis;
    dpp->inputs_count = params->inputs_count;

    dp_input = calloc(1, dpp->inputs_count * sizeof(struct csi_tensor_dp1k *));

    for (int i = 0; i < dpp->inputs_count; i++) {
        dp_input[i] = csi_tensor_dp1k_convert(input[i]);
    }

    csi_dp1000_concat(dp_input, dp_output, dpp);

    return CSINN_TRUE; 
}

int csi_dp1k_conv2d(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct csi_tensor *kernel,
                    struct csi_tensor *bias,
                    struct conv2d_params *params)
{
    struct csi_tensor_dp1k *dp_input = csi_tensor_dp1k_convert(input);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct csi_tensor_dp1k *dp_kernel = csi_tensor_dp1k_convert_const(kernel);
    struct csi_tensor_dp1k *dp_bias = csi_tensor_dp1k_convert_const(bias);
    struct conv2d_params_dp1k *dpp = calloc(1, sizeof(struct conv2d_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    dpp->pad_down = params->pad_down;
    dpp->pad_left = params->pad_left;
    dpp->pad_right = params->pad_right;
    dpp->pad_top = params->pad_top;
    dpp->stride_height = params->stride_height;
    dpp->stride_width = params->stride_width;
    dpp->dilation_width = params->dilation_width;
    dpp->dilation_height = params->dilation_height;
    dpp->group = params->group;
    csi_dp1000_conv2d(dp_input, dp_output, dp_kernel, dp_bias, dpp);
    return CSINN_TRUE;
}

int csi_dp1k_deconv2d(struct csi_tensor *input,
                      struct csi_tensor *output,
                      struct csi_tensor *kernel,
                      struct csi_tensor *bias,
                      struct conv2d_params *params)
{
    struct csi_tensor_dp1k *dp_input = csi_tensor_dp1k_convert(input);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct csi_tensor_dp1k *dp_kernel = csi_tensor_dp1k_convert_const(kernel);
    struct csi_tensor_dp1k *dp_bias = csi_tensor_dp1k_convert_const(bias);
    struct conv2d_params_dp1k *dpp = calloc(1, sizeof(struct conv2d_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    dpp->pad_down = params->pad_down;
    dpp->pad_left = params->pad_left;
    dpp->pad_right = params->pad_right;
    dpp->pad_top = params->pad_top;
    dpp->stride_height = params->stride_height;
    dpp->stride_width = params->stride_width;
    dpp->dilation_width = params->dilation_width;
    dpp->dilation_height = params->dilation_height;
    dpp->group = params->group;
    csi_dp1000_deconv2d(dp_input, dp_output, dp_kernel, dp_bias, dpp);
    return CSINN_TRUE;
}

int csi_dp1k_fullyconnected(struct csi_tensor *input,
                            struct csi_tensor *output,
                            struct csi_tensor *weights,
                            struct csi_tensor *bias,
                            struct fc_params *params)
{
    struct csi_tensor_dp1k *dp_input = csi_tensor_dp1k_convert(input);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct csi_tensor_dp1k *dp_kernel = csi_tensor_dp1k_convert_const(weights);
    struct csi_tensor_dp1k *dp_bias = csi_tensor_dp1k_convert_const(bias);
    struct fc_params_dp1k *dpp = calloc(1, sizeof(struct fc_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    csi_dp1000_fullyconnected(dp_input, dp_output, dp_kernel, dp_bias, dpp);
    return CSINN_TRUE;
}

int csi_dp1k_leaky_relu(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct relu_params *params)
{
    struct csi_tensor_dp1k *dp_input = csi_tensor_dp1k_convert(input);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct relu_params_dp1k *dpp = calloc(1, sizeof(struct relu_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    dpp->n = params->n;
    dpp->n_multiplier = params->n_multiplier;
    dpp->n_shift = params->n_shift;
    csi_dp1000_leaky_relu(dp_input, dp_output, dpp);
    return CSINN_TRUE;
}

int csi_dp1k_maxpool(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct pool_params *params)
{
    struct csi_tensor_dp1k *dp_input = csi_tensor_dp1k_convert(input);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct pool_params_dp1k *dpp = calloc(1, sizeof(struct pool_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    dpp->filter_depth = params->filter_depth;
    dpp->filter_height = params->filter_height;
    dpp->filter_width = params->filter_width;
    dpp->pad_back = params->pad_back;
    dpp->pad_down = params->pad_down;
    dpp->pad_front = params->pad_front;
    dpp->pad_left = params->pad_left;
    dpp->pad_right = params->pad_right;
    dpp->pad_top = params->pad_top;
    dpp->pool_type = params->pool_type;
    dpp->stride_depth = params->stride_depth;
    dpp->stride_height = params->stride_height;
    dpp->stride_width = params->stride_width;
    csi_dp1000_maxpool(dp_input, dp_output, dpp);
    return CSINN_TRUE;
}

int csi_dp1k_prelu(struct csi_tensor *input,
                   struct csi_tensor *alpha,
                   struct csi_tensor *output,
                   struct prelu_params *params)
{
    struct csi_tensor_dp1k *dp_input = csi_tensor_dp1k_convert(input);
    struct csi_tensor_dp1k *dp_alpha = csi_tensor_dp1k_convert_const(alpha);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct prelu_params_dp1k *dpp = calloc(1, sizeof(struct prelu_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    dpp->name = params->base.name;
    dpp->axis = params->axis;
    csi_dp1000_prelu(dp_input, dp_alpha, dp_output, dpp);
    return CSINN_TRUE;
}

int csi_dp1k_mul(struct csi_tensor *input0,
                 struct csi_tensor *input1,
                 struct csi_tensor *output,
                 struct diso_params *params)
{
    struct csi_tensor_dp1k *dp_input0 = csi_tensor_dp1k_convert(input0);
    struct csi_tensor_dp1k *dp_input1 = csi_tensor_dp1k_convert(input1);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct diso_params_dp1k *dpp = calloc(1, sizeof(struct diso_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    csi_dp1000_mul(dp_input0, dp_input1, dp_output, dpp);
    return CSINN_TRUE;
}

int csi_dp1k_relu(struct csi_tensor *input,
                  struct csi_tensor *output,
                  struct relu_params *params)
{
    struct csi_tensor_dp1k *dp_input = csi_tensor_dp1k_convert(input);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct relu_params_dp1k *dpp = calloc(1, sizeof(struct relu_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    dpp->n = params->n;
    dpp->n_multiplier = params->n_multiplier;
    dpp->n_shift = params->n_shift;
    csi_dp1000_relu(dp_input, dp_output, dpp);
    return CSINN_TRUE;
}

int csi_dp1k_reshape(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct reshape_params *params)
{
    struct csi_tensor_dp1k *dp_input = csi_tensor_dp1k_convert(input);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct reshape_params_dp1k *dpp = calloc(1, sizeof(struct reshape_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    csi_dp1000_reshape(dp_input, dp_output, dpp);
    return CSINN_TRUE;
}

int csi_dp1k_resize(struct csi_tensor *input,
                    struct csi_tensor *output,
                    struct resize_params *params)
{
    struct csi_tensor_dp1k *dp_input = csi_tensor_dp1k_convert(input);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct resize_params_dp1k *dpp = calloc(1, sizeof(struct resize_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    dpp->align_corners = params->align_corners;
    dpp->resize_mode = params->resize_mode;
    csi_dp1000_resize(dp_input, dp_output, dpp);
    return CSINN_TRUE;
}

int csi_dp1k_sigmoid(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct sigmoid_params *params)
{
    struct csi_tensor_dp1k *dp_input = csi_tensor_dp1k_convert(input);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct sigmoid_params_dp1k *dpp = calloc(1, sizeof(struct sigmoid_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    csi_dp1000_sigmoid(dp_input, dp_output, dpp);
    return CSINN_TRUE;
}

int csi_dp1k_softmax(struct csi_tensor *input,
                     struct csi_tensor *output,
                     struct softmax_params *params)
{
    struct csi_tensor_dp1k *dp_input = csi_tensor_dp1k_convert(input);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct softmax_params_dp1k *dpp = calloc(1, sizeof(struct softmax_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    dpp->axis = params->axis;
    csi_dp1000_softmax(dp_input, dp_output, dpp);
    return CSINN_TRUE;
}

int csi_dp1k_transpose(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct transpose_params *params)
{
    struct csi_tensor_dp1k *dp_input = csi_tensor_dp1k_convert(input);
    struct csi_tensor_dp1k *dp_output = csi_tensor_dp1k_convert(output);
    struct transpose_params_dp1k *dpp = calloc(1, sizeof(struct transpose_params_dp1k));
    dpp->api = params->base.api;
    dpp->bc = params->base.bc;
    dpp->layout = params->base.layout;
    dpp->permute = params->permute;
    csi_dp1000_transpose(dp_input, dp_output, dpp);
    return CSINN_TRUE;
}
