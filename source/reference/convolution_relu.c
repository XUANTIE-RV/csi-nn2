/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
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

/* CSI-NN2 version 1.12.x */

#include "csi_ref.h"

int csi_ref_conv2d_relu_f32(struct csi_tensor *input, struct csi_tensor *output,
                            struct csi_tensor *kernel, struct csi_tensor *bias,
                            struct conv2d_params *params)
{
    csi_ref_conv2d_f32(input, output, kernel, bias, params);
    float *data = output->data;
    int size = csi_tensor_size(output);
    for (int i = 0; i < size; i++) {
        data[i] = data[i] > 0 ? data[i] : 0;
    }
    return CSINN_TRUE;
}

int csi_ref_conv2d_relu_quant(struct csi_tensor *input, struct csi_tensor *output,
                              struct csi_tensor *kernel, struct csi_tensor *bias,
                              struct conv2d_params *params)
{
    csi_ref_conv2d_quant(input, output, kernel, bias, params);
    struct relu_params *rp = csi_mem_alloc(sizeof(struct relu_params));
    memcpy(&(rp->base), &(params->base), sizeof(struct csi_params_base));
    csi_relu_init(output, output, rp);
    csi_relu(output, output, rp);
    return CSINN_TRUE;
}

int csi_ref_depthwise_conv2d_relu_f32(struct csi_tensor *input, struct csi_tensor *output,
                                      struct csi_tensor *kernel, struct csi_tensor *bias,
                                      struct conv2d_params *params)
{
    csi_ref_depthwise_conv2d_f32(input, output, kernel, bias, params);
    float *data = output->data;
    int size = csi_tensor_size(output);
    for (int i = 0; i < size; i++) {
        data[i] = data[i] > 0 ? data[i] : 0;
    }
    return CSINN_TRUE;
}

int csi_ref_depthwise_conv2d_relu_quant(struct csi_tensor *input, struct csi_tensor *output,
                                        struct csi_tensor *kernel, struct csi_tensor *bias,
                                        struct conv2d_params *params)
{
    csi_ref_depthwise_conv2d_quant(input, output, kernel, bias, params);
    struct relu_params *rp = csi_mem_alloc(sizeof(struct relu_params));
    memcpy(&(rp->base), &(params->base), sizeof(struct csi_params_base));
    csi_relu_init(output, output, rp);
    csi_relu(output, output, rp);
    return CSINN_TRUE;
}

int csi_ref_group_conv2d_relu_quant(struct csi_tensor *input, struct csi_tensor *output,
                                    struct csi_tensor *kernel, struct csi_tensor *bias,
                                    struct conv2d_params *params)
{
    csi_ref_group_conv2d_quant(input, output, kernel, bias, params);
    struct relu_params *rp = csi_mem_alloc(sizeof(struct relu_params));
    memcpy(&(rp->base), &(params->base), sizeof(struct csi_params_base));
    csi_relu_init(output, output, rp);
    csi_relu(output, output, rp);

    return CSINN_TRUE;
}
