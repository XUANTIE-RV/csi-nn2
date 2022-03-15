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

#include "test_utils.h"
#include "csi_nn.h"
#include "math_snr.h"

void verify_conv2d_q7(void *input_data,
                        void *kernel_data,
                        void *bias_data,
                        void *ref_data,
                        uint16_t batch,
                        uint16_t in_h,
                        uint16_t in_w,
                        uint16_t in_c,
                        uint16_t out_h,
                        uint16_t out_w,
                        uint16_t out_c,
                        uint16_t kernel_h,
                        uint16_t kernel_w,
                        uint16_t stride_h,
                        uint16_t stride_w,
                        uint16_t pad_x,
                        uint16_t pad_y,
                        uint16_t bias_shift,
                        uint16_t out_shift,
                        float difference)
{
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size, out_size, kernel_size = 0, bias_size = 0;

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    input->dim[0] = batch;  // N
    input->dim[1] = in_h;   // H
    input->dim[2] = in_w;   // W
    input->dim[3] = in_c;   // C
    input->dim_count = 4;
    input->dtype = CSINN_DTYPE_INT8;
    input->name = "input";
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];

    struct csi_tensor *kernel = csi_alloc_tensor(NULL);
    kernel->dim[0] = out_c;     // O
    kernel->dim[1] = in_c;      // I
    kernel->dim[2] = kernel_h;  // H
    kernel->dim[3] = kernel_w;  // W
    kernel->dim_count = 4;
    kernel->dtype = CSINN_DTYPE_INT8;
    kernel->name = "kernel";
    kernel_size = kernel->dim[0] * kernel->dim[1] * kernel->dim[2] * kernel->dim[3];

    struct csi_tensor *bias = csi_alloc_tensor(NULL);
    bias->dim[0] = out_c;   // O
    bias->dim_count = 1;
    bias->dtype = CSINN_DTYPE_INT8;
    bias->name = "bias";
    bias_size = bias->dim[0];
    bias->qinfo->shift = bias_shift;

    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = out_h;
    output->dim[2] = out_w;
    output->dim[3] = out_c;
    output->dim_count = 4;
    output->dtype = CSINN_DTYPE_INT8;
    output->name = "output";
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    output->qinfo->shift = out_shift;

    struct conv2d_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NHWC;
    params.base.run_mode = CSINN_RM_LAYER;
    params.stride_height = stride_h;
    params.stride_width  = stride_w;
    params.pad_left   = pad_x;
    params.pad_right  = pad_x;
    params.pad_top    = pad_y;
    params.pad_down   = pad_y;
    params.dilation_width  = 0;
    params.dilation_height = 0;
    params.group      = 1;
    params.conv_extra.kernel_tm = NULL;
    params.conv_extra.conv_mode = CSINN_DIRECT;

    input->data      = (uint8_t *)input_data;
    kernel->data     = (uint8_t *)kernel_data;
    bias->data       = (uint8_t *)bias_data;
    reference->data  = (uint8_t *)ref_data;
    // uint8_t *output_tmp = (uint8_t *)malloc(out_size * sizeof(uint8_t));
    uint8_t output_tmp[out_size];
    output->data     = output_tmp;

    if (csi_conv2d_init(input, output, kernel, bias, &params) == CSINN_TRUE) {
        csi_conv2d(input, output, kernel, bias, &params);
    }
    result_verify_q7(reference->data, output->data, input->data, difference, out_size, false);
    // free(output_tmp);
    free(input);
    free(kernel);
    free(bias);
    free(output);
    free(reference);
}
