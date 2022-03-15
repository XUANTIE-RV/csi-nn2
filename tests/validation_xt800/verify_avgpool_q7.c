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


void verify_avgpool2d_q7(void *input_data,
                       void *output_data,
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
                       uint16_t out_lshift,
                       float difference)

{
    struct csi_tensor *reference = csi_alloc_tensor(NULL);
    int in_size, out_size;

    struct csi_tensor *input = csi_alloc_tensor(NULL);
    input->dim[0] = batch;  // N
    input->dim[1] = in_h;   // H
    input->dim[2] = in_w;   // W
    input->dim[3] = in_c;   // C
    input->dim_count = 4;
    input->dtype = CSINN_DTYPE_INT8;
    input->name = "input";
    in_size = input->dim[0] * input->dim[1] * input->dim[2] * input->dim[3];

    struct csi_tensor *output = csi_alloc_tensor(NULL);
    output->dim[0] = input->dim[0];
    output->dim[1] = out_h;
    output->dim[2] = out_w;
    output->dim[3] = out_c;
    output->dim_count = 4;
    output->dtype = CSINN_DTYPE_INT8;
    output->name = "output";
    out_size = output->dim[0] * output->dim[1] * output->dim[2] * output->dim[3];
    output->qinfo->shift = out_lshift;

    struct pool_params params;
    params.base.api = CSINN_API;
    params.base.name = "params";
    params.base.layout = CSINN_LAYOUT_NCHW;
    params.base.run_mode = CSINN_RM_LAYER;
    params.ceil_mode = 0;
    params.stride_height = stride_h;
    params.stride_width  = stride_w;
    params.filter_height = kernel_h;
    params.filter_width  = kernel_w;
    params.pad_left  = pad_x;
    params.pad_right = pad_x;
    params.pad_top   = pad_y;
    params.pad_down  = pad_y;

    input->data      = (uint8_t *)input_data;
    reference->data  = (uint8_t *)output_data;
    uint8_t *output_tmp = (uint8_t *)malloc(out_size * sizeof(uint8_t));
    output->data = output_tmp;

    if (csi_avgpool2d_init(input, output, &params) == CSINN_TRUE) {
        csi_avgpool2d(input, output, &params);
    }

    result_verify_q7(reference->data, output->data, input->data, difference, out_size, false);
    free(output_tmp);
    free(input);
    free(output);
    free(reference);
}
