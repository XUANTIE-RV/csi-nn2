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

/* SHL version 2.1.x */

#include <shl_ref.h>

int main(int argc, char **argv)
{
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_run_mode = CSINN_RM_LAYER;
    struct csinn_tensor *input = csinn_alloc_tensor(sess);
    struct csinn_tensor *output = csinn_alloc_tensor(sess);
    struct csinn_tensor *kernel = csinn_alloc_tensor(sess);
    struct csinn_tensor *bias = csinn_alloc_tensor(sess);
    struct csinn_conv2d_params *params =
        csinn_alloc_params(sizeof(struct csinn_conv2d_params), sess);

    input->dim[0] = 1;    // batch
    input->dim[1] = 512;  // in_channel
    input->dim[2] = 14;   // height
    input->dim[3] = 14;   // width
    kernel->dim[0] = 512;
    kernel->dim[1] = 512;
    kernel->dim[2] = 1;
    kernel->dim[3] = 1;
    bias->dim[0] = 512;
    output->dim[0] = 1;    // batch
    output->dim[1] = 512;  // out_channel
    output->dim[2] = 14;   // height
    output->dim[3] = 14;   // width

    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 0;
    params->pad_right = 0;
    params->pad_top = 0;
    params->pad_down = 0;
    params->dilation_width = 0;
    params->dilation_height = 0;
    params->base.layout = CSINN_LAYOUT_NCHW;
    params->group = 1;
    params->conv_extra.fuse_zp2bias = false;

    input->dim_count = 4;
    input->layout = CSINN_LAYOUT_NCHW;
    input->is_const = 0;
    input->quant_channel = 1;

    kernel->dim_count = 4;
    kernel->layout = CSINN_LAYOUT_OIHW;
    kernel->is_const = 1;
    kernel->quant_channel = 1;

    bias->dim_count = 1;
    bias->layout = CSINN_LAYOUT_O;
    bias->is_const = 1;
    bias->quant_channel = 1;

    output->dim_count = 4;
    output->layout = CSINN_LAYOUT_NCHW;
    output->is_const = 0;
    output->quant_channel = 1;

    input->dtype = CSINN_DTYPE_FLOAT32;
    kernel->dtype = CSINN_DTYPE_FLOAT32;
    bias->dtype = CSINN_DTYPE_FLOAT32;
    output->dtype = CSINN_DTYPE_FLOAT32;

    params->base.api = CSINN_C906;

    /* alloc random input */
    input->data = malloc(14 * 14 * 512 * 4);
    /* alloc random kernel */
    kernel->data = malloc(512 * 512 * 1 * 1 * 4);
    /* alloc random bias */
    bias->data = malloc(512 * 4);
    /* alloc random output */
    output->data = malloc(14 * 14 * 512 * 4);

    csinn_conv2d_init(input, output, kernel, bias, params);

    uint64_t start_time, end_time;
    start_time = shl_get_timespec();
    csinn_conv2d(input, output, kernel, bias, params);
    end_time = shl_get_timespec();
    printf("Run graph execution time: %.5fms, FPS=%.2f\n",
           ((float)(end_time - start_time)) / 1000000,
           1000000000.0 / ((float)(end_time - start_time)));

    return 0;
}
