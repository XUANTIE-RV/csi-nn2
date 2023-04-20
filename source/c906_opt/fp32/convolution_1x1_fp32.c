/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
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

#include "shl_c906.h"

void shl_c906_conv1x1s1_sgemm_transform_kernel(struct csinn_tensor *kernel,
                                               struct csinn_conv2d_params *params)
{
    float *kernel_data = (float *)kernel->data;
    int group = params->group;

    int m = kernel->dim[0] / group;  // out_ch / group
    int k = kernel->dim[1];          // in_ch ( kernel->dim[2] = kernel->dim[3] = 1)

    float *pa_reorder = (float *)shl_mem_alloc(group * m * k * sizeof(float));
    for (int g = 0; g < group; g++) {
        shl_c906_reorder_kernel(kernel_data + g * m * k, pa_reorder + g * m * k, m, k, k);
    }
    memcpy(kernel_data, pa_reorder, group * m * k * sizeof(float));
    shl_mem_free(pa_reorder);
}

static int shl_c906_conv1x1s1_sgemm_base(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params, bool fuse_relu)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];  // assert(batch == 1);
    int32_t in_ch = input->dim[1];
    int32_t out_ch = kernel->dim[0];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim_count == 4 ? output->dim[3] : 1;  // adapt conv1d1s1

    int32_t m = out_ch / group;
    int32_t k = in_ch / group;
    int32_t n = out_h * out_w;

    float *pb_reorder = (float *)shl_mem_alloc(k * n * sizeof(float));

    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            float *pa = kernel_data + g * m * k;
            float *pb = pb_reorder;
            float *pc = output_data;
            // pack
            shl_c906_reorder_input_1(input_data, pb, k, n, n);
            // GEMM
            shl_c906_sgemm_kernel_f32(pc, pa, pb, m, k, n, n, bias_data + g * m, fuse_relu);
            input_data += k * n;
            output_data += m * n;
        }
    }
    shl_mem_free(pb_reorder);
    return CSINN_TRUE;
}

int shl_c906_conv1x1s1_sgemm(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_tensor *kernel, struct csinn_tensor *bias,
                             struct csinn_conv2d_params *params)
{
    bool fuse_relu = 0;
    return shl_c906_conv1x1s1_sgemm_base(input, output, kernel, bias, params, fuse_relu);
}

int shl_c906_conv1x1s1_sgemm_fuse_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params)
{
    bool fuse_relu = 1;
    return shl_c906_conv1x1s1_sgemm_base(input, output, kernel, bias, params, fuse_relu);
}