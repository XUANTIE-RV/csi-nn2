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

#include "sgemm.h"

static void conv1x1s1_sgemm_transform_kernel(struct csi_tensor *kernel)
{
    float *kernel_data = (float *)kernel->data;
    int m = kernel->dim[0];   // out_ch
    int k = kernel->dim[1];   // in_ch ( kernel->dim[2] = kernel->dim[3] = 1)

    float* pa_reorder = (float *)malloc(m * k * sizeof(float));
    reorder_a(kernel_data, pa_reorder, m, k, k);
    memcpy(kernel_data, pa_reorder, m * k * sizeof(float));
    free(pa_reorder);
}

static int conv1x1s1_sgemm(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct csi_tensor *kernel,
                           struct csi_tensor *bias,
                           struct conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    int32_t batch = input->dim[0];      // assert(batch == 1);
    int32_t in_channel = input->dim[1];
    int32_t out_channel = kernel->dim[0];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int32_t m = out_channel;
    int32_t k = in_channel;
    int32_t n = out_h * out_w;

    float* pb_reorder = (float *)malloc(k * n * sizeof(float));
    const float *pa = kernel_data;

    for(int i = 0; i < batch; i++) {
        // pack
        reorder_b(input_data + i * k * n, pb_reorder, k, n, n);

        // GEMM
        const float *pb = pb_reorder;
        float *pc = output_data + i * m * n;

        sgemm_kernel_f32(pc, pa, pb, m, k, n, n, bias_data);
    }
    free(pb_reorder);
    return CSINN_TRUE;
}
