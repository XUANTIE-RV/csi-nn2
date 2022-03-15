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

#include "csi_thead_rvv.h"

void csi_nn_rvv_conv1x1s1_gemm_transform_kernel_int8(struct csi_tensor *kernel,
                                                     struct conv2d_params *params)
{
    int8_t *kernel_data = (int8_t *)kernel->data;
    int group = params->group;

    int m = kernel->dim[0] / group;  // out_ch
    int k = kernel->dim[1];          // in_ch ( kernel->dim[2] = kernel->dim[3] = 1)
    int k4 = (k % 4 != 0) ? ((k / 4 + 1) * 4) : k;

    params->conv_extra.kernel_tm->data = (int8_t *)csi_mem_alloc(group * m * k4 * sizeof(int8_t));
    int8_t *pa_reorder = (int8_t *)params->conv_extra.kernel_tm->data;

    for (int g = 0; g < group; g++) {
        csi_nn_rvv_reorder_kernel_n8_int8(kernel_data + g * m * k, pa_reorder + g * m * k4, m, k,
                                          k);
    }
    // FIXME: free params->conv_extra.kernel_tm->data
    // memcpy(kernel_data, pa_reorder, group * m * k * sizeof(int8_t));
    // csi_mem_free(pa_reorder);
}

int csi_nn_rvv_conv1x1s1_gemm_int8(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *kernel, struct csi_tensor *bias,
                                   struct conv2d_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)params->conv_extra.kernel_tm->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];  // assert(batch == 1);
    int32_t in_ch = input->dim[1];
    int32_t out_ch = kernel->dim[0];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int32_t m = out_ch / group;
    int32_t k = in_ch / group;
    int32_t n = out_h * out_w;
    int32_t k4 = (k % 4 != 0) ? ((k / 4 + 1) * 4) : k;

    int8_t *pb_reorder = (int8_t *)csi_mem_alloc(k4 * n * sizeof(int8_t));
    int32_t *multiplier = (int32_t *)csi_mem_alloc(m * sizeof(int32_t));
    int32_t *shift = (int32_t *)csi_mem_alloc(m * sizeof(int32_t));

    int j = 0;
    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            int8_t *pa = kernel_data + g * m * k4;
            int8_t *pb = pb_reorder;
            int8_t *pc = output_data;

            if (kernel->quant_channel > 1) {
                for (int c = 0; c < m; c++, j++) {
                    multiplier[c] = kernel->qinfo[j].multiplier;
                    shift[c] = kernel->qinfo[j].shift;
                }
            } else if (kernel->quant_channel == 1) {
                for (int c = 0; c < m; c++) {
                    multiplier[c] = kernel->qinfo[0].multiplier;
                    shift[c] = kernel->qinfo[0].shift;
                }
            }

            // pack
            csi_nn_rvv_reorder_input_z8_int8(input_data, pb, k, n, n);
            // GEMM
            csi_nn_rvv_gemm_8x8_int8(pc, pa, pb, m, k4, n, n, bias_data + g * m,
                                     output->qinfo->zero_point, multiplier, shift);

            input_data += k * n;
            output_data += m * n;
        }
    }
    csi_mem_free(pb_reorder);
    csi_mem_free(multiplier);
    csi_mem_free(shift);
    return CSINN_TRUE;
}
