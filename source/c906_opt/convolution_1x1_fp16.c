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

#include "shl_c906.h"

void shl_c906_conv1x1s1_sgemm_transform_kernel_fp16(struct csinn_tensor *kernel,
                                                    struct csinn_conv2d_params *params)
{
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    int group = params->group;

    int m = kernel->dim[0] / group;  // out_ch
    int k = kernel->dim[1];          // in_ch ( kernel->dim[2] = kernel->dim[3] = 1)

    __fp16 *pa_reorder = (__fp16 *)shl_mem_alloc(group * m * k * sizeof(__fp16));
    for (int g = 0; g < group; g++) {
        shl_c906_reorder_kernel_fp16(kernel_data + g * m * k, pa_reorder + g * m * k, m, k, k);
    }
    memcpy(kernel_data, pa_reorder, group * m * k * sizeof(__fp16));

    shl_mem_free(pa_reorder);
}

int shl_c906_conv1x1s1_sgemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];  // assert(batch == 1);
    int32_t in_ch = input->dim[1];
    int32_t out_ch = kernel->dim[0];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim_count == 4 ? output->dim[3] : 1;  // adapt conv1d1s1

    int32_t m = out_ch / group;
    int32_t k = in_ch / group;
    int32_t n = out_h * out_w;

    __fp16 *pb_reorder = (__fp16 *)shl_mem_alloc(k * n * sizeof(__fp16));

    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            __fp16 *pa = kernel_data + g * m * k;
            __fp16 *pb = pb_reorder;
            __fp16 *pc = output_data;
            // pack
            shl_rvv_reorder_input_z16_fp16(input_data, pb, k, n, n);
            // shl_c906_reorder_input_fp16_1(input_data, pb, k, n, n);
            // GEMM
            shl_rvv_gemm_8x16_fp16(pc, pa, pb, bias_data + g * m, m, k, n, n);
            // shl_c906_sgemm_kernel_fp16(pc, pa, pb, m, k, n, n, bias_data + g * m);

            input_data += k * n;
            output_data += m * n;
        }
    }
    shl_mem_free(pb_reorder);
    return CSINN_TRUE;
}

/*
    matrix: input data matrix
    vector: kernel data row
*/
int shl_c906_conv1x1s1_batch_gemv_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];  // assert(batch == 1);
    int32_t in_ch = input->dim[1];
    int32_t out_ch = kernel->dim[0];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int32_t m = out_ch / group;
    int32_t k = in_ch / group;
    int32_t n = out_h * out_w;

    bool flag_bias = 1;  // default: conv2d layer include bias
    if (bias_data == NULL) {
        flag_bias = 0;
        bias_data = (__fp16 *)shl_mem_alloc(out_ch * sizeof(__fp16));
    }

    __fp16 *pb_reorder = (__fp16 *)shl_mem_alloc(k * n * sizeof(__fp16));

    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            __fp16 *pa = kernel_data + g * m * k;
            __fp16 *pb = pb_reorder;
            __fp16 *pc = output_data;
            __fp16 *bias_tmp = bias_data + g * m;

            // pack/reorder
            shl_c906_reorder_matrix_z16_fp16(input_data, pb, k, n, n);
            // batch GEMV
            for (int j = 0; j < m; j++) {
                shl_c906_gemv_trans_pack16_fp16(pc + j * n, pa + j * k, pb, k, n, n, bias_tmp + j);
            }

            input_data += k * n;
            output_data += m * n;
        }
    }
    shl_mem_free(pb_reorder);

    if (!flag_bias) {
        shl_mem_free(bias_data);
        bias_data = NULL;
    }
    return CSINN_TRUE;
}
