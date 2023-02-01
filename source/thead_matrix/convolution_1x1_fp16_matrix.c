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

#include "shl_thead_rvm.h"

void shl_rvm_conv1x1s1_gemm_reorder_kernel_fp16(struct csinn_tensor *kernel,
                                                struct csinn_conv2d_params *params)
{
    shl_rvm_conv_im2col_gemm_reorder_kernel_fp16(kernel, params);
}

static void align_input_channel_fp16(__fp16 *dst, const __fp16 *src, int out_hw, int channel,
                                     int alinged_ch)
{
    for (int i = 0; i < out_hw; i++) {
        int size = channel;
        while (size > 0) {
            int vl = vsetvl_e16m1(size);
            vfloat16m1_t _input = vle16_v_f16m1(src, vl);
            src += vl;
            vse16_v_f16m1(dst, _input, vl);
            dst += vl;
            size -= vl;
        }
        dst += alinged_ch - channel;
    }
}

int shl_rvm_conv1x1s1_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)params->conv_extra.kernel_tm->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[3];
    int32_t out_h = output->dim[1];
    int32_t out_w = output->dim[2];
    int32_t out_c = kernel->dim[0];

    int32_t m = out_h * out_w;
    int32_t n = out_c;
    int32_t k = in_c;

    // k_align = 8 and n_align = 8 for MLEN = 128
    int32_t k_align = ((k - 1) & -(csrr_xmlenb() / 2)) + csrr_xmlenb() / 2;
    // int32_t n_align = ((n - 1) & -(csrr_xmlenb() / 2)) + csrr_xmlenb() / 2;

    __fp16 *input_align_buf = input_data;
    if (k_align != k) {
        input_align_buf = (__fp16 *)shl_mem_alloc(m * k_align * sizeof(__fp16));
    }
    for (int b = 0; b < batch; b++) {
        __fp16 *kernel_ptr = kernel_data;
        __fp16 *in_ptr = input_align_buf;
        if (k_align != k) {
            align_input_channel_fp16(input_align_buf, input_data, m, k, k_align);
        }
        __fp16 *out_ptr = output_data;
        __fp16 *bias_ptr = bias_data ? bias_data : NULL;
        // gemm
        shl_rvm_nhwc_gemm_fp16(out_ptr, kernel_ptr, in_ptr, bias_ptr, m, k_align, n);
        input_data += m * k;
        output_data += m * n;
    }
    if (k_align != k) {
        shl_mem_free(input_align_buf);
    }
    return CSINN_TRUE;
}

// Split the group conv2d into multiple common conv2ds on HHB
int shl_rvm_group_conv1x1s1_gemm_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params)
{
    return CSINN_FALSE;
}
