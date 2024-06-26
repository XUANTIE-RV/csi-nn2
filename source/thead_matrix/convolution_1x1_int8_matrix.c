/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

#include "rvm/rvm.h"

void shl_rvm_conv1x1s1_gemm_reorder_kernel_int8(struct csinn_tensor *kernel,
                                                struct csinn_conv2d_params *params)
{
    shl_rvm_conv_im2col_gemm_reorder_kernel_int8(kernel, params);
}

static void align_input_channel_int8(int8_t *dst, const int8_t *src, int out_hw, int channel,
                                     int alinged_ch)
{
    for (int i = 0; i < out_hw; i++) {
        int size = channel;
        while (size > 0) {
            int vl = vsetvl_e16m1(size);
            vint8m1_t _input = vle8_v_i8m1(src, vl);
            src += vl;
            vse8_v_i8m1(dst, _input, vl);
            dst += vl;
            size -= vl;
        }
        dst += alinged_ch - channel;
    }
}

int shl_rvm_conv1x1s1_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)params->conv_extra.kernel_tm->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];
    int32_t in_ch = input->dim[3];
    int32_t out_ch = kernel->dim[0];
    int32_t out_h = output->dim[1];
    int32_t out_w = output->dim[2];

    int32_t m = out_h * out_w;
    int32_t k = in_ch;
    int32_t n = out_ch;

    // k_align = 16 and n_align = 4 for MLEN = 128
    int32_t k_align = ((k - 1) & -csrr_xrlenb()) + csrr_xrlenb();
    int32_t n_align = ((n - 1) & -(csrr_xrlenb() / 4)) + csrr_xrlenb() / 4;

    int32_t *multiplier = (int32_t *)shl_mem_alloc(n * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(n * sizeof(int32_t));

    if (kernel->quant_channel > 1) {
        for (int c = 0; c < n; c++) {
            multiplier[c] = kernel->qinfo[c].multiplier;
            shift[c] = -1 - kernel->qinfo[c].shift;  // shift pretreat
        }
    } else if (kernel->quant_channel == 1) {
        for (int c = 0; c < n; c++) {
            multiplier[c] = kernel->qinfo[0].multiplier;
            shift[c] = -1 - kernel->qinfo[0].shift;  // right shift
        }
    }

    int8_t *input_align_buf = input_data;
    if (k_align != k) {
        input_align_buf = (int8_t *)shl_mem_alloc(m * k_align * sizeof(int8_t));
    }
    for (int i = 0; i < batch; i++) {
        int8_t *kernel_ptr = kernel_data;
        int8_t *in_ptr = input_align_buf;
        if (k_align != k) {
            align_input_channel_int8(input_align_buf, input_data, m, k, k_align);
        } else {
            in_ptr = input_data;
        }
        int8_t *out_ptr = output_data;
        int32_t *bias_ptr = bias_data;  // bias_data != NULL with fusing zp to bias
        // gemm
        shl_rvm_nhwc_gemm_int8(out_ptr, kernel_ptr, in_ptr, bias_ptr, m, k_align, n,
                               output->qinfo->zero_point, multiplier, shift);
        input_data += m * k;
        output_data += m * n;
    }
    shl_mem_free(multiplier);
    shl_mem_free(shift);
    if (k_align != k) {
        shl_mem_free(input_align_buf);
    }
    return CSINN_TRUE;
}

// Split the group conv2d into multiple common conv2ds on HHB
int shl_rvm_group_conv1x1s1_gemm_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params)
{
    return CSINN_FALSE;
}
