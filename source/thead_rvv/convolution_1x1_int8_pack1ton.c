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

#include "shl_thead_rvv.h"

/*************************************************************************************
 * reorder kernel_data inplace, means the origin kernel_data be destoried.
 * The reason to do this is that the packaging process must not consume more memory.
 **************************************************************************************/
void shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_int8(struct csinn_tensor *kernel,
                                                         struct csinn_conv2d_params *params)
{
    shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_int8(kernel, params);
}

#ifdef SHL_USE_DOT_INT8
static void reorder_input_pack1ton_align4_int8(const int8_t *src, int8_t *dst, int inc, int inh,
                                               int inw)
{
    const int packn = csrr_vlenb() / sizeof(int8_t) / 2;
    int vl = vsetvl_e8mf2(packn);
    const int in_size = inh * inw;  // per-channel size

    while (inc > 0) {
        vl = vsetvl_e8mf2(inc);
        int vl4 = ((vl - 1) & -4) + 4;
        int8_t *in_ptr = (int8_t *)src;
        for (int i = 0; i < inh; i++) {
            for (int j = 0; j < inw; j++) {
                vint8mf2_t _tmp = vlse8_v_i8mf2(in_ptr, in_size * sizeof(int8_t), vl);
                in_ptr++;
                vse8_v_i8mf2(dst, _tmp, vl);
                dst += vl4;
            }
        }
        src += in_size * vl;
        inc -= vl;
    }
}
#endif  // SHL_USE_DOT_INT8

int shl_rvv_conv1x1s1_gemm_pack1ton_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;
    int8_t *kernel_data = (int8_t *)params->conv_extra.kernel_tm->data;
    int32_t *bias_data = (int32_t *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];
    int32_t out_c = kernel->dim[0];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int32_t m = out_c / group;
    int32_t k = in_c / group;
    int32_t n = out_h * out_w;

#ifdef SHL_USE_DOT_INT8
    int32_t k4 = ((k - 1) & -4) + 4;
    int8_t *pb_reorder = (int8_t *)shl_mem_alloc(k4 * n * sizeof(int8_t));
    int8_t *input_ncxhwx = (int8_t *)shl_mem_alloc(k4 * n * sizeof(int8_t));
#else
    int8_t *pb_reorder = (int8_t *)shl_mem_alloc(k * n * sizeof(int8_t));
    int8_t *input_ncxhwx = (int8_t *)shl_mem_alloc(k * n * sizeof(int8_t));
#endif  // SHL_USE_DOT_INT8

    int32_t *multiplier = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));
    int32_t *shift = (int32_t *)shl_mem_alloc(m * sizeof(int32_t));

    for (int i = 0; i < batch; i++) {
        for (int g = 0, j = 0; g < group; g++) {
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
            int8_t *in_ptr = pb_reorder;
            int8_t *out_ptr = output_data;
            int32_t *bias_ptr = bias_data ? (bias_data + g * m) : NULL;

#ifdef SHL_USE_DOT_INT8
            int8_t *kernel_ptr = kernel_data + g * m * k4;
            reorder_input_pack1ton_align4_int8(input_data, input_ncxhwx, k, out_h, out_w);
            shl_rvv_reorder_input_z12_pack1ton_int8_dot(input_ncxhwx, in_ptr, k4, 1, n, n);
            shl_rvv_ncxhwx_gemm_12xpackn_int8_dot(out_ptr, kernel_ptr, in_ptr, bias_ptr, m, k4, n,
                                                  n, output->qinfo->zero_point, multiplier, shift);
#else
            int8_t *kernel_ptr = kernel_data + g * m * k;
            shl_rvv_reorder_input_pack1ton_int8(input_data, input_ncxhwx, k, out_h, out_w);
            shl_rvv_reorder_input_z4_pack1ton_int8(input_ncxhwx, in_ptr, k, 1, n, n);
            shl_rvv_ncxhwx_gemm_4xpack2n_int8(output_data, kernel_ptr, in_ptr, bias_ptr, m, k, n, n,
                                              output->qinfo->zero_point, multiplier, shift);
#endif  // SHL_USE_DOT_INT8
            input_data += k * n;
            output_data += m * n;
        }
    }
    shl_mem_free(multiplier);
    shl_mem_free(shift);
    shl_mem_free(pb_reorder);
    shl_mem_free(input_ncxhwx);
    return CSINN_TRUE;
}
