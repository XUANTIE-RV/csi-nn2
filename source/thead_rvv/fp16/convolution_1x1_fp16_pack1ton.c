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

#include "rvv/rvv.h"

/*************************************************************************************
 * reorder kernel_data inplace, means the origin kernel_data be destoried.
 * The reason to do this is that the packaging process must not consume more memory.
 **************************************************************************************/
void shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp16(struct csinn_tensor *kernel,
                                                         struct csinn_conv2d_params *params)
{
    shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp16(kernel, params);
}

/*************************************************************************************
 * reorder kernel_data inplace, means the origin kernel_data be destoried.
 * The reason to do this is that the packaging process must not consume more memory.
 **************************************************************************************/
void shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp16_w_int8(struct csinn_tensor *kernel,
                                                                struct csinn_conv2d_params *params)
{
    shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp16_w_int8(kernel, params);
}

int shl_rvv_common_conv1x1_gemm_pack1ton_fp16(
    struct csinn_tensor *input, struct csinn_tensor *output, struct csinn_tensor *kernel,
    struct csinn_tensor *bias, struct csinn_conv2d_params *params,
    void (*reorder_input)(__fp16 *, __fp16 *, int, int, int, int),
    void (*gemm)(__fp16 *, const __fp16 *, const __fp16 *, __fp16 *, int, int, int, bool))
{
    if (input->layout == CSINN_LAYOUT_NC1HWC0) {
        shl_rvv_tensor_nc1xc0_to_ndarray_replace_fp16(input);
    }
    if (output->layout == CSINN_LAYOUT_NCHW) {
        const int packn = csrr_vlenb() / sizeof(__fp16);
        output->dim[1] /= packn;
        output->dim[4] = packn;
        output->dim_count = 5;
        output->layout = CSINN_LAYOUT_NC1HWC0;
    }
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = NULL;
    __fp16 *bias_data = (__fp16 *)bias->data;

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

    __fp16 *kernel_fp16 = NULL;
    if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
        int size = csinn_tensor_size(kernel);
        kernel_fp16 = (__fp16 *)shl_mem_alloc(size * sizeof(__fp16));
        if (kernel->quant_channel > 1) {
            shl_rvv_conv_im2col_gemm_pack1ton_dequantize_per_channel_i8_to_f16(kernel, params,
                                                                               kernel_fp16);
        } else {
            int8_t *kernel_int8 = (int8_t *)kernel->data;
            int32_t zp = kernel->qinfo->zero_point;
            float scale = kernel->qinfo->scale;
            shl_rvv_dequantize_i8_to_f16(kernel_int8, kernel_fp16, size, zp, scale);
        }
        kernel_data = kernel_fp16;
    } else if (kernel->dtype == CSINN_DTYPE_FLOAT16) {
        kernel_data = (__fp16 *)kernel->data;
    } else {
        shl_debug_error("kernel unsupport dtype: %d\n", kernel->dtype);
        return CSINN_FALSE;
    }

    __fp16 *pb_reorder = (__fp16 *)shl_mem_alloc(k * n * sizeof(__fp16));
    __fp16 *input_ncxhwx = (__fp16 *)shl_mem_alloc(k * n * sizeof(__fp16));

    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            __fp16 *kernel_ptr = kernel_data + g * m * k;
            __fp16 *in_ptr = pb_reorder;
            __fp16 *out_ptr = output_data;
            __fp16 *bias_ptr = bias_data ? (bias_data + g * m) : NULL;

            shl_rvv_reorder_input_pack1ton_fp16(input_data, input_ncxhwx, k, out_h, out_w);

            // reorder(pack)
            reorder_input(input_ncxhwx, in_ptr, k, 1, n, n);

            // gemm
            gemm(out_ptr, kernel_ptr, in_ptr, bias_ptr, m, k, n, false);

            input_data += k * n;
            output_data += m * n;
        }
    }
    shl_mem_free(pb_reorder);
    shl_mem_free(input_ncxhwx);
    if (kernel->is_const && kernel->dtype == CSINN_DTYPE_INT8) {
        shl_mem_free(kernel_fp16);
        return CSINN_TRUE;
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}

int shl_rvv_conv1x1s1_gemm_pack1ton_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params)
{
    return shl_rvv_common_conv1x1_gemm_pack1ton_fp16(input, output, kernel, bias, params,
                                                     shl_rvv_reorder_input_z12_pack1ton_fp16,
                                                     shl_rvv_ncxhwx_gemm_12xpack2n_fp16);
}
