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

#include "rvv/perf.h"
#include "rvv/rvv.h"

static struct shl_function_map shl_rvv_kernel_map[] = {
    {shl_rvv_common_conv_gemm_fp32, "shl_rvv_common_conv_gemm_fp32"},
    {shl_rvv_common_conv_gemm_fp16, "shl_rvv_common_conv_gemm_fp16"},
    {shl_rvv_common_conv_gemm_int8, "shl_rvv_common_conv_gemm_int8"},
    {shl_rvv_common_conv_gemm_packn_fp32, "shl_rvv_common_conv_gemm_packn_fp32"},
    {shl_rvv_common_conv_gemm_packn_fp16, "shl_rvv_common_conv_gemm_packn_fp16"},
    {shl_rvv_common_conv_gemm_packn_int8, "shl_rvv_common_conv_gemm_packn_int8"},
    {shl_rvv_common_conv_gemm_pack1ton_fp32, "shl_rvv_common_conv_gemm_pack1ton_fp32"},
    {shl_rvv_common_conv_gemm_pack1ton_fp16, "shl_rvv_common_conv_gemm_pack1ton_fp16"},
    {shl_rvv_common_conv_gemm_pack1ton_int8, "shl_rvv_common_conv_gemm_pack1ton_int8"},
    {shl_rvv_common_conv_gemm_packnto1_fp32, "shl_rvv_common_conv_gemm_packnto1_fp32"},
    {shl_rvv_common_conv_gemm_packnto1_fp16, "shl_rvv_common_conv_gemm_packnto1_fp16"},
    {shl_rvv_common_conv_gemm_packnto1_int8, "shl_rvv_common_conv_gemm_packnto1_int8"},
    {shl_rvv_common_conv1x1_gemm_fp32, "shl_rvv_common_conv1x1_gemm_fp32"},
    {shl_rvv_common_conv1x1_gemm_fp16, "shl_rvv_common_conv1x1_gemm_fp16"},
    {shl_rvv_common_conv1x1_gemm_int8, "shl_rvv_common_conv1x1_gemm_int8"},
    {shl_rvv_common_conv1x1_gemm_packn_fp32, "shl_rvv_common_conv1x1_gemm_packn_fp32"},
    {shl_rvv_common_conv1x1_gemm_packn_fp16, "shl_rvv_common_conv1x1_gemm_packn_fp16"},
    {shl_rvv_common_conv1x1_gemm_packn_int8, "shl_rvv_common_conv1x1_gemm_packn_int8"},
    {shl_rvv_common_conv1x1_gemm_pack1ton_fp32, "shl_rvv_common_conv1x1_gemm_pack1ton_fp32"},
    {shl_rvv_common_conv1x1_gemm_pack1ton_fp16, "shl_rvv_common_conv1x1_gemm_pack1ton_fp16"},
    {shl_rvv_common_conv1x1_gemm_pack1ton_int8, "shl_rvv_common_conv1x1_gemm_pack1ton_int8"},
    {shl_rvv_common_conv1x1_gemm_packnto1_fp32, "shl_rvv_common_conv1x1_gemm_packnto1_fp32"},
    {shl_rvv_common_conv1x1_gemm_packnto1_fp16, "shl_rvv_common_conv1x1_gemm_packnto1_fp16"},
    {shl_rvv_common_conv1x1_gemm_packnto1_int8, "shl_rvv_common_conv1x1_gemm_packnto1_int8"},
    {shl_rvv_conv1d_im2col_gemm_reorder_kernel_fp32,
     "shl_rvv_conv1d_im2col_gemm_reorder_kernel_fp32"},
    {shl_rvv_conv1d_im2col_gemm_reorder_kernel_fp16,
     "shl_rvv_conv1d_im2col_gemm_reorder_kernel_fp16"},
    {shl_rvv_conv1d_im2col_gemm_reorder_kernel_fp16_w_int8,
     "shl_rvv_conv1d_im2col_gemm_reorder_kernel_fp16_w_int8"},
    {shl_rvv_conv1d_im2col_gemm_dequantize_per_channel_i8_to_f16,
     "shl_rvv_conv1d_im2col_gemm_dequantize_per_channel_i8_to_f16"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_fp32, "shl_rvv_conv_im2col_gemm_reorder_kernel_fp32"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_fp16, "shl_rvv_conv_im2col_gemm_reorder_kernel_fp16"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_fp16_w_int8,
     "shl_rvv_conv_im2col_gemm_reorder_kernel_fp16_w_int8"},
    {shl_rvv_conv_im2col_gemm_dequantize_per_channel_i8_to_f16,
     "shl_rvv_conv_im2col_gemm_dequantize_per_channel_i8_to_f16"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_int8, "shl_rvv_conv_im2col_gemm_reorder_kernel_int8"},
    {shl_rvv_conv1d_im2col_gemm_fp32, "shl_rvv_conv1d_im2col_gemm_fp32"},
    {shl_rvv_conv1d_im2col_gemm_fp16, "shl_rvv_conv1d_im2col_gemm_fp16"},
    {shl_rvv_conv_im2col_gemm_fp32, "shl_rvv_conv_im2col_gemm_fp32"},
    {shl_rvv_conv_im2col_gemm_fp16, "shl_rvv_conv_im2col_gemm_fp16"},
    {shl_rvv_conv_im2col_gemm_int8, "shl_rvv_conv_im2col_gemm_int8"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp32,
     "shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp32"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp16,
     "shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp16"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp16_w_int8,
     "shl_rvv_conv_im2col_gemm_reorder_kernel_packn_fp16_w_int8"},
    {shl_rvv_conv_im2col_gemm_packn_dequantize_per_channel_i8_to_f16,
     "shl_rvv_conv_im2col_gemm_packn_dequantize_per_channel_i8_to_f16"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_packn_int8,
     "shl_rvv_conv_im2col_gemm_reorder_kernel_packn_int8"},
    {shl_rvv_conv_im2col_gemm_packn_fp32, "shl_rvv_conv_im2col_gemm_packn_fp32"},
    {shl_rvv_conv_im2col_gemm_packn_fp16, "shl_rvv_conv_im2col_gemm_packn_fp16"},
    {shl_rvv_conv_im2col_gemm_packn_int8, "shl_rvv_conv_im2col_gemm_packn_int8"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp32,
     "shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp32"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp16,
     "shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp16"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp16_w_int8,
     "shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_fp16_w_int8"},
    {shl_rvv_conv_im2col_gemm_pack1ton_dequantize_per_channel_i8_to_f16,
     "shl_rvv_conv_im2col_gemm_pack1ton_dequantize_per_channel_i8_to_f16"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_int8,
     "shl_rvv_conv_im2col_gemm_reorder_kernel_pack1ton_int8"},
    {shl_rvv_conv_im2col_gemm_pack1ton_fp32, "shl_rvv_conv_im2col_gemm_pack1ton_fp32"},
    {shl_rvv_conv_im2col_gemm_pack1ton_fp16, "shl_rvv_conv_im2col_gemm_pack1ton_fp16"},
    {shl_rvv_conv_im2col_gemm_pack1ton_int8, "shl_rvv_conv_im2col_gemm_pack1ton_int8"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_fp32,
     "shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_fp32"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_fp16,
     "shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_fp16"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_fp16_w_int8,
     "shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_fp16_w_int8"},
    {shl_rvv_conv_im2col_gemm_packnto1_dequantize_per_channel_i8_to_f16,
     "shl_rvv_conv_im2col_gemm_packnto1_dequantize_per_channel_i8_to_f16"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_int8,
     "shl_rvv_conv_im2col_gemm_reorder_kernel_packnto1_int8"},
    {shl_rvv_conv_im2col_gemm_packnto1_fp32, "shl_rvv_conv_im2col_gemm_packnto1_fp32"},
    {shl_rvv_conv_im2col_gemm_packnto1_fp16, "shl_rvv_conv_im2col_gemm_packnto1_fp16"},
    {shl_rvv_conv_im2col_gemm_packnto1_int8, "shl_rvv_conv_im2col_gemm_packnto1_int8"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_fp32, "shl_rvv_conv1x1s1_gemm_reorder_kernel_fp32"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_fp16, "shl_rvv_conv1x1s1_gemm_reorder_kernel_fp16"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_fp16_w_int8,
     "shl_rvv_conv1x1s1_gemm_reorder_kernel_fp16_w_int8"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_int8, "shl_rvv_conv1x1s1_gemm_reorder_kernel_int8"},
    {shl_rvv_conv1x1s1_gemm_fp32, "shl_rvv_conv1x1s1_gemm_fp32"},
    {shl_rvv_conv1x1s1_gemm_fp16, "shl_rvv_conv1x1s1_gemm_fp16"},
    {shl_rvv_conv1x1s1_gemm_int8, "shl_rvv_conv1x1s1_gemm_int8"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_fp32,
     "shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_fp32"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_fp16,
     "shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_fp16"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_fp16_w_int8,
     "shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_fp16_w_int8"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_int8,
     "shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_int8"},
    {shl_rvv_conv1x1s1_gemm_packn_fp32, "shl_rvv_conv1x1s1_gemm_packn_fp32"},
    {shl_rvv_conv1x1s1_gemm_packn_fp16, "shl_rvv_conv1x1s1_gemm_packn_fp16"},
    {shl_rvv_conv1x1s1_gemm_packn_int8, "shl_rvv_conv1x1s1_gemm_packn_int8"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp32,
     "shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp32"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp16,
     "shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp16"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp16_w_int8,
     "shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_fp16_w_int8"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_int8,
     "shl_rvv_conv1x1s1_gemm_reorder_kernel_pack1ton_int8"},
    {shl_rvv_conv1x1s1_gemm_pack1ton_fp32, "shl_rvv_conv1x1s1_gemm_pack1ton_fp32"},
    {shl_rvv_conv1x1s1_gemm_pack1ton_fp16, "shl_rvv_conv1x1s1_gemm_pack1ton_fp16"},
    {shl_rvv_conv1x1s1_gemm_pack1ton_int8, "shl_rvv_conv1x1s1_gemm_pack1ton_int8"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_fp32,
     "shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_fp32"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_fp16,
     "shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_fp16"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_fp16_w_int8,
     "shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_fp16_w_int8"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_int8,
     "shl_rvv_conv1x1s1_gemm_reorder_kernel_packnto1_int8"},
    {shl_rvv_conv1x1s1_gemm_packnto1_fp32, "shl_rvv_conv1x1s1_gemm_packnto1_fp32"},
    {shl_rvv_conv1x1s1_gemm_packnto1_fp16, "shl_rvv_conv1x1s1_gemm_packnto1_fp16"},
    {shl_rvv_conv1x1s1_gemm_packnto1_int8, "shl_rvv_conv1x1s1_gemm_packnto1_int8"},
    {shl_rvv_wg_b6f3s1_trans_kernel_packn_fp32, "shl_rvv_wg_b6f3s1_trans_kernel_packn_fp32"},
    {shl_rvv_wg_b6f3s1_trans_kernel_packn_fp16, "shl_rvv_wg_b6f3s1_trans_kernel_packn_fp16"},
    {shl_rvv_wg_b6f3s1_packn_fp32, "shl_rvv_wg_b6f3s1_packn_fp32"},
    {shl_rvv_wg_b6f3s1_packn_fp16, "shl_rvv_wg_b6f3s1_packn_fp16"},
    {shl_rvv_wg_b4f3s1_trans_kernel_packn_fp32, "shl_rvv_wg_b4f3s1_trans_kernel_packn_fp32"},
    {shl_rvv_wg_b4f3s1_trans_kernel_packn_fp16, "shl_rvv_wg_b4f3s1_trans_kernel_packn_fp16"},
    {shl_rvv_wg_b4f3s1_trans_kernel_packn_int8, "shl_rvv_wg_b4f3s1_trans_kernel_packn_int8"},
    {shl_rvv_wg_b4f3s1_packn_fp32, "shl_rvv_wg_b4f3s1_packn_fp32"},
    {shl_rvv_wg_b4f3s1_packn_fp16, "shl_rvv_wg_b4f3s1_packn_fp16"},
    {shl_rvv_wg_b4f3s1_packn_int8, "shl_rvv_wg_b4f3s1_packn_int8"},
    {shl_rvv_conv3x3s1_direct_reorder_kernel_pack4n_fp16,
     "shl_rvv_conv3x3s1_direct_reorder_kernel_pack4n_fp16"},
    {shl_rvv_conv3x3s1_direct_fp16_nhwc, "shl_rvv_conv3x3s1_direct_fp16_nhwc"},
    {shl_rvv_dwconv3x3s1_fp32, "shl_rvv_dwconv3x3s1_fp32"},
    {shl_rvv_dwconv3x3s2_fp32, "shl_rvv_dwconv3x3s2_fp32"},
    {shl_rvv_dwconv3x3s1_fp16, "shl_rvv_dwconv3x3s1_fp16"},
    {shl_rvv_dwconv3x3s2_fp16, "shl_rvv_dwconv3x3s2_fp16"},
    {shl_rvv_dwconv3x3s1_int8, "shl_rvv_dwconv3x3s1_int8"},
    {shl_rvv_dwconv3x3s2_int8, "shl_rvv_dwconv3x3s2_int8"},
    {shl_rvv_dwconv3x3s1_int4, "shl_rvv_dwconv3x3s1_int4"},
    {shl_rvv_dwconv3x3s2_int4, "shl_rvv_dwconv3x3s2_int4"},
    {shl_rvv_dwconv_reorder_kernel_packn_fp32, "shl_rvv_dwconv_reorder_kernel_packn_fp32"},
    {shl_rvv_dwconv_reorder_kernel_packn_fp16, "shl_rvv_dwconv_reorder_kernel_packn_fp16"},
    {shl_rvv_dwconv_reorder_kernel_packn_fp16_w_int8,
     "shl_rvv_dwconv_reorder_kernel_packn_fp16_w_int8"},
    {shl_rvv_dwconv_reorder_kernel_packn_int8, "shl_rvv_dwconv_reorder_kernel_packn_int8"},
    {shl_rvv_dwconv3x3s1_packn_fp32, "shl_rvv_dwconv3x3s1_packn_fp32"},
    {shl_rvv_dwconv3x3s2_packn_fp32, "shl_rvv_dwconv3x3s2_packn_fp32"},
    {shl_rvv_dwconv3x3s1_packn_fp16, "shl_rvv_dwconv3x3s1_packn_fp16"},
    {shl_rvv_dwconv3x3s2_packn_fp16, "shl_rvv_dwconv3x3s2_packn_fp16"},
    {shl_rvv_dwconv3x3s1_packn_int8, "shl_rvv_dwconv3x3s1_packn_int8"},
    {shl_rvv_dwconv3x3s2_packn_int8, "shl_rvv_dwconv3x3s2_packn_int8"},
    {shl_rvv_dwconv_packn_fp32, "shl_rvv_dwconv_packn_fp32"},
    {shl_rvv_dwconv_packn_fp16, "shl_rvv_dwconv_packn_fp16"},
    {shl_rvv_dwconv_packn_int8, "shl_rvv_dwconv_packn_int8"},
    {shl_rvv_dwconv_nhwc_fp32, "shl_rvv_dwconv_nhwc_fp32"},
    {shl_rvv_dwconv_nhwc_fp16, "shl_rvv_dwconv_nhwc_fp16"},
    {shl_rvv_dwconv_nhwc_int8, "shl_rvv_dwconv_nhwc_int8"},
    {shl_rvv_deconv2d_gemm_col2im_reorder_kernel_fp32,
     "shl_rvv_deconv2d_gemm_col2im_reorder_kernel_fp32"},
    {shl_rvv_deconv2d_gemm_col2im_fp32, "shl_rvv_deconv2d_gemm_col2im_fp32"},
    {shl_rvv_deconv2d_gemm_col2im_reorder_kernel_fp16,
     "shl_rvv_deconv2d_gemm_col2im_reorder_kernel_fp16"},
    {shl_rvv_deconv2d_gemm_col2im_reorder_kernel_fp16_w_int8,
     "shl_rvv_deconv2d_gemm_col2im_reorder_kernel_fp16_w_int8"},
    {shl_rvv_deconv2d_gemm_col2im_dequantize_per_channel_i8_to_f16,
     "shl_rvv_deconv2d_gemm_col2im_dequantize_per_channel_i8_to_f16"},
    {shl_rvv_deconv2d_gemm_col2im_fp16, "shl_rvv_deconv2d_gemm_col2im_fp16"},
    {shl_rvv_reorder_kernel_n8_fp32, "shl_rvv_reorder_kernel_n8_fp32"},
    {shl_rvv_reorder_input_z8_fp32, "shl_rvv_reorder_input_z8_fp32"},
    {shl_rvv_gemm_8x8_fp32, "shl_rvv_gemm_8x8_fp32"},
    {shl_rvv_reorder_kernel_n8_fp16, "shl_rvv_reorder_kernel_n8_fp16"},
    {shl_rvv_reorder_input_z16_fp16, "shl_rvv_reorder_input_z16_fp16"},
    {shl_rvv_gemm_8x16_fp16, "shl_rvv_gemm_8x16_fp16"},
    {shl_rvv_reorder_kernel_n8_int8_dot, "shl_rvv_reorder_kernel_n8_int8_dot"},
    {shl_rvv_reorder_input_z8_int8_dot, "shl_rvv_reorder_input_z8_int8_dot"},
#ifdef SHL_USE_DOT_INT8
    {shl_rvv_gemm_8x8_int8_dot, "shl_rvv_gemm_8x8_int8_dot"},
    {shl_rvv_ncxhwx_gemm_12xpackn_int8_dot, "shl_rvv_ncxhwx_gemm_12xpackn_int8_dot"},
    {shl_rvv_gemm_a0b1_8xmf2_int8_dot, "shl_rvv_gemm_a0b1_8xmf2_int8_dot"},
    {shl_rvv_matmul_reorder_mat0_n8z4_int8_dot, "shl_rvv_matmul_reorder_mat0_n8z4_int8_dot"},
    {shl_rvv_matmul_reorder_mat1_zmf2n4_int8_dot, "shl_rvv_matmul_reorder_mat1_zmf2n4_int8_dot"},
    {shl_rvv_matmul_8xmf2_int8_dot, "shl_rvv_matmul_8xmf2_int8_dot"},
#endif
    {shl_rvv_reorder_kernel_n4_int8_v128, "shl_rvv_reorder_kernel_n4_int8_v128"},
    {shl_rvv_reorder_input_z16_int8_v128, "shl_rvv_reorder_input_z16_int8_v128"},
    {shl_rvv_gemm_4x16_int8_v128, "shl_rvv_gemm_4x16_int8_v128"},
#ifdef SHL_USE_DOT_INT4
    {shl_rvv_reorder_input_n8_int4_dot, "shl_rvv_reorder_input_n8_int4_dot"},
    {shl_rvv_reorder_kernel_n8_int4, "shl_rvv_reorder_kernel_n8_int4"},
    {shl_rvv_gemm_8x8_int4_dot, "shl_rvv_gemm_8x8_int4_dot"},
    {shl_rvv_ncxhwx_gemm_8xpackn_int4, "shl_rvv_ncxhwx_gemm_8xpackn_int4"},
    {shl_rvv_ncxhwx_gemm_12xpackn_int4, "shl_rvv_ncxhwx_gemm_12xpackn_int4"},
    {shl_rvv_gemm_a0b1_4xpackn_int8, "shl_rvv_gemm_a0b1_4xpackn_int8"},
#endif
    {shl_rvv_reorder_kernel_packn_fp32, "shl_rvv_reorder_kernel_packn_fp32"},
    {shl_rvv_reorder_input_z8_packn_fp32, "shl_rvv_reorder_input_z8_packn_fp32"},
    {shl_rvv_reorder_input_z12_packn_fp32, "shl_rvv_reorder_input_z12_packn_fp32"},
    {shl_rvv_ncxhwx_gemm_12xpack2n_fp32, "shl_rvv_ncxhwx_gemm_12xpack2n_fp32"},
    {shl_rvv_reorder_kernel_packn_fp16, "shl_rvv_reorder_kernel_packn_fp16"},
    {shl_rvv_reorder_input_z8_packn_fp16, "shl_rvv_reorder_input_z8_packn_fp16"},
    {shl_rvv_reorder_input_z12_packn_fp16, "shl_rvv_reorder_input_z12_packn_fp16"},
    {shl_rvv_ncxhwx_gemm_12xpack2n_fp16, "shl_rvv_ncxhwx_gemm_12xpack2n_fp16"},
    {shl_rvv_reorder_input_z8_packn_int8_dot, "shl_rvv_reorder_input_z8_packn_int8_dot"},
    {shl_rvv_reorder_input_z12_packn_int8_dot, "shl_rvv_reorder_input_z12_packn_int8_dot"},
    {shl_rvv_reorder_input_z8_packn_int4, "shl_rvv_reorder_input_z8_packn_int4"},
    {shl_rvv_reorder_input_z12_packn_int4, "shl_rvv_reorder_input_z12_packn_int4"},
    {shl_rvv_reorder_input_z12_pack1ton_fp32, "shl_rvv_reorder_input_z12_pack1ton_fp32"},
    {shl_rvv_reorder_input_z12_pack1ton_fp16, "shl_rvv_reorder_input_z12_pack1ton_fp16"},
    {shl_rvv_reorder_input_z4_pack1ton_int8, "shl_rvv_reorder_input_z4_pack1ton_int8"},
    {shl_rvv_reorder_input_z12_pack1ton_int8_dot, "shl_rvv_reorder_input_z12_pack1ton_int8_dot"},
    {shl_rvv_reorder_input_z4_packn_int8, "shl_rvv_reorder_input_z4_packn_int8"},
    {shl_rvv_ncxhwx_gemm_4xpack2n_int8, "shl_rvv_ncxhwx_gemm_4xpack2n_int8"},
    {shl_rvv_reorder_a_block_12xk_fp32, "shl_rvv_reorder_a_block_12xk_fp32"},
    {shl_rvv_reorder_b_block_pack2nxk_fp32, "shl_rvv_reorder_b_block_pack2nxk_fp32"},
    {shl_rvv_gemm_block_12xpack2n_fp32, "shl_rvv_gemm_block_12xpack2n_fp32"},
    {shl_rvv_reorder_a_block_12xk_fp16, "shl_rvv_reorder_a_block_12xk_fp16"},
    {shl_rvv_reorder_b_block_pack2nxk_fp16, "shl_rvv_reorder_b_block_pack2nxk_fp16"},
    {shl_rvv_gemm_block_12xpack2n_fp16, "shl_rvv_gemm_block_12xpack2n_fp16"},
    {shl_rvv_avgpool2x2s2_fp32, "shl_rvv_avgpool2x2s2_fp32"},
    {shl_rvv_avgpool2x2s2_fp16, "shl_rvv_avgpool2x2s2_fp16"},
    {shl_rvv_avgpool2x2s2_p1_fp32, "shl_rvv_avgpool2x2s2_p1_fp32"},
    {shl_rvv_avgpool2x2s2_p1_fp16, "shl_rvv_avgpool2x2s2_p1_fp16"},
    {shl_rvv_avgpool3x3s2_fp32, "shl_rvv_avgpool3x3s2_fp32"},
    {shl_rvv_avgpool3x3s2_fp16, "shl_rvv_avgpool3x3s2_fp16"},
    {shl_rvv_avgpool3x3s2_p1_fp32, "shl_rvv_avgpool3x3s2_p1_fp32"},
    {shl_rvv_avgpool3x3s2_p1_fp16, "shl_rvv_avgpool3x3s2_p1_fp16"},
    {shl_rvv_avgpool3x3s1_p1_fp32, "shl_rvv_avgpool3x3s1_p1_fp32"},
    {shl_rvv_avgpool3x3s1_p1_fp16, "shl_rvv_avgpool3x3s1_p1_fp16"},
    {shl_rvv_maxpool2x2s2_fp32, "shl_rvv_maxpool2x2s2_fp32"},
    {shl_rvv_maxpool2x2s2_fp16, "shl_rvv_maxpool2x2s2_fp16"},
    {shl_rvv_maxpool2x2s2_int8, "shl_rvv_maxpool2x2s2_int8"},
    {shl_rvv_maxpool2x2s2_p1_fp32, "shl_rvv_maxpool2x2s2_p1_fp32"},
    {shl_rvv_maxpool2x2s2_p1_fp16, "shl_rvv_maxpool2x2s2_p1_fp16"},
    {shl_rvv_maxpool2x2s2_p1_int8, "shl_rvv_maxpool2x2s2_p1_int8"},
    {shl_rvv_maxpool3x3s2_fp32, "shl_rvv_maxpool3x3s2_fp32"},
    {shl_rvv_maxpool3x3s2_fp16, "shl_rvv_maxpool3x3s2_fp16"},
    {shl_rvv_maxpool3x3s2_int8, "shl_rvv_maxpool3x3s2_int8"},
    {shl_rvv_maxpool3x3s2_p1_fp32, "shl_rvv_maxpool3x3s2_p1_fp32"},
    {shl_rvv_maxpool3x3s2_p1_fp16, "shl_rvv_maxpool3x3s2_p1_fp16"},
    {shl_rvv_maxpool3x3s2_p1_int8, "shl_rvv_maxpool3x3s2_p1_int8"},
    {shl_rvv_maxpool3x3s1_p1_fp32, "shl_rvv_maxpool3x3s1_p1_fp32"},
    {shl_rvv_maxpool3x3s1_p1_fp16, "shl_rvv_maxpool3x3s1_p1_fp16"},
    {shl_rvv_maxpool3x3s1_p1_int8, "shl_rvv_maxpool3x3s1_p1_int8"},
    {shl_rvv_global_avgpool2d_fp32, "shl_rvv_global_avgpool2d_fp32"},
    {shl_rvv_global_avgpool2d_fp16, "shl_rvv_global_avgpool2d_fp16"},
    {shl_rvv_global_maxpool2d_fp32, "shl_rvv_global_maxpool2d_fp32"},
    {shl_rvv_global_maxpool2d_fp16, "shl_rvv_global_maxpool2d_fp16"},
    {shl_rvv_global_maxpool2d_packn_fp32, "shl_rvv_global_maxpool2d_packn_fp32"},
    {shl_rvv_global_maxpool2d_packn_fp16, "shl_rvv_global_maxpool2d_packn_fp16"},
    {shl_rvv_global_maxpool2d_packn_int8, "shl_rvv_global_maxpool2d_packn_int8"},
    {shl_rvv_global_avgpool2d_packn_fp32, "shl_rvv_global_avgpool2d_packn_fp32"},
    {shl_rvv_global_avgpool2d_packn_fp16, "shl_rvv_global_avgpool2d_packn_fp16"},
    {shl_rvv_global_avgpool2d_packn_int8, "shl_rvv_global_avgpool2d_packn_int8"},
    {shl_rvv_maxpool_packn_fp32, "shl_rvv_maxpool_packn_fp32"},
    {shl_rvv_maxpool_packn_fp16, "shl_rvv_maxpool_packn_fp16"},
    {shl_rvv_maxpool_packn_int8, "shl_rvv_maxpool_packn_int8"},
    {shl_rvv_avgpool_packn_fp32, "shl_rvv_avgpool_packn_fp32"},
    {shl_rvv_avgpool_packn_fp16, "shl_rvv_avgpool_packn_fp16"},
    {shl_rvv_avgpool_packn_int8, "shl_rvv_avgpool_packn_int8"},
    {shl_rvv_maxpool_nhwc_fp32, "shl_rvv_maxpool_nhwc_fp32"},
    {shl_rvv_maxpool_nhwc_fp16, "shl_rvv_maxpool_nhwc_fp16"},
    {shl_rvv_maxpool_nhwc_int8, "shl_rvv_maxpool_nhwc_int8"},
    {shl_rvv_avgpool_nhwc_fp32, "shl_rvv_avgpool_nhwc_fp32"},
    {shl_rvv_avgpool_nhwc_fp16, "shl_rvv_avgpool_nhwc_fp16"},
    {shl_rvv_avgpool_nhwc_int8, "shl_rvv_avgpool_nhwc_int8"},
    {shl_rvv_global_maxpool2d_nhwc_fp32, "shl_rvv_global_maxpool2d_nhwc_fp32"},
    {shl_rvv_global_maxpool2d_nhwc_fp16, "shl_rvv_global_maxpool2d_nhwc_fp16"},
    {shl_rvv_global_maxpool2d_nhwc_int8, "shl_rvv_global_maxpool2d_nhwc_int8"},
    {shl_rvv_global_avgpool2d_nhwc_fp32, "shl_rvv_global_avgpool2d_nhwc_fp32"},
    {shl_rvv_global_avgpool2d_nhwc_fp16, "shl_rvv_global_avgpool2d_nhwc_fp16"},
    {shl_rvv_global_avgpool2d_nhwc_int8, "shl_rvv_global_avgpool2d_nhwc_int8"},
    {shl_rvv_fc_gemm_reorder_weight_fp32, "shl_rvv_fc_gemm_reorder_weight_fp32"},
    {shl_rvv_fc_gemm_reorder_weight_fp16, "shl_rvv_fc_gemm_reorder_weight_fp16"},
    {shl_rvv_fc_gemm_reorder_weight_fp16_w_int8, "shl_rvv_fc_gemm_reorder_weight_fp16_w_int8"},
    {shl_rvv_fc_gemm_reorder_weight_int8, "shl_rvv_fc_gemm_reorder_weight_int8"},
    {shl_rvv_gemm_a0b1_12xpack2n_fp32, "shl_rvv_gemm_a0b1_12xpack2n_fp32"},
    {shl_rvv_gemm_a0b1_12xpack2n_fp16, "shl_rvv_gemm_a0b1_12xpack2n_fp16"},
    {shl_rvv_fullyconnected_gemm_fp32, "shl_rvv_fullyconnected_gemm_fp32"},
    {shl_rvv_fullyconnected_gemm_fp16, "shl_rvv_fullyconnected_gemm_fp16"},
    {shl_rvv_fullyconnected_gemm_int8, "shl_rvv_fullyconnected_gemm_int8"},
    {shl_rvv_relu_fp32, "shl_rvv_relu_fp32"},
    {shl_rvv_relu_fp16, "shl_rvv_relu_fp16"},
    {shl_rvv_relu_int8, "shl_rvv_relu_int8"},
    {shl_rvv_relu6_fp32, "shl_rvv_relu6_fp32"},
    {shl_rvv_relu6_fp16, "shl_rvv_relu6_fp16"},
    {shl_rvv_relu6_int8, "shl_rvv_relu6_int8"},
    {shl_rvv_leaky_relu_fp32, "shl_rvv_leaky_relu_fp32"},
    {shl_rvv_leaky_relu_fp16, "shl_rvv_leaky_relu_fp16"},
    {shl_rvv_leaky_relu_int8, "shl_rvv_leaky_relu_int8"},
    {shl_rvv_sigmoid_fp32, "shl_rvv_sigmoid_fp32"},
    {shl_rvv_sigmoid_fp16, "shl_rvv_sigmoid_fp16"},
    {shl_rvv_sigmoid_int8, "shl_rvv_sigmoid_int8"},
    {shl_rvv_softmax_fp32, "shl_rvv_softmax_fp32"},
    {shl_rvv_softmax_fp16, "shl_rvv_softmax_fp16"},
    {shl_rvv_softmax_int8, "shl_rvv_softmax_int8"},
    {shl_rvv_prelu_fp32, "shl_rvv_prelu_fp32"},
    {shl_rvv_prelu_fp16, "shl_rvv_prelu_fp16"},
    {shl_rvv_prelu_int8, "shl_rvv_prelu_int8"},
    {shl_rvv_clip_fp32, "shl_rvv_clip_fp32"},
    {shl_rvv_clip_fp16, "shl_rvv_clip_fp16"},
    {shl_rvv_clip_int8, "shl_rvv_clip_int8"},
    {shl_rvv_silu_fp32, "shl_rvv_silu_fp32"},
    {shl_rvv_silu_fp16, "shl_rvv_silu_fp16"},
    {shl_rvv_silu_int8, "shl_rvv_silu_int8"},
    {shl_rvv_concat_fp32, "shl_rvv_concat_fp32"},
    {shl_rvv_concat_fp16, "shl_rvv_concat_fp16"},
    {shl_rvv_concat_int8, "shl_rvv_concat_int8"},
    {shl_rvv_split_fp32, "shl_rvv_split_fp32"},
    {shl_rvv_split_fp16, "shl_rvv_split_fp16"},
    {shl_rvv_split_int8, "shl_rvv_split_int8"},
    {shl_rvv_reshape_fp32, "shl_rvv_reshape_fp32"},
    {shl_rvv_reshape_fp16, "shl_rvv_reshape_fp16"},
    {shl_rvv_reshape_int8, "shl_rvv_reshape_int8"},
    {shl_rvv_transpose_fp32, "shl_rvv_transpose_fp32"},
    {shl_rvv_transpose_fp16, "shl_rvv_transpose_fp16"},
    {shl_rvv_transpose_int8, "shl_rvv_transpose_int8"},
    {shl_rvv_gather_fp32, "shl_rvv_gather_fp32"},
    {shl_rvv_gather_fp16, "shl_rvv_gather_fp16"},
    {shl_rvv_gather_int8, "shl_rvv_gather_int8"},
    {shl_rvv_strided_slice_fp16, "shl_rvv_strided_slice_fp16"},
    {shl_rvv_add_fp32, "shl_rvv_add_fp32"},
    {shl_rvv_add_fp16, "shl_rvv_add_fp16"},
    {shl_rvv_add_int8, "shl_rvv_add_int8"},
    {shl_rvv_sub_fp32, "shl_rvv_sub_fp32"},
    {shl_rvv_sub_fp16, "shl_rvv_sub_fp16"},
    {shl_rvv_sub_int8, "shl_rvv_sub_int8"},
    {shl_rvv_mul_fp32, "shl_rvv_mul_fp32"},
    {shl_rvv_mul_fp16, "shl_rvv_mul_fp16"},
    {shl_rvv_mul_int8, "shl_rvv_mul_int8"},
    {shl_rvv_div_fp32, "shl_rvv_div_fp32"},
    {shl_rvv_div_fp16, "shl_rvv_div_fp16"},
    {shl_rvv_div_int8, "shl_rvv_div_int8"},
    {shl_rvv_reduce_sum_int8, "shl_rvv_reduce_sum_int8"},
    {shl_rvv_erf_fp32, "shl_rvv_erf_fp32"},
    {shl_rvv_erf_fp16, "shl_rvv_erf_fp16"},
    {shl_rvv_erf_int8, "shl_rvv_erf_int8"},
    {shl_rvv_layer_norm_fp32, "shl_rvv_layer_norm_fp32"},
    {shl_rvv_layer_norm_fp16, "shl_rvv_layer_norm_fp16"},
    {shl_rvv_layer_norm_int8, "shl_rvv_layer_norm_int8"},
    {shl_rvv_rms_norm_fp32, "shl_rvv_rms_norm_fp32"},
    {shl_rvv_rms_norm_fp16, "shl_rvv_rms_norm_fp16"},
    {shl_rvv_rms_norm_int8, "shl_rvv_rms_norm_int8"},
    {shl_rvv_matmul_reorder_weight_fp32, "shl_rvv_matmul_reorder_weight_fp32"},
    {shl_rvv_matmul_reorder_weight_fp16, "shl_rvv_matmul_reorder_weight_fp16"},
    {shl_rvv_matmul_reorder_weight_fp16_w_int8, "shl_rvv_matmul_reorder_weight_fp16_w_int8"},
    {shl_rvv_matmul_reorder_weight_int8, "shl_rvv_matmul_reorder_weight_int8"},
    {shl_rvv_matmul_block_fp32, "shl_rvv_matmul_block_fp32"},
    {shl_rvv_matmul_block_fp16, "shl_rvv_matmul_block_fp16"},
    {shl_rvv_matmul_block_fp16_w_int8, "shl_rvv_matmul_block_fp16_w_int8"},
    {shl_rvv_matmul_common_int8, "shl_rvv_matmul_common_int8"},
    {shl_rvv_matmul_reorder_mat0_n4_int8, "shl_rvv_matmul_reorder_mat0_n4_int8"},
    {shl_rvv_matmul_reorder_mat1_zpackn_int8, "shl_rvv_matmul_reorder_mat1_zpackn_int8"},
    {shl_rvv_matmul_4xpackn_int8, "shl_rvv_matmul_4xpackn_int8"},
    {shl_rvv_matmul_fp32, "shl_rvv_matmul_fp32"},
    {shl_rvv_matmul_fp16, "shl_rvv_matmul_fp16"},
    {shl_rvv_matmul_int8, "shl_rvv_matmul_int8"},
    {shl_rvv_pad_input_fp32, "shl_rvv_pad_input_fp32"},
    {shl_rvv_pad_input_fp16, "shl_rvv_pad_input_fp16"},
    {shl_rvv_pad_input_int8, "shl_rvv_pad_input_int8"},
    {shl_rvv_pad_input_packn_fp32, "shl_rvv_pad_input_packn_fp32"},
    {shl_rvv_pad_input_packn_fp16, "shl_rvv_pad_input_packn_fp16"},
    {shl_rvv_pad_input_packn_int8, "shl_rvv_pad_input_packn_int8"},
    {shl_rvv_pad_input_pack1ton_fp32, "shl_rvv_pad_input_pack1ton_fp32"},
    {shl_rvv_pad_input_pack1ton_fp16, "shl_rvv_pad_input_pack1ton_fp16"},
    {shl_rvv_pad_input_pack1ton_int8, "shl_rvv_pad_input_pack1ton_int8"},
    {shl_rvv_pad_input_nhwc_fp32, "shl_rvv_pad_input_nhwc_fp32"},
    {shl_rvv_pad_input_nhwc_fp16, "shl_rvv_pad_input_nhwc_fp16"},
    {shl_rvv_pad_input_nhwc_int8, "shl_rvv_pad_input_nhwc_int8"},
    {shl_rvv_avgpool_get_window_size, "shl_rvv_avgpool_get_window_size"},
    {shl_rvv_conv1d_gemm_reorder_kernel_int8, "shl_rvv_conv1d_gemm_reorder_kernel_int8"},
    {shl_rvv_conv1d_gemm_int8, "shl_rvv_conv1d_gemm_int8"},
    {shl_rvv_dwconv1d_int8, "shl_rvv_dwconv1d_int8"},
    {shl_rvv_transpose_get_tail, "shl_rvv_transpose_get_tail"},
    {shl_rvv_transpose_get_in_index, "shl_rvv_transpose_get_in_index"},
    {shl_rvv_transpose_get_out_index, "shl_rvv_transpose_get_out_index"},
    {shl_rvv_binary_op_broadcast_fp32, "shl_rvv_binary_op_broadcast_fp32"},
    {shl_rvv_binary_op_broadcast_fp16, "shl_rvv_binary_op_broadcast_fp16"},
    {shl_rvv_binary_op_broadcast_int8, "shl_rvv_binary_op_broadcast_int8"},
    {shl_rvv_embedding_int32, "shl_rvv_embedding_int32"},
    {shl_rvv_expand_dims_fp32, "shl_rvv_expand_dims_fp32"},
    {shl_rvv_expand_dims_fp16, "shl_rvv_expand_dims_fp16"},
    {shl_rvv_rope_fp32, "shl_rvv_rope_fp32"},
    {shl_rvv_rope_fp16, "shl_rvv_rope_fp16"},
    {shl_rvv_scaled_dot_product_attention_fp32, "shl_rvv_scaled_dot_product_attention_fp32"},
    {shl_rvv_scaled_dot_product_attention_fp16, "shl_rvv_scaled_dot_product_attention_fp16"},
    {shl_rvv_llm_pos_fp16, "shl_rvv_llm_pos_fp16"},
#ifdef SHL_USE_DOT_INT4
    {shl_rvv_conv2d_init_int4, "shl_rvv_conv2d_init_int4"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_int4, "shl_rvv_conv_im2col_gemm_reorder_kernel_int4"},
    {shl_rvv_conv_im2col_gemm_int4, "shl_rvv_conv_im2col_gemm_int4"},
    {shl_rvv_conv_im2col_gemm_reorder_kernel_packn_int4,
     "shl_rvv_conv_im2col_gemm_reorder_kernel_packn_int4"},
    {shl_rvv_conv_im2col_gemm_packn_int4, "shl_rvv_conv_im2col_gemm_packn_int4"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_int4, "shl_rvv_conv1x1s1_gemm_reorder_kernel_int4"},
    {shl_rvv_conv1x1s1_gemm_int4, "shl_rvv_conv1x1s1_gemm_int4"},
    {shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_int4,
     "shl_rvv_conv1x1s1_gemm_reorder_kernel_packn_int4"},
    {shl_rvv_conv1x1s1_gemm_packn_int4, "shl_rvv_conv1x1s1_gemm_packn_int4"},
    {shl_rvv_fc_gemv_transform_weight_int4_dot, "shl_rvv_fc_gemv_transform_weight_int4_dot"},
    {shl_rvv_fullyconnected_packn_int4_dot, "shl_rvv_fullyconnected_packn_int4_dot"},
#endif
    {NULL, NULL}};

char *shl_ref_get_kernel_name(void *exec);

char *shl_rvv_get_kernel_name(void *exec)
{
    char *name = shl_find_function_name(shl_rvv_kernel_map, exec);
    if (name == NULL) {
        name = shl_ref_get_kernel_name(exec);
    }
    return name;
}

int shl_rvv_conv2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                        struct csinn_conv2d_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_depthwise_conv2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                  struct csinn_conv2d_params *params,
                                  struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_conv1d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                        struct csinn_conv1d_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_deconv2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv2d_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_fullyconnected_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *weights, struct csinn_tensor *bias,
                                struct csinn_fc_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_maxpool2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_avgpool2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_add_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params,
                     struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_sub_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params,
                     struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_mul_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params,
                     struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_div_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params,
                     struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_concat_perf(struct csinn_tensor **input, struct csinn_tensor *output,
                        struct csinn_clip_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_leaky_relu_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_relu_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_relu_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_relu_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_relu6_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_global_avgpool2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params,
                                  struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_global_maxpool2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_pool_params *params,
                                  struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_reshape_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reshape_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_sigmoid_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_sigmoid_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_softmax_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_softmax_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_reduce_sum_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_reduce_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_prelu_perf(struct csinn_tensor *input, struct csinn_tensor *alpha,
                       struct csinn_tensor *output, struct csinn_prelu_params *params,
                       struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_layer_norm_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *gamma, struct csinn_tensor *beta,
                            struct csinn_layer_norm_params *params,
                            struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_clip_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_clip_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_transpose_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_transpose_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_matmul_perf(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                        struct csinn_tensor *output, struct csinn_matmul_params *params,
                        struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_gather_perf(struct csinn_tensor *input, struct csinn_tensor *indices,
                        struct csinn_tensor *output, struct csinn_gather_params *params,
                        struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_erf_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_clip_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_strided_slice_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_strided_slice_params *params,
                               struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_split_perf(struct csinn_tensor *input, struct csinn_tensor **output,
                       struct csinn_split_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_silu_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_sigmoid_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_rms_norm_perf(struct csinn_tensor *input, struct csinn_tensor *weight,
                          struct csinn_tensor *output, struct csinn_rms_norm_params *params,
                          struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_embedding_perf(struct csinn_tensor *input, struct csinn_tensor *weight,
                           struct csinn_tensor *output, struct csinn_diso_params *params,
                           struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_expand_dims_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_expand_dims_params *params,
                             struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_rope_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_rope_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_scaled_dot_product_attention_perf(struct csinn_tensor *query, struct csinn_tensor *key,
                                              struct csinn_tensor *value,
                                              struct csinn_tensor *output_tensor,
                                              struct csinn_scale_dot_attention_params *params,
                                              struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_rvv_llm_pos_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_llm_pos_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_rvv_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}
