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

/* ----------------------------------------------------------------------
 * Title:        csi_nnfunctions.h
 * Description:  Public header file for CSI NN Library
 *
 * -------------------------------------------------------------------- */

#ifndef INCLUDE_INCLUDE_XT800_CSI_I805_NNFUNCTION_H_
#define INCLUDE_INCLUDE_XT800_CSI_I805_NNFUNCTION_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "csky_vdsp2_nnfunctions.h"

/**
 * @brief u8 asym quant generic convolution optimized function
 * @param[in]       input_data            pointer to input tensor data
 * @param[in]       kernel_data           pointer to kernel tensor data
 * @param[in]       bias_data             pointer to bias tensor data
 * @param[in,out]   output_data           pointer to output tensor data
 * @param[in,out]   bufferA               pointer to buffer for input/im2col data
 * @param[in]       input_h               input height
 * @param[in]       input_w               input width
 * @param[in]       input_ch              input channel / output_channel
 * @param[in]       kernel_h              kernel height
 * @param[in]       kernel_w              kernel width
 * @param[in]       pad_h                 pad on height
 * @param[in]       pad_w                 pad on width
 * @param[in]       stride_h              stride on height
 * @param[in]       stride_w              stride on width
 * @param[in]       out_h                 output height
 * @param[in]       out_w                 output width
 * @param[in]       input_zero_point      input zero_point
 * @param[in]       kernel_zero_point     weight zero_point
 * @param[in]       output_zero_point     output zero_point
 * @param[in]       dst_mult              multiplier for s1 * s2 / s3
 * @param[in]       dst_shift             output shift for s1 * s2 / s3, shift_right
 * @return          none.
 * bufferA size: 2*input_ch*kernel_h*kernel_w
 */
void csi_i805_conv2d_opt_u8(uint8_t *input_data, uint8_t *kernel_data, int32_t *bias_data,
                            uint8_t *output_data, uint8_t *bufferA, int32_t input_h,
                            int32_t input_w, int32_t input_ch, int32_t kernel_h, int32_t kernel_w,
                            int32_t pad_h, int32_t pad_w, int32_t stride_h, int32_t stride_w,
                            int32_t out_h, int32_t out_w, int32_t out_c, int32_t input_zero_point,
                            int32_t weight_zero_point, int32_t output_zero_point, int32_t out_mult,
                            int32_t out_shift);

/**
 * @brief u8 asym quant 1x1 kernel_size convolution (pointwise convolution) optimized function
 * @param[in]       input_data            pointer to input tensor data
 * @param[in]       kernel_data           pointer to kernel tensor data
 * @param[in]       bias_data             pointer to bias tensor data
 * @param[in,out]   output_data           pointer to output tensor data
 * @param[in]       input_hxw             input height mul width
 * @param[in]       input_ch              input channel
 * @param[in]       output_ch             output_channel
 * @param[in]       input_zero_point      input zero_point
 * @param[in]       kernel_zero_point     weight zero_point
 * @param[in]       output_zero_point     output zero_point
 * @param[in]       dst_mult              multiplier for s1 * s2 / s3
 * @param[in]       dst_shift             output shift for s1 * s2 / s3, shift_right
 * @return          none.
 *
 */
void csi_i805_pwconv2d_opt_u8(uint8_t *input_data, uint8_t *kernel_data, int32_t *bias_data,
                              uint8_t *output_data, int32_t input_hxw, int32_t input_ch,
                              int32_t output_ch, int32_t input_zero_point,
                              int32_t weight_zero_point, int32_t output_zero_point,
                              int32_t out_mult, int32_t out_shift);

/**
 * @brief u8 asym quant depthwise convolution optimized function
 * @param[in]       input_data            pointer to input tensor data
 * @param[in]       kernel_data           pointer to kernel tensor data
 * @param[in]       bias_data             pointer to bias tensor data
 * @param[in,out]   output_data           pointer to output tensor data
 * @param[in,out]   bufferA               pointer to buffer for input/im2col data
 * @param[in]       input_h               input height
 * @param[in]       input_w               input width
 * @param[in]       input_ch              input channel / output_channel
 * @param[in]       kernel_h              kernel height
 * @param[in]       kernel_w              kernel width
 * @param[in]       pad_h                 pad on height
 * @param[in]       pad_w                 pad on width
 * @param[in]       stride_h              stride on height
 * @param[in]       stride_w              stride on width
 * @param[in]       out_h                 output height
 * @param[in]       out_w                 output width
 * @param[in]       input_zero_point      input zero_point
 * @param[in]       kernel_zero_point     weight zero_point
 * @param[in]       output_zero_point     output zero_point
 * @param[in]       dst_mult              multiplier for s1 * s2 / s3
 * @param[in]       dst_shift             output shift for s1 * s2 / s3, shift_right
 * @return          none.
 * bufferA size: 4*input_ch*kernel_h*kernel_w
 */
void csi_i805_dwconv2d_opt_u8(uint8_t *input_data, uint8_t *kernel_data, int32_t *bias_data,
                              uint8_t *output_data, uint8_t *bufferA, int32_t input_h,
                              int32_t input_w, int32_t input_ch, int32_t kernel_h, int32_t kernel_w,
                              int32_t pad_h, int32_t pad_w, int32_t stride_h, int32_t stride_w,
                              int32_t out_h, int32_t out_w, int32_t input_zero_point,
                              int32_t weight_zero_point, int32_t output_zero_point,
                              int32_t out_mult, int32_t out_shift);

/**
 * @brief u8 asym quant depthwise convolution 3x3 kernel_size and 1 stride optimized function
 * @param[in]       input            pointer to input tensor data
 * @param[in]       kernel           pointer to kernel tensor data
 * @param[in]       bias             pointer to bias tensor data
 * @param[in,out]   output           pointer to output tensor data
 * @param[in]       input_zero_point input zero_point
 * @param[in]       kernel_zero_point weight zero_point
 * @param[in]       output_zero_point output zero_point
 * @param[in]       dst_mult         multiplier for s1 * s2 / s3
 * @param[in]       dst_shift        output shift for s1 * s2 / s3, shift_right
 * @return          none.
 *
 */
void csi_i805_dwconv2d_3x3_opt_u8(uint8_t *input, uint8_t *kernel, int32_t *bias, uint8_t *output,
                                  int32_t input_zero_point, int32_t kernel_zero_point,
                                  int32_t output_zero_point, int32_t dst_mult, int32_t dst_shift);

/**
 * @brief u8 asym quant fullyconnected optimized function
 * @param[in]       input_data             pointer to input tensor data
 * @param[in]       weight_data            pointer to weight tensor data
 * @param[in]       bias_data              pointer to bias tensor data
 * @param[in,out]   output_data            pointer to output tensor data
 * @param[in]       in_nodes               input nodes (weight cols)
 * @param[in]       out_nodes              output nodes (weight rows)
 * @param[in]       input_zero_point       input zero_point
 * @param[in]       weight_zero_point      weight zero_point
 * @param[in]       output_zero_point      output zero_point
 * @param[in]       output_mult            multiplier for s1 * s2 / s3
 * @param[in]       output_shift           output shift for s1 * s2 / s3. shift_right
 * @return          none.
 *
 */
void csi_i805_fullyconnected_opt_u8(uint8_t *input_data, uint8_t *weight_data, int32_t *bias_data,
                                    uint8_t *output_data, int32_t in_nodes, int32_t out_nodes,
                                    int32_t input_zero_point, int32_t weight_zero_point,
                                    int32_t output_zero_point, int32_t output_mult,
                                    int32_t output_shift);

/**
 * @brief u8 asym quant generic maxpool optimized function
 * @param[in]       input_data            pointer to input tensor data
 * @param[in,out]   output_data           pointer to output tensor data
 * @param[in]       input_h               input height
 * @param[in]       input_w               input width
 * @param[in]       input_ch              input channel / output_channel
 * @param[in]       kernel_h              kernel height
 * @param[in]       kernel_w              kernel width
 * @param[in]       pad_h                 pad on height
 * @param[in]       pad_w                 pad on width
 * @param[in]       stride_h              stride on height
 * @param[in]       stride_w              stride on width
 * @param[in]       out_h                 output height
 * @param[in]       out_w                 output width
 * @return          none.
 * bufferA size: 2*input_ch*kernel_h*kernel_w
 */
void csi_i805_maxpool2d_opt_u8(uint8_t *input_data, uint8_t *output_data, int32_t input_h,
                               int32_t input_w, int32_t input_ch, int32_t kernel_h,
                               int32_t kernel_w, int32_t pad_h, int32_t pad_w, int32_t stride_h,
                               int32_t stride_w, int32_t output_h, int32_t output_w);

/**
 * @brief u8 asym quant relu optimized function
 * @param[in,out]   data                pointer to input/output tensor data, compute inplace
 * @param[in]       size                input tensor size, tensor length
 * @param[in]       input_zeropoint     input zero_point
 * @param[in]       out_multiplier      multiplier for sacle_in / scale_out
 * @param[in]       out_shift           shift left > 0
 * @return          none.
 * can be fused with conv/fc
 */
void csi_i805_relu_opt_u8(uint8_t *data, int32_t size, int32_t input_zeropoint,
                          int32_t out_multiplier, int32_t out_shift);

/**
 * @brief u8 asym quant relu6 optimized function
 * @param[in,out]   data                pointer to input/output tensor data, compute inplace
 * @param[in]       size                input tensor size, tensor length
 * @param[in]       input_zeropoint     input zero_point
 * @param[in]       out_multiplier      multiplier for sacle_in / scale_out
 * @param[in]       out_shift           shift left > 0
 * @return          none.
 * can be fused with conv/fc
 */
void csi_i805_relu6_opt_u8(uint8_t *data, int32_t size, int32_t input_zeropoint,
                           int32_t out_multiplier, int32_t out_shift);

/**
 * @brief u8 asym quant clip optimized function
 * @param[in]       input_data          pointer to input tensor data
 * @param[in,out]   output_data         pointer to output tensor data
 * @param[in]       size                input tensor size, tensor length
 * @param[in]       clip_qmin           clip min value(quant)
 * @param[in]       clip_qmax           clip max value(quant)
 * @param[in]       input_zeropoint     input zero_point
 * @param[in]       output_zeropoint    output zero_point
 * @param[in]       out_multiplier      multiplier for sacle_in / scale_out
 * @param[in]       out_shift           shift left > 0
 * @return          none.
 * can be fused with conv/fc
 */
void csi_i805_clip_opt_u8(uint8_t *input_data, uint8_t *output_data, int32_t size, int32_t clip_min,
                          int32_t clip_max, int32_t input_zeropoint, int32_t output_zeropoint,
                          int32_t out_multiplier, int32_t out_shift);

/**
 * @brief u8 asym quant element add optimized function
 * @param[in]       input_0             pointer to input_0 tensor data
 * @param[in]       input_1             pointer to input_1 tensor data
 * @param[in,out]   output              pointer to output tensor data
 * @param[in]       size                input tensor size, tensor length, element size
 * @param[in]       input_0_zeroponit   input_0 zero_point. Range: Range: -255 to 0
 * @param[in]       input_0_mult        multiplier for sacle_input_0
 * @param[in]       input_0_shift       input_0 shift
 * @param[in]       input_1_zeropoint   input_1 zero_point. Range: Range: -255 to 0
 * @param[in]       input_1_mult        multiplier for sacle_input_1
 * @param[in]       input_1_shift       input_1 shift
 * @param[in]       output_zeropoint    output zero_point
 * @param[in]       output_mult         multiplier for scale_output
 * @param[in]       output_shift        output shift
 * @return          none.
 *
 */
void csi_i805_elementwise_add_opt_u8(uint8_t *input_0, uint8_t *input_1, uint8_t *output,
                                     int32_t size, int32_t input_0_zeroponit, int32_t input_0_mult,
                                     int32_t input_0_shift, int32_t input_1_zeropoint,
                                     int32_t input_1_mult, int32_t input_1_shift,
                                     int32_t output_zeropoint, int32_t output_mult,
                                     int32_t output_shift);

/**
 * @brief u8 asym quant element mul optimized function
 * @param[in]       input_0             pointer to input_0 tensor data
 * @param[in]       input_1             pointer to input_1 tensor data
 * @param[in,out]   output              pointer to output tensor data
 * @param[in]       size                input tensor size, tensor length, element size
 * @param[in]       input_0_zeroponit   input_0 zero_point
 * @param[in]       input_1_zeropoint   input_1 zero_point
 * @param[in]       output_zeropoint    output zero_point
 * @param[in]       output_mult         multiplier for s1 * s2 / s3
 * @param[in]       output_shift        output shift for s1 * s2 / s3
 * @return          none.
 *
 */
void csi_i805_elementwise_mul_opt_u8(uint8_t *input_0, uint8_t *input_1, uint8_t *output,
                                     int32_t size, int32_t input_0_zeroponit,
                                     int32_t input_1_zeropoint, int32_t output_zeropoint,
                                     int32_t output_mult, int32_t output_shift);

/**
 * @brief u8 asym quant softmax optimized function
 * @param[in]       input_data             pointer to input tensor data
 * @param[in,out]   output_data            pointer to output tensor data
 * @param[in]       size                   tensor size
 * @param[in]       out_mult               multiplier
 * @param[in]       out_shift              output shift
 * @return          none.
 *
 */
void csi_i805_softmax_opt_u8(uint8_t *input_data, uint8_t *output_data, int32_t size,
                             int32_t out_mult, int32_t out_shift);

/**
 * @brief u8 asym quant reshape optimized function
 * @param[in]       input_data             pointer to input tensor data
 * @param[in,out]   output_data            pointer to output tensor data
 * @param[in]       size                   tensor size
 * @return          none.
 *
 */
void csi_i805_reshape_opt_u8(uint8_t *input_data, uint8_t *output_data, int32_t size);

/**
 * @brief u8 asym quant vec and matrix mul optimized function
 * @param[in]       lhs              pointer to input tensor data
 * @param[in]       rhs              pointer to weight tensor data
 * @param[in]       bias             pointer to bias tensor data
 * @param[in,out]   dst              pointer to output tensor data
 * @param[in]       rhs_col          input nodes (weight cols)
 * @param[in]       rhs_row          output nodes (weight rows)
 * @param[in]       lhs_zero_point   input zero_point
 * @param[in]       rhs_zero_point   weight zero_point
 * @param[in]       dst_zero_point   output zero_point
 * @param[in]       dst_mult         multiplier for s1 * s2 / s3
 * @param[in]       dst_shift        output shift for s1 * s2 / s3
 * @return          none.
 *
 */
void csi_i805_vec_mat_mult_opt_u8(uint8_t *lhs, uint8_t *rhs, int32_t *bias, uint8_t *dst,
                                  int32_t rhs_col, int32_t rhs_row, int32_t lhs_zero_point,
                                  int32_t rhs_zero_point, int32_t dst_zero_point, int32_t dst_mult,
                                  int32_t dst_shift);

/**
 * @brief u8 asym quant matrix mul(A * B_trans) optimized function
 * @param[in]       lhs              pointer to input tensor data
 * @param[in]       rhs              pointer to weight tensor data
 * @param[in]       bias             pointer to bias tensor data
 * @param[in,out]   dst              pointer to output tensor data
 * @param[in]       lhs_row          input row / m
 * @param[in]       lhs_col          input col / k
 * @param[in]       rhs_row          weight row / n
 * @param[in]       lhs_zero_point   input zero_point
 * @param[in]       rhs_zero_point   weight zero_point
 * @param[in]       dst_zero_point   output zero_point
 * @param[in]       dst_mult         multiplier for s1 * s2 / s3
 * @param[in]       dst_shift        output shift for s1 * s2 / s3
 * @return          none.
 *
 */
void csi_i805_mat_mult_nt_t_opt_u8(uint8_t *lhs, uint8_t *rhs, int32_t *bias, uint8_t *dst,
                                   int32_t lhs_row, int32_t lhs_col, int32_t rhs_row,
                                   int32_t lhs_zero_point, int32_t rhs_zero_point,
                                   int32_t dst_zero_point, int32_t dst_mult, int32_t dst_shift);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_INCLUDE_XT800_CSI_I805_NNFUNCTION_H_
