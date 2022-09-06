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

#ifndef SOURCE_I805_OPT_I805_FUNCTION_H_
#define SOURCE_I805_OPT_I805_FUNCTION_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * @brief 8-bit fractional data type in 1.7 format.
 */
typedef int8_t q7_t;

/**
 * @brief 16-bit fractional data type in 1.15 format.
 */
typedef int16_t q15_t;

/**
 * @brief 32-bit fractional data type in 1.31 format.
 */
typedef int32_t q31_t;

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
void shl_i805_conv2d_opt_u8(uint8_t *input_data, uint8_t *kernel_data, int32_t *bias_data,
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
void shl_i805_pwconv2d_opt_u8(uint8_t *input_data, uint8_t *kernel_data, int32_t *bias_data,
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
void shl_i805_dwconv2d_opt_u8(uint8_t *input_data, uint8_t *kernel_data, int32_t *bias_data,
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
void shl_i805_dwconv2d_3x3_opt_u8(uint8_t *input, uint8_t *kernel, int32_t *bias, uint8_t *output,
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
void shl_i805_fullyconnected_opt_u8(uint8_t *input_data, uint8_t *weight_data, int32_t *bias_data,
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
void shl_i805_maxpool2d_opt_u8(uint8_t *input_data, uint8_t *output_data, int32_t input_h,
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
void shl_i805_relu_opt_u8(uint8_t *data, int32_t size, int32_t input_zeropoint,
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
void shl_i805_relu6_opt_u8(uint8_t *data, int32_t size, int32_t input_zeropoint,
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
void shl_i805_clip_opt_u8(uint8_t *input_data, uint8_t *output_data, int32_t size, int32_t clip_min,
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
void shl_i805_elementwise_add_opt_u8(uint8_t *input_0, uint8_t *input_1, uint8_t *output,
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
void shl_i805_elementwise_mul_opt_u8(uint8_t *input_0, uint8_t *input_1, uint8_t *output,
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
void shl_i805_softmax_opt_u8(uint8_t *input_data, uint8_t *output_data, int32_t size,
                             int32_t out_mult, int32_t out_shift);

/**
 * @brief u8 asym quant reshape optimized function
 * @param[in]       input_data             pointer to input tensor data
 * @param[in,out]   output_data            pointer to output tensor data
 * @param[in]       size                   tensor size
 * @return          none.
 *
 */
void shl_i805_reshape_opt_u8(uint8_t *input_data, uint8_t *output_data, int32_t size);

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
void shl_i805_vec_mat_mult_opt_u8(uint8_t *lhs, uint8_t *rhs, int32_t *bias, uint8_t *dst,
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
void shl_i805_mat_mult_nt_t_opt_u8(uint8_t *lhs, uint8_t *rhs, int32_t *bias, uint8_t *dst,
                                   int32_t lhs_row, int32_t lhs_col, int32_t rhs_row,
                                   int32_t lhs_zero_point, int32_t rhs_zero_point,
                                   int32_t dst_zero_point, int32_t dst_mult, int32_t dst_shift);

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
void shl_i805_conv2d_opt_u8(uint8_t *input_data, uint8_t *kernel_data, int32_t *bias_data,
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
void shl_i805_pwconv2d_opt_u8(uint8_t *input_data, uint8_t *kernel_data, int32_t *bias_data,
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
void shl_i805_dwconv2d_opt_u8(uint8_t *input_data, uint8_t *kernel_data, int32_t *bias_data,
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
void shl_i805_dwconv2d_3x3_opt_u8(uint8_t *input, uint8_t *kernel, int32_t *bias, uint8_t *output,
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
void shl_i805_fullyconnected_opt_u8(uint8_t *input_data, uint8_t *weight_data, int32_t *bias_data,
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
void shl_i805_maxpool2d_opt_u8(uint8_t *input_data, uint8_t *output_data, int32_t input_h,
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
void shl_i805_relu_opt_u8(uint8_t *data, int32_t size, int32_t input_zeropoint,
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
void shl_i805_relu6_opt_u8(uint8_t *data, int32_t size, int32_t input_zeropoint,
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
void shl_i805_clip_opt_u8(uint8_t *input_data, uint8_t *output_data, int32_t size, int32_t clip_min,
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
void shl_i805_elementwise_add_opt_u8(uint8_t *input_0, uint8_t *input_1, uint8_t *output,
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
void shl_i805_elementwise_mul_opt_u8(uint8_t *input_0, uint8_t *input_1, uint8_t *output,
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
void shl_i805_softmax_opt_u8(uint8_t *input_data, uint8_t *output_data, int32_t size,
                             int32_t out_mult, int32_t out_shift);

/**
 * @brief u8 asym quant reshape optimized function
 * @param[in]       input_data             pointer to input tensor data
 * @param[in,out]   output_data            pointer to output tensor data
 * @param[in]       size                   tensor size
 * @return          none.
 *
 */
void shl_i805_reshape_opt_u8(uint8_t *input_data, uint8_t *output_data, int32_t size);

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
void shl_i805_vec_mat_mult_opt_u8(uint8_t *lhs, uint8_t *rhs, int32_t *bias, uint8_t *dst,
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
void shl_i805_mat_mult_nt_t_opt_u8(uint8_t *lhs, uint8_t *rhs, int32_t *bias, uint8_t *dst,
                                   int32_t lhs_row, int32_t lhs_col, int32_t rhs_row,
                                   int32_t lhs_zero_point, int32_t rhs_zero_point,
                                   int32_t dst_zero_point, int32_t dst_mult, int32_t dst_shift);

/**
 * @brief Struct for specifying activation function types
 *
 */
typedef enum {
    CSKY_SIGMOID = 0, /**< Sigmoid activation function */
    CSKY_TANH = 1,    /**< Tanh activation function */
} csky_vdsp2_nn_activation_type;

/**
 * @brief Basic Q7 convolution function
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in   input tensor dimention
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       wt          pointer to kernel weights
 * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel  filter kernel size
 * @param[in]       padding     padding sizes
 * @param[in]       stride      convolution stride
 * @param[in]       bias        pointer to bias
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in,out]   Im_out      pointer to output tensor
 * @param[in]       dim_im_out  output tensor dimension
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @return          none.
 *
 */

void csky_vdsp2_convolve_HWC_q7_basic(const q7_t *Im_in, const uint16_t dim_im_in,
                                      const uint16_t ch_im_in, const q7_t *wt,
                                      const uint16_t ch_im_out, const uint16_t dim_kernel,
                                      const uint16_t padding, const uint16_t stride,
                                      const q7_t *bias, const uint16_t bias_shift,
                                      const uint16_t out_shift, q7_t *Im_out,
                                      const uint16_t dim_im_out, q15_t *bufferA);

/**
 * @brief Basic Q15 convolution function
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in   input tensor dimention
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       wt          pointer to kernel weights
 * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel  filter kernel size
 * @param[in]       padding     padding sizes
 * @param[in]       stride      convolution stride
 * @param[in]       bias        pointer to bias
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in,out]   Im_out      pointer to output tensor
 * @param[in]       dim_im_out  output tensor dimension
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @return          none.
 *
 */

void csky_vdsp2_convolve_HWC_q15_basic(const q15_t *Im_in, const uint16_t dim_im_in,
                                       const uint16_t ch_im_in, const q15_t *wt,
                                       const uint16_t ch_im_out, const uint16_t dim_kernel,
                                       const uint16_t padding, const uint16_t stride,
                                       const q15_t *bias, const uint16_t bias_shift,
                                       const uint16_t out_shift, q15_t *Im_out,
                                       const uint16_t dim_im_out, q15_t *bufferA);

/**
 * @brief Fast Q7 convolution function (non-sqaure shape)
 * @param[in]       Im_in        pointer to input tensor
 * @param[in]       dim_im_in_x  input tensor dimention x
 * @param[in]       dim_im_in_y  input tensor dimention y
 * @param[in]       ch_im_in     number of input tensor channels
 * @param[in]       wt           pointer to kernel weights
 * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel_x filter kernel size x
 * @param[in]       dim_kernel_y filter kernel size y
 * @param[in]       padding_x    padding size x
 * @param[in]       padding_y    padding size y
 * @param[in]       stride_x     convolution stride x
 * @param[in]       stride_y     convolution stride y
 * @param[in]       bias         pointer to bias
 * @param[in]       bias_shift   amount of left-shift for bias
 * @param[in]       out_shift    amount of right-shift for output
 * @param[in,out]   Im_out       pointer to output tensor
 * @param[in]       dim_im_out_x output tensor dimension x
 * @param[in]       dim_im_out_y output tensor dimension y
 * @param[in,out]   bufferA      pointer to buffer space for input
 * @return          none.
 *
 * This function is the version with full list of optimization tricks, but with
 * some contraints:
 *   ch_im_in is multiple of 4
 *   ch_im_out is multiple of 2
 */

void csky_vdsp2_convolve_HWC_q7_fast_nonsquare(
    const q7_t *Im_in, const uint16_t dim_im_in_x, const uint16_t dim_im_in_y,
    const uint16_t ch_im_in, const q7_t *wt, const uint16_t ch_im_out, const uint16_t dim_kernel_x,
    const uint16_t dim_kernel_y, const uint16_t padding_x, const uint16_t padding_y,
    const uint16_t stride_x, const uint16_t stride_y, const q7_t *bias, const uint16_t bias_shift,
    const uint16_t out_shift, q7_t *Im_out, const uint16_t dim_im_out_x,
    const uint16_t dim_im_out_y, q15_t *bufferA);

/**
 * @brief Fast Q7 version of 1x1 convolution (non-sqaure shape)
 * @param[in]       Im_in        pointer to input tensor
 * @param[in]       dim_im_in_x  input tensor dimention x
 * @param[in]       dim_im_in_y  input tensor dimention y
 * @param[in]       ch_im_in     number of input tensor channels
 * @param[in]       wt           pointer to kernel weights
 * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel_x filter kernel size x
 * @param[in]       dim_kernel_y filter kernel size y
 * @param[in]       padding_x    padding size x
 * @param[in]       padding_y    padding size y
 * @param[in]       stride_x     convolution stride x
 * @param[in]       stride_y     convolution stride y
 * @param[in]       bias         pointer to bias
 * @param[in]       bias_shift   amount of left-shift for bias
 * @param[in]       out_shift    amount of right-shift for output
 * @param[in,out]   Im_out       pointer to output tensor
 * @param[in]       dim_im_out_x output tensor dimension x
 * @param[in]       dim_im_out_y output tensor dimension y
 * @param[in,out]   bufferA      pointer to buffer space for input
 * @return          none.
 *
 * This function implement convolution with 1x1 kernel size (i.e., dim_kernel_x=1
 * and dim_kernel_y=1). It can be used for
 * second half of MobileNets after depthwise separable convolution.
 *
 * This function is the version with full list of optimization tricks, but with
 * some contraints:
 *   ch_im_in is multiple of 4
 *   ch_im_out is multiple of 2
 */
void csky_vdsp2_convolve_1x1_HWC_q7_fast(const q7_t *Im_in, const uint16_t dim_im_in_x,
                                         const uint16_t dim_im_in_y, const uint16_t ch_im_in,
                                         const q7_t *wt, const uint16_t ch_im_out, const q7_t *bias,
                                         const uint16_t bias_shift, const uint16_t out_shift,
                                         q7_t *Im_out, const uint16_t dim_im_out_x,
                                         const uint16_t dim_im_out_y, q15_t *bufferA);

/**
 * @brief Q7 version of convolution for RGB image
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in   input tensor dimention
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       wt          pointer to kernel weights
 * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel  filter kernel size
 * @param[in]       padding     padding sizes
 * @param[in]       stride      convolution stride
 * @param[in]       bias        pointer to bias
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in,out]   Im_out      pointer to output tensor
 * @param[in]       dim_im_out  output tensor dimension
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @return          none.
 *
 * This kernel is written exclusively for convolution with ch_im_in
 * equals 3. This applies on the first layer of CNNs which has input
 * image with RGB format.
 */

void csky_vdsp2_convolve_HWC_q7_RGB(const q7_t *Im_in, const uint16_t dim_im_in, const q7_t *wt,
                                    const uint16_t ch_im_out, const uint16_t dim_kernel,
                                    const uint16_t padding, const uint16_t stride, const q7_t *bias,
                                    const uint16_t bias_shift, const uint16_t out_shift,
                                    q7_t *Im_out, const uint16_t dim_im_out, q15_t *bufferA);

/**
 * @brief Q7 depthwise separable convolution function
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in   input tensor dimention
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       wt          pointer to kernel weights
 * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel  filter kernel size
 * @param[in]       padding     padding sizes
 * @param[in]       stride      convolution stride
 * @param[in]       bias        pointer to bias
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in,out]   Im_out      pointer to output tensor
 * @param[in]       dim_im_out  output tensor dimension
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @return          none.
 *
 * This function is the version with full list of optimization tricks, but with
 * some contraints:
 *   ch_im_in is multiple of 2
 *   ch_im_out is multiple of 2
 */

void csky_vdsp2_depthwise_separable_conv_HWC_q7(const q7_t *Im_in, const uint16_t dim_im_in,
                                                const uint16_t ch_im_in, const q7_t *wt,
                                                const uint16_t ch_im_out, const uint16_t dim_kernel,
                                                const uint16_t padding, const uint16_t stride,
                                                const q7_t *bias, const uint16_t bias_shift,
                                                const uint16_t out_shift, q7_t *Im_out,
                                                const uint16_t dim_im_out, q15_t *bufferA);

/**
 * @brief Q7 depthwise separable convolution function (non-square shape)
 * @param[in]       Im_in         pointer to input tensor
 * @param[in]       dim_im_in_x   input tensor dimention x
 * @param[in]       dim_im_in_y   input tensor dimention y
 * @param[in]       ch_im_in      number of input tensor channels
 * @param[in]       wt            pointer to kernel weights
 * @param[in]       ch_im_out     number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel_x  filter kernel size x
 * @param[in]       dim_kernel_y  filter kernel size y
 * @param[in]       padding_x     padding sizes x
 * @param[in]       padding_y     padding sizes y
 * @param[in]       stride_x      convolution stride x
 * @param[in]       stride_y      convolution stride y
 * @param[in]       bias          pointer to bias
 * @param[in]       bias_shift    amount of left-shift for bias
 * @param[in]       out_shift     amount of right-shift for output
 * @param[in,out]   Im_out        pointer to output tensor
 * @param[in]       dim_im_out_x  output tensor dimension x
 * @param[in]       dim_im_out_y  output tensor dimension y
 * @param[in,out]   bufferA       pointer to buffer space for input
 * @return          none.
 *
 * This function is the version with full list of optimization tricks, but with
 * some contraints:
 *   ch_im_in is multiple of 2
 *   ch_im_out is multiple of 2
 */
void csky_vdsp2_depthwise_separable_conv_HWC_q7_nonsquare(
    const q7_t *Im_in, const uint16_t dim_im_in_x, const uint16_t dim_im_in_y,
    const uint16_t ch_im_in, const q7_t *wt, const uint16_t ch_im_out, const uint16_t dim_kernel_x,
    const uint16_t dim_kernel_y, const uint16_t padding_x, const uint16_t padding_y,
    const uint16_t stride_x, const uint16_t stride_y, const q7_t *bias, const uint16_t bias_shift,
    const uint16_t out_shift, q7_t *Im_out, const uint16_t dim_im_out_x,
    const uint16_t dim_im_out_y, q15_t *bufferA);

/**
 * @brief Q7 basic fully-connected layer function
 * @param[in]       pV          pointer to input vector
 * @param[in]       pM          pointer to matrix weights
 * @param[in]       dim_vec     length of the vector
 * @param[in]       num_of_rows number of rows in weight matrix
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in]       bias        pointer to bias
 * @param[in,out]   pOut        pointer to output vector
 * @return          none.
 */

void csky_vdsp2_fully_connected_q7(const q7_t *pV, const q7_t *pM, const uint16_t dim_vec,
                                   const uint16_t num_of_rows, const uint16_t bias_shift,
                                   const uint16_t out_shift, const q7_t *bias, q7_t *pOut);

/**
 * @brief Q15 basic fully-connected layer function
 * @param[in]       pV          pointer to input vector
 * @param[in]       pM          pointer to matrix weights
 * @param[in]       dim_vec     length of the vector
 * @param[in]       num_of_rows number of rows in weight matrix
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in]       bias        pointer to bias
 * @param[in,out]   pOut        pointer to output vector
 * @return          none.
 *
 */

void csky_vdsp2_fully_connected_q15(const q15_t *pV, const q15_t *pM, const uint16_t dim_vec,
                                    const uint16_t num_of_rows, const uint16_t bias_shift,
                                    const uint16_t out_shift, const q15_t *bias, q15_t *pOut);

/**
 * @brief Mixed Q15-Q7 fully-connected layer function
 * @param[in]       pV          pointer to input vector
 * @param[in]       pM          pointer to matrix weights
 * @param[in]       dim_vec     length of the vector
 * @param[in]       num_of_rows number of rows in weight matrix
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in]       bias        pointer to bias
 * @param[in,out]   pOut        pointer to output vector
 * @return          none.
 *
 */

void csky_vdsp2_fully_connected_mat_q7_vec_q15(const q15_t *pV, const q7_t *pM,
                                               const uint16_t dim_vec, const uint16_t num_of_rows,
                                               const uint16_t bias_shift, const uint16_t out_shift,
                                               const q7_t *bias, q15_t *pOut);

/**
 * @brief Q7 RELU function
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 * @return none.
 */

void csky_vdsp2_relu_q7(q7_t *data, uint16_t size);

/**
 * @brief Q15 RELU function
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 * @return none.
 */

void csky_vdsp2_relu_q15(q15_t *data, uint16_t size);

/**
 * @brief Q7 neural network activation function using direct table look-up
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 * @param[in]       int_width   bit-width of the integer part, assume to be smaller than 3
 * @param[in]       type        type of activation functions
 * @return none.
 */

void csky_vdsp2_nn_activations_direct_q7(q7_t *data, uint16_t size, uint16_t int_width,
                                         csky_vdsp2_nn_activation_type type);

/**
 * @brief Q15 neural network activation function using direct table look-up
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 * @param[in]       int_width   bit-width of the integer part, assume to be smaller than 3
 * @param[in]       type        type of activation functions
 * @return none.
 */

void csky_vdsp2_nn_activations_direct_q15(q15_t *data, uint16_t size, uint16_t int_width,
                                          csky_vdsp2_nn_activation_type type);

/**
 * @brief Q7 max pooling function
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in   input tensor dimention
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       dim_kernel  filter kernel size
 * @param[in]       padding     padding sizes
 * @param[in]       stride      convolution stride
 * @param[in]       dim_im_out  output tensor dimension
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @param[in,out]   Im_out      pointer to output tensor
 * @return none.
 *
 */

void csky_vdsp2_maxpool2d_q7_HWC(q7_t *Im_in, const uint16_t dim_im_in, const uint16_t ch_im_in,
                                 const uint16_t dim_kernel, const uint16_t padding,
                                 const uint16_t stride, const uint16_t dim_im_out, q7_t *bufferA,
                                 q7_t *Im_out);

/**
 * @brief Q7 average pooling function
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in   input tensor dimention
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       dim_kernel  filter kernel size
 * @param[in]       padding     padding sizes
 * @param[in]       stride      convolution stride
 * @param[in]       dim_im_out  output tensor dimension
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @param[in,out]   Im_out      pointer to output tensor
 * @return none.
 *
 */

void csky_vdsp2_avepool_q7_HWC(q7_t *Im_in, const uint16_t dim_im_in, const uint16_t ch_im_in,
                               const uint16_t dim_kernel, const uint16_t padding,
                               const uint16_t stride, const uint16_t dim_im_out, q7_t *bufferA,
                               q7_t *Im_out);

void csky_vdsp2_avepool_q7_HWC_nonsquare(q7_t *Im_in,                 // input image
                                         const uint16_t dim_im_in_x,  // input image dimension
                                         const uint16_t dim_im_in_y,  // input image dimension
                                         const uint16_t ch_im_in,  // number of input image channels
                                         const uint16_t dim_kernel_x,  // window kernel size
                                         const uint16_t dim_kernel_y,  // window kernel size
                                         const uint16_t padding_x,     // padding sizes
                                         const uint16_t padding_y,     // padding sizes
                                         const uint16_t stride_x,      // stride
                                         const uint16_t stride_y,      // stride
                                         const uint16_t dim_im_out_x,  // output image dimension
                                         const uint16_t dim_im_out_y,  // output image dimension
                                         q7_t *bufferA,                // a buffer for local storage
                                         q7_t *Im_out,                 // output feature
                                         const uint16_t out_lshift);  // output left shift (scaling)

/**
 * @brief Q7 softmax function
 * @param[in]       vec_in      pointer to input vector
 * @param[in]       dim_vec     input vector dimention
 * @param[out]      p_out       pointer to output vector
 * @return none.
 *
 */

void csky_vdsp2_softmax_q7(const q7_t *vec_in, const uint16_t dim_vec, q7_t *p_out);

/**
 * @brief Q15 softmax function
 * @param[in]       vec_in      pointer to input vector
 * @param[in]       dim_vec     input vector dimention
 * @param[out]      p_out       pointer to output vector
 * @return none.
 *
 */

void csky_vdsp2_softmax_q15(const q15_t *vec_in, const uint16_t dim_vec, q15_t *p_out);

#ifdef __cplusplus
}
#endif

#endif  // SOURCE_I805_OPT_I805_FUNCTION_H_
