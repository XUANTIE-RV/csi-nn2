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

/**
 * @file csi_nn.h
 */

#ifndef INCLUDE_CSI_NN_H_
#define INCLUDE_CSI_NN_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "csinn_data_structure.h"
#include "csinn_runtime.h"
#include "shl_debug.h"
#include "shl_memory.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup NN NN function
 * @defgroup INIT Initialize function
 */

/**
 * @brief       Two-dimensional convolution initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_conv2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                      struct csinn_conv2d_params *params);

/**
 * @brief       Two-dimensional convolution function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 *
 * @details
 *      - Support floating point and 8-bit fixed point.
 *      - Support NCHW and NHWC layouts.
 */
int csinn_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                 struct csinn_conv2d_params *params);

/**
 * @brief       Two-dimensional depthwise convolution initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_depthwise_conv2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);

/**
 * @brief       Two-dimensional depthwise convolution function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_depthwise_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                           struct csinn_conv2d_params *params);

/**
 * @brief       Two-dimensional group convolution initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_group_conv2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                            struct csinn_conv2d_params *params);

/**
 * @brief       Two-dimensional group convolution function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_group_conv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                       struct csinn_conv2d_params *params);

/**
 * @brief       Two-dimensional convolution and ReLU fusion initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_conv2d_relu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_tensor *kernel, struct csinn_tensor *bias,
                           struct csinn_conv2d_params *params);

/**
 * @brief       Two-dimensional convolution and ReLU fusion function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_conv2d_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                      struct csinn_conv2d_params *params);

/**
 * @brief       Two-dimensional depthwise convolution and ReLU fusion initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_depthwise_conv2d_relu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                     struct csinn_conv2d_params *params);

/**
 * @brief       Two-dimensional depthwise convolution and ReLU fusion function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_depthwise_conv2d_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_conv2d_params *params);

/**
 * @brief       Two-dimensional convolution and ReLU6 fusion initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_conv2d_relu6_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                            struct csinn_conv2d_params *params);

/**
 * @brief       Two-dimensional convolution and ReLU6 fusion function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_conv2d_relu6(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                       struct csinn_conv2d_params *params);

/**
 * @brief       Two-dimensional deconvolution initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_deconv2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                        struct csinn_conv2d_params *params);

/**
 * @brief       Two-dimensional deconvolution function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_deconv2d(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                   struct csinn_conv2d_params *params);

/**
 * @brief       Three-dimensional convolution initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_conv3d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                      struct csinn_conv3d_params *params);

/**
 * @brief       Three-dimensional convolution function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_conv3d(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                 struct csinn_conv3d_params *params);

/**
 * @brief       Three-dimensional deconvolution initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_deconv3d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tensor *kernel, struct csinn_tensor *bias,
                        struct csinn_conv3d_params *params);

/**
 * @brief       Three-dimensional deconvolution function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_deconv3d(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                   struct csinn_conv3d_params *params);

/**
 * @brief       Feedforward Sequential Memory Network initialization function
 *
 * @param[in]   frame           Point to the current input frame data
 * @param[in]   l_filter        Left coefficient matrix, used for matrix calculation with past
 *                              frames
 * @param[in]   r_filter        Right coefficient matrix for matrix calculation with future frames
 * @param[in]   frame_sequence  Point to all currently calculated frame data
 * @param[in]   frame_counter   Frame counter
 * @param[out]  output          Pointer to the output tensor
 * @param[in]   params          FSMN parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_fsmn_init(struct csinn_tensor *frame, struct csinn_tensor *l_filter,
                    struct csinn_tensor *r_filter, struct csinn_tensor *frame_sequence,
                    struct csinn_tensor *frame_counter, struct csinn_tensor *output,
                    struct csinn_fsmn_params *params);

/**
 * @brief       Feedforward Sequential Memory Network function
 *
 * @param[in]   frame           Point to the current input frame data
 * @param[in]   l_filter        Left coefficient matrix, used for matrix calculation with past
 *                              frames
 * @param[in]   r_filter        Right coefficient matrix for matrix calculation with future frames
 * @param[in]   frame_sequence  Point to all currently calculated frame data
 * @param[in]   frame_counter   Frame counter
 * @param[out]  output          Pointer to the output tensor
 * @param[in]   params          FSMN parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 *
 * @details     FSMN is essentially a feedforward full connection network (FNN). Its
 *              innovation lies in the addition of a memory block in its hidden layer.
 *              The function of the memory module is to encode the front and rear units
 *              of each hidden state together, so as to capture the front and rear relations
 *              of the sequence.
 */
int csinn_fsmn(struct csinn_tensor *frame, struct csinn_tensor *l_filter,
               struct csinn_tensor *r_filter, struct csinn_tensor *frame_sequence,
               struct csinn_tensor *frame_counter, struct csinn_tensor *output,
               struct csinn_fsmn_params *params);

/**
 * @brief       Fully Connected initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   weights Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Fully Connected parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_fullyconnected_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *weights, struct csinn_tensor *bias,
                              struct csinn_fc_params *params);

/**
 * @brief       Fully Connected function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   weights Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Fully Connected parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_fullyconnected(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *weights, struct csinn_tensor *bias,
                         struct csinn_fc_params *params);

/**
 * @brief       Fully Connected and ReLU fusion initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   weights Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Fully Connected parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_fullyconnected_relu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *weights, struct csinn_tensor *bias,
                                   struct csinn_fc_params *params);

/**
 * @brief       Fully Connected and ReLU fusion function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   weights Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Fully Connected parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_fullyconnected_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *weights, struct csinn_tensor *bias,
                              struct csinn_fc_params *params);

/**
 * @brief       Two-dimensional max pooling initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_maxpool2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_pool_params *params);

/**
 * @brief       Two-dimensional max pooling function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_maxpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_pool_params *params);

/**
 * @brief       Three-dimensional max pooling initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_maxpool3d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_pool_params *params);

/**
 * @brief       Three-dimensional max pooling function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_maxpool3d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_pool_params *params);

/**
 * @brief       Two-dimensional global max pooling initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_global_maxpool2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);

/**
 * @brief       Two-dimensional global max pooling function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 *
 * @details     <code>csinn_global_maxpool2d</code> is a special case for
 *              <code>csinn_maxpool2d</code>.
 */
int csinn_global_maxpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params);

/**
 * @brief       Two-dimensional average pooling initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_avgpool2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_pool_params *params);

/**
 * @brief       Two-dimensional average pooling function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_avgpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_pool_params *params);

/**
 * @brief       Three-dimensional average pooling initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_avgpool3d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_pool_params *params);

/**
 * @brief       Three-dimensional average pooling function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_avgpool3d(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_pool_params *params);

/**
 * @brief       Two-dimensional global average pooling initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_global_avgpool2d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);

/**
 * @brief       Two-dimensional global average pooling function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 *
 * @details     <code>csinn_global_avgpool2d</code> is a special case for
 *              <code>csinn_avgpool2d</code>.
 */
int csinn_global_avgpool2d(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params);

/**
 * @brief       L2 pooling initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_l2pool_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_pool_params *params);

/**
 * @brief       L2 pooling function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_l2pool(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_pool_params *params);

/**
 * @brief       Pooling and argmax fusion initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_pool_with_argmax_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_pool_params *params);

/**
 * @brief       Pooling and argmax fusion function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_pool_with_argmax(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_pool_params *params);

/**
 * @brief       Max pooling and with locating information initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_maxpool2d_locat_init(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_pool_params *params);

/**
 * @brief       Max pooling and with locating information function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_maxpool2d_locat(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_pool_params *params);

/**
 * @brief       Unpooling initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[in]   mask    Locating information
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Unpooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_unpooling_init(struct csinn_tensor *input, struct csinn_tensor *mask,
                         struct csinn_tensor *output, struct csinn_unpooling_params *params);

/**
 * @brief       Unpooling function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[in]   mask    Locating information
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Unpooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_unpooling(struct csinn_tensor *input, struct csinn_tensor *mask,
                    struct csinn_tensor *output, struct csinn_unpooling_params *params);

/**
 * @brief       ROI align initialization function
 *
 * @param[in]   data    Pointer to the input tensor
 * @param[in]   rois    ROI information
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  ROI align parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_roi_align_init(struct csinn_tensor *data, struct csinn_tensor *rois,
                         struct csinn_tensor *output, struct csinn_roi_align_params *params);

/**
 * @brief       ROI align function
 *
 * @param[in]   data    Pointer to the input tensor
 * @param[in]   rois    ROI information
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  ROI align parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_roi_align(struct csinn_tensor *data, struct csinn_tensor *rois,
                    struct csinn_tensor *output, struct csinn_roi_align_params *params);

/**
 * @brief       Negative initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Negative parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_negative_init(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>negtive</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Negative parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_negative(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

/**
 * @brief       Floor initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Floor parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_floor_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>floor</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Floor parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_floor(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

/**
 * @brief       Ceil initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Ceil parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_ceil_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>ceil</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Ceil parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_ceil(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

/**
 * @brief       Sign initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Sign parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sign_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>sign</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Sign parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sign(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

/**
 * @brief       Trunc initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Trunc parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_trunc_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>trunc</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Trunc parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_trunc(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

/**
 * @brief       Round initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Round parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_round_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>round</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Round parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_round(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

/**
 * @brief       Abs initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Abs parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_abs_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>abs</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Abs parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_abs(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

/**
 * @brief       Isnan bool initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Isnan bool parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_isnan_bool_init(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>isnan bool</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Isnan bool parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_isnan_bool(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

/**
 * @brief       Exp initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Exp parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_exp_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>exp</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Exp parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_exp(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

/**
 * @brief       Expm1 initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Expm1 parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_expm1_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>expm1</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Expm1 parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_expm1(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

/**
 * @brief       sin initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  sin parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sin_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>sin</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  sin parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sin(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

/**
 * @brief       col initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  col parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_cos_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>cos</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  col parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_cos(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

/**
 * @brief       tanh initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  tanh parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_tanh_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>tan</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  tanh parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_tanh(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

/**
 * @brief       Log initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Log parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_log_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>log</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Log parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_log(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

/**
 * @brief       Sqrt initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Sqrt parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sqrt_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>sqrt</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Sqrt parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sqrt(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

/**
 * @brief       Rsqrt initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Rsqrt parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_rsqrt_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>rsqrt</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Rsqrt parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_rsqrt(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

/**
 * @brief       Square initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Square parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_square_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>square</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Square parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_square(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_siso_params *params);

/**
 * @brief       Sigmoid initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Sigmoid parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sigmoid_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_sigmoid_params *params);

/**
 * @brief       Calculate <code>sigmoid</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Sigmoid parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sigmoid(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_sigmoid_params *params);

/**
 * @brief       Hard sigmoid initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Hard sigmoid parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_hard_sigmoid_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_sigmoid_params *params);

/**
 * @brief       Calculate <code>hard sigmoid</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Hard sigmoid parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_hard_sigmoid(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_sigmoid_params *params);

/**
 * @brief       ELU initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  ELU parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_elu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_relu_params *params);

/**
 * @brief       Calculate <code>ELU</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  ELU parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_elu(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_relu_params *params);

/**
 * @brief       ReLU initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  ReLU parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_relu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_relu_params *params);

/**
 * @brief       Calculate <code>ReLU</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  ReLU parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_relu(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_relu_params *params);

/**
 * @brief       ReLU1 initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  ReLU1 parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_relu1_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_relu_params *params);

/**
 * @brief       Calculate <code>ReLU1</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  ReLU1 parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_relu1(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_relu_params *params);

/**
 * @brief       ReLU6 initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  ReLU6 parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_relu6_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_relu_params *params);

/**
 * @brief       Calculate <code>ReLU6</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  ReLU6 parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_relu6(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_relu_params *params);

/**
 * @brief       ReLUn initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  ReLUn parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_relun_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_relu_params *params);

/**
 * @brief       Calculate <code>ReLUn</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  ReLUn parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_relun(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_relu_params *params);

/**
 * @brief       Leaky ReLU initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Leaky ReLU parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_leaky_relu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_relu_params *params);

/**
 * @brief       Calculate <code>Leaky ReLU</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Leaky ReLU parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_leaky_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_relu_params *params);

/**
 * @brief       Soft ReLU initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Soft ReLU parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_softrelu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_relu_params *params);

/**
 * @brief       Calculate <code>Soft ReLU</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Soft ReLU parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_softrelu(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_relu_params *params);

/**
 * @brief       PReLU initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[in]   alpha   Pointer to the α coefficient tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  PReLU parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_prelu_init(struct csinn_tensor *input, struct csinn_tensor *alpha,
                     struct csinn_tensor *output, struct csinn_prelu_params *params);

/**
 * @brief       Calculate <code>PReLU</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[in]   alpha   Pointer to the α coefficient tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  PReLU parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_prelu(struct csinn_tensor *input, struct csinn_tensor *alpha, struct csinn_tensor *output,
                struct csinn_prelu_params *params);

/**
 * @brief       Softplus initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Softplus parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_softplus_init(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>Softplus</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Softplus parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_softplus(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

/**
 * @brief       Softmax initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Softmax parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_softmax_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_softmax_params *params);

/**
 * @brief       Calculate <code>Softmax</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Softmax parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_softmax(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_softmax_params *params);

/**
 * @brief       Log softmax initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Log softmax parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_log_softmax_init(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_softmax_params *params);

/**
 * @brief       Calculate <code>Log Softmax</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Log softmax parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_log_softmax(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_softmax_params *params);

/**
 * @brief       Batch normalization initialization function
 *
 * @param[in]   input       Pointer to the input tensor
 * @param[in]   mean        Mean value used to calculate BN
 * @param[in]   variance    Variance used to caculate BN
 * @param[in]   gamma       γ coefficient used to caculate BN
 * @param[in]   beta        β coefficient used to caculate BN
 * @param[out]  output      Pointer to the output tensor
 * @param[in]   params      Batch normalization parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_batch_normalization_init(struct csinn_tensor *input, struct csinn_tensor *mean,
                                   struct csinn_tensor *variance, struct csinn_tensor *gamma,
                                   struct csinn_tensor *beta, struct csinn_tensor *output,
                                   struct csinn_bn_params *params);

/**
 * @brief       Batch normalization function
 *
 * @param[in]   input       Pointer to the input tensor
 * @param[in]   mean        Mean value used to calculate BN
 * @param[in]   variance    Variance used to caculate BN
 * @param[in]   gamma       γ coefficient used to caculate BN
 * @param[in]   beta        β coefficient used to caculate BN
 * @param[out]  output      Pointer to the output tensor
 * @param[in]   params      Batch normalization parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 *
 * @details     Usually, after deploying tools to process the model,
 *              the BN layer can be merged with the previous convolution layer
 */
int csinn_batch_normalization(struct csinn_tensor *input, struct csinn_tensor *mean,
                              struct csinn_tensor *variance, struct csinn_tensor *gamma,
                              struct csinn_tensor *beta, struct csinn_tensor *output,
                              struct csinn_bn_params *params);

/**
 * @brief       L2 normalization initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  L2 normalization parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_l2_normalization_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_l2n_params *params);

/**
 * @brief       Calculate <code>L2 Normalization</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  L2 normalization parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_l2_normalization(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_l2n_params *params);

/**
 * @brief       Local Response Normalization initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  LRN parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_lrn_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_lrn_params *params);

/**
 * @brief       Calculate <code>Local Response Normalization</code> for each element of input
 *              tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  LRN parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_lrn(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_lrn_params *params);

/**
 * @brief       Matmul initialization function
 *
 * @param[in]   mat0    Pointer to the input0 tensor
 * @param[in]   mat1    Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Matmul parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_matmul_init(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                      struct csinn_tensor *output, struct csinn_matmul_params *params);

/**
 * @brief       Calculate <code>Matmul</code> for each element of two input tensors.
 *
 * @param[in]   mat0    Pointer to the input0 tensor
 * @param[in]   mat1    Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Matmul parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_matmul(struct csinn_tensor *mat0, struct csinn_tensor *mat1, struct csinn_tensor *output,
                 struct csinn_matmul_params *params);

/**
 * @brief       Add initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Add parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_add_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Add each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Add parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_add(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params);

/**
 * @brief       Sub initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Sub parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sub_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Subtract each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Sub parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sub(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params);

/**
 * @brief       Mul initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Mul parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_mul_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Multiply each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Mul parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_mul(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params);

/**
 * @brief       Div initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Div parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_div_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Divide each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Div parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_div(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params);

/**
 * @brief       Floor div initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Floor div parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_floor_divide_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                            struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Floor divide each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Floor div parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_floor_divide(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Floor mod initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Floor mod parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_floor_mod_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Calculate <code>Floor MOD</code> for each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Floor mod parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_floor_mod(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Mod initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Mod parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_mod_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Calculate <code>MOD</code> for each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Mod parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_mod(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params);

/**
 * @brief       Maximum initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Maximum parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_maximum_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Calculate <code>MAX</code> for each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Maximum parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_maximum(struct csinn_tensor *input0, struct csinn_tensor *input1,
                  struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Minimum initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Minimum parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_minimum_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Calculate <code>MIN</code> for each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Minimum parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_minimum(struct csinn_tensor *input0, struct csinn_tensor *input1,
                  struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Power initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Power parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_power_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Power function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Power parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 *
 * @details     The first tensor is the base of <code>POW</code>, and the
 *              second tensor is the exponential of <code>POW</code>.
 */
int csinn_power(struct csinn_tensor *input0, struct csinn_tensor *input1,
                struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Greater initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Greater parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_greater_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Compare whether input0 is greater than input1 for each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Greater parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_greater(struct csinn_tensor *input0, struct csinn_tensor *input1,
                  struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Less initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Less parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_less_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Compare whether input0 is less than input1 for each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Less parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_less(struct csinn_tensor *input0, struct csinn_tensor *input1,
               struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Logical AND initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Logical AND parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_logical_and_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Calculate <code>Logical AND</code> for each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Logical AND parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_logical_and(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Logical OR initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Logical OR parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_logical_or_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                          struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Calculate <code>Logical OR</code> for each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Logical OR parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_logical_or(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Logical NOT initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Logical NOT parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_logical_not_init(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>Logical NOT</code> for each element of two input tensors.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Logical NOT parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_logical_not(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params);

/**
 * @brief       Logical XOR initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Logical XOR parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_logical_xor_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Calculate <code>Logical XOR</code> for each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Logical XOR parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_logical_xor(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Equal initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Equal parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_equal_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Compare whether input0 is equal to input1 for each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Equal parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Not equal initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Not equal parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_not_equal_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Compare whether input0 is not equal to input1 for each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Not equal parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_not_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                    struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Greater equal initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Greater equal parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_greater_equal_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                             struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Compare whether input0 is greater than or equal to input1 for each element of two
 *              input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Greater equal parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_greater_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Less equal initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Less equal parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_less_equal_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                          struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Compare whether input0 is less than or equal to input1 for each element of two input
 *              tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Less equal parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_less_equal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                     struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Select initialization function
 *
 * @param[in]   condition   Pointer to the condition tensor
 * @param[in]   input0      Pointer to the input0 tensor
 * @param[in]   input1      Pointer to the input1 tensor
 * @param[out]  output      Pointer to the output tensor
 * @param[in]	params      Select parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_select_init(struct csinn_tensor *condition, struct csinn_tensor *input0,
                      struct csinn_tensor *input1, struct csinn_tensor *output,
                      struct csinn_select_params *params);

/**
 * @brief       Select each element from two input tensors according to the condition tensor.
 *
 * @param[in]   condition   Pointer to the condition tensor
 * @param[in]   input0      Pointer to the input0 tensor
 * @param[in]   input1      Pointer to the input1 tensor
 * @param[out]  output      Pointer to the output tensor
 * @param[in]	params      Select parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_select(struct csinn_tensor *condition, struct csinn_tensor *input0,
                 struct csinn_tensor *input1, struct csinn_tensor *output,
                 struct csinn_select_params *params);

/**
 * @brief       AND initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  AND parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_and_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Calculate <code>AND</code> for each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  AND parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_and(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params);

/**
 * @brief       OR initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  OR parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_or_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                  struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Calculate <code>OR</code> for each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  OR parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_or(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
             struct csinn_diso_params *params);

/**
 * @brief       XOR initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  XOR parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_xor_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                   struct csinn_tensor *output, struct csinn_diso_params *params);

/**
 * @brief       Calculate <code>XOR</code> for each element of two input tensors.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  XOR parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_xor(struct csinn_tensor *input0, struct csinn_tensor *input1, struct csinn_tensor *output,
              struct csinn_diso_params *params);

/**
 * @brief       NOT initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  NOT parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_not_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>NOT</code> for each element of two input tensors.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  NOT parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_not(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

/**
 * @brief       Pad initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Pad parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_pad_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_pad_params *params);

/**
 * @brief       Padding input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Pad parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_pad(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_pad_params *params);

/**
 * @brief       Resize initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Resize parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_resize_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_resize_params *params);

/**
 * @brief       Resize input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Resize parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_resize(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_resize_params *params);

/**
 * @brief       Concat initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Concat parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_concat_init(struct csinn_tensor **input, struct csinn_tensor *output,
                      struct csinn_concat_params *params);

/**
 * @brief       Concat multiple input tensors according to the specified dimension.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Concat parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_concat(struct csinn_tensor **input, struct csinn_tensor *output,
                 struct csinn_concat_params *params);

/**
 * @brief       Proposal initialization function
 *
 * @param[in]   cls_prob    Pointer to the classification input tensor
 * @param[in]   bbox_pred   Pointer to the box input tensor
 * @param[in]   im_info     Pointer to the image input tensor
 * @param[out]  output      Pointer to the output tensor
 * @param[in]	params      Proposal parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_proposal_init(struct csinn_tensor *cls_prob, struct csinn_tensor *bbox_pred,
                        struct csinn_tensor *im_info, struct csinn_tensor *output,
                        struct csinn_proposal_params *params);

/**
 * @brief       Proposal initialization function
 *
 * @param[in]   cls_prob    Pointer to the classification input tensor
 * @param[in]   bbox_pred   Pointer to the box input tensor
 * @param[in]   im_info     Pointer to the image input tensor
 * @param[out]  output      Pointer to the output tensor
 * @param[in]	params      Proposal parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 *
 * @details     Special layer from the Faster RCNN.
 */
int csinn_proposal(struct csinn_tensor *cls_prob, struct csinn_tensor *bbox_pred,
                   struct csinn_tensor *im_info, struct csinn_tensor *output,
                   struct csinn_proposal_params *params);

/**
 * @brief       PS ROI pooling initialization function
 *
 * @param[in]   data    Pointer to the input tensor
 * @param[in]   rois    ROI information
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  PS ROI pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_psroipooling_init(struct csinn_tensor *data, struct csinn_tensor *rois,
                            struct csinn_tensor *output, struct csinn_psroipooling_params *params);

/**
 * @brief       PS ROI pooling function
 *
 * @param[in]   data    Pointer to the input tensor
 * @param[in]   rois    ROI information
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  PS ROI pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_psroipooling(struct csinn_tensor *data, struct csinn_tensor *rois,
                       struct csinn_tensor *output, struct csinn_psroipooling_params *params);

/**
 * @brief       Transpose initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Transpose parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_transpose_init(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_transpose_params *params);

/**
 * @brief       Transpose function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Transpose parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 *
 * @details     Transpose the dimension order of input tensor. Transpose is basically the same as
 *              permute. Permute is not implemented for the time being. Instead, transpose is used.
 */
int csinn_transpose(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_transpose_params *params);

/**
 * @brief       Reshape initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reshape parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reshape_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_reshape_params *params);

/**
 * @brief       Reset the dimensions of input tensor. Reshape is essentially a memcpy.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reshape parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reshape(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_reshape_params *params);

/**
 * @brief       Shape initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Shape parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_shape_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_shape_params *params);

/**
 * @brief       Get the dimension information of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Shape parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_shape(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_shape_params *params);

/**
 * @brief       Expand dims initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Expand dims parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_expand_dims_init(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_expand_dims_params *params);

/**
 * @brief       Expand the dimension of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Expand dims parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_expand_dims(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_expand_dims_params *params);

/**
 * @brief       Reverse initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reverse parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reverse_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_reverse_params *params);

/**
 * @brief       Flip the specified dimension of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reverse parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reverse(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_reverse_params *params);

/**
 * @brief       Flatten initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Flatten parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_flatten_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_flatten_params *params);

/**
 * @brief       Flatten input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Flatten parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_flatten(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_flatten_params *params);

/**
 * @brief       Crop initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Crop parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_crop_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_crop_params *params);

/**
 * @brief       Crop input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Crop parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_crop(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_crop_params *params);

/**
 * @brief       Slice initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Slice parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_slice_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_slice_params *params);

/**
 * @brief       Slice input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Slice parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_slice(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_slice_params *params);

/**
 * @brief       Split initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Split parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_split_init(struct csinn_tensor *input, struct csinn_tensor **output,
                     struct csinn_split_params *params);

/**
 * @brief       Split input tensor into multiple tensors according to the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Split parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_split(struct csinn_tensor *input, struct csinn_tensor **output,
                struct csinn_split_params *params);

/**
 * @brief       Stack initialization function
 *
 * @param[in]   inputs  Pointer to the input tensors
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Stack parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_stack_init(struct csinn_tensor **inputs, struct csinn_tensor *output,
                     struct csinn_stack_params *params);

/**
 * @brief       Stack/splice several input tensors.
 *
 * @param[in]   inputs  Pointer to the input tensors
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Stack parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 *
 * @details     Note that different from <code>concat</code>, the dimensions of
 *              <code>concat</code> output tensor and input tensor remain unchanged,
 *              and the output tensor of <code>stack</code> is more dimensional
 *              than input tensor.
 */
int csinn_stack(struct csinn_tensor **inputs, struct csinn_tensor *output,
                struct csinn_stack_params *params);

/**
 * @brief       Unstack initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Unstack parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_unstack_init(struct csinn_tensor *input, struct csinn_tensor **output,
                       struct csinn_unstack_params *params);

/**
 * @brief       Unstack input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Unstack parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 *
 * @details     Note that different from <code>split</code>, the dimension of
 *              <code>split</code> output tensor and input tensor remains unchanged,
 *              and the <code>stack</code> output tensor is one dimension less than
 *              input tensor.
 */
int csinn_unstack(struct csinn_tensor *input, struct csinn_tensor **output,
                  struct csinn_unstack_params *params);

/**
 * @brief       Tile initialization function
 *
 * @param[in]   inputs  Pointer to the input tensors
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Tile parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_tile_init(struct csinn_tensor *inputs, struct csinn_tensor *output,
                    struct csinn_tile_params *params);

/**
 * @brief       Repeat input tensor.
 *
 * @param[in]   inputs  Pointer to the input tensors
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Tile parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_tile(struct csinn_tensor *inputs, struct csinn_tensor *output,
               struct csinn_tile_params *params);

/**
 * @brief       Arange initialization function
 *
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Arange parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_arange_init(struct csinn_tensor *output, struct csinn_arange_params *params);

/**
 * @brief       Return the position information within the range according to the parameters.
 *
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Arange parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_arange(struct csinn_tensor *output, struct csinn_arange_params *params);

/**
 * @brief       Where initialization function
 *
 * @param[in]   condition   Pointer to the condition tensor
 * @param[in]   x           Pointer to the x tensor
 * @param[in]   y           Pointer to the y tensor
 * @param[out]  output      Pointer to the output tensor
 * @param[in]   params      Where parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_where_init(struct csinn_tensor *condition, struct csinn_tensor *x, struct csinn_tensor *y,
                     struct csinn_tensor *output, struct csinn_where_params *params);

/**
 * @brief       Select each element from two input tensors according to the condition tensor.
 *
 * @param[in]   condition   Pointer to the condition tensor
 * @param[in]   x           Pointer to the x tensor
 * @param[in]   y           Pointer to the y tensor
 * @param[out]  output      Pointer to the output tensor
 * @param[in]   params      Where parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_where(struct csinn_tensor *condition, struct csinn_tensor *x, struct csinn_tensor *y,
                struct csinn_tensor *output, struct csinn_where_params *params);

/**
 * @brief       Where initialization function
 *
 * @param[in]   condition   Pointer to the condition tensor
 * @param[in]   y           Pointer to the y tensor
 * @param[out]  output      Pointer to the output tensor
 * @param[in]   params      Where parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_where_softmax_init(struct csinn_tensor *condition, struct csinn_tensor *y,
                             struct csinn_tensor *output,
                             struct csinn_where_softmax_params *params);

/**
 * @brief       Select each element from two input tensors according to the condition tensor.
 *
 * @param[in]   condition   Pointer to the condition tensor
 * @param[in]   y           Pointer to the y tensor
 * @param[out]  output      Pointer to the output tensor
 * @param[in]   params      Where parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_where_softmax(struct csinn_tensor *condition, struct csinn_tensor *y,
                        struct csinn_tensor *output, struct csinn_where_softmax_params *params);

/**
 * @brief       Gather initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[in]   indices Pointer to the index tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Gather parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_gather_init(struct csinn_tensor *input, struct csinn_tensor *indices,
                      struct csinn_tensor *output, struct csinn_gather_params *params);

/**
 * @brief       Gather the data in input tensor according to the specified index.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[in]   indices Pointer to the index tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Gather parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_gather(struct csinn_tensor *input, struct csinn_tensor *indices,
                 struct csinn_tensor *output, struct csinn_gather_params *params);

/**
 * @brief       Gather_nd initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[in]   indices Pointer to the index tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Gather_nd parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_gather_nd_init(struct csinn_tensor *input, struct csinn_tensor *indices,
                         struct csinn_tensor *output, struct csinn_gather_nd_params *params);

/**
 * @brief       Gather the data in input tensor according to the specified index.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[in]   indices Pointer to the index tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Gather_nd parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_gather_nd(struct csinn_tensor *input, struct csinn_tensor *indices,
                    struct csinn_tensor *output, struct csinn_gather_nd_params *params);

/**
 * @brief       Squeeze initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Squeeze parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_squeeze_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_squeeze_params *params);

/**
 * @brief       Tile the data in input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Squeeze parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_squeeze(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_squeeze_params *params);

/**
 * @brief       Ndarray size initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Ndarray size parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_ndarray_size_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_ndarray_size_params *params);

/**
 * @brief       Calculate the size of input tensor data.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Ndarray size parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_ndarray_size(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_ndarray_size_params *params);

/**
 * @brief       Space to batch initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Space to batch parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_space_to_batch_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_space_to_batch_params *params);

/**
 * @brief       Fill the batch according to the height and width of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Space to batch parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_space_to_batch(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_space_to_batch_params *params);

/**
 * @brief       Space to batch nd initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Space to batch nd parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_space_to_batch_nd_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_space_to_batch_nd_params *params);

/**
 * @brief       Fill the batch according to the spatial dimensions such as height and width of input
 *              tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Space to batch nd parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_space_to_batch_nd(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_space_to_batch_nd_params *params);

/**
 * @brief       Batch to space initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Batch to space parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_batch_to_space_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_batch_to_space_params *params);

/**
 * @brief       Fill the batch into the height and width of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Batch to space parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_batch_to_space(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_batch_to_space_params *params);

/**
 * @brief       Batch to space nd initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Batch to space nd parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_batch_to_space_nd_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_batch_to_space_nd_params *params);

/**
 * @brief       Fill the batch into the spatial dimensions such as height and width of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Batch to space nd parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_batch_to_space_nd(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_batch_to_space_nd_params *params);

/**
 * @brief       Space to depth initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Space to depth parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_space_to_depth_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_space_to_depth_params *params);

/**
 * @brief       Fill the depth according to the height and width of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Space to depth parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_space_to_depth(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_space_to_depth_params *params);

/**
 * @brief       Depth to space initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Depth to space parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_depth_to_space_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_depth_to_space_params *params);

/**
 * @brief       Fill the depth into the height and width of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Depth to space parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_depth_to_space(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_depth_to_space_params *params);

/**
 * @brief       One Hot initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  One Hot parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_one_hot_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_one_hot_params *params);

/**
 * @brief       Return One-Hot eigenvector.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  One Hot parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_one_hot(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_one_hot_params *params);

/**
 * @brief       Sequence mask initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Sequence mask parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sequence_mask_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                             struct csinn_tensor *output,
                             struct csinn_sequence_mask_params *params);

/**
 * @brief       Get the mask of sequence.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Sequence mask parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 *
 * @details     The output is usually of bool type, which can be used to fill in sentences and
 *              words.
 */
int csinn_sequence_mask(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_sequence_mask_params *params);

/**
 * @brief       im2col initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  im2col parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_im2col_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_im2col_params *params);

/**
 * @brief       Convert image to columns.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  im2col parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_im2col(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_im2col_params *params);

/**
 * @brief       col2im initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]	params  col2im parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_col2im_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tensor *kernel, struct csinn_col2im_params *params);

/**
 * @brief       Convert columns to image.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]	params  col2im parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_col2im(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_tensor *kernel, struct csinn_col2im_params *params);

/**
 * @brief       Sum initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Sum parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sum_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_reduce_params *params);

/**
 * @brief       Calculate the sum of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Sum parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sum(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_reduce_params *params);

/**
 * @brief       Mean initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Mean parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_mean_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_reduce_params *params);

/**
 * @brief       Calculate the mean value of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Mean parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_mean(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_reduce_params *params);

/**
 * @brief       Max initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Max parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_max_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_reduce_params *params);

/**
 * @brief       Calculate the maximum value of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Max parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_max(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_reduce_params *params);

/**
 * @brief       Min initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Min parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_min_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_reduce_params *params);

/**
 * @brief       Calculate the min value of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Min parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_min(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_reduce_params *params);

/**
 * @brief       Prod initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Prod parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_prod_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_reduce_params *params);

/**
 * @brief       Calculate the product value of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Prod parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_prod(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_reduce_params *params);

/**
 * @brief       Argmin initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Argmin parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_argmin_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_reduce_params *params);

/**
 * @brief       Calculate the index of the minimum value of input tensor on the specified
 *              dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Argmin parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_argmin(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_reduce_params *params);

/**
 * @brief       Argmax initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Argmax parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_argmax_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_reduce_params *params);

/**
 * @brief       Calculate the index of the maximum value of input tensor on the specified
 *              dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Argmax parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_argmax(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_reduce_params *params);

/**
 * @brief       All initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  All parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_all_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_reduce_params *params);

/**
 * @brief       Calculate <code>Reduce AND</code> of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  All parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_all(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_reduce_params *params);

/**
 * @brief       Any initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Any parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_any_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_reduce_params *params);

/**
 * @brief       Calculate <code>Reduce OR</code> of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Any parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_any(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_reduce_params *params);

/**
 * @brief       Reorg initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reorg parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reorg_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_reorg_params *params);

/**
 * @brief       Cut the height and width, then splice them along the channel.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reorg parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reorg(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_reorg_params *params);

/**
 * @brief       YUV RGB scale initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  YUV RGB scale parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_yuv_rgb_scale_init(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_siso_params *params);

/**
 * @brief       Convert YUV to RGB.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  YUV RGB scale parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_yuv_rgb_scale(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

/**
 * @brief       Segment max initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Segment max parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_segment_max_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_segment_params *params);

/**
 * @brief       Compare the maximum value of input tensor at the specified positions.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Segment max parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_segment_max(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_segment_params *params);

/**
 * @brief       Segment min initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Segment min parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_segment_min_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_segment_params *params);

/**
 * @brief       Compare the minimum value of input tensor at the specified positions.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Segment min parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_segment_min(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_segment_params *params);

/**
 * @brief       Segment sum initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Segment sum parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_segment_sum_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_segment_params *params);

/**
 * @brief       Calculate the sum of input tensor at the specified positions.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Segment sum parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_segment_sum(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_segment_params *params);

/**
 * @brief       Segment mean initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Segment mean parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_segment_mean_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                            struct csinn_tensor *output, struct csinn_segment_params *params);

/**
 * @brief       Calculate the mean value of input tensor at the specified positions.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Segment mean parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_segment_mean(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_segment_params *params);

/**
 * @brief       Segment prod initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Segment prod parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_segment_prod_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                            struct csinn_tensor *output, struct csinn_segment_params *params);

/**
 * @brief       Calculate the product value of input tensor at the specified positions.
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Segment prod parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_segment_prod(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_segment_params *params);

/**
 * @brief       Threshold ReLU initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Threshold ReLU parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_threshold_relu_init(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_relu_params *params);

/**
 * @brief       Calculate <code>Threshold ReLU</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Threshold ReLU parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_threshold_relu(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_relu_params *params);

/**
 * @brief       acos initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  acos parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_acos_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>acos</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  acos parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_acos(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

/**
 * @brief       acosh initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  acosh parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_acosh_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>acosh</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  acosh parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_acosh(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

/**
 * @brief       asin initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  asin parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_asin_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>asin</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  asin parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_asin(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

/**
 * @brief       asinh initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  asinh parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_asinh_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>asinh</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  asinh parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_asinh(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

/**
 * @brief       atan initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  atan parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_atan_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>atan</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  atan parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_atan(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

/**
 * @brief       atanh initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  atanh parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_atanh_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>atanh</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  atanh parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_atanh(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

/**
 * @brief       colh function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  colh parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_cosh_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>cosh</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  colh parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_cosh(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

/**
 * @brief       sinh initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  sinh parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sinh_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>sinh</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  sinh parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_sinh(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_siso_params *params);

/**
 * @brief       tan initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  tan parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_tan_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>tan</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  tan parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_tan(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

/**
 * @brief       Log1p initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Log1p parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_log1p_init(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>log1p</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Log1p parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_log1p(struct csinn_tensor *input, struct csinn_tensor *output,
                struct csinn_siso_params *params);

/**
 * @brief       Softsign initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Softsign parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_softsign_init(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>softsign</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Softsign parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_softsign(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

/**
 * @brief       Erf initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Erf parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_erf_init(struct csinn_tensor *input, struct csinn_tensor *output,
                   struct csinn_siso_params *params);

/**
 * @brief       Calculate <code>erf</code> for each element of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Erf parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_erf(struct csinn_tensor *input, struct csinn_tensor *output,
              struct csinn_siso_params *params);

/**
 * @brief       Cumsum initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Cumsum parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_cumsum_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_cumsum_params *params);

/**
 * @brief       Calculate the cumulative sum of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Cumsum parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_cumsum(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_cumsum_params *params);

/**
 * @brief       Cumprod initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Cumprod parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_cumprod_init(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_cumprod_params *params);

/**
 * @brief       Calculate the cumulative product value of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Cumprod parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_cumprod(struct csinn_tensor *input, struct csinn_tensor *output,
                  struct csinn_cumprod_params *params);

/**
 * @brief       Reduce max initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reduce max parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reduce_max_init(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_reduce_params *params);

/**
 * @brief       Calculate the reduced maximum value of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reduce max parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reduce_max(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_reduce_params *params);

/**
 * @brief       Reduce min initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reduce min parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reduce_min_init(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_reduce_params *params);

/**
 * @brief       Calculate the reduced minimum value of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reduce min parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reduce_min(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_reduce_params *params);

/**
 * @brief       Reduce mean initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reduce mean parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reduce_mean_init(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reduce_params *params);

/**
 * @brief       Calculate the reduced mean value of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reduce mean parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reduce_mean(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_reduce_params *params);

/**
 * @brief       Reduce sum initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reduce sum parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reduce_sum_init(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_reduce_params *params);

/**
 * @brief       Calculate the reduced sum of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reduce sum parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reduce_sum(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_reduce_params *params);

/**
 * @brief       Reduce prod initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reduce prod parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reduce_prod_init(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reduce_params *params);

/**
 * @brief       Calculate the reduced product value of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reduce prod parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reduce_prod(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_reduce_params *params);

/**
 * @brief       Reduce LogSumExp initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reduce LogSumExp parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reduce_logsumexp_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_reduce_params *params);

/**
 * @brief       Calculate the reduced LogSumExp value of input tensor on the specified dimensions.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Reduce LogSumExp parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_reduce_logsumexp(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reduce_params *params);

/**
 * @brief       Broadcast to initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Broadcast to parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_broadcast_to_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_broadcast_to_params *params);

/**
 * @brief       Broadcast the input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Broadcast to parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_broadcast_to(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_broadcast_to_params *params);

/**
 * @brief       Scatter nd initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[in]   indices Pointer to the index tensor
 * @param[in]   updates Pointer to the distribution of update tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Broadcast to parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_scatter_nd_init(struct csinn_tensor *input, struct csinn_tensor *indices,
                          struct csinn_tensor *updates, struct csinn_tensor *output,
                          struct csinn_scatter_nd_params *params);

/**
 * @brief       Scatter nd function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[in]   indices Pointer to the index tensor
 * @param[in]   updates Pointer to the distribution of update tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Broadcast to parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 *
 * @details     The index applies sparse updates to a single value or slice
 *              in the zero tensor of a given shape to create a new tensor,
 *              the inverse operation of gather_nd.
 */
int csinn_scatter_nd(struct csinn_tensor *input, struct csinn_tensor *indices,
                     struct csinn_tensor *updates, struct csinn_tensor *output,
                     struct csinn_scatter_nd_params *params);

/**
 * @brief       Clip initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Clip parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_clip_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_clip_params *params);

/**
 * @brief       Saturate the input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Clip parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_clip(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_clip_params *params);

/**
 * @brief       Stride slice initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Stride slice parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_strided_slice_init(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_strided_slice_params *params);

/**
 * @brief       Slice the input tensor by stride.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Stride slice parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_strided_slice(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_strided_slice_params *params);

/**
 * @brief       TOP-k initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output1 Pointer to the output1 tensor
 * @param[out]  output2 Pointer to the output2 tensor
 * @param[in]   params  TOP-k parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_topk_init(struct csinn_tensor *input, struct csinn_tensor *output1,
                    struct csinn_tensor *output2, struct csinn_topk_params *params);

/**
 * @brief       Find the maximum k elements and their indices of input tensor.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output1 Pointer to the output1 tensor
 * @param[out]  output2 Pointer to the output2 tensor
 * @param[in]   params  TOP-k parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_topk(struct csinn_tensor *input, struct csinn_tensor *output1,
               struct csinn_tensor *output2, struct csinn_topk_params *params);

/**
 * @brief       Non-max suppression initialization function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Non-max suppression parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_non_max_suppression_init(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                   struct csinn_tensor *output,
                                   struct csinn_non_max_suppression_params *params);

/**
 * @brief       Non-max suppression function
 *
 * @param[in]   input0  Pointer to the input0 tensor
 * @param[in]   input1  Pointer to the input1 tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Non-max suppression parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 *
 * @details     Non-max suppression is generally used for post-processing of
 *              detection models to filter out redundant candidate frames.
 */
int csinn_non_max_suppression(struct csinn_tensor *input0, struct csinn_tensor *input1,
                              struct csinn_tensor *output,
                              struct csinn_non_max_suppression_params *params);

/**
 * @brief       Shuffle channel initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Shuffle channel parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_shuffle_channel_init(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_shuffle_channel_params *params);

/**
 * @brief       Group input tensor and shuffle them by group.
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]	params  Shuffle channel parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 *
 * @details     Combined with grouping convolution, it alleviates the problem of channel
 *              locality among different groups and increases the generalization ability
 *              of the model, which is proposed in shuffelNet.
 */
int csinn_shuffle_channel(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_shuffle_channel_params *params);

/**
 * @brief       ROI pooling initialization function
 *
 * @param[in]   data    Pointer to the input tensor
 * @param[in]   rois    ROI information
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  ROI pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_roipool_init(struct csinn_tensor *data, struct csinn_tensor *rois,
                       struct csinn_tensor *output, struct csinn_roi_pool_params *params);

/**
 * @brief       ROI pooling function
 *
 * @param[in]   data    Pointer to the input tensor
 * @param[in]   rois    ROI information
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  ROI pooling parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_roipool(struct csinn_tensor *data, struct csinn_tensor *rois, struct csinn_tensor *output,
                  struct csinn_roi_pool_params *params);

/**
 * @brief       Layer normalization initialization function
 *
 * @param[in]   input       Pointer to the input tensor
 * @param[out]  output      Pointer to the output tensor
 * @param[in]   gamma       γ coefficient used to caculate LN
 * @param[in]   beta        β coefficient used to caculate LN
 * @param[in]   params      Layer normalization parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_layer_norm_init(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *gamma, struct csinn_tensor *beta,
                          struct csinn_layer_norm_params *params);

/**
 * @brief       Layer normalization function
 *
 * @param[in]   input       Pointer to the input tensor
 * @param[out]  output      Pointer to the output tensor
 * @param[in]   gamma       γ coefficient used to caculate LN
 * @param[in]   beta        β coefficient used to caculate LN
 * @param[in]   params      Layer normalization parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_layer_norm(struct csinn_tensor *input, struct csinn_tensor *output,
                     struct csinn_tensor *gamma, struct csinn_tensor *beta,
                     struct csinn_layer_norm_params *params);

/**
 * @brief       Cache matmul initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   weight  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Cache matmul parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_cache_matmul_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *weight, struct csinn_tensor *bias,
                            struct csinn_cache_matmul_params *params);

/**
 * @brief       Cache matmul function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   weight  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Cache matmul parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_cache_matmul(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *weight, struct csinn_tensor *bias,
                       struct csinn_cache_matmul_params *params);

/**
 * @brief       Cache conv1d initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   weight  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Cache conv1d parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_cache_conv1d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_tensor *weight, struct csinn_tensor *bias,
                            struct csinn_cache_conv1d_params *params);

/**
 * @brief       Cache conv1d function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   weight  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Cache conv1d parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_cache_conv1d(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_tensor *weight, struct csinn_tensor *bias,
                       struct csinn_cache_conv1d_params *params);

/**
 * @brief       One-dimensional convolution initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_conv1d_init(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                      struct csinn_conv1d_params *params);

/**
 * @brief       One-dimensional convolution function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   kernel  Pointer to the weight tensor
 * @param[in]   bias    Pointer to the bias tensor
 * @param[in]   params  Convolution parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_conv1d(struct csinn_tensor *input, struct csinn_tensor *output,
                 struct csinn_tensor *kernel, struct csinn_tensor *bias,
                 struct csinn_conv1d_params *params);

/**
 * @brief       Data convert initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Data Cast parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_cast_init(struct csinn_tensor *input, struct csinn_tensor *output,
                    struct csinn_cast_params *params);

/**
 * @brief       Data convert function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Data convert parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_cast(struct csinn_tensor *input, struct csinn_tensor *output,
               struct csinn_cast_params *params);
/**
 * @brief       Data convert initialization function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Data convert parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_data_convert_init(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_siso_params *params);

/**
 * @brief       Data convert function
 *
 * @param[in]   input   Pointer to the input tensor
 * @param[out]  output  Pointer to the output tensor
 * @param[in]   params  Data convert parameter descriptor
 * @return      On success, the return value is 1.
 *              If an error occurred while executing the function, the return value is less than or
 *              equal to 0.
 */
int csinn_data_convert(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_siso_params *params);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_CSI_NN_H_
