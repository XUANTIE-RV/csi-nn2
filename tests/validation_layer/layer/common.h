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

/* SHL version 2.1.x */

#include <stddef.h>

#include "csi_nn.h"
#include "test_utils.h"

#define LAYER_QUANT_TEST_SISO(MACRO)                                             \
    MACRO(abs, CSINN_QUANT_FLOAT32, csinn_siso_params)                           \
    MACRO(abs, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                        \
    MACRO(abs, CSINN_QUANT_INT8_SYM, csinn_siso_params)                          \
    MACRO(acos, CSINN_QUANT_FLOAT32, csinn_siso_params)                          \
    MACRO(acos, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                       \
    MACRO(acos, CSINN_QUANT_INT8_SYM, csinn_siso_params)                         \
    MACRO(acosh, CSINN_QUANT_FLOAT32, csinn_siso_params)                         \
    MACRO(acosh, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                      \
    MACRO(acosh, CSINN_QUANT_INT8_SYM, csinn_siso_params)                        \
    MACRO(asin, CSINN_QUANT_FLOAT32, csinn_siso_params)                          \
    MACRO(asin, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                       \
    MACRO(asin, CSINN_QUANT_INT8_SYM, csinn_siso_params)                         \
    MACRO(asinh, CSINN_QUANT_FLOAT32, csinn_siso_params)                         \
    MACRO(asinh, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                      \
    MACRO(asinh, CSINN_QUANT_INT8_SYM, csinn_siso_params)                        \
    MACRO(atan, CSINN_QUANT_FLOAT32, csinn_siso_params)                          \
    MACRO(atan, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                       \
    MACRO(atan, CSINN_QUANT_INT8_SYM, csinn_siso_params)                         \
    MACRO(atanh, CSINN_QUANT_FLOAT32, csinn_siso_params)                         \
    MACRO(atanh, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                      \
    MACRO(atanh, CSINN_QUANT_INT8_SYM, csinn_siso_params)                        \
    MACRO(ceil, CSINN_QUANT_FLOAT32, csinn_siso_params)                          \
    MACRO(ceil, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                       \
    MACRO(ceil, CSINN_QUANT_INT8_SYM, csinn_siso_params)                         \
    MACRO(cos, CSINN_QUANT_FLOAT32, csinn_siso_params)                           \
    MACRO(cos, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                        \
    MACRO(cos, CSINN_QUANT_INT8_SYM, csinn_siso_params)                          \
    MACRO(cosh, CSINN_QUANT_FLOAT32, csinn_siso_params)                          \
    MACRO(cosh, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                       \
    MACRO(cosh, CSINN_QUANT_INT8_SYM, csinn_siso_params)                         \
    MACRO(erf, CSINN_QUANT_FLOAT32, csinn_siso_params)                           \
    MACRO(erf, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                        \
    MACRO(erf, CSINN_QUANT_INT8_SYM, csinn_siso_params)                          \
    MACRO(exp, CSINN_QUANT_FLOAT32, csinn_siso_params)                           \
    MACRO(exp, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                        \
    MACRO(exp, CSINN_QUANT_INT8_SYM, csinn_siso_params)                          \
    MACRO(expm1, CSINN_QUANT_FLOAT32, csinn_siso_params)                         \
    MACRO(expm1, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                      \
    MACRO(expm1, CSINN_QUANT_INT8_SYM, csinn_siso_params)                        \
    MACRO(floor, CSINN_QUANT_FLOAT32, csinn_siso_params)                         \
    MACRO(floor, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                      \
    MACRO(floor, CSINN_QUANT_INT8_SYM, csinn_siso_params)                        \
    MACRO(log, CSINN_QUANT_FLOAT32, csinn_siso_params)                           \
    MACRO(log, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                        \
    MACRO(log, CSINN_QUANT_INT8_SYM, csinn_siso_params)                          \
    MACRO(log1p, CSINN_QUANT_FLOAT32, csinn_siso_params)                         \
    MACRO(log1p, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                      \
    MACRO(log1p, CSINN_QUANT_INT8_SYM, csinn_siso_params)                        \
    MACRO(logical_not, CSINN_QUANT_FLOAT32, csinn_siso_params)                   \
    MACRO(logical_not, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                \
    MACRO(logical_not, CSINN_QUANT_INT8_SYM, csinn_siso_params)                  \
    MACRO(round, CSINN_QUANT_FLOAT32, csinn_siso_params)                         \
    MACRO(round, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                      \
    MACRO(round, CSINN_QUANT_INT8_SYM, csinn_siso_params)                        \
    MACRO(rsqrt, CSINN_QUANT_FLOAT32, csinn_siso_params)                         \
    MACRO(rsqrt, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                      \
    MACRO(rsqrt, CSINN_QUANT_INT8_SYM, csinn_siso_params)                        \
    MACRO(sign, CSINN_QUANT_FLOAT32, csinn_siso_params)                          \
    MACRO(sign, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                       \
    MACRO(sign, CSINN_QUANT_INT8_SYM, csinn_siso_params)                         \
    MACRO(negative, CSINN_QUANT_FLOAT32, csinn_siso_params)                      \
    MACRO(negative, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                   \
    MACRO(negative, CSINN_QUANT_INT8_SYM, csinn_siso_params)                     \
    MACRO(sin, CSINN_QUANT_FLOAT32, csinn_siso_params)                           \
    MACRO(sin, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                        \
    MACRO(sin, CSINN_QUANT_INT8_SYM, csinn_siso_params)                          \
    MACRO(sinh, CSINN_QUANT_FLOAT32, csinn_siso_params)                          \
    MACRO(sinh, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                       \
    MACRO(sinh, CSINN_QUANT_INT8_SYM, csinn_siso_params)                         \
    MACRO(softplus, CSINN_QUANT_FLOAT32, csinn_siso_params)                      \
    MACRO(softplus, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                   \
    MACRO(softplus, CSINN_QUANT_INT8_SYM, csinn_siso_params)                     \
    MACRO(softsign, CSINN_QUANT_FLOAT32, csinn_siso_params)                      \
    MACRO(softsign, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                   \
    MACRO(softsign, CSINN_QUANT_INT8_SYM, csinn_siso_params)                     \
    MACRO(sqrt, CSINN_QUANT_FLOAT32, csinn_siso_params)                          \
    MACRO(sqrt, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                       \
    MACRO(sqrt, CSINN_QUANT_INT8_SYM, csinn_siso_params)                         \
    MACRO(square, CSINN_QUANT_FLOAT32, csinn_siso_params)                        \
    MACRO(square, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                     \
    MACRO(square, CSINN_QUANT_INT8_SYM, csinn_siso_params)                       \
    MACRO(tan, CSINN_QUANT_FLOAT32, csinn_siso_params)                           \
    MACRO(tan, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                        \
    MACRO(tan, CSINN_QUANT_INT8_SYM, csinn_siso_params)                          \
    MACRO(tanh, CSINN_QUANT_FLOAT32, csinn_siso_params)                          \
    MACRO(tanh, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                       \
    MACRO(tanh, CSINN_QUANT_INT8_SYM, csinn_siso_params)                         \
    MACRO(trunc, CSINN_QUANT_FLOAT32, csinn_siso_params)                         \
    MACRO(trunc, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                      \
    MACRO(trunc, CSINN_QUANT_INT8_SYM, csinn_siso_params)                        \
    MACRO(yuv_rgb_scale, CSINN_QUANT_FLOAT32, csinn_siso_params)                 \
    MACRO(yuv_rgb_scale, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)              \
    MACRO(yuv_rgb_scale, CSINN_QUANT_INT8_SYM, csinn_siso_params)                \
    MACRO(not, CSINN_QUANT_FLOAT32, csinn_siso_params)                           \
    MACRO(not, CSINN_QUANT_UINT8_ASYM, csinn_siso_params)                        \
    MACRO(not, CSINN_QUANT_INT8_SYM, csinn_siso_params)                          \
    MACRO(avgpool2d, CSINN_QUANT_FLOAT32, csinn_pool_params)                     \
    MACRO(avgpool2d, CSINN_QUANT_UINT8_ASYM, csinn_pool_params)                  \
    MACRO(avgpool2d, CSINN_QUANT_INT8_SYM, csinn_pool_params)                    \
    MACRO(avgpool3d, CSINN_QUANT_FLOAT32, csinn_pool_params)                     \
    MACRO(avgpool3d, CSINN_QUANT_UINT8_ASYM, csinn_pool_params)                  \
    MACRO(avgpool3d, CSINN_QUANT_INT8_SYM, csinn_pool_params)                    \
    MACRO(clip, CSINN_QUANT_FLOAT32, csinn_clip_params)                          \
    MACRO(clip, CSINN_QUANT_UINT8_ASYM, csinn_clip_params)                       \
    MACRO(clip, CSINN_QUANT_INT8_SYM, csinn_clip_params)                         \
    MACRO(batch_to_space, CSINN_QUANT_FLOAT32, csinn_batch_to_space_params)      \
    MACRO(batch_to_space, CSINN_QUANT_UINT8_ASYM, csinn_batch_to_space_params)   \
    MACRO(batch_to_space, CSINN_QUANT_INT8_SYM, csinn_batch_to_space_params)     \
    MACRO(cumprod, CSINN_QUANT_FLOAT32, csinn_cumprod_params)                    \
    MACRO(cumprod, CSINN_QUANT_UINT8_ASYM, csinn_cumprod_params)                 \
    MACRO(cumprod, CSINN_QUANT_INT8_SYM, csinn_cumprod_params)                   \
    MACRO(cumsum, CSINN_QUANT_FLOAT32, csinn_cumsum_params)                      \
    MACRO(cumsum, CSINN_QUANT_UINT8_ASYM, csinn_cumsum_params)                   \
    MACRO(cumsum, CSINN_QUANT_INT8_SYM, csinn_cumsum_params)                     \
    MACRO(depth_to_space, CSINN_QUANT_FLOAT32, csinn_depth_to_space_params)      \
    MACRO(depth_to_space, CSINN_QUANT_UINT8_ASYM, csinn_depth_to_space_params)   \
    MACRO(depth_to_space, CSINN_QUANT_INT8_SYM, csinn_depth_to_space_params)     \
    MACRO(elu, CSINN_QUANT_FLOAT32, csinn_relu_params)                           \
    MACRO(elu, CSINN_QUANT_UINT8_ASYM, csinn_relu_params)                        \
    MACRO(elu, CSINN_QUANT_INT8_SYM, csinn_relu_params)                          \
    MACRO(expand_dims, CSINN_QUANT_FLOAT32, csinn_expand_dims_params)            \
    MACRO(expand_dims, CSINN_QUANT_UINT8_ASYM, csinn_expand_dims_params)         \
    MACRO(expand_dims, CSINN_QUANT_INT8_SYM, csinn_expand_dims_params)           \
    MACRO(flatten, CSINN_QUANT_FLOAT32, csinn_flatten_params)                    \
    MACRO(flatten, CSINN_QUANT_UINT8_ASYM, csinn_flatten_params)                 \
    MACRO(flatten, CSINN_QUANT_INT8_SYM, csinn_flatten_params)                   \
    MACRO(global_avgpool2d, CSINN_QUANT_FLOAT32, csinn_pool_params)              \
    MACRO(global_avgpool2d, CSINN_QUANT_UINT8_ASYM, csinn_pool_params)           \
    MACRO(global_avgpool2d, CSINN_QUANT_INT8_SYM, csinn_pool_params)             \
    MACRO(global_maxpool2d, CSINN_QUANT_FLOAT32, csinn_pool_params)              \
    MACRO(global_maxpool2d, CSINN_QUANT_UINT8_ASYM, csinn_pool_params)           \
    MACRO(global_maxpool2d, CSINN_QUANT_INT8_SYM, csinn_pool_params)             \
    MACRO(hard_sigmoid, CSINN_QUANT_FLOAT32, csinn_sigmoid_params)               \
    MACRO(hard_sigmoid, CSINN_QUANT_UINT8_ASYM, csinn_sigmoid_params)            \
    MACRO(hard_sigmoid, CSINN_QUANT_INT8_SYM, csinn_sigmoid_params)              \
    MACRO(im2col, CSINN_QUANT_FLOAT32, csinn_im2col_params)                      \
    MACRO(im2col, CSINN_QUANT_UINT8_ASYM, csinn_im2col_params)                   \
    MACRO(im2col, CSINN_QUANT_INT8_SYM, csinn_im2col_params)                     \
    MACRO(l2_normalization, CSINN_QUANT_FLOAT32, csinn_l2n_params)               \
    MACRO(l2_normalization, CSINN_QUANT_UINT8_ASYM, csinn_l2n_params)            \
    MACRO(l2_normalization, CSINN_QUANT_INT8_SYM, csinn_l2n_params)              \
    MACRO(leaky_relu, CSINN_QUANT_FLOAT32, csinn_relu_params)                    \
    MACRO(leaky_relu, CSINN_QUANT_UINT8_ASYM, csinn_relu_params)                 \
    MACRO(leaky_relu, CSINN_QUANT_INT8_SYM, csinn_relu_params)                   \
    MACRO(log_softmax, CSINN_QUANT_FLOAT32, csinn_softmax_params)                \
    MACRO(log_softmax, CSINN_QUANT_UINT8_ASYM, csinn_softmax_params)             \
    MACRO(log_softmax, CSINN_QUANT_INT8_SYM, csinn_softmax_params)               \
    MACRO(lrn, CSINN_QUANT_FLOAT32, csinn_lrn_params)                            \
    MACRO(lrn, CSINN_QUANT_UINT8_ASYM, csinn_lrn_params)                         \
    MACRO(lrn, CSINN_QUANT_INT8_SYM, csinn_lrn_params)                           \
    MACRO(max, CSINN_QUANT_FLOAT32, csinn_reduce_params)                         \
    MACRO(max, CSINN_QUANT_UINT8_ASYM, csinn_reduce_params)                      \
    MACRO(max, CSINN_QUANT_INT8_SYM, csinn_reduce_params)                        \
    MACRO(maxpool2d, CSINN_QUANT_FLOAT32, csinn_pool_params)                     \
    MACRO(maxpool2d, CSINN_QUANT_UINT8_ASYM, csinn_pool_params)                  \
    MACRO(maxpool2d, CSINN_QUANT_INT8_SYM, csinn_pool_params)                    \
    MACRO(maxpool3d, CSINN_QUANT_FLOAT32, csinn_pool_params)                     \
    MACRO(maxpool3d, CSINN_QUANT_UINT8_ASYM, csinn_pool_params)                  \
    MACRO(maxpool3d, CSINN_QUANT_INT8_SYM, csinn_pool_params)                    \
    MACRO(mean, CSINN_QUANT_FLOAT32, csinn_reduce_params)                        \
    MACRO(mean, CSINN_QUANT_UINT8_ASYM, csinn_reduce_params)                     \
    MACRO(mean, CSINN_QUANT_INT8_SYM, csinn_reduce_params)                       \
    MACRO(min, CSINN_QUANT_FLOAT32, csinn_reduce_params)                         \
    MACRO(min, CSINN_QUANT_UINT8_ASYM, csinn_reduce_params)                      \
    MACRO(min, CSINN_QUANT_INT8_SYM, csinn_reduce_params)                        \
    MACRO(pad, CSINN_QUANT_FLOAT32, csinn_pad_params)                            \
    MACRO(pad, CSINN_QUANT_UINT8_ASYM, csinn_pad_params)                         \
    MACRO(pad, CSINN_QUANT_INT8_SYM, csinn_pad_params)                           \
    MACRO(prod, CSINN_QUANT_FLOAT32, csinn_reduce_params)                        \
    MACRO(prod, CSINN_QUANT_UINT8_ASYM, csinn_reduce_params)                     \
    MACRO(prod, CSINN_QUANT_INT8_SYM, csinn_reduce_params)                       \
    MACRO(reduce_logsumexp, CSINN_QUANT_FLOAT32, csinn_reduce_params)            \
    MACRO(reduce_logsumexp, CSINN_QUANT_UINT8_ASYM, csinn_reduce_params)         \
    MACRO(reduce_logsumexp, CSINN_QUANT_INT8_SYM, csinn_reduce_params)           \
    MACRO(reduce_max, CSINN_QUANT_FLOAT32, csinn_reduce_params)                  \
    MACRO(reduce_max, CSINN_QUANT_UINT8_ASYM, csinn_reduce_params)               \
    MACRO(reduce_max, CSINN_QUANT_INT8_SYM, csinn_reduce_params)                 \
    MACRO(reduce_mean, CSINN_QUANT_FLOAT32, csinn_reduce_params)                 \
    MACRO(reduce_mean, CSINN_QUANT_UINT8_ASYM, csinn_reduce_params)              \
    MACRO(reduce_mean, CSINN_QUANT_INT8_SYM, csinn_reduce_params)                \
    MACRO(reduce_min, CSINN_QUANT_FLOAT32, csinn_reduce_params)                  \
    MACRO(reduce_min, CSINN_QUANT_UINT8_ASYM, csinn_reduce_params)               \
    MACRO(reduce_min, CSINN_QUANT_INT8_SYM, csinn_reduce_params)                 \
    MACRO(reduce_prod, CSINN_QUANT_FLOAT32, csinn_reduce_params)                 \
    MACRO(reduce_prod, CSINN_QUANT_UINT8_ASYM, csinn_reduce_params)              \
    MACRO(reduce_prod, CSINN_QUANT_INT8_SYM, csinn_reduce_params)                \
    MACRO(reduce_sum, CSINN_QUANT_FLOAT32, csinn_reduce_params)                  \
    MACRO(reduce_sum, CSINN_QUANT_UINT8_ASYM, csinn_reduce_params)               \
    MACRO(reduce_sum, CSINN_QUANT_INT8_SYM, csinn_reduce_params)                 \
    MACRO(relu, CSINN_QUANT_FLOAT32, csinn_relu_params)                          \
    MACRO(relu, CSINN_QUANT_UINT8_ASYM, csinn_relu_params)                       \
    MACRO(relu, CSINN_QUANT_INT8_SYM, csinn_relu_params)                         \
    MACRO(relu1, CSINN_QUANT_FLOAT32, csinn_relu_params)                         \
    MACRO(relu1, CSINN_QUANT_UINT8_ASYM, csinn_relu_params)                      \
    MACRO(relu1, CSINN_QUANT_INT8_SYM, csinn_relu_params)                        \
    MACRO(relu6, CSINN_QUANT_FLOAT32, csinn_relu_params)                         \
    MACRO(relu6, CSINN_QUANT_UINT8_ASYM, csinn_relu_params)                      \
    MACRO(relu6, CSINN_QUANT_INT8_SYM, csinn_relu_params)                        \
    MACRO(relun, CSINN_QUANT_FLOAT32, csinn_relu_params)                         \
    MACRO(relun, CSINN_QUANT_UINT8_ASYM, csinn_relu_params)                      \
    MACRO(relun, CSINN_QUANT_INT8_SYM, csinn_relu_params)                        \
    MACRO(reshape, CSINN_QUANT_FLOAT32, csinn_reshape_params)                    \
    MACRO(reshape, CSINN_QUANT_UINT8_ASYM, csinn_reshape_params)                 \
    MACRO(reshape, CSINN_QUANT_INT8_SYM, csinn_reshape_params)                   \
    MACRO(resize, CSINN_QUANT_FLOAT32, csinn_resize_params)                      \
    MACRO(resize, CSINN_QUANT_UINT8_ASYM, csinn_resize_params)                   \
    MACRO(resize, CSINN_QUANT_INT8_SYM, csinn_resize_params)                     \
    MACRO(reverse, CSINN_QUANT_FLOAT32, csinn_reverse_params)                    \
    MACRO(reverse, CSINN_QUANT_UINT8_ASYM, csinn_reverse_params)                 \
    MACRO(reverse, CSINN_QUANT_INT8_SYM, csinn_reverse_params)                   \
    MACRO(shuffle_channel, CSINN_QUANT_FLOAT32, csinn_shuffle_channel_params)    \
    MACRO(shuffle_channel, CSINN_QUANT_UINT8_ASYM, csinn_shuffle_channel_params) \
    MACRO(shuffle_channel, CSINN_QUANT_INT8_SYM, csinn_shuffle_channel_params)   \
    MACRO(sigmoid, CSINN_QUANT_FLOAT32, csinn_sigmoid_params)                    \
    MACRO(sigmoid, CSINN_QUANT_UINT8_ASYM, csinn_sigmoid_params)                 \
    MACRO(sigmoid, CSINN_QUANT_INT8_SYM, csinn_sigmoid_params)                   \
    MACRO(slice, CSINN_QUANT_FLOAT32, csinn_slice_params)                        \
    MACRO(slice, CSINN_QUANT_UINT8_ASYM, csinn_slice_params)                     \
    MACRO(slice, CSINN_QUANT_INT8_SYM, csinn_slice_params)                       \
    MACRO(softmax, CSINN_QUANT_FLOAT32, csinn_softmax_params)                    \
    MACRO(softmax, CSINN_QUANT_UINT8_ASYM, csinn_softmax_params)                 \
    MACRO(softmax, CSINN_QUANT_INT8_SYM, csinn_softmax_params)                   \
    MACRO(softrelu, CSINN_QUANT_FLOAT32, csinn_relu_params)                      \
    MACRO(softrelu, CSINN_QUANT_UINT8_ASYM, csinn_relu_params)                   \
    MACRO(softrelu, CSINN_QUANT_INT8_SYM, csinn_relu_params)                     \
    MACRO(space_to_batch, CSINN_QUANT_FLOAT32, csinn_space_to_batch_params)      \
    MACRO(space_to_batch, CSINN_QUANT_UINT8_ASYM, csinn_space_to_batch_params)   \
    MACRO(space_to_batch, CSINN_QUANT_INT8_SYM, csinn_space_to_batch_params)     \
    MACRO(space_to_depth, CSINN_QUANT_FLOAT32, csinn_space_to_depth_params)      \
    MACRO(space_to_depth, CSINN_QUANT_UINT8_ASYM, csinn_space_to_depth_params)   \
    MACRO(space_to_depth, CSINN_QUANT_INT8_SYM, csinn_space_to_depth_params)     \
    MACRO(squeeze, CSINN_QUANT_FLOAT32, csinn_squeeze_params)                    \
    MACRO(squeeze, CSINN_QUANT_UINT8_ASYM, csinn_squeeze_params)                 \
    MACRO(squeeze, CSINN_QUANT_INT8_SYM, csinn_squeeze_params)                   \
    MACRO(strided_slice, CSINN_QUANT_FLOAT32, csinn_strided_slice_params)        \
    MACRO(strided_slice, CSINN_QUANT_UINT8_ASYM, csinn_strided_slice_params)     \
    MACRO(strided_slice, CSINN_QUANT_INT8_SYM, csinn_strided_slice_params)       \
    MACRO(sum, CSINN_QUANT_FLOAT32, csinn_reduce_params)                         \
    MACRO(sum, CSINN_QUANT_UINT8_ASYM, csinn_reduce_params)                      \
    MACRO(sum, CSINN_QUANT_INT8_SYM, csinn_reduce_params)                        \
    MACRO(threshold_relu, CSINN_QUANT_FLOAT32, csinn_relu_params)                \
    MACRO(threshold_relu, CSINN_QUANT_UINT8_ASYM, csinn_relu_params)             \
    MACRO(threshold_relu, CSINN_QUANT_INT8_SYM, csinn_relu_params)               \
    MACRO(tile, CSINN_QUANT_FLOAT32, csinn_tile_params)                          \
    MACRO(tile, CSINN_QUANT_UINT8_ASYM, csinn_tile_params)                       \
    MACRO(tile, CSINN_QUANT_INT8_SYM, csinn_tile_params)                         \
    MACRO(transpose, CSINN_QUANT_FLOAT32, csinn_transpose_params)                \
    MACRO(transpose, CSINN_QUANT_UINT8_ASYM, csinn_transpose_params)             \
    MACRO(transpose, CSINN_QUANT_INT8_SYM, csinn_transpose_params)               \
    MACRO(argmax, CSINN_QUANT_FLOAT32, csinn_reduce_params)                      \
    MACRO(argmax, CSINN_QUANT_UINT8_ASYM, csinn_reduce_params)                   \
    MACRO(argmax, CSINN_QUANT_INT8_SYM, csinn_reduce_params)                     \
    MACRO(argmin, CSINN_QUANT_FLOAT32, csinn_reduce_params)                      \
    MACRO(argmin, CSINN_QUANT_UINT8_ASYM, csinn_reduce_params)                   \
    MACRO(argmin, CSINN_QUANT_INT8_SYM, csinn_reduce_params)                     \
    MACRO(broadcast_to, CSINN_QUANT_FLOAT32, csinn_broadcast_to_params)          \
    MACRO(broadcast_to, CSINN_QUANT_UINT8_ASYM, csinn_broadcast_to_params)       \
    MACRO(broadcast_to, CSINN_QUANT_INT8_SYM, csinn_broadcast_to_params)

#define LAYER_QUANT_TEST_DISO(MACRO)                                                     \
    MACRO(add, CSINN_QUANT_FLOAT32, csinn_diso_params)                                   \
    MACRO(add, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                                \
    MACRO(add, CSINN_QUANT_INT8_SYM, csinn_diso_params)                                  \
    MACRO(div, CSINN_QUANT_FLOAT32, csinn_diso_params)                                   \
    MACRO(div, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                                \
    MACRO(div, CSINN_QUANT_INT8_SYM, csinn_diso_params)                                  \
    MACRO(equal, CSINN_QUANT_FLOAT32, csinn_diso_params)                                 \
    MACRO(equal, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                              \
    MACRO(equal, CSINN_QUANT_INT8_SYM, csinn_diso_params)                                \
    MACRO(floor_divide, CSINN_QUANT_FLOAT32, csinn_diso_params)                          \
    MACRO(floor_divide, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                       \
    MACRO(floor_divide, CSINN_QUANT_INT8_SYM, csinn_diso_params)                         \
    MACRO(floor_mod, CSINN_QUANT_FLOAT32, csinn_diso_params)                             \
    MACRO(floor_mod, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                          \
    MACRO(floor_mod, CSINN_QUANT_INT8_SYM, csinn_diso_params)                            \
    MACRO(greater_equal, CSINN_QUANT_FLOAT32, csinn_diso_params)                         \
    MACRO(greater_equal, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                      \
    MACRO(greater_equal, CSINN_QUANT_INT8_SYM, csinn_diso_params)                        \
    MACRO(greater, CSINN_QUANT_FLOAT32, csinn_diso_params)                               \
    MACRO(greater, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                            \
    MACRO(greater, CSINN_QUANT_INT8_SYM, csinn_diso_params)                              \
    MACRO(less_equal, CSINN_QUANT_FLOAT32, csinn_diso_params)                            \
    MACRO(less_equal, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                         \
    MACRO(less_equal, CSINN_QUANT_INT8_SYM, csinn_diso_params)                           \
    MACRO(less, CSINN_QUANT_FLOAT32, csinn_diso_params)                                  \
    MACRO(less, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                               \
    MACRO(less, CSINN_QUANT_INT8_SYM, csinn_diso_params)                                 \
    MACRO(logical_and, CSINN_QUANT_FLOAT32, csinn_diso_params)                           \
    MACRO(logical_and, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                        \
    MACRO(logical_and, CSINN_QUANT_INT8_SYM, csinn_diso_params)                          \
    MACRO(logical_or, CSINN_QUANT_FLOAT32, csinn_diso_params)                            \
    MACRO(logical_or, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                         \
    MACRO(logical_or, CSINN_QUANT_INT8_SYM, csinn_diso_params)                           \
    MACRO(logical_xor, CSINN_QUANT_FLOAT32, csinn_diso_params)                           \
    MACRO(logical_xor, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                        \
    MACRO(logical_xor, CSINN_QUANT_INT8_SYM, csinn_diso_params)                          \
    MACRO(mod, CSINN_QUANT_FLOAT32, csinn_diso_params)                                   \
    MACRO(mod, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                                \
    MACRO(mod, CSINN_QUANT_INT8_SYM, csinn_diso_params)                                  \
    MACRO(mul, CSINN_QUANT_FLOAT32, csinn_diso_params)                                   \
    MACRO(mul, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                                \
    MACRO(mul, CSINN_QUANT_INT8_SYM, csinn_diso_params)                                  \
    MACRO(not_equal, CSINN_QUANT_FLOAT32, csinn_diso_params)                             \
    MACRO(not_equal, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                          \
    MACRO(not_equal, CSINN_QUANT_INT8_SYM, csinn_diso_params)                            \
    MACRO(power, CSINN_QUANT_FLOAT32, csinn_diso_params)                                 \
    MACRO(power, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                              \
    MACRO(power, CSINN_QUANT_INT8_SYM, csinn_diso_params)                                \
    MACRO(sub, CSINN_QUANT_FLOAT32, csinn_diso_params)                                   \
    MACRO(sub, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                                \
    MACRO(sub, CSINN_QUANT_INT8_SYM, csinn_diso_params)                                  \
    MACRO(maximum, CSINN_QUANT_FLOAT32, csinn_diso_params)                               \
    MACRO(maximum, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                            \
    MACRO(maximum, CSINN_QUANT_INT8_SYM, csinn_diso_params)                              \
    MACRO(minimum, CSINN_QUANT_FLOAT32, csinn_diso_params)                               \
    MACRO(minimum, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                            \
    MACRO(minimum, CSINN_QUANT_INT8_SYM, csinn_diso_params)                              \
    MACRO(and, CSINN_QUANT_FLOAT32, csinn_diso_params)                                   \
    MACRO(and, CSINN_QUANT_UINT8_ASYM, csinn_diso_params)                                \
    MACRO(and, CSINN_QUANT_INT8_SYM, csinn_diso_params)                                  \
    MACRO(matmul, CSINN_QUANT_FLOAT32, csinn_matmul_params)                              \
    MACRO(matmul, CSINN_QUANT_UINT8_ASYM, csinn_matmul_params)                           \
    MACRO(matmul, CSINN_QUANT_INT8_SYM, csinn_matmul_params)                             \
    MACRO(prelu, CSINN_QUANT_FLOAT32, csinn_prelu_params)                                \
    MACRO(prelu, CSINN_QUANT_UINT8_ASYM, csinn_prelu_params)                             \
    MACRO(prelu, CSINN_QUANT_INT8_SYM, csinn_prelu_params)                               \
    MACRO(non_max_suppression, CSINN_QUANT_FLOAT32, csinn_non_max_suppression_params)    \
    MACRO(non_max_suppression, CSINN_QUANT_UINT8_ASYM, csinn_non_max_suppression_params) \
    MACRO(non_max_suppression, CSINN_QUANT_INT8_SYM, csinn_non_max_suppression_params)   \
    MACRO(psroipooling, CSINN_QUANT_FLOAT32, csinn_psroipooling_params)                  \
    MACRO(psroipooling, CSINN_QUANT_UINT8_ASYM, csinn_psroipooling_params)               \
    MACRO(psroipooling, CSINN_QUANT_INT8_SYM, csinn_psroipooling_params)                 \
    MACRO(roi_align, CSINN_QUANT_FLOAT32, csinn_roi_align_params)                        \
    MACRO(roi_align, CSINN_QUANT_UINT8_ASYM, csinn_roi_align_params)                     \
    MACRO(roi_align, CSINN_QUANT_INT8_SYM, csinn_roi_align_params)                       \
    MACRO(roipool, CSINN_QUANT_FLOAT32, csinn_roi_pool_params)                           \
    MACRO(roipool, CSINN_QUANT_UINT8_ASYM, csinn_roi_pool_params)                        \
    MACRO(roipool, CSINN_QUANT_INT8_SYM, csinn_roi_pool_params)                          \
    MACRO(gather_nd, CSINN_QUANT_FLOAT32, csinn_gather_nd_params)                        \
    MACRO(gather_nd, CSINN_QUANT_UINT8_ASYM, csinn_gather_nd_params)                     \
    MACRO(gather_nd, CSINN_QUANT_INT8_SYM, csinn_gather_nd_params)                       \
    MACRO(gather, CSINN_QUANT_FLOAT32, csinn_gather_params)                              \
    MACRO(gather, CSINN_QUANT_UINT8_ASYM, csinn_gather_params)                           \
    MACRO(gather, CSINN_QUANT_INT8_SYM, csinn_gather_params)

#define LAYER_QUANT_TEST_SEGMENT(MACRO)                               \
    MACRO(segment_max, CSINN_QUANT_FLOAT32, csinn_segment_params)     \
    MACRO(segment_max, CSINN_QUANT_UINT8_ASYM, csinn_segment_params)  \
    MACRO(segment_max, CSINN_QUANT_INT8_SYM, csinn_segment_params)    \
    MACRO(segment_mean, CSINN_QUANT_FLOAT32, csinn_segment_params)    \
    MACRO(segment_mean, CSINN_QUANT_UINT8_ASYM, csinn_segment_params) \
    MACRO(segment_mean, CSINN_QUANT_INT8_SYM, csinn_segment_params)   \
    MACRO(segment_min, CSINN_QUANT_FLOAT32, csinn_segment_params)     \
    MACRO(segment_min, CSINN_QUANT_UINT8_ASYM, csinn_segment_params)  \
    MACRO(segment_min, CSINN_QUANT_INT8_SYM, csinn_segment_params)    \
    MACRO(segment_prod, CSINN_QUANT_FLOAT32, csinn_segment_params)    \
    MACRO(segment_prod, CSINN_QUANT_UINT8_ASYM, csinn_segment_params) \
    MACRO(segment_prod, CSINN_QUANT_INT8_SYM, csinn_segment_params)   \
    MACRO(segment_sum, CSINN_QUANT_FLOAT32, csinn_segment_params)     \
    MACRO(segment_sum, CSINN_QUANT_UINT8_ASYM, csinn_segment_params)  \
    MACRO(segment_sum, CSINN_QUANT_INT8_SYM, csinn_segment_params)

#define LAYER_QUANT_TEST_BATCHNORM(MACRO)                               \
    MACRO(batch_normalization, CSINN_QUANT_FLOAT32, csinn_bn_params)    \
    MACRO(batch_normalization, CSINN_QUANT_UINT8_ASYM, csinn_bn_params) \
    MACRO(batch_normalization, CSINN_QUANT_INT8_SYM, csinn_bn_params)

#define LAYER_QUANT_TEST_CONCAT(MACRO)                         \
    MACRO(concat, CSINN_QUANT_FLOAT32, csinn_concat_params)    \
    MACRO(concat, CSINN_QUANT_UINT8_ASYM, csinn_concat_params) \
    MACRO(concat, CSINN_QUANT_INT8_SYM, csinn_concat_params)   \
    MACRO(stack, CSINN_QUANT_FLOAT32, csinn_stack_params)      \
    MACRO(stack, CSINN_QUANT_UINT8_ASYM, csinn_stack_params)   \
    MACRO(stack, CSINN_QUANT_INT8_SYM, csinn_stack_params)

#define LAYER_QUANT_TEST_CONV2D(MACRO)                               \
    MACRO(conv2d, CSINN_QUANT_FLOAT32, csinn_conv2d_params)          \
    MACRO(conv2d, CSINN_QUANT_UINT8_ASYM, csinn_conv2d_params)       \
    MACRO(conv2d, CSINN_QUANT_INT8_SYM, csinn_conv2d_params)         \
    MACRO(conv3d, CSINN_QUANT_FLOAT32, csinn_conv3d_params)          \
    MACRO(conv3d, CSINN_QUANT_UINT8_ASYM, csinn_conv3d_params)       \
    MACRO(conv3d, CSINN_QUANT_INT8_SYM, csinn_conv3d_params)         \
    MACRO(conv2d_relu, CSINN_QUANT_FLOAT32, csinn_conv2d_params)     \
    MACRO(conv2d_relu, CSINN_QUANT_UINT8_ASYM, csinn_conv2d_params)  \
    MACRO(conv2d_relu, CSINN_QUANT_INT8_SYM, csinn_conv2d_params)    \
    MACRO(conv2d_relu6, CSINN_QUANT_FLOAT32, csinn_conv2d_params)    \
    MACRO(conv2d_relu6, CSINN_QUANT_UINT8_ASYM, csinn_conv2d_params) \
    MACRO(conv2d_relu6, CSINN_QUANT_INT8_SYM, csinn_conv2d_params)   \
    MACRO(deconv2d, CSINN_QUANT_FLOAT32, csinn_conv2d_params)        \
    MACRO(deconv2d, CSINN_QUANT_UINT8_ASYM, csinn_conv2d_params)     \
    MACRO(deconv2d, CSINN_QUANT_INT8_SYM, csinn_conv2d_params)       \
    MACRO(deconv3d, CSINN_QUANT_FLOAT32, csinn_conv3d_params)        \
    MACRO(deconv3d, CSINN_QUANT_UINT8_ASYM, csinn_conv3d_params)     \
    MACRO(deconv3d, CSINN_QUANT_INT8_SYM, csinn_conv3d_params)       \
    MACRO(fullyconnected, CSINN_QUANT_FLOAT32, csinn_fc_params)      \
    MACRO(fullyconnected, CSINN_QUANT_UINT8_ASYM, csinn_fc_params)   \
    MACRO(fullyconnected, CSINN_QUANT_INT8_SYM, csinn_fc_params)

#define LAYER_QUANT_TEST_TISO(MACRO)                           \
    MACRO(select, CSINN_QUANT_FLOAT32, csinn_select_params)    \
    MACRO(select, CSINN_QUANT_UINT8_ASYM, csinn_select_params) \
    MACRO(select, CSINN_QUANT_INT8_SYM, csinn_select_params)

#define LAYER_QUANT_TEST_SPLIT(MACRO)                        \
    MACRO(split, CSINN_QUANT_FLOAT32, csinn_split_params)    \
    MACRO(split, CSINN_QUANT_UINT8_ASYM, csinn_split_params) \
    MACRO(split, CSINN_QUANT_INT8_SYM, csinn_split_params)

#define LAYER_QUANT_TEST_UNSTACK(MACRO)                          \
    MACRO(unstack, CSINN_QUANT_FLOAT32, csinn_unstack_params)    \
    MACRO(unstack, CSINN_QUANT_UINT8_ASYM, csinn_unstack_params) \
    MACRO(unstack, CSINN_QUANT_INT8_SYM, csinn_unstack_params)

#define LAYER_QUANT_TEST_ARANGE(MACRO)                         \
    MACRO(arange, CSINN_QUANT_FLOAT32, csinn_arange_params)    \
    MACRO(arange, CSINN_QUANT_UINT8_ASYM, csinn_arange_params) \
    MACRO(arange, CSINN_QUANT_INT8_SYM, csinn_arange_params)
