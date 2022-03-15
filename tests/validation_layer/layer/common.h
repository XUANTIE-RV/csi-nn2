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

/* CSI-NN2 version 1.12.x */

#include <stddef.h>

#include "csi_nn.h"
#include "math_snr.h"
#include "test_utils.h"

#define LAYER_QUANT_TEST_SISO(MACRO)                                       \
    MACRO(abs, CSINN_QUANT_FLOAT32, siso_params)                           \
    MACRO(abs, CSINN_QUANT_UINT8_ASYM, siso_params)                        \
    MACRO(abs, CSINN_QUANT_INT8_SYM, siso_params)                          \
    MACRO(acos, CSINN_QUANT_FLOAT32, siso_params)                          \
    MACRO(acos, CSINN_QUANT_UINT8_ASYM, siso_params)                       \
    MACRO(acos, CSINN_QUANT_INT8_SYM, siso_params)                         \
    MACRO(acosh, CSINN_QUANT_FLOAT32, siso_params)                         \
    MACRO(acosh, CSINN_QUANT_UINT8_ASYM, siso_params)                      \
    MACRO(acosh, CSINN_QUANT_INT8_SYM, siso_params)                        \
    MACRO(asin, CSINN_QUANT_FLOAT32, siso_params)                          \
    MACRO(asin, CSINN_QUANT_UINT8_ASYM, siso_params)                       \
    MACRO(asin, CSINN_QUANT_INT8_SYM, siso_params)                         \
    MACRO(asinh, CSINN_QUANT_FLOAT32, siso_params)                         \
    MACRO(asinh, CSINN_QUANT_UINT8_ASYM, siso_params)                      \
    MACRO(asinh, CSINN_QUANT_INT8_SYM, siso_params)                        \
    MACRO(atan, CSINN_QUANT_FLOAT32, siso_params)                          \
    MACRO(atan, CSINN_QUANT_UINT8_ASYM, siso_params)                       \
    MACRO(atan, CSINN_QUANT_INT8_SYM, siso_params)                         \
    MACRO(atanh, CSINN_QUANT_FLOAT32, siso_params)                         \
    MACRO(atanh, CSINN_QUANT_UINT8_ASYM, siso_params)                      \
    MACRO(atanh, CSINN_QUANT_INT8_SYM, siso_params)                        \
    MACRO(ceil, CSINN_QUANT_FLOAT32, siso_params)                          \
    MACRO(ceil, CSINN_QUANT_UINT8_ASYM, siso_params)                       \
    MACRO(ceil, CSINN_QUANT_INT8_SYM, siso_params)                         \
    MACRO(cos, CSINN_QUANT_FLOAT32, siso_params)                           \
    MACRO(cos, CSINN_QUANT_UINT8_ASYM, siso_params)                        \
    MACRO(cos, CSINN_QUANT_INT8_SYM, siso_params)                          \
    MACRO(cosh, CSINN_QUANT_FLOAT32, siso_params)                          \
    MACRO(cosh, CSINN_QUANT_UINT8_ASYM, siso_params)                       \
    MACRO(cosh, CSINN_QUANT_INT8_SYM, siso_params)                         \
    MACRO(erf, CSINN_QUANT_FLOAT32, siso_params)                           \
    MACRO(erf, CSINN_QUANT_UINT8_ASYM, siso_params)                        \
    MACRO(erf, CSINN_QUANT_INT8_SYM, siso_params)                          \
    MACRO(exp, CSINN_QUANT_FLOAT32, siso_params)                           \
    MACRO(exp, CSINN_QUANT_UINT8_ASYM, siso_params)                        \
    MACRO(exp, CSINN_QUANT_INT8_SYM, siso_params)                          \
    MACRO(expm1, CSINN_QUANT_FLOAT32, siso_params)                         \
    MACRO(expm1, CSINN_QUANT_UINT8_ASYM, siso_params)                      \
    MACRO(expm1, CSINN_QUANT_INT8_SYM, siso_params)                        \
    MACRO(floor, CSINN_QUANT_FLOAT32, siso_params)                         \
    MACRO(floor, CSINN_QUANT_UINT8_ASYM, siso_params)                      \
    MACRO(floor, CSINN_QUANT_INT8_SYM, siso_params)                        \
    MACRO(log, CSINN_QUANT_FLOAT32, siso_params)                           \
    MACRO(log, CSINN_QUANT_UINT8_ASYM, siso_params)                        \
    MACRO(log, CSINN_QUANT_INT8_SYM, siso_params)                          \
    MACRO(log1p, CSINN_QUANT_FLOAT32, siso_params)                         \
    MACRO(log1p, CSINN_QUANT_UINT8_ASYM, siso_params)                      \
    MACRO(log1p, CSINN_QUANT_INT8_SYM, siso_params)                        \
    MACRO(logical_not, CSINN_QUANT_FLOAT32, siso_params)                   \
    MACRO(logical_not, CSINN_QUANT_UINT8_ASYM, siso_params)                \
    MACRO(logical_not, CSINN_QUANT_INT8_SYM, siso_params)                  \
    MACRO(round, CSINN_QUANT_FLOAT32, siso_params)                         \
    MACRO(round, CSINN_QUANT_UINT8_ASYM, siso_params)                      \
    MACRO(round, CSINN_QUANT_INT8_SYM, siso_params)                        \
    MACRO(rsqrt, CSINN_QUANT_FLOAT32, siso_params)                         \
    MACRO(rsqrt, CSINN_QUANT_UINT8_ASYM, siso_params)                      \
    MACRO(rsqrt, CSINN_QUANT_INT8_SYM, siso_params)                        \
    MACRO(sign, CSINN_QUANT_FLOAT32, siso_params)                          \
    MACRO(sign, CSINN_QUANT_UINT8_ASYM, siso_params)                       \
    MACRO(sign, CSINN_QUANT_INT8_SYM, siso_params)                         \
    MACRO(negative, CSINN_QUANT_FLOAT32, siso_params)                      \
    MACRO(negative, CSINN_QUANT_UINT8_ASYM, siso_params)                   \
    MACRO(negative, CSINN_QUANT_INT8_SYM, siso_params)                     \
    MACRO(sin, CSINN_QUANT_FLOAT32, siso_params)                           \
    MACRO(sin, CSINN_QUANT_UINT8_ASYM, siso_params)                        \
    MACRO(sin, CSINN_QUANT_INT8_SYM, siso_params)                          \
    MACRO(sinh, CSINN_QUANT_FLOAT32, siso_params)                          \
    MACRO(sinh, CSINN_QUANT_UINT8_ASYM, siso_params)                       \
    MACRO(sinh, CSINN_QUANT_INT8_SYM, siso_params)                         \
    MACRO(softplus, CSINN_QUANT_FLOAT32, siso_params)                      \
    MACRO(softplus, CSINN_QUANT_UINT8_ASYM, siso_params)                   \
    MACRO(softplus, CSINN_QUANT_INT8_SYM, siso_params)                     \
    MACRO(softsign, CSINN_QUANT_FLOAT32, siso_params)                      \
    MACRO(softsign, CSINN_QUANT_UINT8_ASYM, siso_params)                   \
    MACRO(softsign, CSINN_QUANT_INT8_SYM, siso_params)                     \
    MACRO(sqrt, CSINN_QUANT_FLOAT32, siso_params)                          \
    MACRO(sqrt, CSINN_QUANT_UINT8_ASYM, siso_params)                       \
    MACRO(sqrt, CSINN_QUANT_INT8_SYM, siso_params)                         \
    MACRO(square, CSINN_QUANT_FLOAT32, siso_params)                        \
    MACRO(square, CSINN_QUANT_UINT8_ASYM, siso_params)                     \
    MACRO(square, CSINN_QUANT_INT8_SYM, siso_params)                       \
    MACRO(tan, CSINN_QUANT_FLOAT32, siso_params)                           \
    MACRO(tan, CSINN_QUANT_UINT8_ASYM, siso_params)                        \
    MACRO(tan, CSINN_QUANT_INT8_SYM, siso_params)                          \
    MACRO(tanh, CSINN_QUANT_FLOAT32, siso_params)                          \
    MACRO(tanh, CSINN_QUANT_UINT8_ASYM, siso_params)                       \
    MACRO(tanh, CSINN_QUANT_INT8_SYM, siso_params)                         \
    MACRO(trunc, CSINN_QUANT_FLOAT32, siso_params)                         \
    MACRO(trunc, CSINN_QUANT_UINT8_ASYM, siso_params)                      \
    MACRO(trunc, CSINN_QUANT_INT8_SYM, siso_params)                        \
    MACRO(yuv_rgb_scale, CSINN_QUANT_FLOAT32, siso_params)                 \
    MACRO(yuv_rgb_scale, CSINN_QUANT_UINT8_ASYM, siso_params)              \
    MACRO(yuv_rgb_scale, CSINN_QUANT_INT8_SYM, siso_params)                \
    MACRO(not, CSINN_QUANT_FLOAT32, siso_params)                           \
    MACRO(not, CSINN_QUANT_UINT8_ASYM, siso_params)                        \
    MACRO(not, CSINN_QUANT_INT8_SYM, siso_params)                          \
    MACRO(avgpool2d, CSINN_QUANT_FLOAT32, pool_params)                     \
    MACRO(avgpool2d, CSINN_QUANT_UINT8_ASYM, pool_params)                  \
    MACRO(avgpool2d, CSINN_QUANT_INT8_SYM, pool_params)                    \
    MACRO(avgpool3d, CSINN_QUANT_FLOAT32, pool_params)                     \
    MACRO(avgpool3d, CSINN_QUANT_UINT8_ASYM, pool_params)                  \
    MACRO(avgpool3d, CSINN_QUANT_INT8_SYM, pool_params)                    \
    MACRO(clip, CSINN_QUANT_FLOAT32, clip_params)                          \
    MACRO(clip, CSINN_QUANT_UINT8_ASYM, clip_params)                       \
    MACRO(clip, CSINN_QUANT_INT8_SYM, clip_params)                         \
    MACRO(batch_to_space, CSINN_QUANT_FLOAT32, batch_to_space_params)      \
    MACRO(batch_to_space, CSINN_QUANT_UINT8_ASYM, batch_to_space_params)   \
    MACRO(batch_to_space, CSINN_QUANT_INT8_SYM, batch_to_space_params)     \
    MACRO(cumprod, CSINN_QUANT_FLOAT32, cumprod_params)                    \
    MACRO(cumprod, CSINN_QUANT_UINT8_ASYM, cumprod_params)                 \
    MACRO(cumprod, CSINN_QUANT_INT8_SYM, cumprod_params)                   \
    MACRO(cumsum, CSINN_QUANT_FLOAT32, cumsum_params)                      \
    MACRO(cumsum, CSINN_QUANT_UINT8_ASYM, cumsum_params)                   \
    MACRO(cumsum, CSINN_QUANT_INT8_SYM, cumsum_params)                     \
    MACRO(depth_to_space, CSINN_QUANT_FLOAT32, depth_to_space_params)      \
    MACRO(depth_to_space, CSINN_QUANT_UINT8_ASYM, depth_to_space_params)   \
    MACRO(depth_to_space, CSINN_QUANT_INT8_SYM, depth_to_space_params)     \
    MACRO(elu, CSINN_QUANT_FLOAT32, relu_params)                           \
    MACRO(elu, CSINN_QUANT_UINT8_ASYM, relu_params)                        \
    MACRO(elu, CSINN_QUANT_INT8_SYM, relu_params)                          \
    MACRO(expand_dims, CSINN_QUANT_FLOAT32, expand_dims_params)            \
    MACRO(expand_dims, CSINN_QUANT_UINT8_ASYM, expand_dims_params)         \
    MACRO(expand_dims, CSINN_QUANT_INT8_SYM, expand_dims_params)           \
    MACRO(flatten, CSINN_QUANT_FLOAT32, flatten_params)                    \
    MACRO(flatten, CSINN_QUANT_UINT8_ASYM, flatten_params)                 \
    MACRO(flatten, CSINN_QUANT_INT8_SYM, flatten_params)                   \
    MACRO(global_avgpool2d, CSINN_QUANT_FLOAT32, pool_params)              \
    MACRO(global_avgpool2d, CSINN_QUANT_UINT8_ASYM, pool_params)           \
    MACRO(global_avgpool2d, CSINN_QUANT_INT8_SYM, pool_params)             \
    MACRO(global_maxpool2d, CSINN_QUANT_FLOAT32, pool_params)              \
    MACRO(global_maxpool2d, CSINN_QUANT_UINT8_ASYM, pool_params)           \
    MACRO(global_maxpool2d, CSINN_QUANT_INT8_SYM, pool_params)             \
    MACRO(hard_sigmoid, CSINN_QUANT_FLOAT32, sigmoid_params)               \
    MACRO(hard_sigmoid, CSINN_QUANT_UINT8_ASYM, sigmoid_params)            \
    MACRO(hard_sigmoid, CSINN_QUANT_INT8_SYM, sigmoid_params)              \
    MACRO(im2col, CSINN_QUANT_FLOAT32, im2col_params)                      \
    MACRO(im2col, CSINN_QUANT_UINT8_ASYM, im2col_params)                   \
    MACRO(im2col, CSINN_QUANT_INT8_SYM, im2col_params)                     \
    MACRO(l2_normalization, CSINN_QUANT_FLOAT32, l2n_params)               \
    MACRO(l2_normalization, CSINN_QUANT_UINT8_ASYM, l2n_params)            \
    MACRO(l2_normalization, CSINN_QUANT_INT8_SYM, l2n_params)              \
    MACRO(leaky_relu, CSINN_QUANT_FLOAT32, relu_params)                    \
    MACRO(leaky_relu, CSINN_QUANT_UINT8_ASYM, relu_params)                 \
    MACRO(leaky_relu, CSINN_QUANT_INT8_SYM, relu_params)                   \
    MACRO(log_softmax, CSINN_QUANT_FLOAT32, softmax_params)                \
    MACRO(log_softmax, CSINN_QUANT_UINT8_ASYM, softmax_params)             \
    MACRO(log_softmax, CSINN_QUANT_INT8_SYM, softmax_params)               \
    MACRO(lrn, CSINN_QUANT_FLOAT32, lrn_params)                            \
    MACRO(lrn, CSINN_QUANT_UINT8_ASYM, lrn_params)                         \
    MACRO(lrn, CSINN_QUANT_INT8_SYM, lrn_params)                           \
    MACRO(max, CSINN_QUANT_FLOAT32, reduce_params)                         \
    MACRO(max, CSINN_QUANT_UINT8_ASYM, reduce_params)                      \
    MACRO(max, CSINN_QUANT_INT8_SYM, reduce_params)                        \
    MACRO(maxpool2d, CSINN_QUANT_FLOAT32, pool_params)                     \
    MACRO(maxpool2d, CSINN_QUANT_UINT8_ASYM, pool_params)                  \
    MACRO(maxpool2d, CSINN_QUANT_INT8_SYM, pool_params)                    \
    MACRO(maxpool3d, CSINN_QUANT_FLOAT32, pool_params)                     \
    MACRO(maxpool3d, CSINN_QUANT_UINT8_ASYM, pool_params)                  \
    MACRO(maxpool3d, CSINN_QUANT_INT8_SYM, pool_params)                    \
    MACRO(mean, CSINN_QUANT_FLOAT32, reduce_params)                        \
    MACRO(mean, CSINN_QUANT_UINT8_ASYM, reduce_params)                     \
    MACRO(mean, CSINN_QUANT_INT8_SYM, reduce_params)                       \
    MACRO(min, CSINN_QUANT_FLOAT32, reduce_params)                         \
    MACRO(min, CSINN_QUANT_UINT8_ASYM, reduce_params)                      \
    MACRO(min, CSINN_QUANT_INT8_SYM, reduce_params)                        \
    MACRO(pad, CSINN_QUANT_FLOAT32, pad_params)                            \
    MACRO(pad, CSINN_QUANT_UINT8_ASYM, pad_params)                         \
    MACRO(pad, CSINN_QUANT_INT8_SYM, pad_params)                           \
    MACRO(prod, CSINN_QUANT_FLOAT32, reduce_params)                        \
    MACRO(prod, CSINN_QUANT_UINT8_ASYM, reduce_params)                     \
    MACRO(prod, CSINN_QUANT_INT8_SYM, reduce_params)                       \
    MACRO(reduce_logsumexp, CSINN_QUANT_FLOAT32, reduce_params)            \
    MACRO(reduce_logsumexp, CSINN_QUANT_UINT8_ASYM, reduce_params)         \
    MACRO(reduce_logsumexp, CSINN_QUANT_INT8_SYM, reduce_params)           \
    MACRO(reduce_max, CSINN_QUANT_FLOAT32, reduce_params)                  \
    MACRO(reduce_max, CSINN_QUANT_UINT8_ASYM, reduce_params)               \
    MACRO(reduce_max, CSINN_QUANT_INT8_SYM, reduce_params)                 \
    MACRO(reduce_mean, CSINN_QUANT_FLOAT32, reduce_params)                 \
    MACRO(reduce_mean, CSINN_QUANT_UINT8_ASYM, reduce_params)              \
    MACRO(reduce_mean, CSINN_QUANT_INT8_SYM, reduce_params)                \
    MACRO(reduce_min, CSINN_QUANT_FLOAT32, reduce_params)                  \
    MACRO(reduce_min, CSINN_QUANT_UINT8_ASYM, reduce_params)               \
    MACRO(reduce_min, CSINN_QUANT_INT8_SYM, reduce_params)                 \
    MACRO(reduce_prod, CSINN_QUANT_FLOAT32, reduce_params)                 \
    MACRO(reduce_prod, CSINN_QUANT_UINT8_ASYM, reduce_params)              \
    MACRO(reduce_prod, CSINN_QUANT_INT8_SYM, reduce_params)                \
    MACRO(reduce_sum, CSINN_QUANT_FLOAT32, reduce_params)                  \
    MACRO(reduce_sum, CSINN_QUANT_UINT8_ASYM, reduce_params)               \
    MACRO(reduce_sum, CSINN_QUANT_INT8_SYM, reduce_params)                 \
    MACRO(relu, CSINN_QUANT_FLOAT32, relu_params)                          \
    MACRO(relu, CSINN_QUANT_UINT8_ASYM, relu_params)                       \
    MACRO(relu, CSINN_QUANT_INT8_SYM, relu_params)                         \
    MACRO(relu1, CSINN_QUANT_FLOAT32, relu_params)                         \
    MACRO(relu1, CSINN_QUANT_UINT8_ASYM, relu_params)                      \
    MACRO(relu1, CSINN_QUANT_INT8_SYM, relu_params)                        \
    MACRO(relu6, CSINN_QUANT_FLOAT32, relu_params)                         \
    MACRO(relu6, CSINN_QUANT_UINT8_ASYM, relu_params)                      \
    MACRO(relu6, CSINN_QUANT_INT8_SYM, relu_params)                        \
    MACRO(relun, CSINN_QUANT_FLOAT32, relu_params)                         \
    MACRO(relun, CSINN_QUANT_UINT8_ASYM, relu_params)                      \
    MACRO(relun, CSINN_QUANT_INT8_SYM, relu_params)                        \
    MACRO(reshape, CSINN_QUANT_FLOAT32, reshape_params)                    \
    MACRO(reshape, CSINN_QUANT_UINT8_ASYM, reshape_params)                 \
    MACRO(reshape, CSINN_QUANT_INT8_SYM, reshape_params)                   \
    MACRO(resize, CSINN_QUANT_FLOAT32, resize_params)                      \
    MACRO(resize, CSINN_QUANT_UINT8_ASYM, resize_params)                   \
    MACRO(resize, CSINN_QUANT_INT8_SYM, resize_params)                     \
    MACRO(reverse, CSINN_QUANT_FLOAT32, reverse_params)                    \
    MACRO(reverse, CSINN_QUANT_UINT8_ASYM, reverse_params)                 \
    MACRO(reverse, CSINN_QUANT_INT8_SYM, reverse_params)                   \
    MACRO(shuffle_channel, CSINN_QUANT_FLOAT32, shuffle_channel_params)    \
    MACRO(shuffle_channel, CSINN_QUANT_UINT8_ASYM, shuffle_channel_params) \
    MACRO(shuffle_channel, CSINN_QUANT_INT8_SYM, shuffle_channel_params)   \
    MACRO(sigmoid, CSINN_QUANT_FLOAT32, sigmoid_params)                    \
    MACRO(sigmoid, CSINN_QUANT_UINT8_ASYM, sigmoid_params)                 \
    MACRO(sigmoid, CSINN_QUANT_INT8_SYM, sigmoid_params)                   \
    MACRO(slice, CSINN_QUANT_FLOAT32, slice_params)                        \
    MACRO(slice, CSINN_QUANT_UINT8_ASYM, slice_params)                     \
    MACRO(slice, CSINN_QUANT_INT8_SYM, slice_params)                       \
    MACRO(softmax, CSINN_QUANT_FLOAT32, softmax_params)                    \
    MACRO(softmax, CSINN_QUANT_UINT8_ASYM, softmax_params)                 \
    MACRO(softmax, CSINN_QUANT_INT8_SYM, softmax_params)                   \
    MACRO(softrelu, CSINN_QUANT_FLOAT32, relu_params)                      \
    MACRO(softrelu, CSINN_QUANT_UINT8_ASYM, relu_params)                   \
    MACRO(softrelu, CSINN_QUANT_INT8_SYM, relu_params)                     \
    MACRO(space_to_batch, CSINN_QUANT_FLOAT32, space_to_batch_params)      \
    MACRO(space_to_batch, CSINN_QUANT_UINT8_ASYM, space_to_batch_params)   \
    MACRO(space_to_batch, CSINN_QUANT_INT8_SYM, space_to_batch_params)     \
    MACRO(space_to_depth, CSINN_QUANT_FLOAT32, space_to_depth_params)      \
    MACRO(space_to_depth, CSINN_QUANT_UINT8_ASYM, space_to_depth_params)   \
    MACRO(space_to_depth, CSINN_QUANT_INT8_SYM, space_to_depth_params)     \
    MACRO(squeeze, CSINN_QUANT_FLOAT32, squeeze_params)                    \
    MACRO(squeeze, CSINN_QUANT_UINT8_ASYM, squeeze_params)                 \
    MACRO(squeeze, CSINN_QUANT_INT8_SYM, squeeze_params)                   \
    MACRO(strided_slice, CSINN_QUANT_FLOAT32, strided_slice_params)        \
    MACRO(strided_slice, CSINN_QUANT_UINT8_ASYM, strided_slice_params)     \
    MACRO(strided_slice, CSINN_QUANT_INT8_SYM, strided_slice_params)       \
    MACRO(sum, CSINN_QUANT_FLOAT32, reduce_params)                         \
    MACRO(sum, CSINN_QUANT_UINT8_ASYM, reduce_params)                      \
    MACRO(sum, CSINN_QUANT_INT8_SYM, reduce_params)                        \
    MACRO(threshold_relu, CSINN_QUANT_FLOAT32, relu_params)                \
    MACRO(threshold_relu, CSINN_QUANT_UINT8_ASYM, relu_params)             \
    MACRO(threshold_relu, CSINN_QUANT_INT8_SYM, relu_params)               \
    MACRO(tile, CSINN_QUANT_FLOAT32, tile_params)                          \
    MACRO(tile, CSINN_QUANT_UINT8_ASYM, tile_params)                       \
    MACRO(tile, CSINN_QUANT_INT8_SYM, tile_params)                         \
    MACRO(transpose, CSINN_QUANT_FLOAT32, transpose_params)                \
    MACRO(transpose, CSINN_QUANT_UINT8_ASYM, transpose_params)             \
    MACRO(transpose, CSINN_QUANT_INT8_SYM, transpose_params)               \
    MACRO(argmax, CSINN_QUANT_FLOAT32, reduce_params)                      \
    MACRO(argmax, CSINN_QUANT_UINT8_ASYM, reduce_params)                   \
    MACRO(argmax, CSINN_QUANT_INT8_SYM, reduce_params)                     \
    MACRO(argmin, CSINN_QUANT_FLOAT32, reduce_params)                      \
    MACRO(argmin, CSINN_QUANT_UINT8_ASYM, reduce_params)                   \
    MACRO(argmin, CSINN_QUANT_INT8_SYM, reduce_params)                     \
    MACRO(broadcast_to, CSINN_QUANT_FLOAT32, broadcast_to_params)          \
    MACRO(broadcast_to, CSINN_QUANT_UINT8_ASYM, broadcast_to_params)       \
    MACRO(broadcast_to, CSINN_QUANT_INT8_SYM, broadcast_to_params)

#define LAYER_QUANT_TEST_DISO(MACRO)                                               \
    MACRO(add, CSINN_QUANT_FLOAT32, diso_params)                                   \
    MACRO(add, CSINN_QUANT_UINT8_ASYM, diso_params)                                \
    MACRO(add, CSINN_QUANT_INT8_SYM, diso_params)                                  \
    MACRO(div, CSINN_QUANT_FLOAT32, diso_params)                                   \
    MACRO(div, CSINN_QUANT_UINT8_ASYM, diso_params)                                \
    MACRO(div, CSINN_QUANT_INT8_SYM, diso_params)                                  \
    MACRO(equal, CSINN_QUANT_FLOAT32, diso_params)                                 \
    MACRO(equal, CSINN_QUANT_UINT8_ASYM, diso_params)                              \
    MACRO(equal, CSINN_QUANT_INT8_SYM, diso_params)                                \
    MACRO(floor_divide, CSINN_QUANT_FLOAT32, diso_params)                          \
    MACRO(floor_divide, CSINN_QUANT_UINT8_ASYM, diso_params)                       \
    MACRO(floor_divide, CSINN_QUANT_INT8_SYM, diso_params)                         \
    MACRO(floor_mod, CSINN_QUANT_FLOAT32, diso_params)                             \
    MACRO(floor_mod, CSINN_QUANT_UINT8_ASYM, diso_params)                          \
    MACRO(floor_mod, CSINN_QUANT_INT8_SYM, diso_params)                            \
    MACRO(greater_equal, CSINN_QUANT_FLOAT32, diso_params)                         \
    MACRO(greater_equal, CSINN_QUANT_UINT8_ASYM, diso_params)                      \
    MACRO(greater_equal, CSINN_QUANT_INT8_SYM, diso_params)                        \
    MACRO(greater, CSINN_QUANT_FLOAT32, diso_params)                               \
    MACRO(greater, CSINN_QUANT_UINT8_ASYM, diso_params)                            \
    MACRO(greater, CSINN_QUANT_INT8_SYM, diso_params)                              \
    MACRO(less_equal, CSINN_QUANT_FLOAT32, diso_params)                            \
    MACRO(less_equal, CSINN_QUANT_UINT8_ASYM, diso_params)                         \
    MACRO(less_equal, CSINN_QUANT_INT8_SYM, diso_params)                           \
    MACRO(less, CSINN_QUANT_FLOAT32, diso_params)                                  \
    MACRO(less, CSINN_QUANT_UINT8_ASYM, diso_params)                               \
    MACRO(less, CSINN_QUANT_INT8_SYM, diso_params)                                 \
    MACRO(logical_and, CSINN_QUANT_FLOAT32, diso_params)                           \
    MACRO(logical_and, CSINN_QUANT_UINT8_ASYM, diso_params)                        \
    MACRO(logical_and, CSINN_QUANT_INT8_SYM, diso_params)                          \
    MACRO(logical_or, CSINN_QUANT_FLOAT32, diso_params)                            \
    MACRO(logical_or, CSINN_QUANT_UINT8_ASYM, diso_params)                         \
    MACRO(logical_or, CSINN_QUANT_INT8_SYM, diso_params)                           \
    MACRO(logical_xor, CSINN_QUANT_FLOAT32, diso_params)                           \
    MACRO(logical_xor, CSINN_QUANT_UINT8_ASYM, diso_params)                        \
    MACRO(logical_xor, CSINN_QUANT_INT8_SYM, diso_params)                          \
    MACRO(mod, CSINN_QUANT_FLOAT32, diso_params)                                   \
    MACRO(mod, CSINN_QUANT_UINT8_ASYM, diso_params)                                \
    MACRO(mod, CSINN_QUANT_INT8_SYM, diso_params)                                  \
    MACRO(mul, CSINN_QUANT_FLOAT32, diso_params)                                   \
    MACRO(mul, CSINN_QUANT_UINT8_ASYM, diso_params)                                \
    MACRO(mul, CSINN_QUANT_INT8_SYM, diso_params)                                  \
    MACRO(not_equal, CSINN_QUANT_FLOAT32, diso_params)                             \
    MACRO(not_equal, CSINN_QUANT_UINT8_ASYM, diso_params)                          \
    MACRO(not_equal, CSINN_QUANT_INT8_SYM, diso_params)                            \
    MACRO(power, CSINN_QUANT_FLOAT32, diso_params)                                 \
    MACRO(power, CSINN_QUANT_UINT8_ASYM, diso_params)                              \
    MACRO(power, CSINN_QUANT_INT8_SYM, diso_params)                                \
    MACRO(sub, CSINN_QUANT_FLOAT32, diso_params)                                   \
    MACRO(sub, CSINN_QUANT_UINT8_ASYM, diso_params)                                \
    MACRO(sub, CSINN_QUANT_INT8_SYM, diso_params)                                  \
    MACRO(maximum, CSINN_QUANT_FLOAT32, diso_params)                               \
    MACRO(maximum, CSINN_QUANT_UINT8_ASYM, diso_params)                            \
    MACRO(maximum, CSINN_QUANT_INT8_SYM, diso_params)                              \
    MACRO(minimum, CSINN_QUANT_FLOAT32, diso_params)                               \
    MACRO(minimum, CSINN_QUANT_UINT8_ASYM, diso_params)                            \
    MACRO(minimum, CSINN_QUANT_INT8_SYM, diso_params)                              \
    MACRO(and, CSINN_QUANT_FLOAT32, diso_params)                                   \
    MACRO(and, CSINN_QUANT_UINT8_ASYM, diso_params)                                \
    MACRO(and, CSINN_QUANT_INT8_SYM, diso_params)                                  \
    MACRO(matmul, CSINN_QUANT_FLOAT32, matmul_params)                              \
    MACRO(matmul, CSINN_QUANT_UINT8_ASYM, matmul_params)                           \
    MACRO(matmul, CSINN_QUANT_INT8_SYM, matmul_params)                             \
    MACRO(prelu, CSINN_QUANT_FLOAT32, prelu_params)                                \
    MACRO(prelu, CSINN_QUANT_UINT8_ASYM, prelu_params)                             \
    MACRO(prelu, CSINN_QUANT_INT8_SYM, prelu_params)                               \
    MACRO(non_max_suppression, CSINN_QUANT_FLOAT32, non_max_suppression_params)    \
    MACRO(non_max_suppression, CSINN_QUANT_UINT8_ASYM, non_max_suppression_params) \
    MACRO(non_max_suppression, CSINN_QUANT_INT8_SYM, non_max_suppression_params)   \
    MACRO(psroipooling, CSINN_QUANT_FLOAT32, psroipooling_params)                  \
    MACRO(psroipooling, CSINN_QUANT_UINT8_ASYM, psroipooling_params)               \
    MACRO(psroipooling, CSINN_QUANT_INT8_SYM, psroipooling_params)                 \
    MACRO(roi_align, CSINN_QUANT_FLOAT32, roi_align_params)                        \
    MACRO(roi_align, CSINN_QUANT_UINT8_ASYM, roi_align_params)                     \
    MACRO(roi_align, CSINN_QUANT_INT8_SYM, roi_align_params)                       \
    MACRO(roipool, CSINN_QUANT_FLOAT32, roi_pool_params)                           \
    MACRO(roipool, CSINN_QUANT_UINT8_ASYM, roi_pool_params)                        \
    MACRO(roipool, CSINN_QUANT_INT8_SYM, roi_pool_params)                          \
    MACRO(gather_nd, CSINN_QUANT_FLOAT32, gather_nd_params)                        \
    MACRO(gather_nd, CSINN_QUANT_UINT8_ASYM, gather_nd_params)                     \
    MACRO(gather_nd, CSINN_QUANT_INT8_SYM, gather_nd_params)                       \
    MACRO(gather, CSINN_QUANT_FLOAT32, gather_params)                              \
    MACRO(gather, CSINN_QUANT_UINT8_ASYM, gather_params)                           \
    MACRO(gather, CSINN_QUANT_INT8_SYM, gather_params)

#define LAYER_QUANT_TEST_SEGMENT(MACRO)                         \
    MACRO(segment_max, CSINN_QUANT_FLOAT32, segment_params)     \
    MACRO(segment_max, CSINN_QUANT_UINT8_ASYM, segment_params)  \
    MACRO(segment_max, CSINN_QUANT_INT8_SYM, segment_params)    \
    MACRO(segment_mean, CSINN_QUANT_FLOAT32, segment_params)    \
    MACRO(segment_mean, CSINN_QUANT_UINT8_ASYM, segment_params) \
    MACRO(segment_mean, CSINN_QUANT_INT8_SYM, segment_params)   \
    MACRO(segment_min, CSINN_QUANT_FLOAT32, segment_params)     \
    MACRO(segment_min, CSINN_QUANT_UINT8_ASYM, segment_params)  \
    MACRO(segment_min, CSINN_QUANT_INT8_SYM, segment_params)    \
    MACRO(segment_prod, CSINN_QUANT_FLOAT32, segment_params)    \
    MACRO(segment_prod, CSINN_QUANT_UINT8_ASYM, segment_params) \
    MACRO(segment_prod, CSINN_QUANT_INT8_SYM, segment_params)   \
    MACRO(segment_sum, CSINN_QUANT_FLOAT32, segment_params)     \
    MACRO(segment_sum, CSINN_QUANT_UINT8_ASYM, segment_params)  \
    MACRO(segment_sum, CSINN_QUANT_INT8_SYM, segment_params)

#define LAYER_QUANT_TEST_BATCHNORM(MACRO)                         \
    MACRO(batch_normalization, CSINN_QUANT_FLOAT32, bn_params)    \
    MACRO(batch_normalization, CSINN_QUANT_UINT8_ASYM, bn_params) \
    MACRO(batch_normalization, CSINN_QUANT_INT8_SYM, bn_params)

#define LAYER_QUANT_TEST_CONCAT(MACRO)                   \
    MACRO(concat, CSINN_QUANT_FLOAT32, concat_params)    \
    MACRO(concat, CSINN_QUANT_UINT8_ASYM, concat_params) \
    MACRO(concat, CSINN_QUANT_INT8_SYM, concat_params)   \
    MACRO(stack, CSINN_QUANT_FLOAT32, stack_params)      \
    MACRO(stack, CSINN_QUANT_UINT8_ASYM, stack_params)   \
    MACRO(stack, CSINN_QUANT_INT8_SYM, stack_params)

#define LAYER_QUANT_TEST_CONV2D(MACRO)                         \
    MACRO(conv2d, CSINN_QUANT_FLOAT32, conv2d_params)          \
    MACRO(conv2d, CSINN_QUANT_UINT8_ASYM, conv2d_params)       \
    MACRO(conv2d, CSINN_QUANT_INT8_SYM, conv2d_params)         \
    MACRO(conv3d, CSINN_QUANT_FLOAT32, conv3d_params)          \
    MACRO(conv3d, CSINN_QUANT_UINT8_ASYM, conv3d_params)       \
    MACRO(conv3d, CSINN_QUANT_INT8_SYM, conv3d_params)         \
    MACRO(conv2d_relu, CSINN_QUANT_FLOAT32, conv2d_params)     \
    MACRO(conv2d_relu, CSINN_QUANT_UINT8_ASYM, conv2d_params)  \
    MACRO(conv2d_relu, CSINN_QUANT_INT8_SYM, conv2d_params)    \
    MACRO(conv2d_relu6, CSINN_QUANT_FLOAT32, conv2d_params)    \
    MACRO(conv2d_relu6, CSINN_QUANT_UINT8_ASYM, conv2d_params) \
    MACRO(conv2d_relu6, CSINN_QUANT_INT8_SYM, conv2d_params)   \
    MACRO(deconv2d, CSINN_QUANT_FLOAT32, conv2d_params)        \
    MACRO(deconv2d, CSINN_QUANT_UINT8_ASYM, conv2d_params)     \
    MACRO(deconv2d, CSINN_QUANT_INT8_SYM, conv2d_params)       \
    MACRO(deconv3d, CSINN_QUANT_FLOAT32, conv3d_params)        \
    MACRO(deconv3d, CSINN_QUANT_UINT8_ASYM, conv3d_params)     \
    MACRO(deconv3d, CSINN_QUANT_INT8_SYM, conv3d_params)       \
    MACRO(fullyconnected, CSINN_QUANT_FLOAT32, fc_params)      \
    MACRO(fullyconnected, CSINN_QUANT_UINT8_ASYM, fc_params)   \
    MACRO(fullyconnected, CSINN_QUANT_INT8_SYM, fc_params)

#define LAYER_QUANT_TEST_TISO(MACRO)                     \
    MACRO(select, CSINN_QUANT_FLOAT32, select_params)    \
    MACRO(select, CSINN_QUANT_UINT8_ASYM, select_params) \
    MACRO(select, CSINN_QUANT_INT8_SYM, select_params)

#define LAYER_QUANT_TEST_SPLIT(MACRO)                  \
    MACRO(split, CSINN_QUANT_FLOAT32, split_params)    \
    MACRO(split, CSINN_QUANT_UINT8_ASYM, split_params) \
    MACRO(split, CSINN_QUANT_INT8_SYM, split_params)

#define LAYER_QUANT_TEST_UNSTACK(MACRO)                    \
    MACRO(unstack, CSINN_QUANT_FLOAT32, unstack_params)    \
    MACRO(unstack, CSINN_QUANT_UINT8_ASYM, unstack_params) \
    MACRO(unstack, CSINN_QUANT_INT8_SYM, unstack_params)

#define LAYER_QUANT_TEST_ARANGE(MACRO)                   \
    MACRO(arange, CSINN_QUANT_FLOAT32, arange_params)    \
    MACRO(arange, CSINN_QUANT_UINT8_ASYM, arange_params) \
    MACRO(arange, CSINN_QUANT_INT8_SYM, arange_params)
