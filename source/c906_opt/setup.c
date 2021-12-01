/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

/* CSI-NN2 version 1.8.x */

#include "csi_c906.h"

void *csi_init_map_c906(int op, int dtype)
{
    if (op == CSINN_OP_CONV2D) {
        return csi_c906_conv2d_init;
    } else if (op == CSINN_OP_MAXPOOL2D) {
        return csi_c906_maxpool_init;
    } else if(op == CSINN_OP_AVGPOOL2D) {
        return csi_c906_avgpool_init;
    }

    return NULL;
}

static void *setup_bc_map()
{
    static void* bc_map[CSINN_OP_AND_UTILS_SIZE][2];

    bc_map[CSINN_OP_ABS][0] = csi_ref_abs_quant;
    bc_map[CSINN_OP_ACOS][0] = csi_ref_acos_quant;
    bc_map[CSINN_OP_ACOSH][0] = csi_ref_acosh_quant;
    bc_map[CSINN_OP_ADD][0] = csi_ref_add_quant;
    bc_map[CSINN_OP_AND][0] = csi_ref_and_i8;
    bc_map[CSINN_OP_ARANGE][0] = csi_ref_arange_quant;
    bc_map[CSINN_OP_ARGMAX][0] = csi_ref_argmax_stride_quant;
    bc_map[CSINN_OP_ARGMIN][0] = csi_ref_argmin_stride_quant;
    bc_map[CSINN_OP_ASIN][0] = csi_ref_asin_quant;
    bc_map[CSINN_OP_ASINH][0] = csi_ref_asinh_quant;
    bc_map[CSINN_OP_ATAN][0] = csi_ref_atan_quant;
    bc_map[CSINN_OP_ATANH][0] = csi_ref_atanh_quant;
    bc_map[CSINN_OP_AVGPOOL2D][0] = csi_ref_averagepool_quant;
    bc_map[CSINN_OP_AVGPOOL3D][0] = csi_ref_averagepool3d_quant;
    bc_map[CSINN_OP_BN][0] = csi_ref_batch_normalization_quant;
    bc_map[CSINN_OP_BATCH_TO_SPACE][0] = csi_ref_batch_to_space_quant;
    bc_map[CSINN_OP_BROADCOST][0] = csi_ref_broadcast_to_quant;
    bc_map[CSINN_OP_CEIL][0] = csi_ref_ceil_quant;
    bc_map[CSINN_OP_CLIP][0] = csi_ref_clip_quant;
    bc_map[CSINN_OP_CONCAT][0] = csi_ref_concat_quant;
    bc_map[CSINN_OP_CONV2D][0] = csi_ref_conv2d_quant;
    bc_map[CSINN_OP_CONV2D_RELU][0] = csi_ref_conv2d_relu_quant;
    bc_map[CSINN_OP_CONV2D_RELU6][0] = csi_ref_conv2d_relu6_quant;
    bc_map[CSINN_OP_DEPTHWISE_CONV2D][0] = csi_ref_depthwise_conv2d_quant;
    bc_map[CSINN_OP_DEPTHWISE_CONV2D_RELU][0] = csi_ref_depthwise_conv2d_relu_quant;
    bc_map[CSINN_OP_DEPTHWISE_CONV2D_RELU6][0] = csi_ref_depthwise_conv2d_relu6_quant;
    bc_map[CSINN_OP_GROUP_CONV2D][0] = csi_ref_group_conv2d_quant;
    bc_map[CSINN_OP_CONV3D][0] = csi_ref_conv3d_quant;
    bc_map[CSINN_OP_DECONV2D][0] = csi_ref_deconv2d_quant;
    bc_map[CSINN_OP_DEPTHWISE_DECONV2D][0] = csi_ref_depthwise_deconv2d_quant;
    bc_map[CSINN_OP_DECONV3D][0] = csi_ref_deconv3d_quant;
    bc_map[CSINN_OP_COS][0] = csi_ref_cos_quant;
    bc_map[CSINN_OP_COSH][0] = csi_ref_cosh_quant;
    bc_map[CSINN_OP_CUMPROD][0] = csi_ref_cumprod_quant;
    bc_map[CSINN_OP_CUMSUM][0] = csi_ref_cumsum_quant;
    bc_map[CSINN_OP_DEPTH_TO_SPACE][0] = csi_ref_depth_to_space_quant;
    bc_map[CSINN_OP_DIV][0] = csi_ref_div_quant;
    bc_map[CSINN_OP_ELU][0] = csi_ref_elu_quant;
    bc_map[CSINN_OP_EQUANL][0] = csi_ref_equal_quant;
    bc_map[CSINN_OP_ERF][0] = csi_ref_erf_quant;
    bc_map[CSINN_OP_EXP][0] = csi_ref_exp_quant;
    bc_map[CSINN_OP_EXPAND_DIMS][0] = csi_ref_expand_dims_quant;
    bc_map[CSINN_OP_EXPM1][0] = csi_ref_expm1_quant;
    bc_map[CSINN_OP_FLATTEN][0] = csi_ref_flatten;
    bc_map[CSINN_OP_FLOOR_DIVIDE][0] = csi_ref_floor_divide_quant;
    bc_map[CSINN_OP_FLOOR_MOD][0] = csi_ref_floor_mod_quant;
    bc_map[CSINN_OP_FLOOR][0] = csi_ref_floor_quant;
    bc_map[CSINN_OP_FSMN][0] = csi_ref_fsmn_quant;
    bc_map[CSINN_OP_FULLYCONNECTED][0] = csi_ref_fullyconnected_quant;
    bc_map[CSINN_OP_GATHER_ND][0] = csi_ref_gather_nd_quant;
    bc_map[CSINN_OP_GATHER][0] = csi_ref_gather_quant;
    bc_map[CSINN_OP_GLOBAL_AVGPOOL2D][0] = csi_ref_global_averagepool_quant;
    bc_map[CSINN_OP_GLOBAL_MAXPOOL2D][0] = csi_ref_global_maxpool_quant;
    bc_map[CSINN_OP_GREATHER_EQUAL][0] = csi_ref_greater_equal_quant;
    bc_map[CSINN_OP_GREATHER][0] = csi_ref_greater_quant;
    bc_map[CSINN_OP_HARD_SIGMOID][0] = csi_ref_hard_sigmoid_quant;
    bc_map[CSINN_OP_IM2COL][0] = csi_ref_im2col_quant;
    bc_map[CSINN_OP_L2N][0] = csi_ref_l2_normalization_quant;
    bc_map[CSINN_OP_LEAKY_RELU][0] = csi_ref_leaky_relu_quant;
    bc_map[CSINN_OP_LESS_EQUAL][0] = csi_ref_less_equal_quant;
    bc_map[CSINN_OP_LESS][0] = csi_ref_less_quant;
    bc_map[CSINN_OP_LOG_SOFTMAX][0] = csi_ref_log_softmax_quant;
    bc_map[CSINN_OP_LOG][0] = csi_ref_log_quant;
    bc_map[CSINN_OP_LOG1P][0] = csi_ref_log1p_quant;
    bc_map[CSINN_OP_LOGICAL_AND][0] = csi_ref_logical_and_quant;
    bc_map[CSINN_OP_LOGICAL_NOT][0] = csi_ref_logical_not_quant;
    bc_map[CSINN_OP_LOGICAL_OR][0] = csi_ref_logical_or_quant;
    bc_map[CSINN_OP_LOGICAL_XOR][0] = csi_ref_logical_xor_quant;
    bc_map[CSINN_OP_LRN][0] = csi_ref_lrn_quant;
    bc_map[CSINN_OP_MATMUL][0] = csi_ref_matmul_quant;
    bc_map[CSINN_OP_MAX][0] = csi_ref_max_stride_quant;
    bc_map[CSINN_OP_MAXINUM][0] = csi_ref_maximum_quant;
    bc_map[CSINN_OP_MAXPOOL2D][0] = csi_ref_maxpool_quant;
    bc_map[CSINN_OP_MAXPOOL2D_LOCAT][0] = csi_ref_maxpool2d_locat_quant;
    bc_map[CSINN_OP_MAXPOOL3D][0] = csi_ref_maxpool3d_quant;
    bc_map[CSINN_OP_MEAN][0] = csi_ref_mean_stride_quant;
    bc_map[CSINN_OP_MEAN_STRIDE][0] = csi_ref_mean_stride_quant;
    bc_map[CSINN_OP_MIN][0] = csi_ref_min_stride_quant;
    bc_map[CSINN_OP_MINIMUM][0] = csi_ref_minimum_quant;
    bc_map[CSINN_OP_MOD][0] = csi_ref_mod_quant;
    bc_map[CSINN_OP_MUL][0] = csi_ref_mul_quant;
    bc_map[CSINN_OP_NDARRAY_SIZE][0] = csi_ref_ndarray_size_i8;
    bc_map[CSINN_OP_NEGATIIVE][0] = csi_ref_negative_quant;
    bc_map[CSINN_OP_NOT_EQUAL][0] = csi_ref_not_equal_quant;
    bc_map[CSINN_OP_NOT][0] = csi_ref_not_i8;
    bc_map[CSINN_OP_OR][0] = csi_ref_or_i8;
    bc_map[CSINN_OP_PAD][0] = csi_ref_pad_quant;
    bc_map[CSINN_OP_POWER][0] = csi_ref_power_quant;
    bc_map[CSINN_OP_PRELU][0] = csi_ref_prelu_quant;
    bc_map[CSINN_OP_PROD][0] = csi_ref_prod_stride_quant;
    bc_map[CSINN_OP_PROPOSAL][0] = csi_ref_proposal_quant;
    bc_map[CSINN_OP_PSROIPOOLING][0] = csi_ref_psroipooling_quant;
    bc_map[CSINN_OP_REDUCE_LOGSUMEXP][0] = csi_ref_reduce_logsumexp_quant;
    bc_map[CSINN_OP_REDUCE_MAX][0] = csi_ref_reduce_max_quant;
    bc_map[CSINN_OP_REDUCE_MEAN][0] = csi_ref_reduce_mean_quant;
    bc_map[CSINN_OP_REDUCE_MIN][0] = csi_ref_reduce_min_quant;
    bc_map[CSINN_OP_REDUCE_PROD][0] = csi_ref_reduce_prod_quant;
    bc_map[CSINN_OP_REDUCE_SUM][0] = csi_ref_reduce_sum_quant;
    bc_map[CSINN_OP_RELU][0] = csi_ref_relu_quant;
    bc_map[CSINN_OP_RELU1][0] = csi_ref_relu1_quant;
    bc_map[CSINN_OP_RELU6][0] = csi_ref_relu6_quant;
    bc_map[CSINN_OP_RELUN][0] = csi_ref_relun_quant;
    bc_map[CSINN_OP_RESHAPE][0] = csi_ref_reshape;
    bc_map[CSINN_OP_RESIZE][0] = csi_ref_resize_quant;
    bc_map[CSINN_OP_REVERSE][0] = csi_ref_reverse_quant;
    bc_map[CSINN_OP_ROIPOOL][0] = csi_ref_roipool_quant;
    bc_map[CSINN_OP_ROUND][0] = csi_ref_round_quant;
    bc_map[CSINN_OP_RSQRT][0] = csi_ref_rsqrt_quant;
    bc_map[CSINN_OP_SCATTER_ND][0] = csi_ref_scatter_nd_quant;
    bc_map[CSINN_OP_SEGMENT_MAX][0] = csi_ref_segment_max_quant;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MAX][0] = csi_ref_unsorted_segment_max_quant;
    bc_map[CSINN_OP_SEGMENT_MEAN][0] = csi_ref_segment_mean_quant;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MEAN][0] = csi_ref_unsorted_segment_mean_quant;
    bc_map[CSINN_OP_SEGMENT_MIN][0] = csi_ref_segment_min_quant;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MIN][0] = csi_ref_unsorted_segment_min_quant;
    bc_map[CSINN_OP_SEGMENT_PROD][0] = csi_ref_segment_prod_quant;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_PROD][0] = csi_ref_unsorted_segment_prod_quant;
    bc_map[CSINN_OP_SEGMENT_SUM][0] = csi_ref_segment_sum_quant;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_SUM][0] = csi_ref_unsorted_segment_sum_quant;
    bc_map[CSINN_OP_SELECT][0] = csi_ref_select_i8;
    bc_map[CSINN_OP_SHAPE][0] = csi_ref_shape_i8;
    bc_map[CSINN_OP_SHUFFLE_CHANNEL][0] = csi_ref_shuffle_channel_quant;
    bc_map[CSINN_OP_SIGMOID][0] = csi_ref_sigmoid_quant;
    bc_map[CSINN_OP_SIGN][0] = csi_ref_sign_quant;
    bc_map[CSINN_OP_SIN][0] = csi_ref_sin_quant;
    bc_map[CSINN_OP_SINH][0] = csi_ref_sinh_quant;
    bc_map[CSINN_OP_SLICE][0] = csi_ref_slice_quant;
    bc_map[CSINN_OP_SOFTMAX][0] = csi_ref_softmax_quant;
    bc_map[CSINN_OP_SOFTPLUS][0] = csi_ref_softplus_quant;
    bc_map[CSINN_OP_SOFTRELU][0] = csi_ref_softrelu_quant;
    bc_map[CSINN_OP_SOFTSIGN][0] = csi_ref_softsign_quant;
    bc_map[CSINN_OP_SPACE_TO_BATCH][0] = csi_ref_space_to_batch_quant;
    bc_map[CSINN_OP_SPACE_TO_DEPTH][0] = csi_ref_space_to_depth_quant;
    bc_map[CSINN_OP_SPLIT][0] = csi_ref_split_quant;
    bc_map[CSINN_OP_SQRT][0] = csi_ref_sqrt_quant;
    bc_map[CSINN_OP_SQUEEZE][0] = csi_ref_squeeze;
    bc_map[CSINN_OP_STACK][0] = csi_ref_stack_quant;
    bc_map[CSINN_OP_STRIDED_SLICE][0] = csi_ref_strided_slice_quant;
    bc_map[CSINN_OP_SUB][0] = csi_ref_sub_quant;
    bc_map[CSINN_OP_SUM][0] = csi_ref_sum_stride_quant;
    bc_map[CSINN_OP_TAN][0] = csi_ref_tan_quant;
    bc_map[CSINN_OP_TANH][0] = csi_ref_tanh_quant;
    bc_map[CSINN_OP_THRESHOLD_RELU][0] = csi_ref_threshold_relu_quant;
    bc_map[CSINN_OP_TILE][0] = csi_ref_tile_quant;
    bc_map[CSINN_OP_TOPK][0] = csi_ref_topk_quant;
    bc_map[CSINN_OP_TRUNC][0] = csi_ref_trunc_quant;
    bc_map[CSINN_OP_TRANSPOSE][0] = csi_ref_transpose;
    bc_map[CSINN_OP_TRUNC][0] = csi_ref_trunc_quant;
    bc_map[CSINN_OP_UNPOOLING][0] = csi_ref_unpooling_quant;
    bc_map[CSINN_OP_UNSTACK][0] = csi_ref_unstack_qunat;
    bc_map[CSINN_OP_XOR][0] = csi_ref_xor_i8;
    bc_map[CSINN_OP_YUV_RGB_SCALE][0] = csi_ref_yuv_rgb_scale_quant;

    bc_map[CSINN_OP_ABS][1] = csi_c906_abs_f32;
    bc_map[CSINN_OP_ACOS][1] = csi_ref_acos_f32;
    bc_map[CSINN_OP_ACOSH][1] = csi_ref_acosh_f32;
    bc_map[CSINN_OP_ADD][1] = csi_c906_add_f32;
    bc_map[CSINN_OP_ARANGE][1] = csi_ref_arange_f32;
    bc_map[CSINN_OP_ARGMAX][1] = csi_ref_argmax_stride_i32_f32;
    bc_map[CSINN_OP_ARGMIN][1] = csi_ref_argmin_stride_i32_f32;
    bc_map[CSINN_OP_ASIN][1] = csi_ref_asin_f32;
    bc_map[CSINN_OP_ASINH][1] = csi_ref_asinh_f32;
    bc_map[CSINN_OP_ATAN][1] = csi_ref_atan_f32;
    bc_map[CSINN_OP_ATANH][1] = csi_ref_atanh_f32;
    bc_map[CSINN_OP_AVGPOOL2D][1] = csi_ref_averagepool_f32;
    bc_map[CSINN_OP_AVGPOOL3D][1] = csi_ref_averagepool3d_f32;
    bc_map[CSINN_OP_BN][1] = csi_ref_batch_normalization_f32;
    bc_map[CSINN_OP_BATCH_TO_SPACE][1] = csi_ref_batch_to_space_f32;
    bc_map[CSINN_OP_BROADCOST][1] = csi_ref_broadcast_to_f32;
    bc_map[CSINN_OP_CEIL][1] = csi_ref_ceil_f32;
    bc_map[CSINN_OP_CLIP][1] = csi_c906_clip_f32;
    bc_map[CSINN_OP_COL2IM][1] = csi_ref_col2im_f32;
    bc_map[CSINN_OP_CONCAT][1] = csi_ref_concat_f32;
    bc_map[CSINN_OP_CONV2D][1] = csi_ref_conv2d_f32;
    bc_map[CSINN_OP_DEPTHWISE_CONV2D][1] = csi_ref_depthwise_conv2d_f32;
    bc_map[CSINN_OP_GROUP_CONV2D][1] = csi_ref_group_conv2d_f32;
    bc_map[CSINN_OP_CONV3D][1] = csi_ref_conv3d_f32;
    bc_map[CSINN_OP_DECONV2D][1] = csi_ref_deconv2d_f32;
    bc_map[CSINN_OP_DEPTHWISE_DECONV2D][1] = csi_ref_depthwise_deconv2d_f32;
    bc_map[CSINN_OP_DECONV3D][1] = csi_ref_deconv3d_f32;
    bc_map[CSINN_OP_COS][1] = csi_ref_cos_f32;
    bc_map[CSINN_OP_COSH][1] = csi_ref_cosh_f32;
    bc_map[CSINN_OP_CUMPROD][1] = csi_ref_cumprod_f32;
    bc_map[CSINN_OP_CUMSUM][1] = csi_ref_cumsum_f32;
    bc_map[CSINN_OP_DEPTH_TO_SPACE][1] = csi_ref_depth_to_space_f32;
    bc_map[CSINN_OP_DIV][1] = csi_ref_div_f32;
    bc_map[CSINN_OP_ELU][1] = csi_ref_elu_f32;
    bc_map[CSINN_OP_EQUANL][1] = csi_ref_equal_f32;
    bc_map[CSINN_OP_ERF][1] = csi_ref_erf_f32;
    bc_map[CSINN_OP_EXP][1] = csi_ref_exp_f32;
    bc_map[CSINN_OP_EXPAND_DIMS][1] = csi_ref_expand_dims_f32;
    bc_map[CSINN_OP_EXPM1][1] = csi_ref_expm1_f32;
    bc_map[CSINN_OP_FLATTEN][1] = csi_ref_flatten;
    bc_map[CSINN_OP_FLOOR_DIVIDE][1] = csi_ref_floor_divide_f32;
    bc_map[CSINN_OP_FLOOR_MOD][1] = csi_ref_floor_mod_f32;
    bc_map[CSINN_OP_FLOOR][1] = csi_ref_floor_f32;
    bc_map[CSINN_OP_FSMN][1] = csi_ref_fsmn_f32;
    bc_map[CSINN_OP_FULLYCONNECTED][1] = csi_c906_fullyconnected_f32;
    bc_map[CSINN_OP_GATHER_ND][1] = csi_ref_gather_nd_f32;
    bc_map[CSINN_OP_GATHER][1] = csi_ref_gather_f32;
    bc_map[CSINN_OP_GLOBAL_AVGPOOL2D][1] = csi_c906_global_avgpool_f32;
    bc_map[CSINN_OP_GLOBAL_MAXPOOL2D][1] = csi_c906_global_maxpool_f32;
    bc_map[CSINN_OP_GREATHER_EQUAL][1] = csi_ref_greater_equal_f32;
    bc_map[CSINN_OP_GREATHER][1] = csi_ref_greater_f32;
    bc_map[CSINN_OP_HARD_SIGMOID][1] = csi_ref_hard_sigmoid_f32;
    bc_map[CSINN_OP_IM2COL][1] = csi_ref_im2col_f32;
    bc_map[CSINN_OP_ISNAN][1] = csi_ref_isnan_bool_f32;
    bc_map[CSINN_OP_L2N][1] = csi_ref_l2_normalization_f32;
    bc_map[CSINN_OP_L2POOL2D][1] = csi_ref_l2pool_f32;
    bc_map[CSINN_OP_LEAKY_RELU][1] = csi_c906_leaky_relu_f32;
    bc_map[CSINN_OP_LESS_EQUAL][1] = csi_ref_less_equal_f32;
    bc_map[CSINN_OP_LESS][1] = csi_ref_less_f32;
    bc_map[CSINN_OP_LOG_SOFTMAX][1] = csi_ref_log_softmax_f32;
    bc_map[CSINN_OP_LOG][1] = csi_ref_log_f32;
    bc_map[CSINN_OP_LOG1P][1] = csi_ref_log1p_f32;
    bc_map[CSINN_OP_LOGICAL_AND][1] = csi_ref_logical_and_f32;
    bc_map[CSINN_OP_LOGICAL_NOT][1] = csi_ref_logical_not_f32;
    bc_map[CSINN_OP_LOGICAL_OR][1] = csi_ref_logical_or_f32;
    bc_map[CSINN_OP_LOGICAL_XOR][1] = csi_ref_logical_xor_f32;
    bc_map[CSINN_OP_LRN][1] = csi_ref_lrn_f32;
    bc_map[CSINN_OP_MATMUL][1] = csi_ref_matmul_f32;
    bc_map[CSINN_OP_MAX][1] = csi_ref_max_stride_f32;
    bc_map[CSINN_OP_MAXINUM][1] = csi_ref_maximum_f32;
    bc_map[CSINN_OP_MAXPOOL2D][1] = csi_ref_maxpool_f32;
    bc_map[CSINN_OP_MAXPOOL2D_LOCAT][1] = csi_ref_maxpool2d_locat_f32;
    bc_map[CSINN_OP_MAXPOOL3D][1] = csi_ref_maxpool3d_f32;
    bc_map[CSINN_OP_MEAN][1] = csi_ref_mean_stride_f32;
    bc_map[CSINN_OP_MEAN_STRIDE][1] = csi_ref_mean_stride_f32;
    bc_map[CSINN_OP_MINIMUM][1] = csi_ref_minimum_f32;
    bc_map[CSINN_OP_MOD][1] = csi_ref_mod_f32;
    bc_map[CSINN_OP_MUL][1] = csi_ref_mul_f32;
    bc_map[CSINN_OP_NDARRAY_SIZE][1] = csi_ref_ndarray_size_f32;
    bc_map[CSINN_OP_NEGATIIVE][1] = csi_ref_negative_f32;
    bc_map[CSINN_OP_NOT_EQUAL][1] = csi_ref_not_equal_f32;
    bc_map[CSINN_OP_PAD][1] = csi_ref_pad_f32;
    bc_map[CSINN_OP_POWER][1] = csi_ref_power_f32;
    bc_map[CSINN_OP_PRELU][1] = csi_c906_prelu_f32;
    bc_map[CSINN_OP_PROD][1] = csi_ref_prod_stride_f32;
    bc_map[CSINN_OP_PROPOSAL][1] = csi_ref_proposal_f32;
    bc_map[CSINN_OP_PSROIPOOLING][1] = csi_ref_psroipooling_f32;
    bc_map[CSINN_OP_REDUCE_LOGSUMEXP][1] = csi_ref_reduce_logsumexp_f32;
    bc_map[CSINN_OP_REDUCE_MAX][1] = csi_ref_reduce_max_f32;
    bc_map[CSINN_OP_REDUCE_MEAN][1] = csi_ref_reduce_mean_f32;
    bc_map[CSINN_OP_REDUCE_MIN][1] = csi_ref_reduce_min_f32;
    bc_map[CSINN_OP_REDUCE_PROD][1] = csi_ref_reduce_prod_f32;
    bc_map[CSINN_OP_REDUCE_SUM][1] = csi_ref_reduce_sum_f32;
    bc_map[CSINN_OP_RELU][1] = csi_c906_relu_f32;
    bc_map[CSINN_OP_RELU1][1] = csi_c906_relu1_f32;
    bc_map[CSINN_OP_RELU6][1] = csi_c906_relu6_f32;
    bc_map[CSINN_OP_RELUN][1] = csi_ref_relun_f32;
    bc_map[CSINN_OP_RESHAPE][1] = csi_ref_reshape;
    bc_map[CSINN_OP_RESIZE][1] = csi_ref_resize_f32;
    bc_map[CSINN_OP_REVERSE][1] = csi_ref_reverse_f32;
    bc_map[CSINN_OP_ROIALIGN][1] = csi_ref_roi_align_f32;
    bc_map[CSINN_OP_ROIPOOL][1] = csi_ref_roipool_f32;
    bc_map[CSINN_OP_ROUND][1] = csi_ref_round_f32;
    bc_map[CSINN_OP_RSQRT][1] = csi_ref_rsqrt_f32;
    bc_map[CSINN_OP_SCATTER_ND][1] = csi_ref_scatter_nd_f32;
    bc_map[CSINN_OP_SEGMENT_MAX][1] = csi_ref_segment_max_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MAX][1] = csi_ref_unsorted_segment_max_f32;
    bc_map[CSINN_OP_SEGMENT_MEAN][1] = csi_ref_segment_mean_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MEAN][1] = csi_ref_unsorted_segment_mean_f32;
    bc_map[CSINN_OP_SEGMENT_MIN][1] = csi_ref_segment_min_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MIN][1] = csi_ref_unsorted_segment_min_f32;
    bc_map[CSINN_OP_SEGMENT_PROD][1] = csi_ref_segment_prod_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_PROD][1] = csi_ref_unsorted_segment_prod_f32;
    bc_map[CSINN_OP_SEGMENT_SUM][1] = csi_ref_segment_sum_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_SUM][1] = csi_ref_unsorted_segment_sum_f32;
    bc_map[CSINN_OP_SELECT][1] = csi_ref_select_f32;
    bc_map[CSINN_OP_SHUFFLE_CHANNEL][1] = csi_ref_shuffle_channel_f32;
    bc_map[CSINN_OP_SIGMOID][1] = csi_ref_sigmoid_f32;
    bc_map[CSINN_OP_SIGN][1] = csi_ref_sign_f32;
    bc_map[CSINN_OP_SIN][1] = csi_ref_sin_f32;
    bc_map[CSINN_OP_SINH][1] = csi_ref_sinh_f32;
    bc_map[CSINN_OP_SLICE][1] = csi_ref_slice_f32;
    bc_map[CSINN_OP_SOFTMAX][1] = csi_ref_softmax_f32;
    bc_map[CSINN_OP_SOFTPLUS][1] = csi_ref_softplus_f32;
    bc_map[CSINN_OP_SOFTRELU][1] = csi_ref_softrelu_f32;
    bc_map[CSINN_OP_SOFTSIGN][1] = csi_ref_softsign_f32;
    bc_map[CSINN_OP_SPACE_TO_BATCH][1] = csi_ref_space_to_batch_f32;
    bc_map[CSINN_OP_SPACE_TO_DEPTH][1] = csi_ref_space_to_depth_f32;
    bc_map[CSINN_OP_SPLIT][1] = csi_ref_split_f32;
    bc_map[CSINN_OP_SQRT][1] = csi_ref_sqrt_f32;
    bc_map[CSINN_OP_SQUARE][1] = csi_ref_square_f32;
    bc_map[CSINN_OP_SQUEEZE][1] = csi_ref_squeeze;
    bc_map[CSINN_OP_STACK][1] = csi_ref_stack_f32;
    bc_map[CSINN_OP_STRIDED_SLICE][1] = csi_ref_strided_slice_f32;
    bc_map[CSINN_OP_SUB][1] = csi_ref_sub_f32;
    bc_map[CSINN_OP_SUM][1] = csi_ref_sum_stride_f32;
    bc_map[CSINN_OP_TAN][1] = csi_ref_tan_f32;
    bc_map[CSINN_OP_TANH][1] = csi_ref_tanh_f32;
    bc_map[CSINN_OP_THRESHOLD_RELU][1] = csi_ref_threshold_relu_f32;
    bc_map[CSINN_OP_TILE][1] = csi_ref_tile_f32;
    bc_map[CSINN_OP_TOPK][1] = csi_ref_topk_f32;
    bc_map[CSINN_OP_TRUNC][1] = csi_ref_trunc_f32;
    bc_map[CSINN_OP_TRANSPOSE][1] = csi_ref_transpose;
    bc_map[CSINN_OP_TRUNC][1] = csi_ref_trunc_f32;
    bc_map[CSINN_OP_UNPOOLING][1] = csi_ref_unpooling_f32;
    bc_map[CSINN_OP_UNSTACK][1] = csi_ref_unstack_f32;
    bc_map[CSINN_OP_YUV_RGB_SCALE][1] = csi_ref_yuv_rgb_scale_f32;

    return bc_map;
}

static int get_bc_map_index(int op, int dtype)
{
    switch (dtype) {
    case CSINN_DTYPE_INT8:
        return op * 2;
        break;
    case CSINN_DTYPE_FLOAT32:
        return op * 2 + 1;
        break;
    default:
        return CSINN_UNSUPPORT_DTYPE;
    }
}

void *csi_bc_map_c906(int op, int dtype) {
    static int has_init;
    static void **bc_map_table;
    if (has_init == 0) {
        bc_map_table = setup_bc_map();
        has_init = 1;
    }
    return bc_map_table[get_bc_map_index(op, dtype)];
}
