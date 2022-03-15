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

/* CSI-NN2 version 1.13.x */

#include "csi_thead_rvv.h"

void *csi_init_map_rvv(int op, int dtype)
{
    if (op == CSINN_OP_CONV2D || op == CSINN_OP_GROUP_CONV2D) {
        return csi_nn_rvv_conv2d_init;
    } else if (op == CSINN_OP_DEPTHWISE_CONV2D) {
        return csi_nn_rvv_depthwise_conv2d_init;
    } else if (op == CSINN_OP_MAXPOOL2D) {
        return csi_nn_rvv_maxpool2d_init;
    } else if (op == CSINN_OP_AVGPOOL2D) {
        return csi_nn_rvv_avgpool2d_init;
    } else if (op == CSINN_OP_FULLYCONNECTED) {
        return csi_nn_rvv_fullyconnected_init;
    } else if (op == CSINN_OP_CONV2D_RELU) {
        if (dtype == CSINN_DTYPE_INT8 || dtype == CSINN_DTYPE_INT4) {
            return csi_nn_rvv_conv2d_init;
        }
    } else if (op == CSINN_OP_DEPTHWISE_CONV2D_RELU) {
        if (dtype == CSINN_DTYPE_INT8 || dtype == CSINN_DTYPE_INT4) {
            return csi_nn_rvv_depthwise_conv2d_init;
        }
    }
    return NULL;
}

static void *setup_bc_map()
{
    static void *bc_map[CSINN_OP_AND_UTILS_SIZE][4];

    bc_map[CSINN_OP_ABS][3] = csi_ref_abs_f32;
    bc_map[CSINN_OP_ACOS][3] = csi_ref_acos_f32;
    bc_map[CSINN_OP_ACOSH][3] = csi_ref_acosh_f32;
    bc_map[CSINN_OP_ADD][3] = csi_nn_rvv_add_fp32;
    bc_map[CSINN_OP_ARANGE][3] = csi_ref_arange_f32;
    bc_map[CSINN_OP_ARGMAX][3] = csi_ref_argmax_stride_i32_f32;
    bc_map[CSINN_OP_ARGMIN][3] = csi_ref_argmin_stride_i32_f32;
    bc_map[CSINN_OP_ASIN][3] = csi_ref_asin_f32;
    bc_map[CSINN_OP_ASINH][3] = csi_ref_asinh_f32;
    bc_map[CSINN_OP_ATAN][3] = csi_ref_atan_f32;
    bc_map[CSINN_OP_ATANH][3] = csi_ref_atanh_f32;
    bc_map[CSINN_OP_AVGPOOL2D][3] = csi_ref_avgpool2d_f32;
    bc_map[CSINN_OP_AVGPOOL3D][3] = csi_ref_avgpool3d_f32;
    bc_map[CSINN_OP_BN][3] = csi_ref_batch_normalization_f32;
    bc_map[CSINN_OP_BATCH_TO_SPACE][3] = csi_ref_batch_to_space_f32;
    bc_map[CSINN_OP_BROADCOST][3] = csi_ref_broadcast_to_f32;
    bc_map[CSINN_OP_CEIL][3] = csi_ref_ceil_f32;
    bc_map[CSINN_OP_CLIP][3] = csi_ref_clip_f32;
    bc_map[CSINN_OP_COL2IM][3] = csi_ref_col2im_f32;
    bc_map[CSINN_OP_CONCAT][3] = csi_nn_rvv_concat_fp32;
    bc_map[CSINN_OP_CONV2D][3] = csi_ref_conv2d_f32;
    bc_map[CSINN_OP_CONV2D_RELU][3] = csi_ref_conv2d_relu_f32;
    bc_map[CSINN_OP_DEPTHWISE_CONV2D][3] = csi_ref_depthwise_conv2d_f32;
    bc_map[CSINN_OP_DEPTHWISE_CONV2D_RELU][3] = csi_ref_depthwise_conv2d_relu_f32;
    bc_map[CSINN_OP_GROUP_CONV2D][3] = csi_ref_group_conv2d_f32;
    bc_map[CSINN_OP_CONV3D][3] = csi_ref_conv3d_f32;
    bc_map[CSINN_OP_DECONV2D][3] = csi_ref_deconv2d_f32;
    bc_map[CSINN_OP_DEPTHWISE_DECONV2D][3] = csi_ref_depthwise_deconv2d_f32;
    bc_map[CSINN_OP_DECONV3D][3] = csi_ref_deconv3d_f32;
    bc_map[CSINN_OP_COS][3] = csi_ref_cos_f32;
    bc_map[CSINN_OP_COSH][3] = csi_ref_cosh_f32;
    bc_map[CSINN_OP_CUMPROD][3] = csi_ref_cumprod_f32;
    bc_map[CSINN_OP_CUMSUM][3] = csi_ref_cumsum_f32;
    bc_map[CSINN_OP_DEPTH_TO_SPACE][3] = csi_ref_depth_to_space_f32;
    bc_map[CSINN_OP_DIV][3] = csi_ref_div_f32;
    bc_map[CSINN_OP_ELU][3] = csi_ref_elu_f32;
    bc_map[CSINN_OP_EQUANL][3] = csi_ref_equal_f32;
    bc_map[CSINN_OP_ERF][3] = csi_ref_erf_f32;
    bc_map[CSINN_OP_EXP][3] = csi_ref_exp_f32;
    bc_map[CSINN_OP_EXPAND_DIMS][3] = csi_ref_expand_dims_f32;
    bc_map[CSINN_OP_EXPM1][3] = csi_ref_expm1_f32;
    bc_map[CSINN_OP_FLATTEN][3] = csi_ref_flatten;
    bc_map[CSINN_OP_FLOOR_DIVIDE][3] = csi_ref_floor_divide_f32;
    bc_map[CSINN_OP_FLOOR_MOD][3] = csi_ref_floor_mod_f32;
    bc_map[CSINN_OP_FLOOR][3] = csi_ref_floor_f32;
    bc_map[CSINN_OP_FSMN][3] = csi_ref_fsmn_f32;
    bc_map[CSINN_OP_FULLYCONNECTED][3] = csi_ref_fullyconnected_f32;
    bc_map[CSINN_OP_GATHER_ND][3] = csi_ref_gather_nd_f32;
    bc_map[CSINN_OP_GATHER][3] = csi_ref_gather_f32;
    bc_map[CSINN_OP_GLOBAL_AVGPOOL2D][3] = csi_nn_rvv_global_avgpool2d_fp32;
    bc_map[CSINN_OP_GLOBAL_MAXPOOL2D][3] = csi_ref_global_maxpool2d_f32;
    bc_map[CSINN_OP_GREATHER_EQUAL][3] = csi_ref_greater_equal_f32;
    bc_map[CSINN_OP_GREATHER][3] = csi_ref_greater_f32;
    bc_map[CSINN_OP_HARD_SIGMOID][3] = csi_ref_hard_sigmoid_f32;
    bc_map[CSINN_OP_IM2COL][3] = csi_ref_im2col_f32;
    bc_map[CSINN_OP_ISNAN][3] = csi_ref_isnan_bool_f32;
    bc_map[CSINN_OP_L2N][3] = csi_ref_l2_normalization_f32;
    bc_map[CSINN_OP_L2POOL2D][3] = csi_ref_l2pool_f32;
    bc_map[CSINN_OP_LEAKY_RELU][3] = csi_nn_rvv_leaky_relu_fp32;
    bc_map[CSINN_OP_LESS_EQUAL][3] = csi_ref_less_equal_f32;
    bc_map[CSINN_OP_LESS][3] = csi_ref_less_f32;
    bc_map[CSINN_OP_LOG_SOFTMAX][3] = csi_ref_log_softmax_f32;
    bc_map[CSINN_OP_LOG][3] = csi_ref_log_f32;
    bc_map[CSINN_OP_LOG1P][3] = csi_ref_log1p_f32;
    bc_map[CSINN_OP_LOGICAL_AND][3] = csi_ref_logical_and_f32;
    bc_map[CSINN_OP_LOGICAL_NOT][3] = csi_ref_logical_not_f32;
    bc_map[CSINN_OP_LOGICAL_OR][3] = csi_ref_logical_or_f32;
    bc_map[CSINN_OP_LOGICAL_XOR][3] = csi_ref_logical_xor_f32;
    bc_map[CSINN_OP_LRN][3] = csi_ref_lrn_f32;
    bc_map[CSINN_OP_MATMUL][3] = csi_ref_matmul_f32;
    bc_map[CSINN_OP_MAX][3] = csi_ref_max_stride_f32;
    bc_map[CSINN_OP_MAXIMUM][3] = csi_ref_maximum_f32;
    bc_map[CSINN_OP_MAXPOOL2D][3] = csi_ref_maxpool2d_f32;
    bc_map[CSINN_OP_MAXPOOL2D_LOCAT][3] = csi_ref_maxpool2d_locat_f32;
    bc_map[CSINN_OP_MAXPOOL3D][3] = csi_ref_maxpool3d_f32;
    bc_map[CSINN_OP_MEAN][3] = csi_ref_mean_stride_f32;
    bc_map[CSINN_OP_MEAN_STRIDE][3] = csi_ref_mean_stride_f32;
    bc_map[CSINN_OP_MINIMUM][3] = csi_ref_minimum_f32;
    bc_map[CSINN_OP_MOD][3] = csi_ref_mod_f32;
    bc_map[CSINN_OP_MUL][3] = csi_ref_mul_f32;
    bc_map[CSINN_OP_NDARRAY_SIZE][3] = csi_ref_ndarray_size_f32;
    bc_map[CSINN_OP_NEGATIIVE][3] = csi_ref_negative_f32;
    bc_map[CSINN_OP_NOT_EQUAL][3] = csi_ref_not_equal_f32;
    bc_map[CSINN_OP_PAD][3] = csi_ref_pad_f32;
    bc_map[CSINN_OP_POWER][3] = csi_ref_power_f32;
    bc_map[CSINN_OP_PRELU][3] = csi_ref_prelu_f32;
    bc_map[CSINN_OP_PROD][3] = csi_ref_prod_stride_f32;
    bc_map[CSINN_OP_PROPOSAL][3] = csi_ref_proposal_f32;
    bc_map[CSINN_OP_PSROIPOOLING][3] = csi_ref_psroipooling_f32;
    bc_map[CSINN_OP_REDUCE_LOGSUMEXP][3] = csi_ref_reduce_logsumexp_f32;
    bc_map[CSINN_OP_REDUCE_MAX][3] = csi_ref_reduce_max_f32;
    bc_map[CSINN_OP_REDUCE_MEAN][3] = csi_ref_reduce_mean_f32;
    bc_map[CSINN_OP_REDUCE_MIN][3] = csi_ref_reduce_min_f32;
    bc_map[CSINN_OP_REDUCE_PROD][3] = csi_ref_reduce_prod_f32;
    bc_map[CSINN_OP_REDUCE_SUM][3] = csi_ref_reduce_sum_f32;
    bc_map[CSINN_OP_RELU][3] = csi_nn_rvv_relu_fp32;
    bc_map[CSINN_OP_RELU1][3] = csi_ref_relu1_f32;
    bc_map[CSINN_OP_RELU6][3] = csi_ref_relu6_f32;
    bc_map[CSINN_OP_RELUN][3] = csi_ref_relun_f32;
    bc_map[CSINN_OP_RESHAPE][3] = csi_ref_reshape;
    bc_map[CSINN_OP_RESIZE][3] = csi_ref_resize_f32;
    bc_map[CSINN_OP_REVERSE][3] = csi_ref_reverse_f32;
    bc_map[CSINN_OP_ROIALIGN][3] = csi_ref_roi_align_f32;
    bc_map[CSINN_OP_ROIPOOL][3] = csi_ref_roipool_f32;
    bc_map[CSINN_OP_ROUND][3] = csi_ref_round_f32;
    bc_map[CSINN_OP_RSQRT][3] = csi_ref_rsqrt_f32;
    bc_map[CSINN_OP_SCATTER_ND][3] = csi_ref_scatter_nd_f32;
    bc_map[CSINN_OP_SEGMENT_MAX][3] = csi_ref_segment_max_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MAX][3] = csi_ref_unsorted_segment_max_f32;
    bc_map[CSINN_OP_SEGMENT_MEAN][3] = csi_ref_segment_mean_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MEAN][3] = csi_ref_unsorted_segment_mean_f32;
    bc_map[CSINN_OP_SEGMENT_MIN][3] = csi_ref_segment_min_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MIN][3] = csi_ref_unsorted_segment_min_f32;
    bc_map[CSINN_OP_SEGMENT_PROD][3] = csi_ref_segment_prod_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_PROD][3] = csi_ref_unsorted_segment_prod_f32;
    bc_map[CSINN_OP_SEGMENT_SUM][3] = csi_ref_segment_sum_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_SUM][3] = csi_ref_unsorted_segment_sum_f32;
    bc_map[CSINN_OP_SELECT][3] = csi_ref_select_f32;
    bc_map[CSINN_OP_SHUFFLE_CHANNEL][3] = csi_ref_shuffle_channel_f32;
    bc_map[CSINN_OP_SIGMOID][3] = csi_ref_sigmoid_f32;
    bc_map[CSINN_OP_SIGN][3] = csi_ref_sign_f32;
    bc_map[CSINN_OP_SIN][3] = csi_ref_sin_f32;
    bc_map[CSINN_OP_SINH][3] = csi_ref_sinh_f32;
    bc_map[CSINN_OP_SLICE][3] = csi_ref_slice_f32;
    bc_map[CSINN_OP_SOFTMAX][3] = csi_ref_softmax_f32;
    bc_map[CSINN_OP_SOFTPLUS][3] = csi_ref_softplus_f32;
    bc_map[CSINN_OP_SOFTRELU][3] = csi_ref_softrelu_f32;
    bc_map[CSINN_OP_SOFTSIGN][3] = csi_ref_softsign_f32;
    bc_map[CSINN_OP_SPACE_TO_BATCH][3] = csi_ref_space_to_batch_f32;
    bc_map[CSINN_OP_SPACE_TO_DEPTH][3] = csi_ref_space_to_depth_f32;
    bc_map[CSINN_OP_SPLIT][3] = csi_ref_split_f32;
    bc_map[CSINN_OP_SQRT][3] = csi_ref_sqrt_f32;
    bc_map[CSINN_OP_SQUARE][3] = csi_ref_square_f32;
    bc_map[CSINN_OP_SQUEEZE][3] = csi_ref_squeeze;
    bc_map[CSINN_OP_STACK][3] = csi_ref_stack_f32;
    bc_map[CSINN_OP_STRIDED_SLICE][3] = csi_ref_strided_slice_f32;
    bc_map[CSINN_OP_SUB][3] = csi_ref_sub_f32;
    bc_map[CSINN_OP_SUM][3] = csi_ref_sum_stride_f32;
    bc_map[CSINN_OP_TAN][3] = csi_ref_tan_f32;
    bc_map[CSINN_OP_TANH][3] = csi_ref_tanh_f32;
    bc_map[CSINN_OP_THRESHOLD_RELU][3] = csi_ref_threshold_relu_f32;
    bc_map[CSINN_OP_TILE][3] = csi_ref_tile_f32;
    bc_map[CSINN_OP_TOPK][3] = csi_ref_topk_f32;
    bc_map[CSINN_OP_TRUNC][3] = csi_ref_trunc_f32;
    bc_map[CSINN_OP_TRANSPOSE][3] = csi_ref_transpose;
    bc_map[CSINN_OP_TRUNC][3] = csi_ref_trunc_f32;
    bc_map[CSINN_OP_UNPOOLING][3] = csi_ref_unpooling_f32;
    bc_map[CSINN_OP_UNSTACK][3] = csi_ref_unstack_f32;
    bc_map[CSINN_OP_YUV_RGB_SCALE][3] = csi_ref_yuv_rgb_scale_f32;

    for (int i = 0; i < 3; i++) {
        bc_map[CSINN_OP_ABS][i] = csi_ref_abs_quant;
        bc_map[CSINN_OP_ACOS][i] = csi_ref_acos_quant;
        bc_map[CSINN_OP_ACOSH][i] = csi_ref_acosh_quant;
        bc_map[CSINN_OP_ADD][i] = csi_ref_add_quant;
        bc_map[CSINN_OP_ARANGE][i] = csi_ref_arange_quant;
        bc_map[CSINN_OP_ARGMAX][i] = csi_ref_argmax_stride_quant;
        bc_map[CSINN_OP_ARGMIN][i] = csi_ref_argmin_stride_quant;
        bc_map[CSINN_OP_ASIN][i] = csi_ref_asin_quant;
        bc_map[CSINN_OP_ASINH][i] = csi_ref_asinh_quant;
        bc_map[CSINN_OP_ATAN][i] = csi_ref_atan_quant;
        bc_map[CSINN_OP_ATANH][i] = csi_ref_atanh_quant;
        bc_map[CSINN_OP_AVGPOOL2D][i] = csi_ref_avgpool2d_quant;
        bc_map[CSINN_OP_AVGPOOL3D][i] = csi_ref_avgpool3d_quant;
        bc_map[CSINN_OP_BN][i] = csi_ref_batch_normalization_quant;
        bc_map[CSINN_OP_BATCH_TO_SPACE][i] = csi_ref_batch_to_space_quant;
        bc_map[CSINN_OP_BROADCOST][i] = csi_ref_broadcast_to_quant;
        bc_map[CSINN_OP_CEIL][i] = csi_ref_ceil_quant;
        bc_map[CSINN_OP_CLIP][i] = csi_ref_clip_quant;
        bc_map[CSINN_OP_CONCAT][i] = csi_ref_concat_quant;
        bc_map[CSINN_OP_CONV2D][i] = csi_ref_conv2d_quant;
        bc_map[CSINN_OP_CONV2D_RELU][i] = csi_ref_conv2d_relu_quant;
        bc_map[CSINN_OP_CONV2D_RELU6][i] = csi_ref_conv2d_relu6_quant;
        bc_map[CSINN_OP_DEPTHWISE_CONV2D][i] = csi_ref_depthwise_conv2d_quant;
        bc_map[CSINN_OP_DEPTHWISE_CONV2D_RELU][i] = csi_ref_depthwise_conv2d_relu_quant;
        bc_map[CSINN_OP_DEPTHWISE_CONV2D_RELU6][i] = csi_ref_depthwise_conv2d_relu6_quant;
        bc_map[CSINN_OP_GROUP_CONV2D][i] = csi_ref_group_conv2d_quant;
        bc_map[CSINN_OP_CONV3D][i] = csi_ref_conv3d_quant;
        bc_map[CSINN_OP_DECONV2D][i] = csi_ref_deconv2d_quant;
        bc_map[CSINN_OP_DEPTHWISE_DECONV2D][i] = csi_ref_depthwise_deconv2d_quant;
        bc_map[CSINN_OP_DECONV3D][i] = csi_ref_deconv3d_quant;
        bc_map[CSINN_OP_COS][i] = csi_ref_cos_quant;
        bc_map[CSINN_OP_COSH][i] = csi_ref_cosh_quant;
        bc_map[CSINN_OP_CUMPROD][i] = csi_ref_cumprod_quant;
        bc_map[CSINN_OP_CUMSUM][i] = csi_ref_cumsum_quant;
        bc_map[CSINN_OP_DEPTH_TO_SPACE][i] = csi_ref_depth_to_space_quant;
        bc_map[CSINN_OP_DIV][i] = csi_ref_div_quant;
        bc_map[CSINN_OP_ELU][i] = csi_ref_elu_quant;
        bc_map[CSINN_OP_EQUANL][i] = csi_ref_equal_quant;
        bc_map[CSINN_OP_ERF][i] = csi_ref_erf_quant;
        bc_map[CSINN_OP_EXP][i] = csi_ref_exp_quant;
        bc_map[CSINN_OP_EXPAND_DIMS][i] = csi_ref_expand_dims_quant;
        bc_map[CSINN_OP_EXPM1][i] = csi_ref_expm1_quant;
        bc_map[CSINN_OP_FLATTEN][i] = csi_ref_flatten;
        bc_map[CSINN_OP_FLOOR_DIVIDE][i] = csi_ref_floor_divide_quant;
        bc_map[CSINN_OP_FLOOR_MOD][i] = csi_ref_floor_mod_quant;
        bc_map[CSINN_OP_FLOOR][i] = csi_ref_floor_quant;
        bc_map[CSINN_OP_FSMN][i] = csi_ref_fsmn_quant;
        bc_map[CSINN_OP_FULLYCONNECTED][i] = csi_ref_fullyconnected_quant;
        bc_map[CSINN_OP_GATHER_ND][i] = csi_ref_gather_nd_quant;
        bc_map[CSINN_OP_GATHER][i] = csi_ref_gather_quant;
        bc_map[CSINN_OP_GLOBAL_AVGPOOL2D][i] = csi_ref_global_avgpool2d_quant;
        bc_map[CSINN_OP_GLOBAL_MAXPOOL2D][i] = csi_ref_global_maxpool2d_quant;
        bc_map[CSINN_OP_GREATHER_EQUAL][i] = csi_ref_greater_equal_quant;
        bc_map[CSINN_OP_GREATHER][i] = csi_ref_greater_quant;
        bc_map[CSINN_OP_HARD_SIGMOID][i] = csi_ref_hard_sigmoid_quant;
        bc_map[CSINN_OP_IM2COL][i] = csi_ref_im2col_quant;
        bc_map[CSINN_OP_L2N][i] = csi_ref_l2_normalization_quant;
        bc_map[CSINN_OP_LEAKY_RELU][i] = csi_ref_leaky_relu_quant;
        bc_map[CSINN_OP_LESS_EQUAL][i] = csi_ref_less_equal_quant;
        bc_map[CSINN_OP_LESS][i] = csi_ref_less_quant;
        bc_map[CSINN_OP_LOG_SOFTMAX][i] = csi_ref_log_softmax_quant;
        bc_map[CSINN_OP_LOG][i] = csi_ref_log_quant;
        bc_map[CSINN_OP_LOG1P][i] = csi_ref_log1p_quant;
        bc_map[CSINN_OP_LOGICAL_AND][i] = csi_ref_logical_and_quant;
        bc_map[CSINN_OP_LOGICAL_NOT][i] = csi_ref_logical_not_quant;
        bc_map[CSINN_OP_LOGICAL_OR][i] = csi_ref_logical_or_quant;
        bc_map[CSINN_OP_LOGICAL_XOR][i] = csi_ref_logical_xor_quant;
        bc_map[CSINN_OP_LRN][i] = csi_ref_lrn_quant;
        bc_map[CSINN_OP_MATMUL][i] = csi_ref_matmul_quant;
        bc_map[CSINN_OP_MAX][i] = csi_ref_max_stride_quant;
        bc_map[CSINN_OP_MAXIMUM][i] = csi_ref_maximum_quant;
        bc_map[CSINN_OP_MAXPOOL2D][i] = csi_ref_maxpool2d_quant;
        bc_map[CSINN_OP_MAXPOOL2D_LOCAT][i] = csi_ref_maxpool2d_locat_quant;
        bc_map[CSINN_OP_MAXPOOL3D][i] = csi_ref_maxpool3d_quant;
        bc_map[CSINN_OP_MEAN][i] = csi_ref_mean_stride_quant;
        bc_map[CSINN_OP_MEAN_STRIDE][i] = csi_ref_mean_stride_quant;
        bc_map[CSINN_OP_MIN][i] = csi_ref_min_stride_quant;
        bc_map[CSINN_OP_MINIMUM][i] = csi_ref_minimum_quant;
        bc_map[CSINN_OP_MOD][i] = csi_ref_mod_quant;
        bc_map[CSINN_OP_MUL][i] = csi_ref_mul_quant;
        bc_map[CSINN_OP_NEGATIIVE][i] = csi_ref_negative_quant;
        bc_map[CSINN_OP_NOT_EQUAL][i] = csi_ref_not_equal_quant;
        bc_map[CSINN_OP_PAD][i] = csi_ref_pad_quant;
        bc_map[CSINN_OP_POWER][i] = csi_ref_power_quant;
        bc_map[CSINN_OP_PRELU][i] = csi_ref_prelu_quant;
        bc_map[CSINN_OP_PROD][i] = csi_ref_prod_stride_quant;
        bc_map[CSINN_OP_PROPOSAL][i] = csi_ref_proposal_quant;
        bc_map[CSINN_OP_PSROIPOOLING][i] = csi_ref_psroipooling_quant;
        bc_map[CSINN_OP_REDUCE_LOGSUMEXP][i] = csi_ref_reduce_logsumexp_quant;
        bc_map[CSINN_OP_REDUCE_MAX][i] = csi_ref_reduce_max_quant;
        bc_map[CSINN_OP_REDUCE_MEAN][i] = csi_ref_reduce_mean_quant;
        bc_map[CSINN_OP_REDUCE_MIN][i] = csi_ref_reduce_min_quant;
        bc_map[CSINN_OP_REDUCE_PROD][i] = csi_ref_reduce_prod_quant;
        bc_map[CSINN_OP_REDUCE_SUM][i] = csi_ref_reduce_sum_quant;
        bc_map[CSINN_OP_RELU][i] = csi_ref_relu_quant;
        bc_map[CSINN_OP_RELU1][i] = csi_ref_relu1_quant;
        bc_map[CSINN_OP_RELU6][i] = csi_ref_relu6_quant;
        bc_map[CSINN_OP_RELUN][i] = csi_ref_relun_quant;
        bc_map[CSINN_OP_RESHAPE][i] = csi_ref_reshape_quant;
        bc_map[CSINN_OP_RESIZE][i] = csi_ref_resize_quant;
        bc_map[CSINN_OP_REVERSE][i] = csi_ref_reverse_quant;
        bc_map[CSINN_OP_ROIPOOL][i] = csi_ref_roipool_quant;
        bc_map[CSINN_OP_ROUND][i] = csi_ref_round_quant;
        bc_map[CSINN_OP_RSQRT][i] = csi_ref_rsqrt_quant;
        bc_map[CSINN_OP_SCATTER_ND][i] = csi_ref_scatter_nd_quant;
        bc_map[CSINN_OP_SEGMENT_MAX][i] = csi_ref_segment_max_quant;
        bc_map[CSINN_OP_UNSORTED_SEGMENT_MAX][i] = csi_ref_unsorted_segment_max_quant;
        bc_map[CSINN_OP_SEGMENT_MEAN][i] = csi_ref_segment_mean_quant;
        bc_map[CSINN_OP_UNSORTED_SEGMENT_MEAN][i] = csi_ref_unsorted_segment_mean_quant;
        bc_map[CSINN_OP_SEGMENT_MIN][i] = csi_ref_segment_min_quant;
        bc_map[CSINN_OP_UNSORTED_SEGMENT_MIN][i] = csi_ref_unsorted_segment_min_quant;
        bc_map[CSINN_OP_SEGMENT_PROD][i] = csi_ref_segment_prod_quant;
        bc_map[CSINN_OP_UNSORTED_SEGMENT_PROD][i] = csi_ref_unsorted_segment_prod_quant;
        bc_map[CSINN_OP_SEGMENT_SUM][i] = csi_ref_segment_sum_quant;
        bc_map[CSINN_OP_UNSORTED_SEGMENT_SUM][i] = csi_ref_unsorted_segment_sum_quant;
        bc_map[CSINN_OP_SHUFFLE_CHANNEL][i] = csi_ref_shuffle_channel_quant;
        bc_map[CSINN_OP_SIGMOID][i] = csi_ref_sigmoid_quant;
        bc_map[CSINN_OP_SIGN][i] = csi_ref_sign_quant;
        bc_map[CSINN_OP_SIN][i] = csi_ref_sin_quant;
        bc_map[CSINN_OP_SINH][i] = csi_ref_sinh_quant;
        bc_map[CSINN_OP_SLICE][i] = csi_ref_slice_quant;
        bc_map[CSINN_OP_SOFTMAX][i] = csi_ref_softmax_quant;
        bc_map[CSINN_OP_SOFTPLUS][i] = csi_ref_softplus_quant;
        bc_map[CSINN_OP_SOFTRELU][i] = csi_ref_softrelu_quant;
        bc_map[CSINN_OP_SOFTSIGN][i] = csi_ref_softsign_quant;
        bc_map[CSINN_OP_SPACE_TO_BATCH][i] = csi_ref_space_to_batch_quant;
        bc_map[CSINN_OP_SPACE_TO_DEPTH][i] = csi_ref_space_to_depth_quant;
        bc_map[CSINN_OP_SPLIT][i] = csi_ref_split_quant;
        bc_map[CSINN_OP_SQRT][i] = csi_ref_sqrt_quant;
        bc_map[CSINN_OP_STACK][i] = csi_ref_stack_quant;
        bc_map[CSINN_OP_STRIDED_SLICE][i] = csi_ref_strided_slice_quant;
        bc_map[CSINN_OP_SUB][i] = csi_ref_sub_quant;
        bc_map[CSINN_OP_SUM][i] = csi_ref_sum_stride_quant;
        bc_map[CSINN_OP_TAN][i] = csi_ref_tan_quant;
        bc_map[CSINN_OP_TANH][i] = csi_ref_tanh_quant;
        bc_map[CSINN_OP_THRESHOLD_RELU][i] = csi_ref_threshold_relu_quant;
        bc_map[CSINN_OP_TILE][i] = csi_ref_tile_quant;
        bc_map[CSINN_OP_TOPK][i] = csi_ref_topk_quant;
        bc_map[CSINN_OP_TRUNC][i] = csi_ref_trunc_quant;
        bc_map[CSINN_OP_TRANSPOSE][i] = csi_ref_transpose_quant;
        bc_map[CSINN_OP_TRUNC][i] = csi_ref_trunc_quant;
        bc_map[CSINN_OP_UNPOOLING][i] = csi_ref_unpooling_quant;
        bc_map[CSINN_OP_UNSTACK][i] = csi_ref_unstack_qunat;
        bc_map[CSINN_OP_YUV_RGB_SCALE][i] = csi_ref_yuv_rgb_scale_quant;
    }
    // fp16 opt interface
    bc_map[CSINN_OP_ADD][2] = csi_nn_rvv_add_fp16;
    bc_map[CSINN_OP_CONCAT][2] = csi_nn_rvv_concat_fp16;
    bc_map[CSINN_OP_GLOBAL_AVGPOOL2D][2] = csi_nn_rvv_global_avgpool2d_fp16;
    bc_map[CSINN_OP_LEAKY_RELU][2] = csi_nn_rvv_leaky_relu_fp16;
    bc_map[CSINN_OP_RELU][2] = csi_nn_rvv_relu_fp16;
    // int8 opt interface
    bc_map[CSINN_OP_ADD][1] = csi_nn_rvv_add_int8;
    bc_map[CSINN_OP_CONCAT][1] = csi_nn_rvv_concat_int8;
    bc_map[CSINN_OP_LEAKY_RELU][1] = csi_nn_rvv_leaky_relu_int8;
    bc_map[CSINN_OP_RELU][1] = csi_nn_rvv_relu_int8;
    // int4 opt interface

    return bc_map;
}

static int get_bc_map_index(int op, int dtype)
{
    switch (dtype) {
        case CSINN_DTYPE_INT4:
            return op * 4;
            break;
        case CSINN_DTYPE_INT8:
            return op * 4 + 1;
            break;
        case CSINN_DTYPE_FLOAT16:
            return op * 4 + 2;
            break;
        case CSINN_DTYPE_FLOAT32:
            return op * 4 + 3;
            break;
        default:
            return CSINN_UNSUPPORT_DTYPE;
    }
}

void *csi_bc_map_rvv(int op, int dtype)
{
    static int has_init;
    static void **bc_map_table;
    if (has_init == 0) {
        bc_map_table = setup_bc_map();
        has_init = 1;
    }
    return bc_map_table[get_bc_map_index(op, dtype)];
}
