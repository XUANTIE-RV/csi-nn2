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

/* CSI-NN2 version 1.10.x */

#include "csi_ref.h"

void *csi_init_map_ref(int op, int dtype)
{
    if (op == CSINN_OP_FLATTEN) {
        return csi_ref_flatten_init;
    } else if (op == CSINN_OP_RESHAPE) {
        return csi_ref_reshape_init;
    } else if(op == CSINN_OP_TRANSPOSE) {
        return csi_ref_transpose_init;
    }

    return NULL;
}

void csi_ref_nn_init(struct csi_tensor *input,
                     struct csi_tensor *output)
{
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }
    int q_size = output->quant_channel;
    int inner_size = size / q_size;
    if (output->dtype == CSINN_DTYPE_UINT8){
        float *input_data = input->data;
        uint8_t *output_data = output->data;
        for (int i = 0; i < q_size; i++){
            for (int j = 0; j < inner_size; j++) {
                int index = i * inner_size + j;
                int32_t input_val = round(input_data[index] / output->qinfo[i].scale) + output->qinfo[i].zero_point;
                if (input_val < 0) {
                    input_val = 0;
                } else if (input_val > 255) {
                    input_val = 255;
                }
                output_data[index] = input_val;
            }
        }
    } else if (output->dtype == CSINN_DTYPE_INT8){
        float *input_data = input->data;
        int8_t *output_data = output->data;
        for (int i = 0; i < q_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                int index = i * inner_size + j;
                int32_t input_val = round(input_data[index] / output->qinfo[i].scale) + output->qinfo[i].zero_point;
                if (input_val < -127) {
                    input_val = 0;
                } else if (input_val > 127) {
                    input_val = 127;
                }
                output_data[index] = input_val;
            }
        }
    } else if (output->dtype == CSINN_DTYPE_FLOAT16) {
        float *input_data = input->data;
        int16_t *output_data = output->data;
        for (int i = 0; i < q_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                int index = i * inner_size + j;
                output_data[index] = csi_ref_float32_to_float16(input_data[index]);
            }
        }
    } else {
        csi_debug_error("csi_ref_nn_init: unsupport dtype\n");
    }
}

void csi_ref_nn_deinit(struct csi_tensor *input,
                       struct csi_tensor *output)
{
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }
    int q_size = input->quant_channel;
    int inner_size = size / q_size;
    if (input->dtype == CSINN_DTYPE_UINT8){
        uint8_t *input_data = input->data;
        float *output_data = output->data;
        for (int i = 0; i < q_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                int index = i * inner_size + j;
                float x = input_data[index];
                x -= input->qinfo[i].zero_point;
                output_data[index] = x * input->qinfo[i].scale;
            }
        }
    } else if (input->dtype == CSINN_DTYPE_INT8){
        int8_t *input_data = input->data;
        float *output_data = output->data;
        for (int i = 0; i < q_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                int index = i * inner_size + j;
                float x = input_data[index];
                x -= input->qinfo[i].zero_point;
                output_data[index] = x * input->qinfo[i].scale;
            }
        }
    } else if (input->dtype == CSINN_DTYPE_INT32){
        int size = csi_tensor_size(input);
        memcpy(output->data, input->data, size*4);
    } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
        int16_t *input_data = input->data;
        float *output_data = output->data;
        for (int i = 0; i < q_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                int index = i * inner_size + j;
                output_data[index] = csi_ref_float16_to_float32(input_data[index]);
            }
        }
    } else if (input->dtype == CSINN_DTYPE_BOOL) {
        int size = csi_tensor_size(input);
        memcpy(output->data, input->data, size);
    } else {
        csi_debug_error("csi_ref_nn_deinit: unsupport dtype\n");
    }
}

static void *setup_bc_map()
{
    static void* bc_map[CSINN_OP_AND_UTILS_SIZE][CSINN_DTYPE_SIZE];
    for (int i = CSINN_DTYPE_UINT8; i <= CSINN_DTYPE_FLOAT16; i++) {
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
        bc_map[CSINN_OP_RESHAPE][i] = csi_ref_reshape;
        bc_map[CSINN_OP_RESIZE][i] = csi_ref_resize_quant;
        bc_map[CSINN_OP_REVERSE][i] = csi_ref_reverse_quant;
        bc_map[CSINN_OP_ROIPOOL][i] = csi_ref_roipool_quant;
        bc_map[CSINN_OP_ROUND][i] = csi_ref_round_quant;
        bc_map[CSINN_OP_RSQRT][i] = csi_ref_rsqrt_quant;
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
        bc_map[CSINN_OP_TRANSPOSE][i] = csi_ref_transpose;
        bc_map[CSINN_OP_TRUNC][i] = csi_ref_trunc_quant;
        bc_map[CSINN_OP_UNPOOLING][i] = csi_ref_unpooling_quant;
        bc_map[CSINN_OP_YUV_RGB_SCALE][i] = csi_ref_yuv_rgb_scale_quant;
        bc_map[CSINN_OP_CONV2D][i] = csi_ref_conv2d_quant;
        bc_map[CSINN_OP_CONV2D_RELU][i] = csi_ref_conv2d_relu_quant;
        bc_map[CSINN_OP_CONV2D_RELU6][i] = csi_ref_conv2d_relu6_quant;
        bc_map[CSINN_OP_CONV2D_CHANNEL][i] = csi_ref_conv2d_channel_quant;
        bc_map[CSINN_OP_CONV2D_CHANNEL_RELU][i] = csi_ref_conv2d_channel_relu_quant;
        bc_map[CSINN_OP_CONV2D_CHANNEL_RELU6][i] = csi_ref_conv2d_channel_relu6_quant;
        bc_map[CSINN_OP_DEPTHWISE_CONV2D][i] = csi_ref_depthwise_conv2d_quant;
        bc_map[CSINN_OP_DEPTHWISE_CONV2D_RELU][i] = csi_ref_depthwise_conv2d_relu_quant;
        bc_map[CSINN_OP_DEPTHWISE_CONV2D_RELU6][i] = csi_ref_depthwise_conv2d_relu6_quant;
        bc_map[CSINN_OP_DEPTHWISE_CONV2D_CHANNEL][i] = csi_ref_depthwise_conv2d_channel_quant;
        bc_map[CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU][i] = csi_ref_depthwise_conv2d_channel_relu_quant;
        bc_map[CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6][i] = csi_ref_depthwise_conv2d_channel_relu6_quant;
        bc_map[CSINN_OP_GROUP_CONV2D][i] = csi_ref_group_conv2d_quant;
        bc_map[CSINN_OP_GROUP_CONV2D_RELU][i] = csi_ref_group_conv2d_relu_quant;
        bc_map[CSINN_OP_GROUP_CONV2D_RELU6][i] = csi_ref_group_conv2d_relu6_quant;
        bc_map[CSINN_OP_GROUP_CONV2D_CHANNEL][i] = csi_ref_group_conv2d_channel_quant;
        bc_map[CSINN_OP_GROUP_CONV2D_CHANNEL_RELU][i] = csi_ref_group_conv2d_channel_relu_quant;
        bc_map[CSINN_OP_CONV3D][i] = csi_ref_conv3d_quant;
        bc_map[CSINN_OP_DECONV2D][i] = csi_ref_deconv2d_quant;
        bc_map[CSINN_OP_DEPTHWISE_DECONV2D][i] = csi_ref_depthwise_deconv2d_quant;
        bc_map[CSINN_OP_DECONV3D][i] = csi_ref_deconv3d_quant;
        bc_map[CSINN_OP_FULLYCONNECTED][i] = csi_ref_fullyconnected_quant;
        bc_map[CSINN_OP_SCATTER_ND][i] = csi_ref_scatter_nd_quant;
        bc_map[CSINN_OP_SPLIT][i] = csi_ref_split_quant;
    }

    for (int i = CSINN_DTYPE_UINT8; i <= CSINN_DTYPE_FLOAT64; i++) {
        bc_map[CSINN_OP_SQUEEZE][i] = csi_ref_squeeze;
    }

    bc_map[CSINN_OP_AND][CSINN_DTYPE_UINT8] = csi_ref_and_u8;
    bc_map[CSINN_OP_AND][CSINN_DTYPE_INT8] = csi_ref_and_i8;
    bc_map[CSINN_OP_AND][CSINN_DTYPE_UINT32] = csi_ref_and_u32;
    bc_map[CSINN_OP_NDARRAY_SIZE][CSINN_DTYPE_UINT8] = csi_ref_ndarray_size_u8;
    bc_map[CSINN_OP_NDARRAY_SIZE][CSINN_DTYPE_INT8] = csi_ref_ndarray_size_i8;
    bc_map[CSINN_OP_NDARRAY_SIZE][CSINN_DTYPE_INT32] = csi_ref_ndarray_size_i32;
    bc_map[CSINN_OP_NDARRAY_SIZE][CSINN_DTYPE_FLOAT32] = csi_ref_ndarray_size_f32;
    bc_map[CSINN_OP_NOT][CSINN_DTYPE_UINT8] = csi_ref_not_u8;
    bc_map[CSINN_OP_NOT][CSINN_DTYPE_INT8] = csi_ref_not_i8;
    bc_map[CSINN_OP_NOT][CSINN_DTYPE_UINT32] = csi_ref_not_u32;
    bc_map[CSINN_OP_OR][CSINN_DTYPE_UINT8] = csi_ref_or_u8;
    bc_map[CSINN_OP_OR][CSINN_DTYPE_INT8] = csi_ref_or_i8;
    bc_map[CSINN_OP_OR][CSINN_DTYPE_UINT32] = csi_ref_or_u32;
    bc_map[CSINN_OP_SELECT][CSINN_DTYPE_UINT8] = csi_ref_select_u8;
    bc_map[CSINN_OP_SELECT][CSINN_DTYPE_INT8] = csi_ref_select_i8;
    bc_map[CSINN_OP_SELECT][CSINN_DTYPE_FLOAT32] = csi_ref_select_f32;
    bc_map[CSINN_OP_SHAPE][CSINN_DTYPE_UINT8] = csi_ref_shape_u8;
    bc_map[CSINN_OP_SHAPE][CSINN_DTYPE_INT8] = csi_ref_shape_i8;
    bc_map[CSINN_OP_SHAPE][CSINN_DTYPE_INT32] = csi_ref_shape_i32;
    bc_map[CSINN_OP_XOR][CSINN_DTYPE_UINT8] = csi_ref_xor_u8;
    bc_map[CSINN_OP_XOR][CSINN_DTYPE_INT8] = csi_ref_xor_i8;
    bc_map[CSINN_OP_XOR][CSINN_DTYPE_UINT32] = csi_ref_xor_u32;

    bc_map[CSINN_OP_ABS][CSINN_DTYPE_FLOAT32] = csi_ref_abs_f32;
    bc_map[CSINN_OP_ACOS][CSINN_DTYPE_FLOAT32] = csi_ref_acos_f32;
    bc_map[CSINN_OP_ACOSH][CSINN_DTYPE_FLOAT32] = csi_ref_acosh_f32;
    bc_map[CSINN_OP_ADD][CSINN_DTYPE_FLOAT32] = csi_ref_add_f32;
    bc_map[CSINN_OP_ARANGE][CSINN_DTYPE_FLOAT32] = csi_ref_arange_f32;
    bc_map[CSINN_OP_ARGMAX][CSINN_DTYPE_FLOAT32] = csi_ref_argmax_stride_i32_f32;
    bc_map[CSINN_OP_ARGMIN][CSINN_DTYPE_FLOAT32] = csi_ref_argmin_stride_i32_f32;
    bc_map[CSINN_OP_ASIN][CSINN_DTYPE_FLOAT32] = csi_ref_asin_f32;
    bc_map[CSINN_OP_ASINH][CSINN_DTYPE_FLOAT32] = csi_ref_asinh_f32;
    bc_map[CSINN_OP_ATAN][CSINN_DTYPE_FLOAT32] = csi_ref_atan_f32;
    bc_map[CSINN_OP_ATANH][CSINN_DTYPE_FLOAT32] = csi_ref_atanh_f32;
    bc_map[CSINN_OP_AVGPOOL2D][CSINN_DTYPE_FLOAT32] = csi_ref_avgpool2d_f32;
    bc_map[CSINN_OP_AVGPOOL3D][CSINN_DTYPE_FLOAT32] = csi_ref_avgpool3d_f32;
    bc_map[CSINN_OP_BN][CSINN_DTYPE_FLOAT32] = csi_ref_batch_normalization_f32;
    bc_map[CSINN_OP_BATCH_TO_SPACE][CSINN_DTYPE_FLOAT32] = csi_ref_batch_to_space_f32;
    bc_map[CSINN_OP_BROADCOST][CSINN_DTYPE_FLOAT32] = csi_ref_broadcast_to_f32;
    bc_map[CSINN_OP_CEIL][CSINN_DTYPE_FLOAT32] = csi_ref_ceil_f32;
    bc_map[CSINN_OP_CLIP][CSINN_DTYPE_FLOAT32] = csi_ref_clip_f32;
    bc_map[CSINN_OP_CONCAT][CSINN_DTYPE_FLOAT32] = csi_ref_concat_f32;
    bc_map[CSINN_OP_CONV2D][CSINN_DTYPE_FLOAT32] = csi_ref_conv2d_f32;
    bc_map[CSINN_OP_DEPTHWISE_CONV2D][CSINN_DTYPE_FLOAT32] = csi_ref_depthwise_conv2d_f32;
    bc_map[CSINN_OP_GROUP_CONV2D][CSINN_DTYPE_FLOAT32] = csi_ref_group_conv2d_f32;
    bc_map[CSINN_OP_CONV3D][CSINN_DTYPE_FLOAT32] = csi_ref_conv3d_f32;
    bc_map[CSINN_OP_DECONV2D][CSINN_DTYPE_FLOAT32] = csi_ref_deconv2d_f32;
    bc_map[CSINN_OP_DEPTHWISE_DECONV2D][CSINN_DTYPE_FLOAT32] = csi_ref_depthwise_deconv2d_f32;
    bc_map[CSINN_OP_DECONV3D][CSINN_DTYPE_FLOAT32] = csi_ref_deconv3d_f32;
    bc_map[CSINN_OP_COS][CSINN_DTYPE_FLOAT32] = csi_ref_cos_f32;
    bc_map[CSINN_OP_COSH][CSINN_DTYPE_FLOAT32] = csi_ref_cosh_f32;
    bc_map[CSINN_OP_CUMPROD][CSINN_DTYPE_FLOAT32] = csi_ref_cumprod_f32;
    bc_map[CSINN_OP_CUMSUM][CSINN_DTYPE_FLOAT32] = csi_ref_cumsum_f32;
    bc_map[CSINN_OP_DEPTH_TO_SPACE][CSINN_DTYPE_FLOAT32] = csi_ref_depth_to_space_f32;
    bc_map[CSINN_OP_DIV][CSINN_DTYPE_FLOAT32] = csi_ref_div_f32;
    bc_map[CSINN_OP_ELU][CSINN_DTYPE_FLOAT32] = csi_ref_elu_f32;
    bc_map[CSINN_OP_EQUANL][CSINN_DTYPE_FLOAT32] = csi_ref_equal_f32;
    bc_map[CSINN_OP_ERF][CSINN_DTYPE_FLOAT32] = csi_ref_erf_f32;
    bc_map[CSINN_OP_EXP][CSINN_DTYPE_FLOAT32] = csi_ref_exp_f32;
    bc_map[CSINN_OP_EXPAND_DIMS][CSINN_DTYPE_FLOAT32] = csi_ref_expand_dims_f32;
    bc_map[CSINN_OP_EXPM1][CSINN_DTYPE_FLOAT32] = csi_ref_expm1_f32;
    bc_map[CSINN_OP_FLATTEN][CSINN_DTYPE_FLOAT32] = csi_ref_flatten;
    bc_map[CSINN_OP_FLOOR_DIVIDE][CSINN_DTYPE_FLOAT32] = csi_ref_floor_divide_f32;
    bc_map[CSINN_OP_FLOOR_MOD][CSINN_DTYPE_FLOAT32] = csi_ref_floor_mod_f32;
    bc_map[CSINN_OP_FLOOR][CSINN_DTYPE_FLOAT32] = csi_ref_floor_f32;
    bc_map[CSINN_OP_FSMN][CSINN_DTYPE_FLOAT32] = csi_ref_fsmn_f32;
    bc_map[CSINN_OP_FULLYCONNECTED][CSINN_DTYPE_FLOAT32] = csi_ref_fullyconnected_f32;
    bc_map[CSINN_OP_GATHER_ND][CSINN_DTYPE_FLOAT32] = csi_ref_gather_nd_f32;
    bc_map[CSINN_OP_GATHER][CSINN_DTYPE_FLOAT32] = csi_ref_gather_f32;
    bc_map[CSINN_OP_GLOBAL_AVGPOOL2D][CSINN_DTYPE_FLOAT32] = csi_ref_global_avgpool2d_f32;
    bc_map[CSINN_OP_GLOBAL_MAXPOOL2D][CSINN_DTYPE_FLOAT32] = csi_ref_global_maxpool2d_f32;
    bc_map[CSINN_OP_GREATHER_EQUAL][CSINN_DTYPE_FLOAT32] = csi_ref_greater_equal_f32;
    bc_map[CSINN_OP_GREATHER][CSINN_DTYPE_FLOAT32] = csi_ref_greater_f32;
    bc_map[CSINN_OP_HARD_SIGMOID][CSINN_DTYPE_FLOAT32] = csi_ref_hard_sigmoid_f32;
    bc_map[CSINN_OP_IM2COL][CSINN_DTYPE_FLOAT32] = csi_ref_im2col_f32;
    bc_map[CSINN_OP_L2N][CSINN_DTYPE_FLOAT32] = csi_ref_l2_normalization_f32;
    bc_map[CSINN_OP_LEAKY_RELU][CSINN_DTYPE_FLOAT32] = csi_ref_leaky_relu_f32;
    bc_map[CSINN_OP_LESS_EQUAL][CSINN_DTYPE_FLOAT32] = csi_ref_less_equal_f32;
    bc_map[CSINN_OP_LESS][CSINN_DTYPE_FLOAT32] = csi_ref_less_f32;
    bc_map[CSINN_OP_LOG_SOFTMAX][CSINN_DTYPE_FLOAT32] = csi_ref_log_softmax_f32;
    bc_map[CSINN_OP_LOG][CSINN_DTYPE_FLOAT32] = csi_ref_log_f32;
    bc_map[CSINN_OP_LOG1P][CSINN_DTYPE_FLOAT32] = csi_ref_log1p_f32;
    bc_map[CSINN_OP_LOGICAL_AND][CSINN_DTYPE_FLOAT32] = csi_ref_logical_and_f32;
    bc_map[CSINN_OP_LOGICAL_NOT][CSINN_DTYPE_FLOAT32] = csi_ref_logical_not_f32;
    bc_map[CSINN_OP_LOGICAL_OR][CSINN_DTYPE_FLOAT32] = csi_ref_logical_or_f32;
    bc_map[CSINN_OP_LOGICAL_XOR][CSINN_DTYPE_FLOAT32] = csi_ref_logical_xor_f32;
    bc_map[CSINN_OP_LRN][CSINN_DTYPE_FLOAT32] = csi_ref_lrn_f32;
    bc_map[CSINN_OP_MATMUL][CSINN_DTYPE_FLOAT32] = csi_ref_matmul_f32;
    bc_map[CSINN_OP_MAX][CSINN_DTYPE_FLOAT32] = csi_ref_max_stride_f32;
    bc_map[CSINN_OP_MAXIMUM][CSINN_DTYPE_FLOAT32] = csi_ref_maximum_f32;
    bc_map[CSINN_OP_MAXPOOL2D][CSINN_DTYPE_FLOAT32] = csi_ref_maxpool2d_f32;
    bc_map[CSINN_OP_MAXPOOL2D_LOCAT][CSINN_DTYPE_FLOAT32] = csi_ref_maxpool2d_locat_f32;
    bc_map[CSINN_OP_MAXPOOL3D][CSINN_DTYPE_FLOAT32] = csi_ref_maxpool3d_f32;
    bc_map[CSINN_OP_MEAN][CSINN_DTYPE_FLOAT32] = csi_ref_mean_stride_f32;
    bc_map[CSINN_OP_MEAN_STRIDE][CSINN_DTYPE_FLOAT32] = csi_ref_mean_stride_f32;
    bc_map[CSINN_OP_MIN][CSINN_DTYPE_FLOAT32] = csi_ref_min_stride_f32;
    bc_map[CSINN_OP_MINIMUM][CSINN_DTYPE_FLOAT32] = csi_ref_minimum_f32;
    bc_map[CSINN_OP_MOD][CSINN_DTYPE_FLOAT32] = csi_ref_mod_f32;
    bc_map[CSINN_OP_MUL][CSINN_DTYPE_FLOAT32] = csi_ref_mul_f32;
    bc_map[CSINN_OP_NEGATIIVE][CSINN_DTYPE_FLOAT32] = csi_ref_negative_f32;
    bc_map[CSINN_OP_NON_MAX_SUPPRESSION][CSINN_DTYPE_FLOAT32] = csi_ref_non_max_suppression_std;
    bc_map[CSINN_OP_NOT_EQUAL][CSINN_DTYPE_FLOAT32] = csi_ref_not_equal_f32;
    bc_map[CSINN_OP_PAD][CSINN_DTYPE_FLOAT32] = csi_ref_pad_f32;
    bc_map[CSINN_OP_POWER][CSINN_DTYPE_FLOAT32] = csi_ref_power_f32;
    bc_map[CSINN_OP_PRELU][CSINN_DTYPE_FLOAT32] = csi_ref_prelu_f32;
    bc_map[CSINN_OP_PROD][CSINN_DTYPE_FLOAT32] = csi_ref_prod_stride_f32;
    bc_map[CSINN_OP_PROPOSAL][CSINN_DTYPE_FLOAT32] = csi_ref_proposal_f32;
    bc_map[CSINN_OP_PSROIPOOLING][CSINN_DTYPE_FLOAT32] = csi_ref_psroipooling_f32;
    bc_map[CSINN_OP_REDUCE_LOGSUMEXP][CSINN_DTYPE_FLOAT32] = csi_ref_reduce_logsumexp_f32;
    bc_map[CSINN_OP_REDUCE_MAX][CSINN_DTYPE_FLOAT32] = csi_ref_reduce_max_f32;
    bc_map[CSINN_OP_REDUCE_MEAN][CSINN_DTYPE_FLOAT32] = csi_ref_reduce_mean_f32;
    bc_map[CSINN_OP_REDUCE_MIN][CSINN_DTYPE_FLOAT32] = csi_ref_reduce_min_f32;
    bc_map[CSINN_OP_REDUCE_PROD][CSINN_DTYPE_FLOAT32] = csi_ref_reduce_prod_f32;
    bc_map[CSINN_OP_REDUCE_SUM][CSINN_DTYPE_FLOAT32] = csi_ref_reduce_sum_f32;
    bc_map[CSINN_OP_RELU][CSINN_DTYPE_FLOAT32] = csi_ref_relu_f32;
    bc_map[CSINN_OP_RELU1][CSINN_DTYPE_FLOAT32] = csi_ref_relu1_f32;
    bc_map[CSINN_OP_RELU6][CSINN_DTYPE_FLOAT32] = csi_ref_relu6_f32;
    bc_map[CSINN_OP_RELUN][CSINN_DTYPE_FLOAT32] = csi_ref_relun_f32;
    bc_map[CSINN_OP_RESHAPE][CSINN_DTYPE_FLOAT32] = csi_ref_reshape;
    bc_map[CSINN_OP_RESIZE][CSINN_DTYPE_FLOAT32] = csi_ref_resize_f32;
    bc_map[CSINN_OP_REVERSE][CSINN_DTYPE_FLOAT32] = csi_ref_reverse_f32;
    bc_map[CSINN_OP_ROIALIGN][CSINN_DTYPE_FLOAT32] = csi_ref_roi_align_f32;
    bc_map[CSINN_OP_ROIPOOL][CSINN_DTYPE_FLOAT32] = csi_ref_roipool_f32;
    bc_map[CSINN_OP_ROUND][CSINN_DTYPE_FLOAT32] = csi_ref_round_f32;
    bc_map[CSINN_OP_RSQRT][CSINN_DTYPE_FLOAT32] = csi_ref_rsqrt_f32;
    bc_map[CSINN_OP_SCATTER_ND][CSINN_DTYPE_FLOAT32] = csi_ref_scatter_nd_f32;
    bc_map[CSINN_OP_SEGMENT_MAX][CSINN_DTYPE_FLOAT32] = csi_ref_segment_max_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MAX][CSINN_DTYPE_FLOAT32] = csi_ref_unsorted_segment_max_f32;
    bc_map[CSINN_OP_SEGMENT_MEAN][CSINN_DTYPE_FLOAT32] = csi_ref_segment_mean_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MEAN][CSINN_DTYPE_FLOAT32] = csi_ref_unsorted_segment_mean_f32;
    bc_map[CSINN_OP_SEGMENT_MIN][CSINN_DTYPE_FLOAT32] = csi_ref_segment_min_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_MIN][CSINN_DTYPE_FLOAT32] = csi_ref_unsorted_segment_min_f32;
    bc_map[CSINN_OP_SEGMENT_PROD][CSINN_DTYPE_FLOAT32] = csi_ref_segment_prod_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_PROD][CSINN_DTYPE_FLOAT32] = csi_ref_unsorted_segment_prod_f32;
    bc_map[CSINN_OP_SEGMENT_SUM][CSINN_DTYPE_FLOAT32] = csi_ref_segment_sum_f32;
    bc_map[CSINN_OP_UNSORTED_SEGMENT_SUM][CSINN_DTYPE_FLOAT32] = csi_ref_unsorted_segment_sum_f32;
    bc_map[CSINN_OP_SHUFFLE_CHANNEL][CSINN_DTYPE_FLOAT32] = csi_ref_shuffle_channel_f32;
    bc_map[CSINN_OP_SIGMOID][CSINN_DTYPE_FLOAT32] = csi_ref_sigmoid_f32;
    bc_map[CSINN_OP_SIGN][CSINN_DTYPE_FLOAT32] = csi_ref_sign_f32;
    bc_map[CSINN_OP_SIN][CSINN_DTYPE_FLOAT32] = csi_ref_sin_f32;
    bc_map[CSINN_OP_SINH][CSINN_DTYPE_FLOAT32] = csi_ref_sinh_f32;
    bc_map[CSINN_OP_SLICE][CSINN_DTYPE_FLOAT32] = csi_ref_slice_f32;
    bc_map[CSINN_OP_SOFTMAX][CSINN_DTYPE_FLOAT32] = csi_ref_softmax_f32;
    bc_map[CSINN_OP_SOFTPLUS][CSINN_DTYPE_FLOAT32] = csi_ref_softplus_f32;
    bc_map[CSINN_OP_SOFTRELU][CSINN_DTYPE_FLOAT32] = csi_ref_softrelu_f32;
    bc_map[CSINN_OP_SOFTSIGN][CSINN_DTYPE_FLOAT32] = csi_ref_softsign_f32;
    bc_map[CSINN_OP_SPACE_TO_BATCH][CSINN_DTYPE_FLOAT32] = csi_ref_space_to_batch_f32;
    bc_map[CSINN_OP_SPACE_TO_DEPTH][CSINN_DTYPE_FLOAT32] = csi_ref_space_to_depth_f32;
    bc_map[CSINN_OP_SPLIT][CSINN_DTYPE_FLOAT32] = csi_ref_split_f32;
    bc_map[CSINN_OP_SQRT][CSINN_DTYPE_FLOAT32] = csi_ref_sqrt_f32;
    bc_map[CSINN_OP_SQUARE][CSINN_DTYPE_FLOAT32] = csi_ref_square_f32;
    bc_map[CSINN_OP_STACK][CSINN_DTYPE_FLOAT32] = csi_ref_stack_f32;
    bc_map[CSINN_OP_STRIDED_SLICE][CSINN_DTYPE_FLOAT32] = csi_ref_strided_slice_f32;
    bc_map[CSINN_OP_SUB][CSINN_DTYPE_FLOAT32] = csi_ref_sub_f32;
    bc_map[CSINN_OP_SUM][CSINN_DTYPE_FLOAT32] = csi_ref_sum_stride_f32;
    bc_map[CSINN_OP_TAN][CSINN_DTYPE_FLOAT32] = csi_ref_tan_f32;
    bc_map[CSINN_OP_TANH][CSINN_DTYPE_FLOAT32] = csi_ref_tanh_f32;
    bc_map[CSINN_OP_THRESHOLD_RELU][CSINN_DTYPE_FLOAT32] = csi_ref_threshold_relu_f32;
    bc_map[CSINN_OP_TILE][CSINN_DTYPE_FLOAT32] = csi_ref_tile_f32;
    bc_map[CSINN_OP_TOPK][CSINN_DTYPE_FLOAT32] = csi_ref_topk_f32;
    bc_map[CSINN_OP_TRANSPOSE][CSINN_DTYPE_FLOAT32] = csi_ref_transpose;
    bc_map[CSINN_OP_TRUNC][CSINN_DTYPE_FLOAT32] = csi_ref_trunc_f32;
    bc_map[CSINN_OP_UNPOOLING][CSINN_DTYPE_FLOAT32] = csi_ref_unpooling_f32;
    bc_map[CSINN_OP_YUV_RGB_SCALE][CSINN_DTYPE_FLOAT32] = csi_ref_yuv_rgb_scale_f32;
    bc_map[CSINN_OP_COL2IM][CSINN_DTYPE_FLOAT32] = csi_ref_col2im_f32;
    bc_map[CSINN_OP_ISNAN][CSINN_DTYPE_FLOAT32] = csi_ref_isnan_bool_f32;
    bc_map[CSINN_OP_L2POOL2D][CSINN_DTYPE_FLOAT32] = csi_ref_l2pool_f32;

    return bc_map;
}

static int get_bc_map_index(int op, int dtype)
{
    return op * CSINN_DTYPE_SIZE + dtype;
}

void *csi_bc_map_ref(int op, int dtype)
{
    static int has_init;
    static void **bc_map_table;
    if (has_init == 0) {
        bc_map_table = setup_bc_map();
        has_init = 1;
    }
    return bc_map_table[get_bc_map_index(op, dtype)];
}
