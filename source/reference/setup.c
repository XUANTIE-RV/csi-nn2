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

/* CSI-NN2 version 2.0.x */

#include "shl_ref.h"

void shl_ref_nn_init(struct csinn_tensor *input, struct csinn_tensor *output)
{
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }
    int q_size = output->quant_channel;
    int inner_size = size / q_size;
    if (output->dtype == CSINN_DTYPE_INT4) {
        float *input_data = input->data;
        int8_t *output_data = output->data;
        for (int i = 0; i < q_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                int index = i * inner_size + j;
                int32_t input_val =
                    round(input_data[index] / output->qinfo[i].scale) + output->qinfo[i].zero_point;
                if (input_val < -8) {
                    input_val = -8;
                } else if (input_val > 7) {
                    input_val = 7;
                }
                int out_index = index / 2;
                /* int4 little endian */
                if (index % 2) {
                    output_data[out_index] = (output_data[out_index] & 0xf) | (input_val << 4);
                } else {
                    output_data[out_index] = (output_data[out_index] & 0xf0) | (input_val & 0xf);
                }
            }
        }
    } else if (output->dtype == CSINN_DTYPE_UINT8) {
        float *input_data = input->data;
        uint8_t *output_data = output->data;
        for (int i = 0; i < q_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                int index = i * inner_size + j;
                int32_t input_val =
                    round(input_data[index] / output->qinfo[i].scale) + output->qinfo[i].zero_point;
                if (input_val < 0) {
                    input_val = 0;
                } else if (input_val > 255) {
                    input_val = 255;
                }
                output_data[index] = input_val;
            }
        }
    } else if (output->dtype == CSINN_DTYPE_INT8) {
        float *input_data = input->data;
        int8_t *output_data = output->data;
        for (int i = 0; i < q_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                int index = i * inner_size + j;
                int32_t input_val =
                    round(input_data[index] / output->qinfo[i].scale) + output->qinfo[i].zero_point;
                if (input_val < -127) {
                    input_val = -127;
                } else if (input_val > 127) {
                    input_val = 127;
                }
                output_data[index] = input_val;
            }
        }
    } else if (output->dtype == CSINN_DTYPE_INT16) {
        float *input_data = input->data;
        int16_t *output_data = output->data;
        for (int i = 0; i < q_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                int index = i * inner_size + j;
                int32_t input_val =
                    round(input_data[index] / output->qinfo[i].scale) + output->qinfo[i].zero_point;
                if (input_val < -32768) {
                    input_val = -32768;
                } else if (input_val > 32767) {
                    input_val = 32767;
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
                output_data[index] = shl_ref_float32_to_float16(input_data[index]);
            }
        }
    } else if (output->dtype == CSINN_DTYPE_BFLOAT16) {
        float *input_data = input->data;
        int16_t *output_data = output->data;
        for (int i = 0; i < q_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                int index = i * inner_size + j;
                output_data[index] = shl_ref_float32_to_bfloat16(input_data[index]);
            }
        }
    } else if (output->dtype == CSINN_DTYPE_FLOAT32) {
        float *input_data = input->data;
        float *output_data = output->data;
        memcpy(output_data, input_data, size * 4);
    } else {
        shl_debug_error("shl_ref_nn_init: unsupport dtype\n");
    }
}

void shl_ref_nn_deinit(struct csinn_tensor *input, struct csinn_tensor *output)
{
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }
    int q_size = input->quant_channel;
    int inner_size = size / q_size;
    if (input->dtype == CSINN_DTYPE_INT4) {
        int8_t *input_data = input->data;
        float *output_data = output->data;
        for (int i = 0; i < q_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                int index = i * inner_size + j;
                int in_index = index / 2;
                float x;
                int8_t tmp_in = 0;
                /* int4 little endian */
                if (index % 2) {
                    tmp_in = input_data[in_index] & 0xf0;
                    x = tmp_in >> 4;
                } else {
                    tmp_in = (input_data[in_index] & 0xf) << 4;
                    x = tmp_in >> 4;
                }
                x -= input->qinfo[i].zero_point;
                output_data[index] = x * input->qinfo[i].scale;
            }
        }
    } else if (input->dtype == CSINN_DTYPE_UINT8) {
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
    } else if (input->dtype == CSINN_DTYPE_INT8) {
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
    } else if (input->dtype == CSINN_DTYPE_INT32) {
        int size = csinn_tensor_size(input);
        memcpy(output->data, input->data, size * 4);
    } else if (input->dtype == CSINN_DTYPE_FLOAT16) {
        int16_t *input_data = input->data;
        float *output_data = output->data;
        for (int i = 0; i < q_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                int index = i * inner_size + j;
                output_data[index] = shl_ref_float16_to_float32(input_data[index]);
            }
        }
    } else if (input->dtype == CSINN_DTYPE_BFLOAT16) {
        int16_t *input_data = input->data;
        float *output_data = output->data;
        for (int i = 0; i < q_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                int index = i * inner_size + j;
                output_data[index] = shl_ref_bfloat16_to_float32(input_data[index]);
            }
        }
    } else if (input->dtype == CSINN_DTYPE_BOOL) {
        int size = csinn_tensor_size(input);
        memcpy(output->data, input->data, size);
    } else {
        shl_debug_error("shl_ref_nn_deinit: unsupport dtype\n");
    }
}

static void *setup_cb_map()
{
    static struct csinn_callback cb_map[CSINN_OP_AND_UTILS_SIZE][CSINN_DTYPE_SIZE];
    memset(cb_map, 0, sizeof(struct csinn_callback) * CSINN_OP_AND_UTILS_SIZE * CSINN_DTYPE_SIZE);

    for (int i = CSINN_DTYPE_INT4; i <= CSINN_DTYPE_BFLOAT16; i++) {
        cb_map[CSINN_OP_ABS][i].exec = shl_ref_abs_quant;
        cb_map[CSINN_OP_ACOS][i].exec = shl_ref_acos_quant;
        cb_map[CSINN_OP_ACOSH][i].exec = shl_ref_acosh_quant;
        cb_map[CSINN_OP_ADD][i].exec = shl_ref_add_quant;
        cb_map[CSINN_OP_ARANGE][i].exec = shl_ref_arange_quant;
        cb_map[CSINN_OP_ARGMAX][i].exec = shl_ref_argmax_stride_quant;
        cb_map[CSINN_OP_ARGMIN][i].exec = shl_ref_argmin_stride_quant;
        cb_map[CSINN_OP_ASIN][i].exec = shl_ref_asin_quant;
        cb_map[CSINN_OP_ASINH][i].exec = shl_ref_asinh_quant;
        cb_map[CSINN_OP_ATAN][i].exec = shl_ref_atan_quant;
        cb_map[CSINN_OP_ATANH][i].exec = shl_ref_atanh_quant;
        cb_map[CSINN_OP_AVGPOOL2D][i].exec = shl_ref_avgpool2d_quant;
        cb_map[CSINN_OP_AVGPOOL3D][i].exec = shl_ref_avgpool3d_quant;
        cb_map[CSINN_OP_BN][i].exec = shl_ref_batch_normalization_quant;
        cb_map[CSINN_OP_BATCH_TO_SPACE][i].exec = shl_ref_batch_to_space_quant;
        cb_map[CSINN_OP_BROADCOST][i].exec = shl_ref_broadcast_to_quant;
        cb_map[CSINN_OP_CACHE_MATMUL][i].exec = shl_ref_cache_matmul_quant;
        cb_map[CSINN_OP_CACHE_MATMUL][i].init = shl_ref_cache_matmul_init;
        cb_map[CSINN_OP_CACHE_CONV1D][i].exec = shl_ref_cache_conv1d_quant;
        cb_map[CSINN_OP_CACHE_CONV1D][i].init = shl_ref_cache_conv1d_init;
        cb_map[CSINN_OP_CEIL][i].exec = shl_ref_ceil_quant;
        cb_map[CSINN_OP_CLIP][i].exec = shl_ref_clip_quant;
        cb_map[CSINN_OP_CONCAT][i].exec = shl_ref_concat_quant;
        cb_map[CSINN_OP_COS][i].exec = shl_ref_cos_quant;
        cb_map[CSINN_OP_COSH][i].exec = shl_ref_cosh_quant;
        cb_map[CSINN_OP_CUMPROD][i].exec = shl_ref_cumprod_quant;
        cb_map[CSINN_OP_DATA_CONVERT][i].exec = shl_ref_data_convert_quant;
        cb_map[CSINN_OP_CUMSUM][i].exec = shl_ref_cumsum_quant;
        cb_map[CSINN_OP_DEPTH_TO_SPACE][i].exec = shl_ref_depth_to_space_quant;
        cb_map[CSINN_OP_DIV][i].exec = shl_ref_div_quant;
        cb_map[CSINN_OP_ELU][i].exec = shl_ref_elu_quant;
        cb_map[CSINN_OP_EQUANL][i].exec = shl_ref_equal_quant;
        cb_map[CSINN_OP_ERF][i].exec = shl_ref_erf_quant;
        cb_map[CSINN_OP_EXP][i].exec = shl_ref_exp_quant;
        cb_map[CSINN_OP_EXPAND_DIMS][i].exec = shl_ref_expand_dims_quant;
        cb_map[CSINN_OP_EXPM1][i].exec = shl_ref_expm1_quant;
        cb_map[CSINN_OP_FLATTEN][i].exec = shl_ref_flatten;
        cb_map[CSINN_OP_FLATTEN][i].init = shl_ref_flatten_init;
        cb_map[CSINN_OP_FLOOR_DIVIDE][i].exec = shl_ref_floor_divide_quant;
        cb_map[CSINN_OP_FLOOR_MOD][i].exec = shl_ref_floor_mod_quant;
        cb_map[CSINN_OP_FLOOR][i].exec = shl_ref_floor_quant;
        cb_map[CSINN_OP_FSMN][i].exec = shl_ref_fsmn_quant;
        cb_map[CSINN_OP_GATHER_ND][i].exec = shl_ref_gather_nd_quant;
        cb_map[CSINN_OP_GATHER][i].exec = shl_ref_gather_quant;
        cb_map[CSINN_OP_GLOBAL_AVGPOOL2D][i].exec = shl_ref_global_avgpool2d_quant;
        cb_map[CSINN_OP_GLOBAL_MAXPOOL2D][i].exec = shl_ref_global_maxpool2d_quant;
        cb_map[CSINN_OP_GREATHER_EQUAL][i].exec = shl_ref_greater_equal_quant;
        cb_map[CSINN_OP_GREATHER][i].exec = shl_ref_greater_quant;
        cb_map[CSINN_OP_HARD_SIGMOID][i].exec = shl_ref_hard_sigmoid_quant;
        cb_map[CSINN_OP_IM2COL][i].exec = shl_ref_im2col_quant;
        cb_map[CSINN_OP_L2N][i].exec = shl_ref_l2_normalization_quant;
        cb_map[CSINN_OP_LEAKY_RELU][i].exec = shl_ref_leaky_relu_quant;
        cb_map[CSINN_OP_LESS_EQUAL][i].exec = shl_ref_less_equal_quant;
        cb_map[CSINN_OP_LESS][i].exec = shl_ref_less_quant;
        cb_map[CSINN_OP_LOG_SOFTMAX][i].exec = shl_ref_log_softmax_quant;
        cb_map[CSINN_OP_LOG][i].exec = shl_ref_log_quant;
        cb_map[CSINN_OP_LOG1P][i].exec = shl_ref_log1p_quant;
        cb_map[CSINN_OP_LOGICAL_AND][i].exec = shl_ref_logical_and_quant;
        cb_map[CSINN_OP_LOGICAL_NOT][i].exec = shl_ref_logical_not_quant;
        cb_map[CSINN_OP_LOGICAL_OR][i].exec = shl_ref_logical_or_quant;
        cb_map[CSINN_OP_LOGICAL_XOR][i].exec = shl_ref_logical_xor_quant;
        cb_map[CSINN_OP_LRN][i].exec = shl_ref_lrn_quant;
        cb_map[CSINN_OP_MATMUL][i].exec = shl_ref_matmul_quant;
        cb_map[CSINN_OP_MAX][i].exec = shl_ref_max_stride_quant;
        cb_map[CSINN_OP_MAXIMUM][i].exec = shl_ref_maximum_quant;
        cb_map[CSINN_OP_MAXPOOL2D][i].exec = shl_ref_maxpool2d_quant;
        cb_map[CSINN_OP_MAXPOOL2D_LOCAT][i].exec = shl_ref_maxpool2d_locat_quant;
        cb_map[CSINN_OP_MAXPOOL3D][i].exec = shl_ref_maxpool3d_quant;
        cb_map[CSINN_OP_MEAN][i].exec = shl_ref_mean_stride_quant;
        cb_map[CSINN_OP_MEAN_STRIDE][i].exec = shl_ref_mean_stride_quant;
        cb_map[CSINN_OP_MIN][i].exec = shl_ref_min_stride_quant;
        cb_map[CSINN_OP_MINIMUM][i].exec = shl_ref_minimum_quant;
        cb_map[CSINN_OP_MOD][i].exec = shl_ref_mod_quant;
        cb_map[CSINN_OP_MUL][i].exec = shl_ref_mul_quant;
        cb_map[CSINN_OP_NEGATIIVE][i].exec = shl_ref_negative_quant;
        cb_map[CSINN_OP_NOT_EQUAL][i].exec = shl_ref_not_equal_quant;
        cb_map[CSINN_OP_PAD][i].exec = shl_ref_pad_quant;
        cb_map[CSINN_OP_POWER][i].exec = shl_ref_power_quant;
        cb_map[CSINN_OP_PRELU][i].exec = shl_ref_prelu_quant;
        cb_map[CSINN_OP_PROD][i].exec = shl_ref_prod_stride_quant;
        cb_map[CSINN_OP_PROPOSAL][i].exec = shl_ref_proposal_quant;
        cb_map[CSINN_OP_PSROIPOOLING][i].exec = shl_ref_psroipooling_quant;
        cb_map[CSINN_OP_REDUCE_LOGSUMEXP][i].exec = shl_ref_reduce_logsumexp_quant;
        cb_map[CSINN_OP_REDUCE_MAX][i].exec = shl_ref_reduce_max_quant;
        cb_map[CSINN_OP_REDUCE_MEAN][i].exec = shl_ref_reduce_mean_quant;
        cb_map[CSINN_OP_REDUCE_MIN][i].exec = shl_ref_reduce_min_quant;
        cb_map[CSINN_OP_REDUCE_PROD][i].exec = shl_ref_reduce_prod_quant;
        cb_map[CSINN_OP_REDUCE_SUM][i].exec = shl_ref_reduce_sum_quant;
        cb_map[CSINN_OP_RELU][i].exec = shl_ref_relu_quant;
        cb_map[CSINN_OP_RELU1][i].exec = shl_ref_relu1_quant;
        cb_map[CSINN_OP_RELU6][i].exec = shl_ref_relu6_quant;
        cb_map[CSINN_OP_RELUN][i].exec = shl_ref_relun_quant;
        cb_map[CSINN_OP_RESHAPE][i].exec = shl_ref_reshape;
        cb_map[CSINN_OP_RESHAPE][i].init = shl_ref_reshape_init;
        cb_map[CSINN_OP_RESIZE][i].exec = shl_ref_resize_quant;
        cb_map[CSINN_OP_REVERSE][i].exec = shl_ref_reverse_quant;
        cb_map[CSINN_OP_ROIPOOL][i].exec = shl_ref_roipool_quant;
        cb_map[CSINN_OP_ROUND][i].exec = shl_ref_round_quant;
        cb_map[CSINN_OP_RSQRT][i].exec = shl_ref_rsqrt_quant;
        cb_map[CSINN_OP_SEGMENT_MAX][i].exec = shl_ref_segment_max_quant;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_MAX][i].exec = shl_ref_unsorted_segment_max_quant;
        cb_map[CSINN_OP_SEGMENT_MEAN][i].exec = shl_ref_segment_mean_quant;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_MEAN][i].exec = shl_ref_unsorted_segment_mean_quant;
        cb_map[CSINN_OP_SEGMENT_MIN][i].exec = shl_ref_segment_min_quant;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_MIN][i].exec = shl_ref_unsorted_segment_min_quant;
        cb_map[CSINN_OP_SEGMENT_PROD][i].exec = shl_ref_segment_prod_quant;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_PROD][i].exec = shl_ref_unsorted_segment_prod_quant;
        cb_map[CSINN_OP_SEGMENT_SUM][i].exec = shl_ref_segment_sum_quant;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_SUM][i].exec = shl_ref_unsorted_segment_sum_quant;
        cb_map[CSINN_OP_SHUFFLE_CHANNEL][i].exec = shl_ref_shuffle_channel_quant;
        cb_map[CSINN_OP_SIGMOID][i].exec = shl_ref_sigmoid_quant;
        cb_map[CSINN_OP_SIGN][i].exec = shl_ref_sign_quant;
        cb_map[CSINN_OP_SIN][i].exec = shl_ref_sin_quant;
        cb_map[CSINN_OP_SINH][i].exec = shl_ref_sinh_quant;
        cb_map[CSINN_OP_SLICE][i].exec = shl_ref_slice_quant;
        cb_map[CSINN_OP_SOFTMAX][i].exec = shl_ref_softmax_quant;
        cb_map[CSINN_OP_SOFTPLUS][i].exec = shl_ref_softplus_quant;
        cb_map[CSINN_OP_SOFTRELU][i].exec = shl_ref_softrelu_quant;
        cb_map[CSINN_OP_SOFTSIGN][i].exec = shl_ref_softsign_quant;
        cb_map[CSINN_OP_SPACE_TO_BATCH][i].exec = shl_ref_space_to_batch_quant;
        cb_map[CSINN_OP_SPACE_TO_DEPTH][i].exec = shl_ref_space_to_depth_quant;
        cb_map[CSINN_OP_SQRT][i].exec = shl_ref_sqrt_quant;
        cb_map[CSINN_OP_STACK][i].exec = shl_ref_stack_quant;
        cb_map[CSINN_OP_STRIDED_SLICE][i].exec = shl_ref_strided_slice_quant;
        cb_map[CSINN_OP_SUB][i].exec = shl_ref_sub_quant;
        cb_map[CSINN_OP_SUM][i].exec = shl_ref_sum_stride_quant;
        cb_map[CSINN_OP_TAN][i].exec = shl_ref_tan_quant;
        cb_map[CSINN_OP_TANH][i].exec = shl_ref_tanh_quant;
        cb_map[CSINN_OP_THRESHOLD_RELU][i].exec = shl_ref_threshold_relu_quant;
        cb_map[CSINN_OP_TILE][i].exec = shl_ref_tile_quant;
        cb_map[CSINN_OP_TOPK][i].exec = shl_ref_topk_quant;
        cb_map[CSINN_OP_TRANSPOSE][i].exec = shl_ref_transpose;
        cb_map[CSINN_OP_TRANSPOSE][i].init = shl_ref_transpose_init;
        cb_map[CSINN_OP_TRUNC][i].exec = shl_ref_trunc_quant;
        cb_map[CSINN_OP_UNPOOLING][i].exec = shl_ref_unpooling_quant;
        cb_map[CSINN_OP_YUV_RGB_SCALE][i].exec = shl_ref_yuv_rgb_scale_quant;
        cb_map[CSINN_OP_CONV2D][i].exec = shl_ref_conv2d_quant;
        cb_map[CSINN_OP_CONV2D_RELU][i].exec = shl_ref_conv2d_relu_quant;
        cb_map[CSINN_OP_CONV2D_RELU6][i].exec = shl_ref_conv2d_relu6_quant;
        cb_map[CSINN_OP_CONV2D_CHANNEL][i].exec = shl_ref_conv2d_channel_quant;
        cb_map[CSINN_OP_CONV2D_CHANNEL_RELU][i].exec = shl_ref_conv2d_channel_relu_quant;
        cb_map[CSINN_OP_CONV2D_CHANNEL_RELU6][i].exec = shl_ref_conv2d_channel_relu6_quant;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D][i].exec = shl_ref_depthwise_conv2d_quant;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D_RELU][i].exec = shl_ref_depthwise_conv2d_relu_quant;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D_RELU6][i].exec = shl_ref_depthwise_conv2d_relu6_quant;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D_CHANNEL][i].exec = shl_ref_depthwise_conv2d_channel_quant;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU][i].exec =
            shl_ref_depthwise_conv2d_channel_relu_quant;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6][i].exec =
            shl_ref_depthwise_conv2d_channel_relu6_quant;
        cb_map[CSINN_OP_GROUP_CONV2D][i].exec = shl_ref_group_conv2d_quant;
        cb_map[CSINN_OP_GROUP_CONV2D_RELU][i].exec = shl_ref_group_conv2d_relu_quant;
        cb_map[CSINN_OP_GROUP_CONV2D_RELU6][i].exec = shl_ref_group_conv2d_relu6_quant;
        cb_map[CSINN_OP_GROUP_CONV2D_CHANNEL][i].exec = shl_ref_group_conv2d_channel_quant;
        cb_map[CSINN_OP_GROUP_CONV2D_CHANNEL_RELU][i].exec =
            shl_ref_group_conv2d_channel_relu_quant;
        cb_map[CSINN_OP_CONV3D][i].exec = shl_ref_conv3d_quant;
        cb_map[CSINN_OP_DECONV2D][i].exec = shl_ref_deconv2d_quant;
        cb_map[CSINN_OP_DEPTHWISE_DECONV2D][i].exec = shl_ref_depthwise_deconv2d_quant;
        cb_map[CSINN_OP_DECONV3D][i].exec = shl_ref_deconv3d_quant;
        cb_map[CSINN_OP_FULLYCONNECTED][i].exec = shl_ref_fullyconnected_quant;
        cb_map[CSINN_OP_SCATTER_ND][i].exec = shl_ref_scatter_nd_quant;
        cb_map[CSINN_OP_SPLIT][i].exec = shl_ref_split_quant;
    }

    for (int i = CSINN_DTYPE_UINT8; i <= CSINN_DTYPE_FLOAT64; i++) {
        cb_map[CSINN_OP_SQUEEZE][i].exec = shl_ref_squeeze;
    }

    cb_map[CSINN_OP_AND][CSINN_DTYPE_UINT8].exec = shl_ref_and_u8;
    cb_map[CSINN_OP_AND][CSINN_DTYPE_INT8].exec = shl_ref_and_i8;
    cb_map[CSINN_OP_AND][CSINN_DTYPE_UINT32].exec = shl_ref_and_u32;
    cb_map[CSINN_OP_NDARRAY_SIZE][CSINN_DTYPE_UINT8].exec = shl_ref_ndarray_size_u8;
    cb_map[CSINN_OP_NDARRAY_SIZE][CSINN_DTYPE_INT8].exec = shl_ref_ndarray_size_i8;
    cb_map[CSINN_OP_NDARRAY_SIZE][CSINN_DTYPE_INT32].exec = shl_ref_ndarray_size_i32;
    cb_map[CSINN_OP_NDARRAY_SIZE][CSINN_DTYPE_FLOAT32].exec = shl_ref_ndarray_size_f32;
    cb_map[CSINN_OP_NOT][CSINN_DTYPE_UINT8].exec = shl_ref_not_u8;
    cb_map[CSINN_OP_NOT][CSINN_DTYPE_INT8].exec = shl_ref_not_i8;
    cb_map[CSINN_OP_NOT][CSINN_DTYPE_UINT32].exec = shl_ref_not_u32;
    cb_map[CSINN_OP_OR][CSINN_DTYPE_UINT8].exec = shl_ref_or_u8;
    cb_map[CSINN_OP_OR][CSINN_DTYPE_INT8].exec = shl_ref_or_i8;
    cb_map[CSINN_OP_OR][CSINN_DTYPE_UINT32].exec = shl_ref_or_u32;
    cb_map[CSINN_OP_SELECT][CSINN_DTYPE_UINT8].exec = shl_ref_select_u8;
    cb_map[CSINN_OP_SELECT][CSINN_DTYPE_INT8].exec = shl_ref_select_i8;
    cb_map[CSINN_OP_SELECT][CSINN_DTYPE_FLOAT32].exec = shl_ref_select_f32;
    cb_map[CSINN_OP_SHAPE][CSINN_DTYPE_UINT8].exec = shl_ref_shape_u8;
    cb_map[CSINN_OP_SHAPE][CSINN_DTYPE_INT8].exec = shl_ref_shape_i8;
    cb_map[CSINN_OP_SHAPE][CSINN_DTYPE_INT32].exec = shl_ref_shape_i32;
    cb_map[CSINN_OP_XOR][CSINN_DTYPE_UINT8].exec = shl_ref_xor_u8;
    cb_map[CSINN_OP_XOR][CSINN_DTYPE_INT8].exec = shl_ref_xor_i8;
    cb_map[CSINN_OP_XOR][CSINN_DTYPE_UINT32].exec = shl_ref_xor_u32;

    cb_map[CSINN_OP_ABS][CSINN_DTYPE_FLOAT32].exec = shl_ref_abs_f32;
    cb_map[CSINN_OP_ACOS][CSINN_DTYPE_FLOAT32].exec = shl_ref_acos_f32;
    cb_map[CSINN_OP_ACOSH][CSINN_DTYPE_FLOAT32].exec = shl_ref_acosh_f32;
    cb_map[CSINN_OP_ADD][CSINN_DTYPE_FLOAT32].exec = shl_ref_add_f32;
    cb_map[CSINN_OP_ARANGE][CSINN_DTYPE_FLOAT32].exec = shl_ref_arange_f32;
    cb_map[CSINN_OP_ARGMAX][CSINN_DTYPE_FLOAT32].exec = shl_ref_argmax_stride_i32_f32;
    cb_map[CSINN_OP_ARGMIN][CSINN_DTYPE_FLOAT32].exec = shl_ref_argmin_stride_i32_f32;
    cb_map[CSINN_OP_ASIN][CSINN_DTYPE_FLOAT32].exec = shl_ref_asin_f32;
    cb_map[CSINN_OP_ASINH][CSINN_DTYPE_FLOAT32].exec = shl_ref_asinh_f32;
    cb_map[CSINN_OP_ATAN][CSINN_DTYPE_FLOAT32].exec = shl_ref_atan_f32;
    cb_map[CSINN_OP_ATANH][CSINN_DTYPE_FLOAT32].exec = shl_ref_atanh_f32;
    cb_map[CSINN_OP_AVGPOOL2D][CSINN_DTYPE_FLOAT32].exec = shl_ref_avgpool2d_f32;
    cb_map[CSINN_OP_AVGPOOL3D][CSINN_DTYPE_FLOAT32].exec = shl_ref_avgpool3d_f32;
    cb_map[CSINN_OP_BN][CSINN_DTYPE_FLOAT32].exec = shl_ref_batch_normalization_f32;
    cb_map[CSINN_OP_BATCH_TO_SPACE][CSINN_DTYPE_FLOAT32].exec = shl_ref_batch_to_space_f32;
    cb_map[CSINN_OP_BROADCOST][CSINN_DTYPE_FLOAT32].exec = shl_ref_broadcast_to_f32;
    cb_map[CSINN_OP_CACHE_MATMUL][CSINN_DTYPE_FLOAT32].exec = shl_ref_cache_matmul_f32;
    cb_map[CSINN_OP_CACHE_MATMUL][CSINN_DTYPE_FLOAT32].init = shl_ref_cache_matmul_init;
    cb_map[CSINN_OP_CACHE_CONV1D][CSINN_DTYPE_FLOAT32].exec = shl_ref_cache_conv1d_f32;
    cb_map[CSINN_OP_CACHE_CONV1D][CSINN_DTYPE_FLOAT32].init = shl_ref_cache_conv1d_init;
    cb_map[CSINN_OP_CEIL][CSINN_DTYPE_FLOAT32].exec = shl_ref_ceil_f32;
    cb_map[CSINN_OP_CLIP][CSINN_DTYPE_FLOAT32].exec = shl_ref_clip_f32;
    cb_map[CSINN_OP_CONCAT][CSINN_DTYPE_FLOAT32].exec = shl_ref_concat_f32;
    cb_map[CSINN_OP_CONV2D][CSINN_DTYPE_FLOAT32].exec = shl_ref_conv2d_f32;
    cb_map[CSINN_OP_DEPTHWISE_CONV2D][CSINN_DTYPE_FLOAT32].exec = shl_ref_depthwise_conv2d_f32;
    cb_map[CSINN_OP_GROUP_CONV2D][CSINN_DTYPE_FLOAT32].exec = shl_ref_group_conv2d_f32;
    cb_map[CSINN_OP_CONV3D][CSINN_DTYPE_FLOAT32].exec = shl_ref_conv3d_f32;
    cb_map[CSINN_OP_DECONV2D][CSINN_DTYPE_FLOAT32].exec = shl_ref_deconv2d_f32;
    cb_map[CSINN_OP_DEPTHWISE_DECONV2D][CSINN_DTYPE_FLOAT32].exec = shl_ref_depthwise_deconv2d_f32;
    cb_map[CSINN_OP_DECONV3D][CSINN_DTYPE_FLOAT32].exec = shl_ref_deconv3d_f32;
    cb_map[CSINN_OP_COS][CSINN_DTYPE_FLOAT32].exec = shl_ref_cos_f32;
    cb_map[CSINN_OP_COSH][CSINN_DTYPE_FLOAT32].exec = shl_ref_cosh_f32;
    cb_map[CSINN_OP_CUMPROD][CSINN_DTYPE_FLOAT32].exec = shl_ref_cumprod_f32;
    cb_map[CSINN_OP_CUMSUM][CSINN_DTYPE_FLOAT32].exec = shl_ref_cumsum_f32;
    cb_map[CSINN_OP_DEPTH_TO_SPACE][CSINN_DTYPE_FLOAT32].exec = shl_ref_depth_to_space_f32;
    cb_map[CSINN_OP_DIV][CSINN_DTYPE_FLOAT32].exec = shl_ref_div_f32;
    cb_map[CSINN_OP_ELU][CSINN_DTYPE_FLOAT32].exec = shl_ref_elu_f32;
    cb_map[CSINN_OP_EQUANL][CSINN_DTYPE_FLOAT32].exec = shl_ref_equal_f32;
    cb_map[CSINN_OP_ERF][CSINN_DTYPE_FLOAT32].exec = shl_ref_erf_f32;
    cb_map[CSINN_OP_EXP][CSINN_DTYPE_FLOAT32].exec = shl_ref_exp_f32;
    cb_map[CSINN_OP_EXPAND_DIMS][CSINN_DTYPE_FLOAT32].exec = shl_ref_expand_dims_f32;
    cb_map[CSINN_OP_EXPM1][CSINN_DTYPE_FLOAT32].exec = shl_ref_expm1_f32;
    cb_map[CSINN_OP_FLATTEN][CSINN_DTYPE_FLOAT32].exec = shl_ref_flatten;
    cb_map[CSINN_OP_FLATTEN][CSINN_DTYPE_FLOAT32].init = shl_ref_flatten_init;
    cb_map[CSINN_OP_FLOOR_DIVIDE][CSINN_DTYPE_FLOAT32].exec = shl_ref_floor_divide_f32;
    cb_map[CSINN_OP_FLOOR_MOD][CSINN_DTYPE_FLOAT32].exec = shl_ref_floor_mod_f32;
    cb_map[CSINN_OP_FLOOR][CSINN_DTYPE_FLOAT32].exec = shl_ref_floor_f32;
    cb_map[CSINN_OP_FSMN][CSINN_DTYPE_FLOAT32].exec = shl_ref_fsmn_f32;
    cb_map[CSINN_OP_FULLYCONNECTED][CSINN_DTYPE_FLOAT32].exec = shl_ref_fullyconnected_f32;
    cb_map[CSINN_OP_GATHER_ND][CSINN_DTYPE_FLOAT32].exec = shl_ref_gather_nd_f32;
    cb_map[CSINN_OP_GATHER][CSINN_DTYPE_FLOAT32].exec = shl_ref_gather_f32;
    cb_map[CSINN_OP_GLOBAL_AVGPOOL2D][CSINN_DTYPE_FLOAT32].exec = shl_ref_global_avgpool2d_f32;
    cb_map[CSINN_OP_GLOBAL_MAXPOOL2D][CSINN_DTYPE_FLOAT32].exec = shl_ref_global_maxpool2d_f32;
    cb_map[CSINN_OP_GREATHER_EQUAL][CSINN_DTYPE_FLOAT32].exec = shl_ref_greater_equal_f32;
    cb_map[CSINN_OP_GREATHER][CSINN_DTYPE_FLOAT32].exec = shl_ref_greater_f32;
    cb_map[CSINN_OP_HARD_SIGMOID][CSINN_DTYPE_FLOAT32].exec = shl_ref_hard_sigmoid_f32;
    cb_map[CSINN_OP_IM2COL][CSINN_DTYPE_FLOAT32].exec = shl_ref_im2col_f32;
    cb_map[CSINN_OP_L2N][CSINN_DTYPE_FLOAT32].exec = shl_ref_l2_normalization_f32;
    cb_map[CSINN_OP_LEAKY_RELU][CSINN_DTYPE_FLOAT32].exec = shl_ref_leaky_relu_f32;
    cb_map[CSINN_OP_LESS_EQUAL][CSINN_DTYPE_FLOAT32].exec = shl_ref_less_equal_f32;
    cb_map[CSINN_OP_LESS][CSINN_DTYPE_FLOAT32].exec = shl_ref_less_f32;
    cb_map[CSINN_OP_LOG_SOFTMAX][CSINN_DTYPE_FLOAT32].exec = shl_ref_log_softmax_f32;
    cb_map[CSINN_OP_LOG][CSINN_DTYPE_FLOAT32].exec = shl_ref_log_f32;
    cb_map[CSINN_OP_LOG1P][CSINN_DTYPE_FLOAT32].exec = shl_ref_log1p_f32;
    cb_map[CSINN_OP_LOGICAL_AND][CSINN_DTYPE_FLOAT32].exec = shl_ref_logical_and_f32;
    cb_map[CSINN_OP_LOGICAL_NOT][CSINN_DTYPE_FLOAT32].exec = shl_ref_logical_not_f32;
    cb_map[CSINN_OP_LOGICAL_OR][CSINN_DTYPE_FLOAT32].exec = shl_ref_logical_or_f32;
    cb_map[CSINN_OP_LOGICAL_XOR][CSINN_DTYPE_FLOAT32].exec = shl_ref_logical_xor_f32;
    cb_map[CSINN_OP_LRN][CSINN_DTYPE_FLOAT32].exec = shl_ref_lrn_f32;
    cb_map[CSINN_OP_MATMUL][CSINN_DTYPE_FLOAT32].exec = shl_ref_matmul_f32;
    cb_map[CSINN_OP_MAX][CSINN_DTYPE_FLOAT32].exec = shl_ref_max_stride_f32;
    cb_map[CSINN_OP_MAXIMUM][CSINN_DTYPE_FLOAT32].exec = shl_ref_maximum_f32;
    cb_map[CSINN_OP_MAXPOOL2D][CSINN_DTYPE_FLOAT32].exec = shl_ref_maxpool2d_f32;
    cb_map[CSINN_OP_MAXPOOL2D_LOCAT][CSINN_DTYPE_FLOAT32].exec = shl_ref_maxpool2d_locat_f32;
    cb_map[CSINN_OP_MAXPOOL3D][CSINN_DTYPE_FLOAT32].exec = shl_ref_maxpool3d_f32;
    cb_map[CSINN_OP_MEAN][CSINN_DTYPE_FLOAT32].exec = shl_ref_mean_stride_f32;
    cb_map[CSINN_OP_MEAN_STRIDE][CSINN_DTYPE_FLOAT32].exec = shl_ref_mean_stride_f32;
    cb_map[CSINN_OP_MIN][CSINN_DTYPE_FLOAT32].exec = shl_ref_min_stride_f32;
    cb_map[CSINN_OP_MINIMUM][CSINN_DTYPE_FLOAT32].exec = shl_ref_minimum_f32;
    cb_map[CSINN_OP_MOD][CSINN_DTYPE_FLOAT32].exec = shl_ref_mod_f32;
    cb_map[CSINN_OP_MUL][CSINN_DTYPE_FLOAT32].exec = shl_ref_mul_f32;
    cb_map[CSINN_OP_NEGATIIVE][CSINN_DTYPE_FLOAT32].exec = shl_ref_negative_f32;
    cb_map[CSINN_OP_NON_MAX_SUPPRESSION][CSINN_DTYPE_FLOAT32].exec =
        shl_ref_non_max_suppression_std;
    cb_map[CSINN_OP_NOT_EQUAL][CSINN_DTYPE_FLOAT32].exec = shl_ref_not_equal_f32;
    cb_map[CSINN_OP_PAD][CSINN_DTYPE_FLOAT32].exec = shl_ref_pad_f32;
    cb_map[CSINN_OP_POWER][CSINN_DTYPE_FLOAT32].exec = shl_ref_power_f32;
    cb_map[CSINN_OP_PRELU][CSINN_DTYPE_FLOAT32].exec = shl_ref_prelu_f32;
    cb_map[CSINN_OP_PROD][CSINN_DTYPE_FLOAT32].exec = shl_ref_prod_stride_f32;
    cb_map[CSINN_OP_PROPOSAL][CSINN_DTYPE_FLOAT32].exec = shl_ref_proposal_f32;
    cb_map[CSINN_OP_PSROIPOOLING][CSINN_DTYPE_FLOAT32].exec = shl_ref_psroipooling_f32;
    cb_map[CSINN_OP_REDUCE_LOGSUMEXP][CSINN_DTYPE_FLOAT32].exec = shl_ref_reduce_logsumexp_f32;
    cb_map[CSINN_OP_REDUCE_MAX][CSINN_DTYPE_FLOAT32].exec = shl_ref_reduce_max_f32;
    cb_map[CSINN_OP_REDUCE_MEAN][CSINN_DTYPE_FLOAT32].exec = shl_ref_reduce_mean_f32;
    cb_map[CSINN_OP_REDUCE_MIN][CSINN_DTYPE_FLOAT32].exec = shl_ref_reduce_min_f32;
    cb_map[CSINN_OP_REDUCE_PROD][CSINN_DTYPE_FLOAT32].exec = shl_ref_reduce_prod_f32;
    cb_map[CSINN_OP_REDUCE_SUM][CSINN_DTYPE_FLOAT32].exec = shl_ref_reduce_sum_f32;
    cb_map[CSINN_OP_RELU][CSINN_DTYPE_FLOAT32].exec = shl_ref_relu_f32;
    cb_map[CSINN_OP_RELU1][CSINN_DTYPE_FLOAT32].exec = shl_ref_relu1_f32;
    cb_map[CSINN_OP_RELU6][CSINN_DTYPE_FLOAT32].exec = shl_ref_relu6_f32;
    cb_map[CSINN_OP_RELUN][CSINN_DTYPE_FLOAT32].exec = shl_ref_relun_f32;
    cb_map[CSINN_OP_RESHAPE][CSINN_DTYPE_FLOAT32].exec = shl_ref_reshape;
    cb_map[CSINN_OP_RESHAPE][CSINN_DTYPE_FLOAT32].init = shl_ref_reshape_init;
    cb_map[CSINN_OP_RESIZE][CSINN_DTYPE_FLOAT32].exec = shl_ref_resize_f32;
    cb_map[CSINN_OP_REVERSE][CSINN_DTYPE_FLOAT32].exec = shl_ref_reverse_f32;
    cb_map[CSINN_OP_ROIALIGN][CSINN_DTYPE_FLOAT32].exec = shl_ref_roi_align_f32;
    cb_map[CSINN_OP_ROIPOOL][CSINN_DTYPE_FLOAT32].exec = shl_ref_roipool_f32;
    cb_map[CSINN_OP_ROUND][CSINN_DTYPE_FLOAT32].exec = shl_ref_round_f32;
    cb_map[CSINN_OP_RSQRT][CSINN_DTYPE_FLOAT32].exec = shl_ref_rsqrt_f32;
    cb_map[CSINN_OP_SCATTER_ND][CSINN_DTYPE_FLOAT32].exec = shl_ref_scatter_nd_f32;
    cb_map[CSINN_OP_SEGMENT_MAX][CSINN_DTYPE_FLOAT32].exec = shl_ref_segment_max_f32;
    cb_map[CSINN_OP_UNSORTED_SEGMENT_MAX][CSINN_DTYPE_FLOAT32].exec =
        shl_ref_unsorted_segment_max_f32;
    cb_map[CSINN_OP_SEGMENT_MEAN][CSINN_DTYPE_FLOAT32].exec = shl_ref_segment_mean_f32;
    cb_map[CSINN_OP_UNSORTED_SEGMENT_MEAN][CSINN_DTYPE_FLOAT32].exec =
        shl_ref_unsorted_segment_mean_f32;
    cb_map[CSINN_OP_SEGMENT_MIN][CSINN_DTYPE_FLOAT32].exec = shl_ref_segment_min_f32;
    cb_map[CSINN_OP_UNSORTED_SEGMENT_MIN][CSINN_DTYPE_FLOAT32].exec =
        shl_ref_unsorted_segment_min_f32;
    cb_map[CSINN_OP_SEGMENT_PROD][CSINN_DTYPE_FLOAT32].exec = shl_ref_segment_prod_f32;
    cb_map[CSINN_OP_UNSORTED_SEGMENT_PROD][CSINN_DTYPE_FLOAT32].exec =
        shl_ref_unsorted_segment_prod_f32;
    cb_map[CSINN_OP_SEGMENT_SUM][CSINN_DTYPE_FLOAT32].exec = shl_ref_segment_sum_f32;
    cb_map[CSINN_OP_UNSORTED_SEGMENT_SUM][CSINN_DTYPE_FLOAT32].exec =
        shl_ref_unsorted_segment_sum_f32;
    cb_map[CSINN_OP_SHUFFLE_CHANNEL][CSINN_DTYPE_FLOAT32].exec = shl_ref_shuffle_channel_f32;
    cb_map[CSINN_OP_SIGMOID][CSINN_DTYPE_FLOAT32].exec = shl_ref_sigmoid_f32;
    cb_map[CSINN_OP_SIGN][CSINN_DTYPE_FLOAT32].exec = shl_ref_sign_f32;
    cb_map[CSINN_OP_SIN][CSINN_DTYPE_FLOAT32].exec = shl_ref_sin_f32;
    cb_map[CSINN_OP_SINH][CSINN_DTYPE_FLOAT32].exec = shl_ref_sinh_f32;
    cb_map[CSINN_OP_SLICE][CSINN_DTYPE_FLOAT32].exec = shl_ref_slice_f32;
    cb_map[CSINN_OP_SOFTMAX][CSINN_DTYPE_FLOAT32].exec = shl_ref_softmax_f32;
    cb_map[CSINN_OP_SOFTPLUS][CSINN_DTYPE_FLOAT32].exec = shl_ref_softplus_f32;
    cb_map[CSINN_OP_SOFTRELU][CSINN_DTYPE_FLOAT32].exec = shl_ref_softrelu_f32;
    cb_map[CSINN_OP_SOFTSIGN][CSINN_DTYPE_FLOAT32].exec = shl_ref_softsign_f32;
    cb_map[CSINN_OP_SPACE_TO_BATCH][CSINN_DTYPE_FLOAT32].exec = shl_ref_space_to_batch_f32;
    cb_map[CSINN_OP_SPACE_TO_DEPTH][CSINN_DTYPE_FLOAT32].exec = shl_ref_space_to_depth_f32;
    cb_map[CSINN_OP_SPLIT][CSINN_DTYPE_FLOAT32].exec = shl_ref_split_f32;
    cb_map[CSINN_OP_SQRT][CSINN_DTYPE_FLOAT32].exec = shl_ref_sqrt_f32;
    cb_map[CSINN_OP_SQUARE][CSINN_DTYPE_FLOAT32].exec = shl_ref_square_f32;
    cb_map[CSINN_OP_STACK][CSINN_DTYPE_FLOAT32].exec = shl_ref_stack_f32;
    cb_map[CSINN_OP_STRIDED_SLICE][CSINN_DTYPE_FLOAT32].exec = shl_ref_strided_slice_f32;
    cb_map[CSINN_OP_SUB][CSINN_DTYPE_FLOAT32].exec = shl_ref_sub_f32;
    cb_map[CSINN_OP_SUM][CSINN_DTYPE_FLOAT32].exec = shl_ref_sum_stride_f32;
    cb_map[CSINN_OP_TAN][CSINN_DTYPE_FLOAT32].exec = shl_ref_tan_f32;
    cb_map[CSINN_OP_TANH][CSINN_DTYPE_FLOAT32].exec = shl_ref_tanh_f32;
    cb_map[CSINN_OP_THRESHOLD_RELU][CSINN_DTYPE_FLOAT32].exec = shl_ref_threshold_relu_f32;
    cb_map[CSINN_OP_TILE][CSINN_DTYPE_FLOAT32].exec = shl_ref_tile_f32;
    cb_map[CSINN_OP_TOPK][CSINN_DTYPE_FLOAT32].exec = shl_ref_topk_f32;
    cb_map[CSINN_OP_TRANSPOSE][CSINN_DTYPE_FLOAT32].exec = shl_ref_transpose;
    cb_map[CSINN_OP_TRANSPOSE][CSINN_DTYPE_FLOAT32].init = shl_ref_transpose_init;
    cb_map[CSINN_OP_TRUNC][CSINN_DTYPE_FLOAT32].exec = shl_ref_trunc_f32;
    cb_map[CSINN_OP_UNPOOLING][CSINN_DTYPE_FLOAT32].exec = shl_ref_unpooling_f32;
    cb_map[CSINN_OP_YUV_RGB_SCALE][CSINN_DTYPE_FLOAT32].exec = shl_ref_yuv_rgb_scale_f32;
    cb_map[CSINN_OP_COL2IM][CSINN_DTYPE_FLOAT32].exec = shl_ref_col2im_f32;
    cb_map[CSINN_OP_ISNAN][CSINN_DTYPE_FLOAT32].exec = shl_ref_isnan_bool_f32;
    cb_map[CSINN_OP_L2POOL2D][CSINN_DTYPE_FLOAT32].exec = shl_ref_l2pool_f32;

#ifdef SHL_BUILD_GREF
#include "shl_gref.h"
    shl_register_runtime_callback(CSINN_REF, shl_gref_runtime_callback);
    for (int i = 0; i < CSINN_DTYPE_SIZE; i++) {
        cb_map[CSINN_OP_ABS][i].est = shl_gref_abs;
        cb_map[CSINN_OP_ACOS][i].est = shl_gref_acos;
        cb_map[CSINN_OP_ACOSH][i].est = shl_gref_acosh;
        cb_map[CSINN_OP_ADD][i].est = shl_gref_add;
        cb_map[CSINN_OP_ARANGE][i].est = shl_gref_arange;
        cb_map[CSINN_OP_ARGMAX][i].est = shl_gref_argmax;
        cb_map[CSINN_OP_ARGMIN][i].est = shl_gref_argmin;
        cb_map[CSINN_OP_ASIN][i].est = shl_gref_asin;
        cb_map[CSINN_OP_ASINH][i].est = shl_gref_asinh;
        cb_map[CSINN_OP_ATAN][i].est = shl_gref_atan;
        cb_map[CSINN_OP_ATANH][i].est = shl_gref_atanh;
        cb_map[CSINN_OP_AVGPOOL2D][i].est = shl_gref_avgpool2d;
        cb_map[CSINN_OP_AVGPOOL3D][i].est = shl_gref_avgpool3d;
        cb_map[CSINN_OP_BN][i].est = shl_gref_batch_normalization;
        cb_map[CSINN_OP_BATCH_TO_SPACE][i].est = shl_gref_batch_to_space;
        cb_map[CSINN_OP_BROADCOST][i].est = shl_gref_broadcast_to;
        cb_map[CSINN_OP_CACHE_MATMUL][i].est = shl_gref_cache_matmul;
        cb_map[CSINN_OP_CACHE_CONV1D][i].est = shl_gref_cache_conv1d;
        cb_map[CSINN_OP_CEIL][i].est = shl_gref_ceil;
        cb_map[CSINN_OP_CLIP][i].est = shl_gref_clip;
        cb_map[CSINN_OP_CONCAT][i].est = shl_gref_concat;
        cb_map[CSINN_OP_COS][i].est = shl_gref_cos;
        cb_map[CSINN_OP_COSH][i].est = shl_gref_cosh;
        cb_map[CSINN_OP_CUMPROD][i].est = shl_gref_cumprod;
        cb_map[CSINN_OP_DATA_CONVERT][i].est = shl_gref_data_convert;
        cb_map[CSINN_OP_CUMSUM][i].est = shl_gref_cumsum;
        cb_map[CSINN_OP_DEPTH_TO_SPACE][i].est = shl_gref_depth_to_space;
        cb_map[CSINN_OP_DIV][i].est = shl_gref_div;
        cb_map[CSINN_OP_ELU][i].est = shl_gref_elu;
        cb_map[CSINN_OP_EQUANL][i].est = shl_gref_equal;
        cb_map[CSINN_OP_ERF][i].est = shl_gref_erf;
        cb_map[CSINN_OP_EXP][i].est = shl_gref_exp;
        cb_map[CSINN_OP_EXPAND_DIMS][i].est = shl_gref_expand_dims;
        cb_map[CSINN_OP_EXPM1][i].est = shl_gref_expm1;
        cb_map[CSINN_OP_FLATTEN][i].est = shl_gref_flatten;
        cb_map[CSINN_OP_FLOOR_DIVIDE][i].est = shl_gref_floor_divide;
        cb_map[CSINN_OP_FLOOR_MOD][i].est = shl_gref_floor_mod;
        cb_map[CSINN_OP_FLOOR][i].est = shl_gref_floor;
        cb_map[CSINN_OP_FSMN][i].est = shl_gref_fsmn;
        cb_map[CSINN_OP_GATHER_ND][i].est = shl_gref_gather_nd;
        cb_map[CSINN_OP_GATHER][i].est = shl_gref_gather;
        cb_map[CSINN_OP_GLOBAL_AVGPOOL2D][i].est = shl_gref_global_avgpool2d;
        cb_map[CSINN_OP_GLOBAL_MAXPOOL2D][i].est = shl_gref_global_maxpool2d;
        cb_map[CSINN_OP_GREATHER_EQUAL][i].est = shl_gref_greater_equal;
        cb_map[CSINN_OP_GREATHER][i].est = shl_gref_greater;
        cb_map[CSINN_OP_HARD_SIGMOID][i].est = shl_gref_hard_sigmoid;
        cb_map[CSINN_OP_IM2COL][i].est = shl_gref_im2col;
        cb_map[CSINN_OP_L2N][i].est = shl_gref_l2_normalization;
        cb_map[CSINN_OP_LEAKY_RELU][i].est = shl_gref_leaky_relu;
        cb_map[CSINN_OP_LESS_EQUAL][i].est = shl_gref_less_equal;
        cb_map[CSINN_OP_LESS][i].est = shl_gref_less;
        cb_map[CSINN_OP_LOG_SOFTMAX][i].est = shl_gref_log_softmax;
        cb_map[CSINN_OP_LOG][i].est = shl_gref_log;
        cb_map[CSINN_OP_LOG1P][i].est = shl_gref_log1p;
        cb_map[CSINN_OP_LOGICAL_AND][i].est = shl_gref_logical_and;
        cb_map[CSINN_OP_LOGICAL_NOT][i].est = shl_gref_logical_not;
        cb_map[CSINN_OP_LOGICAL_OR][i].est = shl_gref_logical_or;
        cb_map[CSINN_OP_LOGICAL_XOR][i].est = shl_gref_logical_xor;
        cb_map[CSINN_OP_LRN][i].est = shl_gref_lrn;
        cb_map[CSINN_OP_MATMUL][i].est = shl_gref_matmul;
        cb_map[CSINN_OP_MAX][i].est = shl_gref_max;
        cb_map[CSINN_OP_MAXIMUM][i].est = shl_gref_maximum;
        cb_map[CSINN_OP_MAXPOOL2D][i].est = shl_gref_maxpool2d;
        cb_map[CSINN_OP_MAXPOOL2D_LOCAT][i].est = shl_gref_maxpool2d_locat;
        cb_map[CSINN_OP_MAXPOOL3D][i].est = shl_gref_maxpool3d;
        cb_map[CSINN_OP_MEAN][i].est = shl_gref_mean;
        cb_map[CSINN_OP_MEAN_STRIDE][i].est = shl_gref_mean;
        cb_map[CSINN_OP_MIN][i].est = shl_gref_min;
        cb_map[CSINN_OP_MINIMUM][i].est = shl_gref_minimum;
        cb_map[CSINN_OP_MOD][i].est = shl_gref_mod;
        cb_map[CSINN_OP_MUL][i].est = shl_gref_mul;
        cb_map[CSINN_OP_NEGATIIVE][i].est = shl_gref_negative;
        cb_map[CSINN_OP_NOT_EQUAL][i].est = shl_gref_not_equal;
        cb_map[CSINN_OP_PAD][i].est = shl_gref_pad;
        cb_map[CSINN_OP_POWER][i].est = shl_gref_power;
        cb_map[CSINN_OP_PRELU][i].est = shl_gref_prelu;
        cb_map[CSINN_OP_PROD][i].est = shl_gref_prod;
        cb_map[CSINN_OP_PROPOSAL][i].est = shl_gref_proposal;
        cb_map[CSINN_OP_PSROIPOOLING][i].est = shl_gref_psroipooling;
        cb_map[CSINN_OP_REDUCE_LOGSUMEXP][i].est = shl_gref_reduce_logsumexp;
        cb_map[CSINN_OP_REDUCE_MAX][i].est = shl_gref_reduce_max;
        cb_map[CSINN_OP_REDUCE_MEAN][i].est = shl_gref_reduce_mean;
        cb_map[CSINN_OP_REDUCE_MIN][i].est = shl_gref_reduce_min;
        cb_map[CSINN_OP_REDUCE_PROD][i].est = shl_gref_reduce_prod;
        cb_map[CSINN_OP_REDUCE_SUM][i].est = shl_gref_reduce_sum;
        cb_map[CSINN_OP_RELU][i].est = shl_gref_relu;
        cb_map[CSINN_OP_RELU1][i].est = shl_gref_relu1;
        cb_map[CSINN_OP_RELU6][i].est = shl_gref_relu6;
        cb_map[CSINN_OP_RELUN][i].est = shl_gref_relun;
        cb_map[CSINN_OP_RESHAPE][i].est = shl_gref_reshape;
        cb_map[CSINN_OP_RESIZE][i].est = shl_gref_resize;
        cb_map[CSINN_OP_REVERSE][i].est = shl_gref_reverse;
        cb_map[CSINN_OP_ROIPOOL][i].est = shl_gref_roipool;
        cb_map[CSINN_OP_ROUND][i].est = shl_gref_round;
        cb_map[CSINN_OP_RSQRT][i].est = shl_gref_rsqrt;
        cb_map[CSINN_OP_SEGMENT_MAX][i].est = shl_gref_segment_max;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_MAX][i].est = shl_gref_segment_max;
        cb_map[CSINN_OP_SEGMENT_MEAN][i].est = shl_gref_segment_mean;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_MEAN][i].est = shl_gref_segment_mean;
        cb_map[CSINN_OP_SEGMENT_MIN][i].est = shl_gref_segment_min;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_MIN][i].est = shl_gref_segment_min;
        cb_map[CSINN_OP_SEGMENT_PROD][i].est = shl_gref_segment_prod;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_PROD][i].est = shl_gref_segment_prod;
        cb_map[CSINN_OP_SEGMENT_SUM][i].est = shl_gref_segment_sum;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_SUM][i].est = shl_gref_segment_sum;
        cb_map[CSINN_OP_SHUFFLE_CHANNEL][i].est = shl_gref_shuffle_channel;
        cb_map[CSINN_OP_SIGMOID][i].est = shl_gref_sigmoid;
        cb_map[CSINN_OP_SIGN][i].est = shl_gref_sign;
        cb_map[CSINN_OP_SIN][i].est = shl_gref_sin;
        cb_map[CSINN_OP_SINH][i].est = shl_gref_sinh;
        cb_map[CSINN_OP_SLICE][i].est = shl_gref_slice;
        cb_map[CSINN_OP_SOFTMAX][i].est = shl_gref_softmax;
        cb_map[CSINN_OP_SOFTPLUS][i].est = shl_gref_softplus;
        cb_map[CSINN_OP_SOFTRELU][i].est = shl_gref_softrelu;
        cb_map[CSINN_OP_SOFTSIGN][i].est = shl_gref_softsign;
        cb_map[CSINN_OP_SPACE_TO_BATCH][i].est = shl_gref_space_to_batch;
        cb_map[CSINN_OP_SPACE_TO_DEPTH][i].est = shl_gref_space_to_depth;
        cb_map[CSINN_OP_SQRT][i].est = shl_gref_sqrt;
        cb_map[CSINN_OP_STACK][i].est = shl_gref_stack;
        cb_map[CSINN_OP_STRIDED_SLICE][i].est = shl_gref_strided_slice;
        cb_map[CSINN_OP_SUB][i].est = shl_gref_sub;
        cb_map[CSINN_OP_SUM][i].est = shl_gref_sum;
        cb_map[CSINN_OP_TAN][i].est = shl_gref_tan;
        cb_map[CSINN_OP_TANH][i].est = shl_gref_tanh;
        cb_map[CSINN_OP_THRESHOLD_RELU][i].est = shl_gref_threshold_relu;
        cb_map[CSINN_OP_TILE][i].est = shl_gref_tile;
        cb_map[CSINN_OP_TOPK][i].est = shl_gref_topk;
        cb_map[CSINN_OP_TRANSPOSE][i].est = shl_gref_transpose;
        cb_map[CSINN_OP_TRUNC][i].est = shl_gref_trunc;
        cb_map[CSINN_OP_UNPOOLING][i].est = shl_gref_unpooling;
        cb_map[CSINN_OP_YUV_RGB_SCALE][i].est = shl_gref_yuv_rgb_scale;
        cb_map[CSINN_OP_CONV2D][i].est = shl_gref_conv2d;
        cb_map[CSINN_OP_CONV2D_RELU][i].est = shl_gref_conv2d_relu;
        cb_map[CSINN_OP_CONV2D_RELU6][i].est = shl_gref_conv2d_relu6;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D][i].est = shl_gref_depthwise_conv2d;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D_RELU][i].est = shl_gref_depthwise_conv2d_relu;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D_RELU6][i].est = shl_gref_depthwise_conv2d_relu6;
        cb_map[CSINN_OP_GROUP_CONV2D][i].est = shl_gref_group_conv2d;
        cb_map[CSINN_OP_CONV3D][i].est = shl_gref_conv3d;
        cb_map[CSINN_OP_DECONV2D][i].est = shl_gref_deconv2d;
        cb_map[CSINN_OP_DEPTHWISE_DECONV2D][i].est = shl_gref_depthwise_deconv2d;
        cb_map[CSINN_OP_DECONV3D][i].est = shl_gref_deconv3d;
        cb_map[CSINN_OP_FULLYCONNECTED][i].est = shl_gref_fullyconnected;
        cb_map[CSINN_OP_SCATTER_ND][i].est = shl_gref_scatter_nd;
        cb_map[CSINN_OP_SPLIT][i].est = shl_gref_split;
    }
#endif
    return cb_map;
}

static int get_cb_map_index(int op, int dtype) { return op * CSINN_DTYPE_SIZE + dtype; }
static struct csinn_callback *__cb_map_table_ref;
struct csinn_callback *shl_cb_map_ref(int op, int dtype)
{
    return &__cb_map_table_ref[get_cb_map_index(op, dtype)];
}

void shl_target_init_ref()
{
    __cb_map_table_ref = setup_cb_map();
    shl_register_runtime_callback(CSINN_REF, NULL);
    shl_register_op_callback(CSINN_REF, shl_cb_map_ref);
}
