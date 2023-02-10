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
                int32_t input_val = nearbyint(input_data[index] / output->qinfo[i].scale) +
                                    output->qinfo[i].zero_point;
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
                int32_t input_val = nearbyint(input_data[index] / output->qinfo[i].scale) +
                                    output->qinfo[i].zero_point;
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
                int32_t input_val = nearbyint(input_data[index] / output->qinfo[i].scale) +
                                    output->qinfo[i].zero_point;
                if (input_val < -128) {
                    input_val = -128;
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

    for (int i = CSINN_DTYPE_INT4; i <= CSINN_DTYPE_FLOAT32; i++) {
#ifndef CONFIG_C_REFERENCE_ABS_DISABLED
        cb_map[CSINN_OP_ABS][i].exec = shl_ref_abs_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ACOS_DISABLED
        cb_map[CSINN_OP_ACOS][i].exec = shl_ref_acos_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ACOSH_DISABLED
        cb_map[CSINN_OP_ACOSH][i].exec = shl_ref_acosh_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ADD_DISABLED
        cb_map[CSINN_OP_ADD][i].exec = shl_ref_add_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ARANGE_DISABLED
        cb_map[CSINN_OP_ARANGE][i].exec = shl_ref_arange_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ARGMAX_DISABLED
        cb_map[CSINN_OP_ARGMAX][i].exec = shl_ref_argmax_stride_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ARGMIN_DISABLED
        cb_map[CSINN_OP_ARGMIN][i].exec = shl_ref_argmin_stride_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ASIN_DISABLED
        cb_map[CSINN_OP_ASIN][i].exec = shl_ref_asin_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ASINH_DISABLED
        cb_map[CSINN_OP_ASINH][i].exec = shl_ref_asinh_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ATAN_DISABLED
        cb_map[CSINN_OP_ATAN][i].exec = shl_ref_atan_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ATANH_DISABLED
        cb_map[CSINN_OP_ATANH][i].exec = shl_ref_atanh_quant;
#endif
#ifndef CONFIG_C_REFERENCE_AVERAGEPOOL_DISABLED
        cb_map[CSINN_OP_AVGPOOL2D][i].exec = shl_ref_avgpool2d_quant;
#endif
#ifndef CONFIG_C_REFERENCE_AVERAGEPOOL3D_DISABLED
        cb_map[CSINN_OP_AVGPOOL3D][i].exec = shl_ref_avgpool3d_quant;
#endif
#ifndef CONFIG_C_REFERENCE_BATCH_NORMALIZATION_DISABLED
        cb_map[CSINN_OP_BN][i].exec = shl_ref_batch_normalization_quant;
#endif
#ifndef CONFIG_C_REFERENCE_BATCH_TO_SPACE_DISABLED
        cb_map[CSINN_OP_BATCH_TO_SPACE][i].exec = shl_ref_batch_to_space_quant;
#endif
#ifndef CONFIG_C_REFERENCE_BROADCAST_TO_DISABLED
        cb_map[CSINN_OP_BROADCOST][i].exec = shl_ref_broadcast_to_quant;
#endif
#ifndef CONFIG_C_REFERENCE_CACHE_MATMUL_DISABLED
        cb_map[CSINN_OP_CACHE_MATMUL][i].exec = shl_ref_cache_matmul_quant;
        cb_map[CSINN_OP_CACHE_MATMUL][i].init = shl_ref_cache_matmul_init;
#endif
#ifndef CONFIG_C_REFERENCE_CACHE_CONV1D_DISABLED
        cb_map[CSINN_OP_CACHE_CONV1D][i].exec = shl_ref_cache_conv1d_quant;
        cb_map[CSINN_OP_CACHE_CONV1D][i].init = shl_ref_cache_conv1d_init;
#endif
#ifndef CONFIG_C_REFERENCE_CONV1D_DISABLED
        cb_map[CSINN_OP_CONV1D][i].exec = shl_ref_conv1d_quant;
        cb_map[CSINN_OP_DEPTHWISE_CONV1D][i].exec = shl_ref_conv1d_quant;
        cb_map[CSINN_OP_GROUP_CONV1D][i].exec = shl_ref_conv1d_quant;
#endif
#ifndef CONFIG_C_REFERENCE_CEIL_DISABLED
        cb_map[CSINN_OP_CEIL][i].exec = shl_ref_ceil_quant;
#endif
#ifndef CONFIG_C_REFERENCE_CLIP_DISABLED
        cb_map[CSINN_OP_CLIP][i].exec = shl_ref_clip_quant;
#endif
#ifndef CONFIG_C_REFERENCE_CONCAT_DISABLED
        cb_map[CSINN_OP_CONCAT][i].exec = shl_ref_concat_quant;
#endif
#ifndef CONFIG_C_REFERENCE_COS_DISABLED
        cb_map[CSINN_OP_COS][i].exec = shl_ref_cos_quant;
#endif
#ifndef CONFIG_C_REFERENCE_COSH_DISABLED
        cb_map[CSINN_OP_COSH][i].exec = shl_ref_cosh_quant;
#endif
#ifndef CONFIG_C_REFERENCE_CUMPROD_DISABLED
        cb_map[CSINN_OP_CUMPROD][i].exec = shl_ref_cumprod_quant;
#endif
#ifndef CONFIG_C_REFERENCE_CUMSUM_DISABLED
        cb_map[CSINN_OP_CUMSUM][i].exec = shl_ref_cumsum_quant;
#endif
#ifndef CONFIG_C_REFERENCE_DEPTH_TO_SPACE_DISABLED
        cb_map[CSINN_OP_DEPTH_TO_SPACE][i].exec = shl_ref_depth_to_space_quant;
#endif
#ifndef CONFIG_C_REFERENCE_DIV_DISABLED
        cb_map[CSINN_OP_DIV][i].exec = shl_ref_div_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ELU_DISABLED
        cb_map[CSINN_OP_ELU][i].exec = shl_ref_elu_quant;
#endif
#ifndef CONFIG_C_REFERENCE_EQUAL_DISABLED
        cb_map[CSINN_OP_EQUANL][i].exec = shl_ref_equal_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ERF_DISABLED
        cb_map[CSINN_OP_ERF][i].exec = shl_ref_erf_quant;
#endif
#ifndef CONFIG_C_REFERENCE_EXP_DISABLED
        cb_map[CSINN_OP_EXP][i].exec = shl_ref_exp_quant;
#endif
#ifndef CONFIG_C_REFERENCE_EXPAND_DIMS_DISABLED
        cb_map[CSINN_OP_EXPAND_DIMS][i].exec = shl_ref_expand_dims_quant;
#endif
#ifndef CONFIG_C_REFERENCE_EXPM1_DISABLED
        cb_map[CSINN_OP_EXPM1][i].exec = shl_ref_expm1_quant;
#endif
#ifndef CONFIG_C_REFERENCE_FLATTEN_DISABLED
        cb_map[CSINN_OP_FLATTEN][i].exec = shl_ref_flatten;
        cb_map[CSINN_OP_FLATTEN][i].init = shl_ref_flatten_init;
#endif
#ifndef CONFIG_C_REFERENCE_FLOOR_DIVIDE_DISABLED
        cb_map[CSINN_OP_FLOOR_DIVIDE][i].exec = shl_ref_floor_divide_quant;
#endif
#ifndef CONFIG_C_REFERENCE_FLOOR_MOD_DISABLED
        cb_map[CSINN_OP_FLOOR_MOD][i].exec = shl_ref_floor_mod_quant;
#endif
#ifndef CONFIG_C_REFERENCE_FLOOR_DISABLED
        cb_map[CSINN_OP_FLOOR][i].exec = shl_ref_floor_quant;
#endif
#ifndef CONFIG_C_REFERENCE_FSMN_DISABLED
        cb_map[CSINN_OP_FSMN][i].exec = shl_ref_fsmn_quant;
#endif
#ifndef CONFIG_C_REFERENCE_GATHER_ND_DISABLED
        cb_map[CSINN_OP_GATHER_ND][i].exec = shl_ref_gather_nd_quant;
#endif
#ifndef CONFIG_C_REFERENCE_GATHER_DISABLED
        cb_map[CSINN_OP_GATHER][i].exec = shl_ref_gather_quant;
#endif
#ifndef CONFIG_C_REFERENCE_GLOBAL_AVERAGEPOOL_DISABLED
        cb_map[CSINN_OP_GLOBAL_AVGPOOL2D][i].exec = shl_ref_global_avgpool2d_quant;
#endif
#ifndef CONFIG_C_REFERENCE_GLOBAL_MAXPOOL_DISABLED
        cb_map[CSINN_OP_GLOBAL_MAXPOOL2D][i].exec = shl_ref_global_maxpool2d_quant;
#endif
#ifndef CONFIG_C_REFERENCE_GREATER_EQUAL_DISABLED
        cb_map[CSINN_OP_GREATHER_EQUAL][i].exec = shl_ref_greater_equal_quant;
#endif
#ifndef CONFIG_C_REFERENCE_GREATER_DISABLED
        cb_map[CSINN_OP_GREATHER][i].exec = shl_ref_greater_quant;
#endif
#ifndef CONFIG_C_REFERENCE_HARD_SIGMOID_DISABLED
        cb_map[CSINN_OP_HARD_SIGMOID][i].exec = shl_ref_hard_sigmoid_quant;
#endif
#ifndef CONFIG_C_REFERENCE_IM2COL_DISABLED
        cb_map[CSINN_OP_IM2COL][i].exec = shl_ref_im2col_quant;
#endif
#ifndef CONFIG_C_REFERENCE_L2_NORMALIZATION_DISABLED
        cb_map[CSINN_OP_L2N][i].exec = shl_ref_l2_normalization_quant;
#endif
#ifndef CONFIG_C_REFERENCE_LAYER_NORM_DISABLED
        cb_map[CSINN_OP_LAYER_NORM][i].exec = shl_ref_layer_norm_quant;
#endif
#ifndef CONFIG_C_REFERENCE_LEAKY_RELU_DISABLED
        cb_map[CSINN_OP_LEAKY_RELU][i].exec = shl_ref_leaky_relu_quant;
#endif
#ifndef CONFIG_C_REFERENCE_LESS_EQUAL_DISABLED
        cb_map[CSINN_OP_LESS_EQUAL][i].exec = shl_ref_less_equal_quant;
#endif
#ifndef CONFIG_C_REFERENCE_LESS_DISABLED
        cb_map[CSINN_OP_LESS][i].exec = shl_ref_less_quant;
#endif
#ifndef CONFIG_C_REFERENCE_LOG_SOFTMAX_DISABLED
        cb_map[CSINN_OP_LOG_SOFTMAX][i].exec = shl_ref_log_softmax_quant;
#endif
#ifndef CONFIG_C_REFERENCE_LOG_DISABLED
        cb_map[CSINN_OP_LOG][i].exec = shl_ref_log_quant;
#endif
#ifndef CONFIG_C_REFERENCE_LOG1P_DISABLED
        cb_map[CSINN_OP_LOG1P][i].exec = shl_ref_log1p_quant;
#endif
#ifndef CONFIG_C_REFERENCE_LOGICAL_AND_DISABLED
        cb_map[CSINN_OP_LOGICAL_AND][i].exec = shl_ref_logical_and_quant;
#endif
#ifndef CONFIG_C_REFERENCE_LOGICAL_NOT_DISABLED
        cb_map[CSINN_OP_LOGICAL_NOT][i].exec = shl_ref_logical_not_quant;
#endif
#ifndef CONFIG_C_REFERENCE_LOGICAL_OR_DISABLED
        cb_map[CSINN_OP_LOGICAL_OR][i].exec = shl_ref_logical_or_quant;
#endif
#ifndef CONFIG_C_REFERENCE_LOGICAL_XOR_DISABLED
        cb_map[CSINN_OP_LOGICAL_XOR][i].exec = shl_ref_logical_xor_quant;
#endif
#ifndef CONFIG_C_REFERENCE_LRN_DISABLED
        cb_map[CSINN_OP_LRN][i].exec = shl_ref_lrn_quant;
#endif
#ifndef CONFIG_C_REFERENCE_MATMUL_DISABLED
        cb_map[CSINN_OP_MATMUL][i].exec = shl_ref_matmul_quant;
#endif
#ifndef CONFIG_C_REFERENCE_MAX_DISABLED
        cb_map[CSINN_OP_MAX][i].exec = shl_ref_max_stride_quant;
#endif
#ifndef CONFIG_C_REFERENCE_MAXIMUM_DISABLED
        cb_map[CSINN_OP_MAXIMUM][i].exec = shl_ref_maximum_quant;
#endif
#ifndef CONFIG_C_REFERENCE_MAXPOOL_DISABLED
        cb_map[CSINN_OP_MAXPOOL2D][i].exec = shl_ref_maxpool2d_quant;
#endif
#ifndef CONFIG_C_REFERENCE_MAXPOOL2D_LOCAT_DISABLED
        cb_map[CSINN_OP_MAXPOOL2D_LOCAT][i].exec = shl_ref_maxpool2d_locat_quant;
#endif
#ifndef CONFIG_C_REFERENCE_MAXPOOL3D_DISABLED
        cb_map[CSINN_OP_MAXPOOL3D][i].exec = shl_ref_maxpool3d_quant;
#endif
#ifndef CONFIG_C_REFERENCE_MEAN_DISABLED
        cb_map[CSINN_OP_MEAN][i].exec = shl_ref_mean_stride_quant;
        cb_map[CSINN_OP_MEAN_STRIDE][i].exec = shl_ref_mean_stride_quant;
#endif
#ifndef CONFIG_C_REFERENCE_MIN_DISABLED
        cb_map[CSINN_OP_MIN][i].exec = shl_ref_min_stride_quant;
#endif
#ifndef CONFIG_C_REFERENCE_MINIMUM_DISABLED
        cb_map[CSINN_OP_MINIMUM][i].exec = shl_ref_minimum_quant;
#endif
#ifndef CONFIG_C_REFERENCE_MOD_DISABLED
        cb_map[CSINN_OP_MOD][i].exec = shl_ref_mod_quant;
#endif
#ifndef CONFIG_C_REFERENCE_MUL_DISABLED
        cb_map[CSINN_OP_MUL][i].exec = shl_ref_mul_quant;
#endif
#ifndef CONFIG_C_REFERENCE_NEGATIVE_DISABLED
        cb_map[CSINN_OP_NEGATIVE][i].exec = shl_ref_negative_quant;
#endif
#ifndef CONFIG_C_REFERENCE_NOT_EQUAL_DISABLED
        cb_map[CSINN_OP_NOT_EQUAL][i].exec = shl_ref_not_equal_quant;
#endif
#ifndef CONFIG_C_REFERENCE_PAD_DISABLED
        cb_map[CSINN_OP_PAD][i].exec = shl_ref_pad_quant;
#endif
#ifndef CONFIG_C_REFERENCE_POWER_DISABLED
        cb_map[CSINN_OP_POWER][i].exec = shl_ref_power_quant;
#endif
#ifndef CONFIG_C_REFERENCE_PRELU_DISABLED
        cb_map[CSINN_OP_PRELU][i].exec = shl_ref_prelu_quant;
#endif
#ifndef CONFIG_C_REFERENCE_PROD_DISABLED
        cb_map[CSINN_OP_PROD][i].exec = shl_ref_prod_stride_quant;
#endif
#ifndef CONFIG_C_REFERENCE_PROPOSAL_DISABLED
        cb_map[CSINN_OP_PROPOSAL][i].exec = shl_ref_proposal_quant;
#endif
#ifndef CONFIG_C_REFERENCE_PSROIPOOLING_DISABLED
        cb_map[CSINN_OP_PSROIPOOLING][i].exec = shl_ref_psroipooling_quant;
#endif
#ifndef CONFIG_C_REFERENCE_REDUCE_LOGSUMEXP_DISABLED
        cb_map[CSINN_OP_REDUCE_LOGSUMEXP][i].exec = shl_ref_reduce_logsumexp_quant;
#endif
#ifndef CONFIG_C_REFERENCE_REDUCE_MAX_DISABLED
        cb_map[CSINN_OP_REDUCE_MAX][i].exec = shl_ref_reduce_max_quant;
#endif
#ifndef CONFIG_C_REFERENCE_REDUCE_MEAN_DISABLED
        cb_map[CSINN_OP_REDUCE_MEAN][i].exec = shl_ref_reduce_mean_quant;
#endif
#ifndef CONFIG_C_REFERENCE_REDUCE_MIN_DISABLED
        cb_map[CSINN_OP_REDUCE_MIN][i].exec = shl_ref_reduce_min_quant;
#endif
#ifndef CONFIG_C_REFERENCE_REDUCE_PROD_DISABLED
        cb_map[CSINN_OP_REDUCE_PROD][i].exec = shl_ref_reduce_prod_quant;
#endif
#ifndef CONFIG_C_REFERENCE_REDUCE_SUM_DISABLED
        cb_map[CSINN_OP_REDUCE_SUM][i].exec = shl_ref_reduce_sum_quant;
#endif
#ifndef CONFIG_C_REFERENCE_RELU_DISABLED
        cb_map[CSINN_OP_RELU][i].exec = shl_ref_relu_quant;
#endif
#ifndef CONFIG_C_REFERENCE_RELU1_DISABLED
        cb_map[CSINN_OP_RELU1][i].exec = shl_ref_relu1_quant;
#endif
#ifndef CONFIG_C_REFERENCE_RELU6_DISABLED
        cb_map[CSINN_OP_RELU6][i].exec = shl_ref_relu6_quant;
#endif
#ifndef CONFIG_C_REFERENCE_RELUN_DISABLED
        cb_map[CSINN_OP_RELUN][i].exec = shl_ref_relun_quant;
#endif
#ifndef CONFIG_C_REFERENCE_RESHAPE_DISABLED
        cb_map[CSINN_OP_RESHAPE][i].exec = shl_ref_reshape;
        cb_map[CSINN_OP_RESHAPE][i].init = shl_ref_reshape_init;
#endif
#ifndef CONFIG_C_REFERENCE_RESIZE_DISABLED
        cb_map[CSINN_OP_RESIZE][i].exec = shl_ref_resize_quant;
#endif
#ifndef CONFIG_C_REFERENCE_REVERSE_DISABLED
        cb_map[CSINN_OP_REVERSE][i].exec = shl_ref_reverse_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ROIPOOL_DISABLED
        cb_map[CSINN_OP_ROIPOOL][i].exec = shl_ref_roipool_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ROUND_DISABLED
        cb_map[CSINN_OP_ROUND][i].exec = shl_ref_round_quant;
#endif
#ifndef CONFIG_C_REFERENCE_RSQRT_DISABLED
        cb_map[CSINN_OP_RSQRT][i].exec = shl_ref_rsqrt_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SEGMENT_MAX_DISABLED
        cb_map[CSINN_OP_SEGMENT_MAX][i].exec = shl_ref_segment_max_quant;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_MAX][i].exec = shl_ref_unsorted_segment_max_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SEGMENT_MEAN_DISABLED
        cb_map[CSINN_OP_SEGMENT_MEAN][i].exec = shl_ref_segment_mean_quant;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_MEAN][i].exec = shl_ref_unsorted_segment_mean_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SEGMENT_MIN_DISABLED
        cb_map[CSINN_OP_SEGMENT_MIN][i].exec = shl_ref_segment_min_quant;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_MIN][i].exec = shl_ref_unsorted_segment_min_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SEGMENT_PROD_DISABLED
        cb_map[CSINN_OP_SEGMENT_PROD][i].exec = shl_ref_segment_prod_quant;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_PROD][i].exec = shl_ref_unsorted_segment_prod_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SEGMENT_SUM_DISABLED
        cb_map[CSINN_OP_SEGMENT_SUM][i].exec = shl_ref_segment_sum_quant;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_SUM][i].exec = shl_ref_unsorted_segment_sum_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SHUFFLE_CHANNEL_DISABLED
        cb_map[CSINN_OP_SHUFFLE_CHANNEL][i].exec = shl_ref_shuffle_channel_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SIGMOID_DISABLED
        cb_map[CSINN_OP_SIGMOID][i].exec = shl_ref_sigmoid_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SIGN_DISABLED
        cb_map[CSINN_OP_SIGN][i].exec = shl_ref_sign_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SIN_DISABLED
        cb_map[CSINN_OP_SIN][i].exec = shl_ref_sin_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SINH_DISABLED
        cb_map[CSINN_OP_SINH][i].exec = shl_ref_sinh_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SLICE_DISABLED
        cb_map[CSINN_OP_SLICE][i].exec = shl_ref_slice_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SOFTMAX_DISABLED
        cb_map[CSINN_OP_SOFTMAX][i].exec = shl_ref_softmax_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SOFTPLUS_DISABLED
        cb_map[CSINN_OP_SOFTPLUS][i].exec = shl_ref_softplus_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SOFTRELU_DISABLED
        cb_map[CSINN_OP_SOFTRELU][i].exec = shl_ref_softrelu_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SOFTSIGN_DISABLED
        cb_map[CSINN_OP_SOFTSIGN][i].exec = shl_ref_softsign_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SPACE_TO_BATCH_DISABLED
        cb_map[CSINN_OP_SPACE_TO_BATCH][i].exec = shl_ref_space_to_batch_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SPACE_TO_DEPTH_DISABLED
        cb_map[CSINN_OP_SPACE_TO_DEPTH][i].exec = shl_ref_space_to_depth_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SQRT_DISABLED
        cb_map[CSINN_OP_SQRT][i].exec = shl_ref_sqrt_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SQUARE_DISABLED
        cb_map[CSINN_OP_SQUARE][i].exec = shl_ref_square_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SQUEEZE_DISABLED
        cb_map[CSINN_OP_SQUEEZE][i].exec = shl_ref_squeeze_quant;
#endif
#ifndef CONFIG_C_REFERENCE_STACK_DISABLED
        cb_map[CSINN_OP_STACK][i].exec = shl_ref_stack_quant;
#endif
#ifndef CONFIG_C_REFERENCE_STRIDED_SLICE_DISABLED
        cb_map[CSINN_OP_STRIDED_SLICE][i].exec = shl_ref_strided_slice_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SUB_DISABLED
        cb_map[CSINN_OP_SUB][i].exec = shl_ref_sub_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SUM_DISABLED
        cb_map[CSINN_OP_SUM][i].exec = shl_ref_sum_stride_quant;
#endif
#ifndef CONFIG_C_REFERENCE_TAN_DISABLED
        cb_map[CSINN_OP_TAN][i].exec = shl_ref_tan_quant;
#endif
#ifndef CONFIG_C_REFERENCE_TANH_DISABLED
        cb_map[CSINN_OP_TANH][i].exec = shl_ref_tanh_quant;
#endif
#ifndef CONFIG_C_REFERENCE_THRESHOLD_RELU_DISABLED
        cb_map[CSINN_OP_THRESHOLD_RELU][i].exec = shl_ref_threshold_relu_quant;
#endif
#ifndef CONFIG_C_REFERENCE_TILE_DISABLED
        cb_map[CSINN_OP_TILE][i].exec = shl_ref_tile_quant;
#endif
#ifndef CONFIG_C_REFERENCE_TOPK_DISABLED
        cb_map[CSINN_OP_TOPK][i].exec = shl_ref_topk_quant;
#endif
#ifndef CONFIG_C_REFERENCE_TRANSPOSE_DISABLED
        cb_map[CSINN_OP_TRANSPOSE][i].exec = shl_ref_transpose;
        cb_map[CSINN_OP_TRANSPOSE][i].init = shl_ref_transpose_init;
#endif
#ifndef CONFIG_C_REFERENCE_TRUNC_DISABLED
        cb_map[CSINN_OP_TRUNC][i].exec = shl_ref_trunc_quant;
#endif
#ifndef CONFIG_C_REFERENCE_UNPOOLING_DISABLED
        cb_map[CSINN_OP_UNPOOLING][i].exec = shl_ref_unpooling_quant;
#endif
#ifndef CONFIG_C_REFERENCE_YUV_RGB_SCALE_DISABLED
        cb_map[CSINN_OP_YUV_RGB_SCALE][i].exec = shl_ref_yuv_rgb_scale_quant;
#endif
#ifndef CONFIG_C_REFERENCE_CONVOLUTION_DISABLED
        cb_map[CSINN_OP_CONV2D][i].exec = shl_ref_conv2d_quant;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D][i].exec = shl_ref_depthwise_conv2d_quant;
        cb_map[CSINN_OP_GROUP_CONV2D][i].exec = shl_ref_group_conv2d_quant;
#endif
#ifndef CONFIG_C_REFERENCE_CONVOLUTION_RELU_DISABLED
        cb_map[CSINN_OP_CONV2D_RELU][i].exec = shl_ref_conv2d_relu_quant;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D_RELU][i].exec = shl_ref_depthwise_conv2d_relu_quant;
        cb_map[CSINN_OP_GROUP_CONV2D_RELU][i].exec = shl_ref_group_conv2d_relu_quant;
#endif
#ifndef CONFIG_C_REFERENCE_CONVOLUTION_RELU6_DISABLED
        cb_map[CSINN_OP_CONV2D_RELU6][i].exec = shl_ref_conv2d_relu6_quant;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D_RELU6][i].exec = shl_ref_depthwise_conv2d_relu6_quant;
        cb_map[CSINN_OP_GROUP_CONV2D_RELU6][i].exec = shl_ref_group_conv2d_relu6_quant;
#endif
#ifndef CONFIG_C_REFERENCE_CONVOLUTION_CHANNEL_DISABLED
        cb_map[CSINN_OP_CONV2D_CHANNEL][i].exec = shl_ref_conv2d_channel_quant;
        cb_map[CSINN_OP_CONV2D_CHANNEL_RELU][i].exec = shl_ref_conv2d_channel_relu_quant;
        cb_map[CSINN_OP_CONV2D_CHANNEL_RELU6][i].exec = shl_ref_conv2d_channel_relu6_quant;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D_CHANNEL][i].exec = shl_ref_depthwise_conv2d_channel_quant;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU][i].exec =
            shl_ref_depthwise_conv2d_channel_relu_quant;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6][i].exec =
            shl_ref_depthwise_conv2d_channel_relu6_quant;
        cb_map[CSINN_OP_GROUP_CONV2D_CHANNEL][i].exec = shl_ref_group_conv2d_channel_quant;
        cb_map[CSINN_OP_GROUP_CONV2D_CHANNEL_RELU][i].exec =
            shl_ref_group_conv2d_channel_relu_quant;
#endif
#ifndef CONFIG_C_REFERENCE_CONVOLUTION3D_DISABLED
        cb_map[CSINN_OP_CONV3D][i].exec = shl_ref_conv3d_quant;
#endif
#ifndef CONFIG_C_REFERENCE_DECONVOLUTION_DISABLED
        cb_map[CSINN_OP_DECONV2D][i].exec = shl_ref_deconv2d_quant;
        cb_map[CSINN_OP_DEPTHWISE_DECONV2D][i].exec = shl_ref_depthwise_deconv2d_quant;
        cb_map[CSINN_OP_GROUP_DECONV2D][i].exec = shl_ref_group_deconv2d_quant;
#endif
#ifndef CONFIG_C_REFERENCE_DECONVOLUTION3D_DISABLED
        cb_map[CSINN_OP_DECONV3D][i].exec = shl_ref_deconv3d_quant;
#endif
#ifndef CONFIG_C_REFERENCE_FULLYCONNECTED_DISABLED
        cb_map[CSINN_OP_FULLYCONNECTED][i].exec = shl_ref_fullyconnected_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SCATTER_DISABLED
        cb_map[CSINN_OP_SCATTER_ND][i].exec = shl_ref_scatter_nd_quant;
#endif
#ifndef CONFIG_C_REFERENCE_SPLIT_DISABLED
        cb_map[CSINN_OP_SPLIT][i].exec = shl_ref_split_quant;
#endif
#ifndef CONFIG_C_REFERENCE_ONE_HOT_DISABLED
        cb_map[CSINN_OP_ONE_HOT][i].exec = shl_ref_one_hot_quant;
#endif
#ifndef CONFIG_C_REFERENCE_WHERE_DISABLED
        cb_map[CSINN_OP_WHERE][i].exec = shl_ref_where_quant;
#endif
#ifndef CONFIG_C_REFERENCE_WHERE_SOFTMAX_DISABLED
        cb_map[CSINN_OP_WHERE_SOFTMAX][i].exec = shl_ref_where_softmax_quant;
#endif
#ifndef CONFIG_C_REFERENCE_INSTANCE_NORM_DISABLED
        cb_map[CSINN_OP_INSTANCE_NORM][i].exec = shl_ref_instance_norm_quant;
#endif
    }

    for (int i = CSINN_DTYPE_INT4; i < CSINN_DTYPE_FLOAT64; i++) {
#ifndef CONFIG_C_REFERENCE_DATA_CONVERT_DISABLED
        cb_map[CSINN_OP_DATA_CONVERT][i].exec = shl_ref_data_convert_quant;
#endif
    }

#ifndef CONFIG_C_REFERENCE_RESHAPE_DISABLED
        cb_map[CSINN_OP_RESHAPE][CSINN_DTYPE_INT64].exec = shl_ref_reshape;
        cb_map[CSINN_OP_RESHAPE][CSINN_DTYPE_INT64].init = shl_ref_reshape_init;
#endif

#ifndef CONFIG_C_REFERENCE_CONCAT_DISABLED
    cb_map[CSINN_OP_CONCAT][CSINN_DTYPE_INT64].exec = shl_ref_concat_quant;
#endif

#ifndef CONFIG_C_REFERENCE_AND_DISABLED
    cb_map[CSINN_OP_AND][CSINN_DTYPE_UINT8].exec = shl_ref_and_u8;
    cb_map[CSINN_OP_AND][CSINN_DTYPE_INT8].exec = shl_ref_and_i8;
    cb_map[CSINN_OP_AND][CSINN_DTYPE_UINT32].exec = shl_ref_and_u32;
#endif
#ifndef CONFIG_C_REFERENCE_NDARRAY_SIZE_DISABLED
    cb_map[CSINN_OP_NDARRAY_SIZE][CSINN_DTYPE_UINT8].exec = shl_ref_ndarray_size_u8;
    cb_map[CSINN_OP_NDARRAY_SIZE][CSINN_DTYPE_INT8].exec = shl_ref_ndarray_size_i8;
    cb_map[CSINN_OP_NDARRAY_SIZE][CSINN_DTYPE_INT32].exec = shl_ref_ndarray_size_i32;
    cb_map[CSINN_OP_NDARRAY_SIZE][CSINN_DTYPE_FLOAT32].exec = shl_ref_ndarray_size_f32;
#endif
#ifndef CONFIG_C_REFERENCE_NOT_DISABLED
    cb_map[CSINN_OP_NOT][CSINN_DTYPE_UINT8].exec = shl_ref_not_u8;
    cb_map[CSINN_OP_NOT][CSINN_DTYPE_INT8].exec = shl_ref_not_i8;
    cb_map[CSINN_OP_NOT][CSINN_DTYPE_UINT32].exec = shl_ref_not_u32;
#endif
#ifndef CONFIG_C_REFERENCE_OR_DISABLED
    cb_map[CSINN_OP_OR][CSINN_DTYPE_UINT8].exec = shl_ref_or_u8;
    cb_map[CSINN_OP_OR][CSINN_DTYPE_INT8].exec = shl_ref_or_i8;
    cb_map[CSINN_OP_OR][CSINN_DTYPE_UINT32].exec = shl_ref_or_u32;
#endif
#ifndef CONFIG_C_REFERENCE_SELECT_DISABLED
    cb_map[CSINN_OP_SELECT][CSINN_DTYPE_UINT8].exec = shl_ref_select_u8;
    cb_map[CSINN_OP_SELECT][CSINN_DTYPE_INT8].exec = shl_ref_select_i8;
    cb_map[CSINN_OP_SELECT][CSINN_DTYPE_FLOAT32].exec = shl_ref_select_f32;
#endif
#ifndef CONFIG_C_REFERENCE_SHAPE_DISABLED
    cb_map[CSINN_OP_SHAPE][CSINN_DTYPE_UINT8].exec = shl_ref_shape_u8;
    cb_map[CSINN_OP_SHAPE][CSINN_DTYPE_INT8].exec = shl_ref_shape_i8;
    cb_map[CSINN_OP_SHAPE][CSINN_DTYPE_INT32].exec = shl_ref_shape_i32;
#endif
#ifndef CONFIG_C_REFERENCE_XOR_DISABLED
    cb_map[CSINN_OP_XOR][CSINN_DTYPE_UINT8].exec = shl_ref_xor_u8;
    cb_map[CSINN_OP_XOR][CSINN_DTYPE_INT8].exec = shl_ref_xor_i8;
    cb_map[CSINN_OP_XOR][CSINN_DTYPE_UINT32].exec = shl_ref_xor_u32;
#endif

#ifndef CONFIG_C_REFERENCE_NON_MAX_SUPPRESSION_DISABLED
    cb_map[CSINN_OP_NON_MAX_SUPPRESSION][CSINN_DTYPE_FLOAT32].exec =
        shl_ref_non_max_suppression_std;
#endif

#ifndef CONFIG_C_REFERENCE_ROIALIGN_DISABLED
    cb_map[CSINN_OP_ROIALIGN][CSINN_DTYPE_FLOAT32].exec = shl_ref_roi_align_f32;
#endif

#ifndef CONFIG_C_REFERENCE_SCATTER_DISABLED
    cb_map[CSINN_OP_SCATTER_ND][CSINN_DTYPE_FLOAT32].exec = shl_ref_scatter_nd_f32;
#endif

#ifndef CONFIG_C_REFERENCE_COL2IM_DISABLED
    cb_map[CSINN_OP_COL2IM][CSINN_DTYPE_FLOAT32].exec = shl_ref_col2im_f32;
#endif
#ifndef CONFIG_C_REFERENCE_ISNAN_DISABLED
    cb_map[CSINN_OP_ISNAN][CSINN_DTYPE_FLOAT32].exec = shl_ref_isnan_bool_f32;
#endif
#ifndef CONFIG_C_REFERENCE_L2POOL_DISABLED
    cb_map[CSINN_OP_L2POOL2D][CSINN_DTYPE_FLOAT32].exec = shl_ref_l2pool_f32;
#endif

#ifndef CONFIG_C_REFERENCE_WHERE_DISABLED
    cb_map[CSINN_OP_WHERE][CSINN_DTYPE_BOOL].exec = shl_ref_where_quant;
#endif

#ifndef CONFIG_C_REFERENCE_WHERE_SOFTMAX_DISABLED
    cb_map[CSINN_OP_WHERE_SOFTMAX][CSINN_DTYPE_BOOL].exec = shl_ref_where_softmax_quant;
#endif

#ifndef CONFIG_C_REFERENCE_INSTANCE_NORM_DISABLED
    cb_map[CSINN_OP_INSTANCE_NORM][CSINN_DTYPE_FLOAT32].exec = shl_ref_instance_norm_f32;
#endif

#ifndef CONFIG_C_REFERENCE_CAST_DISABLED
    cb_map[CSINN_OP_CAST][CSINN_DTYPE_UINT8].exec = shl_ref_cast_quant;
    cb_map[CSINN_OP_CAST][CSINN_DTYPE_INT8].exec = shl_ref_cast_quant;
    cb_map[CSINN_OP_CAST][CSINN_DTYPE_INT32].exec = shl_ref_cast_quant;
    cb_map[CSINN_OP_CAST][CSINN_DTYPE_FLOAT16].exec = shl_ref_cast_quant;
    cb_map[CSINN_OP_CAST][CSINN_DTYPE_FLOAT32].exec = shl_ref_cast_f32;
    cb_map[CSINN_OP_CAST][CSINN_DTYPE_BOOL].exec = shl_ref_cast_bool;
    cb_map[CSINN_OP_CAST][CSINN_DTYPE_INT64].exec = shl_ref_cast_i64;
#endif

#ifdef SHL_BUILD_GREF
#include "shl_gref.h"
    shl_register_runtime_callback(CSINN_REF, shl_gref_runtime_callback);
    for (int i = 0; i < CSINN_DTYPE_SIZE; i++) {
#ifndef CONFIG_GRAPH_REFERENCE_ABS_DISABLED
        cb_map[CSINN_OP_ABS][i].est = shl_gref_abs;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ACOS_DISABLED
        cb_map[CSINN_OP_ACOS][i].est = shl_gref_acos;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ACOSH_DISABLED
        cb_map[CSINN_OP_ACOSH][i].est = shl_gref_acosh;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ADD_DISABLED
        cb_map[CSINN_OP_ADD][i].est = shl_gref_add;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ARANGE_DISABLED
        cb_map[CSINN_OP_ARANGE][i].est = shl_gref_arange;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ARGMAX_DISABLED
        cb_map[CSINN_OP_ARGMAX][i].est = shl_gref_argmax;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ARGMIN_DISABLED
        cb_map[CSINN_OP_ARGMIN][i].est = shl_gref_argmin;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ASIN_DISABLED
        cb_map[CSINN_OP_ASIN][i].est = shl_gref_asin;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ASINH_DISABLED
        cb_map[CSINN_OP_ASINH][i].est = shl_gref_asinh;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ATAN_DISABLED
        cb_map[CSINN_OP_ATAN][i].est = shl_gref_atan;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ATANH_DISABLED
        cb_map[CSINN_OP_ATANH][i].est = shl_gref_atanh;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_AVERAGEPOOL_DISABLED
        cb_map[CSINN_OP_AVGPOOL2D][i].est = shl_gref_avgpool2d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_AVERAGEPOOL3D_DISABLED
        cb_map[CSINN_OP_AVGPOOL3D][i].est = shl_gref_avgpool3d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_BATCH_NORMALIZATION_DISABLED
        cb_map[CSINN_OP_BN][i].est = shl_gref_batch_normalization;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_BATCH_TO_SPACE_DISABLED
        cb_map[CSINN_OP_BATCH_TO_SPACE][i].est = shl_gref_batch_to_space;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_BATCH_TO_SPACE_ND_DISABLED
        cb_map[CSINN_OP_BROADCOST][i].est = shl_gref_broadcast_to;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CACHE_MATMUL_DISABLED
        cb_map[CSINN_OP_CACHE_MATMUL][i].est = shl_gref_cache_matmul;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CACHE_CONV1D_DISABLED
        cb_map[CSINN_OP_CACHE_CONV1D][i].est = shl_gref_cache_conv1d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CAST_DISABLED
        cb_map[CSINN_OP_CAST][i].est = shl_gref_cast;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CEIL_DISABLED
        cb_map[CSINN_OP_CEIL][i].est = shl_gref_ceil;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CLIP_DISABLED
        cb_map[CSINN_OP_CLIP][i].est = shl_gref_clip;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONCAT_DISABLED
        cb_map[CSINN_OP_CONCAT][i].est = shl_gref_concat;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_COS_DISABLED
        cb_map[CSINN_OP_COS][i].est = shl_gref_cos;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_COSH_DISABLED
        cb_map[CSINN_OP_COSH][i].est = shl_gref_cosh;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CUMPROD_DISABLED
        cb_map[CSINN_OP_CUMPROD][i].est = shl_gref_cumprod;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_DATA_CONVERT_DISABLED
        cb_map[CSINN_OP_DATA_CONVERT][i].est = shl_gref_data_convert;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CUMSUM_DISABLED
        cb_map[CSINN_OP_CUMSUM][i].est = shl_gref_cumsum;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_DEPTH_TO_SPACE_DISABLED
        cb_map[CSINN_OP_DEPTH_TO_SPACE][i].est = shl_gref_depth_to_space;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_DIV_DISABLED
        cb_map[CSINN_OP_DIV][i].est = shl_gref_div;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ELU_DISABLED
        cb_map[CSINN_OP_ELU][i].est = shl_gref_elu;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_EQUAL_DISABLED
        cb_map[CSINN_OP_EQUANL][i].est = shl_gref_equal;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ERF_DISABLED
        cb_map[CSINN_OP_ERF][i].est = shl_gref_erf;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_EXP_DISABLED
        cb_map[CSINN_OP_EXP][i].est = shl_gref_exp;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_EXPAND_DIMS_DISABLED
        cb_map[CSINN_OP_EXPAND_DIMS][i].est = shl_gref_expand_dims;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_EXPM1_DISABLED
        cb_map[CSINN_OP_EXPM1][i].est = shl_gref_expm1;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FLATTEN_DISABLED
        cb_map[CSINN_OP_FLATTEN][i].est = shl_gref_flatten;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FLOOR_DIVIDE_DISABLED
        cb_map[CSINN_OP_FLOOR_DIVIDE][i].est = shl_gref_floor_divide;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FLOOR_MOD_DISABLED
        cb_map[CSINN_OP_FLOOR_MOD][i].est = shl_gref_floor_mod;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FLOOR_DISABLED
        cb_map[CSINN_OP_FLOOR][i].est = shl_gref_floor;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FSMN_DISABLED
        cb_map[CSINN_OP_FSMN][i].est = shl_gref_fsmn;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GATHER_ND_DISABLED
        cb_map[CSINN_OP_GATHER_ND][i].est = shl_gref_gather_nd;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GATHER_DISABLED
        cb_map[CSINN_OP_GATHER][i].est = shl_gref_gather;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GLOBAL_AVERAGEPOOL_DISABLED
        cb_map[CSINN_OP_GLOBAL_AVGPOOL2D][i].est = shl_gref_global_avgpool2d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GLOBAL_MAXPOOL_DISABLED
        cb_map[CSINN_OP_GLOBAL_MAXPOOL2D][i].est = shl_gref_global_maxpool2d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GREATER_EQUAL_DISABLED
        cb_map[CSINN_OP_GREATHER_EQUAL][i].est = shl_gref_greater_equal;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_GREATER_DISABLED
        cb_map[CSINN_OP_GREATHER][i].est = shl_gref_greater;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_HARD_SIGMOID_DISABLED
        cb_map[CSINN_OP_HARD_SIGMOID][i].est = shl_gref_hard_sigmoid;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_IM2COL_DISABLED
        cb_map[CSINN_OP_IM2COL][i].est = shl_gref_im2col;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_L2_NORMALIZATION_DISABLED
        cb_map[CSINN_OP_L2N][i].est = shl_gref_l2_normalization;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LAYER_NORM_DISABLED
        cb_map[CSINN_OP_LAYER_NORM][i].est = shl_gref_layer_norm;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LEAKY_RELU_DISABLED
        cb_map[CSINN_OP_LEAKY_RELU][i].est = shl_gref_leaky_relu;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LESS_EQUAL_DISABLED
        cb_map[CSINN_OP_LESS_EQUAL][i].est = shl_gref_less_equal;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LESS_DISABLED
        cb_map[CSINN_OP_LESS][i].est = shl_gref_less;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LOG_SOFTMAX_DISABLED
        cb_map[CSINN_OP_LOG_SOFTMAX][i].est = shl_gref_log_softmax;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LOG_DISABLED
        cb_map[CSINN_OP_LOG][i].est = shl_gref_log;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LOG1P_DISABLED
        cb_map[CSINN_OP_LOG1P][i].est = shl_gref_log1p;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LOGICAL_AND_DISABLED
        cb_map[CSINN_OP_LOGICAL_AND][i].est = shl_gref_logical_and;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LOGICAL_NOT_DISABLED
        cb_map[CSINN_OP_LOGICAL_NOT][i].est = shl_gref_logical_not;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LOGICAL_OR_DISABLED
        cb_map[CSINN_OP_LOGICAL_OR][i].est = shl_gref_logical_or;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LOGICAL_XOR_DISABLED
        cb_map[CSINN_OP_LOGICAL_XOR][i].est = shl_gref_logical_xor;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_LRN_DISABLED
        cb_map[CSINN_OP_LRN][i].est = shl_gref_lrn;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MATMUL_DISABLED
        cb_map[CSINN_OP_MATMUL][i].est = shl_gref_matmul;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MAX_DISABLED
        cb_map[CSINN_OP_MAX][i].est = shl_gref_max;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MAXIMUM_DISABLED
        cb_map[CSINN_OP_MAXIMUM][i].est = shl_gref_maximum;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MAXPOOL_DISABLED
        cb_map[CSINN_OP_MAXPOOL2D][i].est = shl_gref_maxpool2d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MAXPOOL2D_LOCAT_DISABLED
        cb_map[CSINN_OP_MAXPOOL2D_LOCAT][i].est = shl_gref_maxpool2d_locat;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MAXPOOL3D_DISABLED
        cb_map[CSINN_OP_MAXPOOL3D][i].est = shl_gref_maxpool3d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MEAN_DISABLED
        cb_map[CSINN_OP_MEAN][i].est = shl_gref_mean;
        cb_map[CSINN_OP_MEAN_STRIDE][i].est = shl_gref_mean;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MIN_DISABLED
        cb_map[CSINN_OP_MIN][i].est = shl_gref_min;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MINIMUM_DISABLED
        cb_map[CSINN_OP_MINIMUM][i].est = shl_gref_minimum;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MOD_DISABLED
        cb_map[CSINN_OP_MOD][i].est = shl_gref_mod;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_MUL_DISABLED
        cb_map[CSINN_OP_MUL][i].est = shl_gref_mul;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_NEGATIVE_DISABLED
        cb_map[CSINN_OP_NEGATIVE][i].est = shl_gref_negative;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_NOT_EQUAL_DISABLED
        cb_map[CSINN_OP_NOT_EQUAL][i].est = shl_gref_not_equal;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_PAD_DISABLED
        cb_map[CSINN_OP_PAD][i].est = shl_gref_pad;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_POWER_DISABLED
        cb_map[CSINN_OP_POWER][i].est = shl_gref_power;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_PRELU_DISABLED
        cb_map[CSINN_OP_PRELU][i].est = shl_gref_prelu;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_PROD_DISABLED
        cb_map[CSINN_OP_PROD][i].est = shl_gref_prod;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_PROPOSAL_DISABLED
        cb_map[CSINN_OP_PROPOSAL][i].est = shl_gref_proposal;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_PSROIPOOLING_DISABLED
        cb_map[CSINN_OP_PSROIPOOLING][i].est = shl_gref_psroipooling;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_REDUCE_LOGSUMEXP_DISABLED
        cb_map[CSINN_OP_REDUCE_LOGSUMEXP][i].est = shl_gref_reduce_logsumexp;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_REDUCE_MAX_DISABLED
        cb_map[CSINN_OP_REDUCE_MAX][i].est = shl_gref_reduce_max;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_REDUCE_MEAN_DISABLED
        cb_map[CSINN_OP_REDUCE_MEAN][i].est = shl_gref_reduce_mean;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_REDUCE_MIN_DISABLED
        cb_map[CSINN_OP_REDUCE_MIN][i].est = shl_gref_reduce_min;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_REDUCE_PROD_DISABLED
        cb_map[CSINN_OP_REDUCE_PROD][i].est = shl_gref_reduce_prod;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_REDUCE_SUM_DISABLED
        cb_map[CSINN_OP_REDUCE_SUM][i].est = shl_gref_reduce_sum;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU_DISABLED
        cb_map[CSINN_OP_RELU][i].est = shl_gref_relu;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU1_DISABLED
        cb_map[CSINN_OP_RELU1][i].est = shl_gref_relu1;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELU6_DISABLED
        cb_map[CSINN_OP_RELU6][i].est = shl_gref_relu6;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RELUN_DISABLED
        cb_map[CSINN_OP_RELUN][i].est = shl_gref_relun;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RESHAPE_DISABLED
        cb_map[CSINN_OP_RESHAPE][i].est = shl_gref_reshape;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RESIZE_DISABLED
        cb_map[CSINN_OP_RESIZE][i].est = shl_gref_resize;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_REVERSE_DISABLED
        cb_map[CSINN_OP_REVERSE][i].est = shl_gref_reverse;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ROIPOOL_DISABLED
        cb_map[CSINN_OP_ROIPOOL][i].est = shl_gref_roipool;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ROUND_DISABLED
        cb_map[CSINN_OP_ROUND][i].est = shl_gref_round;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_RSQRT_DISABLED
        cb_map[CSINN_OP_RSQRT][i].est = shl_gref_rsqrt;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SEGMENT_MAX_DISABLED
        cb_map[CSINN_OP_SEGMENT_MAX][i].est = shl_gref_segment_max;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_MAX][i].est = shl_gref_segment_max;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SEGMENT_MEAN_DISABLED
        cb_map[CSINN_OP_SEGMENT_MEAN][i].est = shl_gref_segment_mean;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_MEAN][i].est = shl_gref_segment_mean;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SEGMENT_MIN_DISABLED
        cb_map[CSINN_OP_SEGMENT_MIN][i].est = shl_gref_segment_min;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_MIN][i].est = shl_gref_segment_min;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SEGMENT_PROD_DISABLED
        cb_map[CSINN_OP_SEGMENT_PROD][i].est = shl_gref_segment_prod;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_PROD][i].est = shl_gref_segment_prod;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SEGMENT_SUM_DISABLED
        cb_map[CSINN_OP_SEGMENT_SUM][i].est = shl_gref_segment_sum;
        cb_map[CSINN_OP_UNSORTED_SEGMENT_SUM][i].est = shl_gref_segment_sum;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SHUFFLE_CHANNEL_DISABLED
        cb_map[CSINN_OP_SHUFFLE_CHANNEL][i].est = shl_gref_shuffle_channel;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SIGMOID_DISABLED
        cb_map[CSINN_OP_SIGMOID][i].est = shl_gref_sigmoid;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SIGN_DISABLED
        cb_map[CSINN_OP_SIGN][i].est = shl_gref_sign;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SIN_DISABLED
        cb_map[CSINN_OP_SIN][i].est = shl_gref_sin;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SINH_DISABLED
        cb_map[CSINN_OP_SINH][i].est = shl_gref_sinh;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SLICE_DISABLED
        cb_map[CSINN_OP_SLICE][i].est = shl_gref_slice;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SOFTMAX_DISABLED
        cb_map[CSINN_OP_SOFTMAX][i].est = shl_gref_softmax;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SOFTPLUS_DISABLED
        cb_map[CSINN_OP_SOFTPLUS][i].est = shl_gref_softplus;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SOFTRELU_DISABLED
        cb_map[CSINN_OP_SOFTRELU][i].est = shl_gref_softrelu;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SOFTSIGN_DISABLED
        cb_map[CSINN_OP_SOFTSIGN][i].est = shl_gref_softsign;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SPACE_TO_BATCH_DISABLED
        cb_map[CSINN_OP_SPACE_TO_BATCH][i].est = shl_gref_space_to_batch;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SPACE_TO_DEPTH_DISABLED
        cb_map[CSINN_OP_SPACE_TO_DEPTH][i].est = shl_gref_space_to_depth;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SQRT_DISABLED
        cb_map[CSINN_OP_SQRT][i].est = shl_gref_sqrt;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_STACK_DISABLED
        cb_map[CSINN_OP_STACK][i].est = shl_gref_stack;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_STRIDED_SLICE_DISABLED
        cb_map[CSINN_OP_STRIDED_SLICE][i].est = shl_gref_strided_slice;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SUB_DISABLED
        cb_map[CSINN_OP_SUB][i].est = shl_gref_sub;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SUM_DISABLED
        cb_map[CSINN_OP_SUM][i].est = shl_gref_sum;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_TAN_DISABLED
        cb_map[CSINN_OP_TAN][i].est = shl_gref_tan;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_TANH_DISABLED
        cb_map[CSINN_OP_TANH][i].est = shl_gref_tanh;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_THRESHOLD_RELU_DISABLED
        cb_map[CSINN_OP_THRESHOLD_RELU][i].est = shl_gref_threshold_relu;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_TILE_DISABLED
        cb_map[CSINN_OP_TILE][i].est = shl_gref_tile;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_TOPK_DISABLED
        cb_map[CSINN_OP_TOPK][i].est = shl_gref_topk;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_TRANSPOSE_DISABLED
        cb_map[CSINN_OP_TRANSPOSE][i].est = shl_gref_transpose;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_TRUNC_DISABLED
        cb_map[CSINN_OP_TRUNC][i].est = shl_gref_trunc;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_UNPOOLING_DISABLED
        cb_map[CSINN_OP_UNPOOLING][i].est = shl_gref_unpooling;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_YUV_RGB_SCALE_DISABLED
        cb_map[CSINN_OP_YUV_RGB_SCALE][i].est = shl_gref_yuv_rgb_scale;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONVOLUTION1D_DISABLED
        cb_map[CSINN_OP_CONV1D][i].est = shl_gref_conv1d;
        cb_map[CSINN_OP_DEPTHWISE_CONV1D][i].est = shl_gref_conv1d;
        cb_map[CSINN_OP_GROUP_CONV1D][i].est = shl_gref_conv1d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONVOLUTION_DISABLED
        cb_map[CSINN_OP_CONV2D][i].est = shl_gref_conv2d;
        cb_map[CSINN_OP_CONV2D_RELU][i].est = shl_gref_conv2d_relu;
        cb_map[CSINN_OP_CONV2D_RELU6][i].est = shl_gref_conv2d_relu6;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D][i].est = shl_gref_depthwise_conv2d;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D_RELU][i].est = shl_gref_depthwise_conv2d_relu;
        cb_map[CSINN_OP_DEPTHWISE_CONV2D_RELU6][i].est = shl_gref_depthwise_conv2d_relu6;
        cb_map[CSINN_OP_GROUP_CONV2D][i].est = shl_gref_group_conv2d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_CONVOLUTION3D_DISABLED
        cb_map[CSINN_OP_CONV3D][i].est = shl_gref_conv3d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_DECONVOLUTION_DISABLED
        cb_map[CSINN_OP_DECONV2D][i].est = shl_gref_deconv2d;
        cb_map[CSINN_OP_DEPTHWISE_DECONV2D][i].est = shl_gref_depthwise_deconv2d;
        cb_map[CSINN_OP_GROUP_DECONV2D][i].est = shl_gref_group_deconv2d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_DECONVOLUTION3D_DISABLED
        cb_map[CSINN_OP_DECONV3D][i].est = shl_gref_deconv3d;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_FULLYCONNECTED_DISABLED
        cb_map[CSINN_OP_FULLYCONNECTED][i].est = shl_gref_fullyconnected;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SCATTER_DISABLED
        cb_map[CSINN_OP_SCATTER_ND][i].est = shl_gref_scatter_nd;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_SPLIT_DISABLED
        cb_map[CSINN_OP_SPLIT][i].est = shl_gref_split;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_ONE_HOT_DISABLED
        cb_map[CSINN_OP_ONE_HOT][i].est = shl_gref_one_hot;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_WHERE_DISABLED
        cb_map[CSINN_OP_WHERE][i].est = shl_gref_where;
#endif
#ifndef CONFIG_GRAPH_REFERENCE_WHERE_SOFTMAX_DISABLED
        cb_map[CSINN_OP_WHERE_SOFTMAX][i].est = shl_gref_where_softmax;
#endif
#ifndef CONFIG_GRAPH__REFERENCE_INSTANCE_NORM_DISABLED
        cb_map[CSINN_OP_INSTANCE_NORM][i].est = shl_gref_instance_norm;
#endif
    }
#endif
    return cb_map;
}

static int get_cb_map_index(int op, int dtype) { return op * CSINN_DTYPE_SIZE + dtype; }
static struct csinn_callback *__cb_map_table_ref;
struct csinn_callback *shl_cb_map_ref(int op, int dtype)
{
    if (__cb_map_table_ref) {
        return &__cb_map_table_ref[get_cb_map_index(op, dtype)];
    } else {
        return NULL;
    }
}

void __attribute__((weak)) shl_target_init_ref()
{
    __cb_map_table_ref = setup_cb_map();
    shl_register_runtime_callback(CSINN_REF, NULL);
    shl_register_op_callback(CSINN_REF, shl_cb_map_ref);
}
