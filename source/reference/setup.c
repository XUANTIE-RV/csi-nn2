/*
 * Copyright (C) 2016-2020 C-SKY Limited. All rights reserved.
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

#include "csi_nn.h"
#include "csi_utils.h"
#include "csi_internal_ref.h"

void csi_nn_init(struct csi_tensor *input,
                 struct csi_tensor *output)
{
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }
    if (output->dtype == CSINN_DTYPE_UINT8){
        float *input_data = input->data;
        uint8_t *output_data = output->data;
        for (int i = 0; i < size; i++) {
            int32_t input_val = round(input_data[i] / output->scale) + output->zero_point;
            if (input_val < 0) {
                input_val = 0;
            } else if (input_val > 255) {
                input_val = 255;
            }
            output_data[i] = input_val;
        }
    }else if (output->dtype == CSINN_DTYPE_INT8){
        float *input_data = input->data;
        int8_t *output_data = output->data;
        for (int i = 0; i < size; i++) {
            int32_t input_val = round(input_data[i] / output->scale) + output->zero_point;
            if (input_val < -127) {
                input_val = 0;
            } else if (input_val > 127) {
                input_val = 127;
            }
            output_data[i] = input_val;
        }
    }
}

void csi_nn_deinit(struct csi_tensor *input,
                   struct csi_tensor *output)
{
    int size = 1;
    for (int i = 0; i < input->dim_count; i++) {
        size = size * input->dim[i];
    }
    if (input->dtype == CSINN_DTYPE_UINT8){
        uint8_t *input_data = input->data;
        float *output_data = output->data;
        for (int i = 0; i < size; i++) {
            float input_val = csi_dequantize_u8_to_f32(input_data[i], input->zero_point, input->multiplier, input->shift);
            output_data[i] = input_val;
        }
    } else if (input->dtype == CSINN_DTYPE_INT8){
        int8_t *input_data = input->data;
        float *output_data = output->data;
        for (int i = 0; i < size; i++) {
            float input_val = csi_dequantize_i8_to_f32(input_data[i], 0, input->multiplier, input->shift);
            output_data[i] = input_val;
        }
    }
}

struct csi_tensor *csi_alloc_tensor(struct csi_session *session)
{
    struct csi_tensor *ret = calloc(1, sizeof(struct csi_tensor));
    ret->dtype = session->base_dtype;
    ret->layout = session->base_layout;
    ret->sess = session;
    return ret;
}

struct csi_session *csi_alloc_session()
{
    return calloc(1, sizeof(struct csi_session));
}

void csi_free_session(struct csi_session *sess)
{
    free(sess);
}

void* csi_bc_map_table_ref[CSINN_OP_SIZE][CSINN_DTYPE_SIZE] = {
    {csi_abs_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_abs_f32, NULL}, /* CSINN_OP_ABS */
    {csi_acos_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_acos_f32, NULL}, /* CSINN_OP_ACOS */
    {csi_acosh_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_acosh_f32, NULL}, /* CSINN_OP_ACOSH */
    {csi_add_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_add_f32, NULL}, /* CSINN_OP_ADD */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_ALL */
    {csi_and_u8, NULL, NULL, NULL, csi_and_u32, NULL, NULL, NULL, NULL}, /* CSINN_OP_AND */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_ANY */
    {csi_arange_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_arange_f32, NULL}, /* CSINN_OP_ARANGE */
    {csi_argmax_stride_i32_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_argmax_stride_i32_f32, NULL}, /* CSINN_OP_ARGMAX */
    {csi_argmin_stride_i32_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_argmin_stride_i32_f32, NULL}, /* CSINN_OP_ARGMIN */
    {csi_asin_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_asin_f32, NULL}, /* CSINN_OP_ASIN */
    {csi_asinh_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_asinh_f32, NULL}, /* CSINN_OP_ASINH */
    {csi_atan_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_atan_f32, NULL}, /* CSINN_OP_ATAN */
    {csi_atanh_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_atanh_f32, NULL}, /* CSINN_OP_ATANH */
    {csi_averagepool_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_averagepool_f32, NULL}, /* CSINN_OP_AVGPOOL2D */
    {csi_averagepool3d_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_averagepool3d_f32, NULL}, /* CSINN_OP_AVGPOOL3D */
    {csi_batch_normalization_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_batch_normalization_f32, NULL}, /* CSINN_OP_BN */
    {csi_batch_to_space_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_batch_to_space_f32, NULL}, /* CSINN_OP_BATCH_TO_SPACE */
    {csi_broadcast_to_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_broadcast_to_f32, NULL}, /* CSINN_OP_BROADCOST */
    {csi_ceil_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_ceil_f32, NULL}, /* CSINN_OP_CEIL */
    {csi_clip_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_clip_f32, NULL}, /* CSINN_OP_CLIP */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, csi_col2im_f32, NULL}, /* CSINN_OP_COL2IM */
    {csi_concat_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_concat_f32, NULL}, /* CSINN_OP_CONCAT */
    {csi_conv2d_u8, csi_conv2d_i8, NULL, NULL, NULL, NULL, NULL, csi_conv2d_f32, NULL}, /* CSINN_OP_CONV2D */
    {csi_conv2d_relu_u8, csi_conv2d_relu_i8, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_CONV2D_RELU */
    {csi_conv2d_relu6_u8, csi_conv2d_relu6_i8, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_CONV2D_RELU6 */
    {csi_conv2d_channel_u8, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_CONV2D_CHANNEL */
    {csi_conv2d_channel_relu_u8, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_CONV2D_CHANNEL_RELU */
    {csi_conv2d_channel_relu6_u8, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_CONV2D_CHANNEL_RELU6 */
    {csi_depthwise_conv2d_u8, csi_depthwise_conv2d_i8, NULL, NULL, NULL, NULL, NULL, csi_depthwise_conv2d_f32, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D */
    {csi_depthwise_conv2d_relu_u8, csi_depthwise_conv2d_relu_i8, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_RELU */
    {csi_depthwise_conv2d_relu6_u8, csi_depthwise_conv2d_relu6_i8, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_RELU6 */
    {csi_depthwise_conv2d_channel_u8, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL */
    {csi_depthwise_conv2d_channel_relu_u8, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU */
    {csi_depthwise_conv2d_channel_relu6_u8, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6 */
    {csi_group_conv2d_u8, csi_group_conv2d_i8, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_GROUP_CONV2D */
    {csi_group_conv2d_relu_u8, csi_group_conv2d_relu_i8, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_GROUP_CONV2D_RELU */
    {csi_group_conv2d_channel_u8, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_GROUP_CONV2D_CHANNEL */
    {csi_group_conv2d_channel_relu_u8, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_GROUP_CONV2D_CHANNEL_RELU */
    {csi_conv3d_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_conv3d_f32, NULL}, /* CSINN_OP_CONV3D */
    {csi_cos_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_cos_f32, NULL}, /* CSINN_OP_COS */
    {csi_cosh_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_cosh_f32, NULL}, /* CSINN_OP_COSH */
    {csi_cumprod_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_cumprod_f32, NULL}, /* CSINN_OP_CUMPROD */
    {csi_cumsum_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_cumsum_f32, NULL}, /* CSINN_OP_CUMSUM */
    {csi_deconv2d_u8, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_DECONV2D */
    {csi_depthwise_deconv2d_u8, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_DEPTHWISE_DECONV2D */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, csi_deconv3d_f32, NULL}, /* CSINN_OP_DECONV3D */
    {csi_depth_to_space_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_depth_to_space_f32, NULL}, /* CSINN_OP_DEPTH_TO_SPACE */
    {csi_div_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_div_f32, NULL}, /* CSINN_OP_DIV */
    {csi_elu_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_elu_f32, NULL}, /* CSINN_OP_ELU */
    {csi_equal_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_equal_f32, NULL}, /* CSINN_OP_EQUANL */
    {csi_erf_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_erf_f32, NULL}, /* CSINN_OP_ERF */
    {csi_exp_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_exp_f32, NULL}, /* CSINN_OP_EXP */
    {csi_expand_dims_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_expand_dims_f32, NULL}, /* CSINN_OP_EXPAND_DIMS */
    {csi_expm1_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_expm1_f32, NULL}, /* CSINN_OP_EXPM1 */
    {csi_flatten_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_flatten_f32, NULL}, /* CSINN_OP_FLATTEN */
    {csi_floor_divide_f32, NULL, NULL, NULL, NULL, NULL, NULL, csi_floor_divide_f32, NULL}, /* CSINN_OP_FLOOR_DIVIDE */
    {csi_floor_mod_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_floor_mod_f32, NULL}, /* CSINN_OP_FLOOR_MOD */
    {csi_floor_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_floor_f32, NULL}, /* CSINN_OP_FLOOR */
    {csi_fullyconnected_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_fullyconnected_f32, NULL}, /* CSINN_OP_FULLYCONNECTED */
    {csi_gather_nd_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_gather_nd_f32, NULL}, /* CSINN_OP_GATHER_ND */
    {csi_gather_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_gather_f32, NULL}, /* CSINN_OP_GATHER */
    {csi_global_averagepool_u8, csi_global_averagepool_i8, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_GLOBAL_AVGPOOL2D */
    {csi_global_maxpool_u8, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_GLOBAL_MAXPOOL2D */
    {csi_greater_equal_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_greater_equal_f32, NULL}, /* CSINN_OP_GREATHER_EQUAL */
    {csi_greater_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_greater_f32, NULL}, /* CSINN_OP_GREATHER */
    {csi_hard_sigmoid_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_hard_sigmoid_f32, NULL}, /* CSINN_OP_HARD_SIGMOID */
    {csi_im2col_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_im2col_f32, NULL}, /* CSINN_OP_IM2COL */
    {csi_isnan_bool_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_isnan_bool_f32, NULL}, /* CSINN_OP_ISNAN */
    {csi_l2_normalization_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_l2_normalization_f32, NULL}, /* CSINN_OP_L2N */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, csi_l2pool_f32, NULL}, /* CSINN_OP_L2POOL2D */
    {csi_leaky_relu_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_leaky_relu_f32, NULL}, /* CSINN_OP_LEAKY_RELU */
    {csi_less_equal_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_less_equal_f32, NULL}, /* CSINN_OP_LESS_EQUAL */
    {csi_less_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_less_f32, NULL}, /* CSINN_OP_LESS */
    {csi_log_softmax_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_log_softmax_f32, NULL}, /* CSINN_OP_LOG_SOFTMAX */
    {csi_log_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_log_f32, NULL}, /* CSINN_OP_LOG */
    {csi_log1p_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_log1p_f32, NULL}, /* CSINN_OP_LOG1P */
    {csi_logical_and_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_logical_and_f32, NULL}, /* CSINN_OP_LOGICAL_AND */
    {csi_logical_not_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_logical_not_f32, NULL}, /* CSINN_OP_LOGICAL_NOT */
    {csi_logical_or_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_logical_or_f32, NULL}, /* CSINN_OP_LOGICAL_OR */
    {csi_logical_xor_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_logical_xor_f32, NULL}, /* CSINN_OP_LOGICAL_XOR */
    {csi_lrn_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_lrn_f32, NULL}, /* CSINN_OP_LRN */
    {csi_matmul_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_matmul_f32, NULL}, /* CSINN_OP_MATMUL */
    {csi_max_stride_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_max_stride_f32, NULL}, /* CSINN_OP_MAX */
    {csi_maximum_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_maximum_f32, NULL}, /* CSINN_OP_MAXINUM */
    {csi_maxpool_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_maxpool_f32, NULL}, /* CSINN_OP_MAXPOOL2D */
    {csi_maxpool2d_locat_i32_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_maxpool2d_locat_f32, NULL}, /* CSINN_OP_MAXPOOL2D_LOCAT */
    {csi_maxpool3d_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_maxpool3d_f32, NULL}, /* CSINN_OP_MAXPOOL3D */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_MEAN */
    {csi_mean_stride_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_mean_stride_f32, NULL}, /* CSINN_OP_MEAN_STRIDE */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_MIN */
    {csi_min_stride_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_min_stride_f32, NULL}, /* CSINN_OP_MIN_STRIDE */
    {csi_minimum_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_minimum_f32, NULL}, /* CSINN_OP_MINIMUM */
    {csi_mod_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_mod_f32, NULL}, /* CSINN_OP_MOD */
    {csi_mul_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_mul_f32, NULL}, /* CSINN_OP_MUL */
    {csi_ndarray_size_u8, NULL, NULL, NULL, NULL, csi_ndarray_size_i32, NULL, csi_ndarray_size_f32, NULL}, /* CSINN_OP_NDARRAY_SIZE */
    {csi_negative_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_negative_f32, NULL}, /* CSINN_OP_NEGATIIVE */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, csi_non_max_suppression_std, NULL}, /* CSINN_OP_NON_MAX_SUPPRESSION */
    {csi_not_equal_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_not_equal_f32, NULL}, /* CSINN_OP_NOT_EQUAL */
    {csi_not_u8, NULL, NULL, NULL, csi_not_u32, NULL, NULL, NULL, NULL}, /* CSINN_OP_NOT */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_ONE_HOT */
    {csi_or_u8, NULL, NULL, NULL, csi_or_u32, NULL, NULL, NULL, NULL}, /* CSINN_OP_OR */
    {csi_pad_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_pad_f32, NULL}, /* CSINN_OP_PAD */
    {csi_power_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_power_f32, NULL}, /* CSINN_OP_POWER */
    {csi_prelu_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_prelu_f32, NULL}, /* CSINN_OP_PRELU */
    {csi_prod_stride_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_prod_stride_f32, NULL}, /* CSINN_OP_PROD */
    {csi_proposal_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_proposal_f32, NULL}, /* CSINN_OP_PROPOSAL */
    {csi_psroipooling_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_psroipooling_f32, NULL}, /* CSINN_OP_PSROIPOOLING */
    {csi_reduce_logsumexp_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_reduce_logsumexp_f32, NULL}, /* CSINN_OP_REDUCE_LOGSUMEXP */
    {csi_reduce_max_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_reduce_max_f32, NULL}, /* CSINN_OP_REDUCE_MAX */
    {csi_reduce_mean_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_reduce_mean_f32, NULL}, /* CSINN_OP_REDUCE_MEAN */
    {csi_reduce_min_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_reduce_min_f32, NULL}, /* CSINN_OP_REDUCE_MIN */
    {csi_reduce_prod_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_reduce_prod_f32, NULL}, /* CSINN_OP_REDUCE_PROD */
    {csi_reduce_sum_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_reduce_sum_f32, NULL}, /* CSINN_OP_REDUCE_SUM */
    {csi_relu_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_relu_f32, NULL}, /* CSINN_OP_RELU */
    {csi_relu1_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_relu1_f32, NULL}, /* CSINN_OP_RELU1 */
    {csi_relu6_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_relu6_f32, NULL}, /* CSINN_OP_RELU6 */
    {csi_relun_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_relun_f32, NULL}, /* CSINN_OP_RELUN */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_REORG */
    {csi_reshape_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_reshape_f32, NULL}, /* CSINN_OP_RESHAPE */
    {csi_resize_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_resize_f32, NULL}, /* CSINN_OP_RESIZE */
    {csi_reverse_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_reverse_f32, NULL}, /* CSINN_OP_REVERSE */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, csi_roi_align_f32, NULL}, /* CSINN_OP_ROIALIGN */
    {csi_roipool_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_roipool_f32, NULL}, /* CSINN_OP_ROIPOOL */
    {csi_round_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_round_f32, NULL}, /* CSINN_OP_ROUND */
    {csi_rsqrt_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_rsqrt_f32, NULL}, /* CSINN_OP_RSQRT */
    {csi_segment_max_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_segment_max_f32, NULL}, /* CSINN_OP_SEGMENT_MAX */
    {csi_unsorted_segment_max_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_unsorted_segment_max_f32, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MAX */
    {csi_segment_mean_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_segment_mean_f32, NULL}, /* CSINN_OP_SEGMENT_MEAN */
    {csi_unsorted_segment_mean_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_unsorted_segment_mean_f32, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MEAN */
    {csi_segment_min_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_segment_min_f32, NULL}, /* CSINN_OP_SEGMENT_MIN */
    {csi_unsorted_segment_min_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_unsorted_segment_min_f32, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MIN */
    {csi_segment_prod_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_segment_prod_f32, NULL}, /* CSINN_OP_SEGMENT_PROD */
    {csi_unsorted_segment_prod_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_unsorted_segment_prod_f32, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_PROD */
    {csi_segment_sum_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_segment_sum_f32, NULL}, /* CSINN_OP_SEGMENT_SUM */
    {csi_unsorted_segment_sum_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_unsorted_segment_sum_f32, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_SUM */
    {csi_select_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_select_f32, NULL}, /* CSINN_OP_SELECT */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_SEQUENCE_MASK */
    {csi_shape_u8, NULL, NULL, NULL, NULL, csi_shape_i32, NULL, NULL, NULL}, /* CSINN_OP_SHAPE */
    {csi_shuffle_channel_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_shuffle_channel_f32, NULL}, /* CSINN_OP_SHUFFLE_CHANNEL */
    {csi_sigmoid_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_sigmoid_f32, NULL}, /* CSINN_OP_SIGMOID */
    {csi_sign_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_sign_f32, NULL}, /* CSINN_OP_SIGN */
    {csi_sin_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_sin_f32, NULL}, /* CSINN_OP_SIN */
    {csi_sinh_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_sinh_f32, NULL}, /* CSINN_OP_SINH */
    {csi_slice_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_slice_f32, NULL}, /* CSINN_OP_SLICE */
    {csi_softmax_u8, csi_softmax_i8, NULL, NULL, NULL, NULL, NULL, csi_softmax_f32, NULL}, /* CSINN_OP_SOFTMAX */
    {csi_softplus_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_softplus_f32, NULL}, /* CSINN_OP_SOFTPLUS */
    {csi_softrelu_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_softrelu_f32, NULL}, /* CSINN_OP_SOFTRELU */
    {csi_softsign_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_softsign_f32, NULL}, /* CSINN_OP_SOFTSIGN */
    {csi_space_to_batch_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_space_to_batch_f32, NULL}, /* CSINN_OP_SPACE_TO_BATCH */
    {csi_space_to_depth_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_space_to_depth_f32, NULL}, /* CSINN_OP_SPACE_TO_DEPTH */
    {csi_split_u8, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_SPLIT */
    {csi_sqrt_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_sqrt_f32, NULL}, /* CSINN_OP_SQRT */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, csi_square_f32, NULL}, /* CSINN_OP_SQUARE */
    {csi_squeeze_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_squeeze_f32, NULL}, /* CSINN_OP_SQUEEZE */
    {csi_stack_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_stack_f32, NULL}, /* CSINN_OP_STACK */
    {csi_strided_slice_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_strided_slice_f32, NULL}, /* CSINN_OP_STRIDED_SLICE */
    {csi_sub_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_sub_f32, NULL}, /* CSINN_OP_SUB */
    {csi_sum_stride_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_sum_stride_f32, NULL}, /* CSINN_OP_SUM */
    {csi_tan_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_tan_f32, NULL}, /* CSINN_OP_TAN */
    {csi_tanh_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_tanh_f32, csi_tanh_f64}, /* CSINN_OP_TANH */
    {csi_threshold_relu_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_threshold_relu_f32, NULL}, /* CSINN_OP_THRESHOLD_RELU */
    {csi_tile_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_tile_f32, NULL}, /* CSINN_OP_TILE */
    {csi_topk_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_topk_f32, NULL}, /* CSINN_OP_TOPK */
    {csi_transpose_u8, csi_transpose_i8, NULL, NULL, NULL, NULL, NULL, csi_transpose_f32, NULL}, /* CSINN_OP_TRANSPOSE */
    {csi_trunc_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_trunc_f32, NULL}, /* CSINN_OP_TRUNC */
    {csi_unpooling_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_unpooling_f32, NULL}, /* CSINN_OP_UNPOOLING */
    {csi_unstack_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_unstack_f32, NULL}, /* CSINN_OP_UNSTACK */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_WHERE */
    {csi_xor_u8, NULL, NULL, NULL, csi_xor_u32, NULL, NULL, NULL, NULL}, /* CSINN_OP_XOR */
    {csi_yuv_rgb_scale_u8, NULL, NULL, NULL, NULL, NULL, NULL, csi_yuv_rgb_scale_f32, NULL}, /* CSINN_OP_YUV_RGB_SCALE */
};

void *csi_bc_map_ref(int op, int dtype)
{
    return csi_bc_map_table_ref[op][dtype];
}

void *csi_bc_map_ovx(int op, int dtype);
void *csi_bc_map_c906(int op, int dtype);
void *csi_bc_map_pnna(int op, int dtype);
void *csi_bc_func_table[CSINN_API_SIZE] = {
    csi_bc_map_ref, /* ref */
    NULL, /* c860 */
#ifdef CSI_BUILD_C906
    csi_bc_map_c906,
#else
    NULL, /* c906 */
#endif
    NULL, /* c910 */
#ifdef CSI_BUILD_OPENVX
    csi_bc_map_ovx,
#else
    NULL, /* anole */
#endif
    NULL, /* tx510 */
#ifdef CSI_BUILD_PNNA
    csi_bc_map_pnna,
#else
    NULL, /* light */
#endif
    NULL, /* tvmgen */
};

void *csi_bc_map(int api, int op, int dtype)
{
    void* (*func)() = csi_bc_func_table[api];
    return func(op, dtype);
}

void csi_session_init(struct csi_session *sess)
{
    void* (*func)();
    func = csi_bc_map(sess->base_api, CSINN_SESSION_INIT, sess->base_dtype);
    if (func != NULL) {
        func(sess);
    }
}

void csi_session_deinit(struct csi_session *sess)
{
    void* (*func)();
    func = csi_bc_map(sess->base_api, CSINN_SESSION_DEINIT, sess->base_dtype);
    if (func != NULL) {
        func(sess);
    }
}

void csi_set_output_number(int number, struct csi_session *sess)
{
    void (*func)();
    func = csi_bc_map(sess->base_api, CSINN_SET_OUTPUT_NUMBER, sess->base_dtype);
    if (func != NULL) {
        func(number, sess);
    }
}

void csi_set_input_number(int number, struct csi_session *sess)
{
    void (*func)();
    func = csi_bc_map(sess->base_api, CSINN_SET_INPUT_NUMBER, sess->base_dtype);
    if (func != NULL) {
        func(number, sess);
    }
}

int csi_get_output_number(struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, CSINN_GET_OUTPUT_NUMBER, sess->base_dtype);
    if (func != NULL) {
        return func(sess);
    }
    return CSINN_FALSE;
}

int csi_get_input_number(struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, CSINN_GET_INPUT_NUMBER, sess->base_dtype);
    if (func != NULL) {
        return func(sess);
    }
    return CSINN_FALSE;
}

int csi_set_output(int index, struct csi_tensor *output, struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, CSINN_SET_OUTPUT, sess->base_dtype);
    if (func != NULL) {
        return func(index, output, sess);
    }
    return CSINN_FALSE;
}

int csi_set_input(int index, struct csi_tensor *input, struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, CSINN_SET_INPUT, sess->base_dtype);
    if (func != NULL) {
        return func(index, input, sess);
    }
    return CSINN_FALSE;
}

int csi_get_output(int index, struct csi_tensor *output, struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, CSINN_GET_OUTPUT, sess->base_dtype);
    if (func != NULL) {
        return func(index, output, sess);
    }
    return CSINN_FALSE;
}

int csi_get_input(int index, struct csi_tensor *input, struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, CSINN_GET_INPUT, sess->base_dtype);
    if (func != NULL) {
        return func(index, input, sess);
    }
    return CSINN_FALSE;
}

int csi_update_input(int index, struct csi_tensor *input, struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, CSINN_UPDATE_INPUT, sess->base_dtype);
    if (func != NULL) {
        return func(index, input, sess);
    }
    return CSINN_FALSE;
}

int csi_session_setup(struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, CSINN_SESSION_SETUP, sess->base_dtype);
    if (func != NULL) {
        return func(sess);
    }
    return CSINN_FALSE;
}

int csi_session_run(struct csi_session *sess)
{
    int (*func)();
    func = csi_bc_map(sess->base_api, CSINN_SESSION_RUN, sess->base_dtype);
    if (func != NULL) {
        return func(sess);
    }
    return CSINN_FALSE;
}
