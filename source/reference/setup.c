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

#include "csi_ref.h"

void csi_ref_nn_init(struct csi_tensor *input,
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
            int32_t input_val = round(input_data[i] / output->qinfo->scale) + output->qinfo->zero_point;
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
            int32_t input_val = round(input_data[i] / output->qinfo->scale) + output->qinfo->zero_point;
            if (input_val < -127) {
                input_val = 0;
            } else if (input_val > 127) {
                input_val = 127;
            }
            output_data[i] = input_val;
        }
    }
}

void csi_ref_nn_deinit(struct csi_tensor *input,
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
            float x = input_data[i];
            x -= input->qinfo->zero_point;
            output_data[i] = x * input->qinfo->scale;
        }
    } else if (input->dtype == CSINN_DTYPE_INT8){
        int8_t *input_data = input->data;
        float *output_data = output->data;
        for (int i = 0; i < size; i++) {
            float x = input_data[i];
            x -= input->qinfo->zero_point;
            output_data[i] = x * input->qinfo->scale;
        }
    }
}

void* csi_ref_bc_map_table[CSINN_OP_AND_UTILS_SIZE][CSINN_DTYPE_SIZE] = {
    {csi_ref_abs_quant, csi_ref_abs_quant, csi_ref_abs_quant, csi_ref_abs_quant, csi_ref_abs_quant, csi_ref_abs_quant, csi_ref_abs_quant, csi_ref_abs_f32, NULL}, /* CSINN_OP_ABS */
    {csi_ref_acos_quant, csi_ref_acos_quant, csi_ref_acos_quant, csi_ref_acos_quant, csi_ref_acos_quant, csi_ref_acos_quant, csi_ref_acos_quant, csi_ref_acos_f32, NULL}, /* CSINN_OP_ACOS */
    {csi_ref_acosh_quant, csi_ref_acosh_quant, csi_ref_acosh_quant, csi_ref_acosh_quant, csi_ref_acosh_quant, csi_ref_acosh_quant, csi_ref_acosh_quant, csi_ref_acosh_f32, NULL}, /* CSINN_OP_ACOSH */
    {csi_ref_add_quant, csi_ref_add_quant, csi_ref_add_quant, csi_ref_add_quant, csi_ref_add_quant, csi_ref_add_quant, csi_ref_add_quant, csi_ref_add_f32, NULL}, /* CSINN_OP_ADD */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_ALL */
    {csi_ref_and_u8, csi_ref_and_i8, NULL, NULL, csi_ref_and_u32, NULL, NULL, NULL, NULL}, /* CSINN_OP_AND */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_ANY */
    {csi_ref_arange_quant, csi_ref_arange_quant, csi_ref_arange_quant, csi_ref_arange_quant, csi_ref_arange_quant, csi_ref_arange_quant, csi_ref_arange_quant, csi_ref_arange_f32, NULL}, /* CSINN_OP_ARANGE */
    {csi_ref_argmax_stride_quant, csi_ref_argmax_stride_quant, csi_ref_argmax_stride_quant, csi_ref_argmax_stride_quant, csi_ref_argmax_stride_quant, csi_ref_argmax_stride_quant, csi_ref_argmax_stride_quant, csi_ref_argmax_stride_i32_f32, NULL}, /* CSINN_OP_ARGMAX */
    {csi_ref_argmin_stride_quant, csi_ref_argmin_stride_quant, csi_ref_argmin_stride_quant, csi_ref_argmin_stride_quant, csi_ref_argmin_stride_quant, csi_ref_argmin_stride_quant, csi_ref_argmin_stride_quant, csi_ref_argmin_stride_i32_f32, NULL}, /* CSINN_OP_ARGMIN */
    {csi_ref_asin_quant, csi_ref_asin_quant, csi_ref_asin_quant, csi_ref_asin_quant, csi_ref_asin_quant, csi_ref_asin_quant, csi_ref_asin_quant, csi_ref_asin_f32, NULL}, /* CSINN_OP_ASIN */
    {csi_ref_asinh_quant, csi_ref_asinh_quant, csi_ref_asinh_quant, csi_ref_asinh_quant, csi_ref_asinh_quant, csi_ref_asinh_quant, csi_ref_asinh_quant, csi_ref_asinh_f32, NULL}, /* CSINN_OP_ASINH */
    {csi_ref_atan_quant, csi_ref_atan_quant, csi_ref_atan_quant, csi_ref_atan_quant, csi_ref_atan_quant, csi_ref_atan_quant, csi_ref_atan_quant, csi_ref_atan_f32, NULL}, /* CSINN_OP_ATAN */
    {csi_ref_atanh_quant, csi_ref_atanh_quant, csi_ref_atanh_quant, csi_ref_atanh_quant, csi_ref_atanh_quant, csi_ref_atanh_quant, csi_ref_atanh_quant, csi_ref_atanh_f32, NULL}, /* CSINN_OP_ATANH */
    {csi_ref_averagepool_quant, csi_ref_averagepool_quant, csi_ref_averagepool_quant, csi_ref_averagepool_quant, csi_ref_averagepool_quant, csi_ref_averagepool_quant, csi_ref_averagepool_quant, csi_ref_averagepool_f32, NULL}, /* CSINN_OP_AVGPOOL2D */
    {csi_ref_averagepool3d_quant, csi_ref_averagepool3d_quant, csi_ref_averagepool3d_quant, csi_ref_averagepool3d_quant, csi_ref_averagepool3d_quant, csi_ref_averagepool3d_quant, csi_ref_averagepool3d_quant, csi_ref_averagepool3d_f32, NULL}, /* CSINN_OP_AVGPOOL3D */
    {csi_ref_batch_normalization_quant, csi_ref_batch_normalization_quant, csi_ref_batch_normalization_quant, csi_ref_batch_normalization_quant, csi_ref_batch_normalization_quant, csi_ref_batch_normalization_quant, csi_ref_batch_normalization_quant, csi_ref_batch_normalization_f32, NULL}, /* CSINN_OP_BN */
    {csi_ref_batch_to_space_quant, csi_ref_batch_to_space_quant, csi_ref_batch_to_space_quant, csi_ref_batch_to_space_quant, csi_ref_batch_to_space_quant, csi_ref_batch_to_space_quant, csi_ref_batch_to_space_quant, csi_ref_batch_to_space_f32, NULL}, /* CSINN_OP_BATCH_TO_SPACE */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_BATCH_TO_SPACE_ND */
    {csi_ref_broadcast_to_quant, csi_ref_broadcast_to_quant, csi_ref_broadcast_to_quant, csi_ref_broadcast_to_quant, csi_ref_broadcast_to_quant, csi_ref_broadcast_to_quant, csi_ref_broadcast_to_quant, csi_ref_broadcast_to_f32, NULL}, /* CSINN_OP_BROADCOST */
    {csi_ref_ceil_quant, csi_ref_ceil_quant, csi_ref_ceil_quant, csi_ref_ceil_quant, csi_ref_ceil_quant, csi_ref_ceil_quant, csi_ref_ceil_quant, csi_ref_ceil_f32, NULL}, /* CSINN_OP_CEIL */
    {csi_ref_clip_quant, csi_ref_clip_quant, csi_ref_clip_quant, csi_ref_clip_quant, csi_ref_clip_quant, csi_ref_clip_quant, csi_ref_clip_quant, csi_ref_clip_f32, NULL}, /* CSINN_OP_CLIP */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, csi_ref_col2im_f32, NULL}, /* CSINN_OP_COL2IM */
    {csi_ref_concat_quant, csi_ref_concat_quant, csi_ref_concat_quant, csi_ref_concat_quant, csi_ref_concat_quant, csi_ref_concat_quant, csi_ref_concat_quant, csi_ref_concat_f32, NULL}, /* CSINN_OP_CONCAT */
    {csi_ref_conv2d_quant, csi_ref_conv2d_quant, NULL, NULL, NULL, NULL, NULL, csi_ref_conv2d_f32, NULL}, /* CSINN_OP_CONV2D */
    {csi_ref_conv2d_relu_quant, csi_ref_conv2d_relu_quant, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_CONV2D_RELU */
    {csi_ref_conv2d_relu6_quant, csi_ref_conv2d_relu6_quant, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_CONV2D_RELU6 */
    {csi_ref_conv2d_channel_quant, csi_ref_conv2d_channel_quant, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_CONV2D_CHANNEL */
    {csi_ref_conv2d_channel_relu_quant, csi_ref_conv2d_channel_relu_quant, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_CONV2D_CHANNEL_RELU */
    {csi_ref_conv2d_channel_relu6_quant, csi_ref_conv2d_channel_relu6_quant, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_CONV2D_CHANNEL_RELU6 */
    {csi_ref_depthwise_conv2d_quant, csi_ref_depthwise_conv2d_quant, NULL, NULL, NULL, NULL, NULL, csi_ref_depthwise_conv2d_f32, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D */
    {csi_ref_depthwise_conv2d_relu_quant, csi_ref_depthwise_conv2d_relu_quant, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_RELU */
    {csi_ref_depthwise_conv2d_relu6_quant, csi_ref_depthwise_conv2d_relu6_quant, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_RELU6 */
    {csi_ref_depthwise_conv2d_channel_quant, csi_ref_depthwise_conv2d_channel_quant, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL */
    {csi_ref_depthwise_conv2d_channel_relu_quant, csi_ref_depthwise_conv2d_channel_relu_quant, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU */
    {csi_ref_depthwise_conv2d_channel_relu6_quant, csi_ref_depthwise_conv2d_channel_relu6_quant, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6 */
    {csi_ref_group_conv2d_quant, csi_ref_group_conv2d_quant, NULL, NULL, NULL, NULL, NULL, csi_ref_group_conv2d_f32, NULL}, /* CSINN_OP_GROUP_CONV2D */
    {csi_ref_group_conv2d_relu_quant, csi_ref_group_conv2d_relu_quant, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_GROUP_CONV2D_RELU */
    {csi_ref_group_conv2d_relu6_quant, csi_ref_group_conv2d_relu6_quant, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_GROUP_CONV2D_RELU6 */
    {csi_ref_group_conv2d_channel_quant, csi_ref_group_conv2d_channel_quant, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_GROUP_CONV2D_CHANNEL */
    {csi_ref_group_conv2d_channel_relu_quant, csi_ref_group_conv2d_channel_relu_quant, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_GROUP_CONV2D_CHANNEL_RELU */
    {csi_ref_conv3d_quant, csi_ref_conv3d_quant, csi_ref_conv3d_quant, csi_ref_conv3d_quant, csi_ref_conv3d_quant, csi_ref_conv3d_quant, csi_ref_conv3d_quant, csi_ref_conv3d_f32, NULL}, /* CSINN_OP_CONV3D */
    {csi_ref_cos_quant, csi_ref_cos_quant, csi_ref_cos_quant, csi_ref_cos_quant, csi_ref_cos_quant, csi_ref_cos_quant, csi_ref_cos_quant, csi_ref_cos_f32, NULL}, /* CSINN_OP_COS */
    {csi_ref_cosh_quant, csi_ref_cosh_quant, csi_ref_cosh_quant, csi_ref_cosh_quant, csi_ref_cosh_quant, csi_ref_cosh_quant, csi_ref_cosh_quant, csi_ref_cosh_f32, NULL}, /* CSINN_OP_COSH */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_CROP */
    {csi_ref_cumprod_quant, csi_ref_cumprod_quant, csi_ref_cumprod_quant, csi_ref_cumprod_quant, csi_ref_cumprod_quant, csi_ref_cumprod_quant, csi_ref_cumprod_quant, csi_ref_cumprod_f32, NULL}, /* CSINN_OP_CUMPROD */
    {csi_ref_cumsum_quant, csi_ref_cumsum_quant, csi_ref_cumsum_quant, csi_ref_cumsum_quant, csi_ref_cumsum_quant, csi_ref_cumsum_quant, csi_ref_cumsum_quant, csi_ref_cumsum_f32, NULL}, /* CSINN_OP_CUMSUM */
    {csi_ref_deconv2d_quant, csi_ref_deconv2d_quant, NULL, NULL, NULL, NULL, NULL, csi_ref_deconv2d_f32, NULL}, /* CSINN_OP_DECONV2D */
    {csi_ref_depthwise_deconv2d_quant, csi_ref_depthwise_deconv2d_quant, NULL, NULL, NULL, NULL, NULL, csi_ref_depthwise_deconv2d_f32, NULL}, /* CSINN_OP_DEPTHWISE_DECONV2D */
    {csi_ref_deconv3d_quant, csi_ref_deconv3d_quant, NULL, NULL, NULL, NULL, NULL, csi_ref_deconv3d_f32, NULL}, /* CSINN_OP_DECONV3D */
    {csi_ref_depth_to_space_quant, csi_ref_depth_to_space_quant, csi_ref_depth_to_space_quant, csi_ref_depth_to_space_quant, csi_ref_depth_to_space_quant, csi_ref_depth_to_space_quant, csi_ref_depth_to_space_quant, csi_ref_depth_to_space_f32, NULL}, /* CSINN_OP_DEPTH_TO_SPACE */
    {csi_ref_div_quant, csi_ref_div_quant, csi_ref_div_quant, csi_ref_div_quant, csi_ref_div_quant, csi_ref_div_quant, csi_ref_div_quant, csi_ref_div_f32, NULL}, /* CSINN_OP_DIV */
    {csi_ref_elu_quant, csi_ref_elu_quant, csi_ref_elu_quant, csi_ref_elu_quant, csi_ref_elu_quant, csi_ref_elu_quant, csi_ref_elu_quant, csi_ref_elu_f32, NULL}, /* CSINN_OP_ELU */
    {csi_ref_equal_quant, csi_ref_equal_quant, csi_ref_equal_quant, csi_ref_equal_quant, csi_ref_equal_quant, csi_ref_equal_quant, csi_ref_equal_quant, csi_ref_equal_f32, NULL}, /* CSINN_OP_EQUANL */
    {csi_ref_erf_quant, csi_ref_erf_quant, csi_ref_erf_quant, csi_ref_erf_quant, csi_ref_erf_quant, csi_ref_erf_quant, csi_ref_erf_quant, csi_ref_erf_f32, NULL}, /* CSINN_OP_ERF */
    {csi_ref_exp_quant, csi_ref_exp_quant, csi_ref_exp_quant, csi_ref_exp_quant, csi_ref_exp_quant, csi_ref_exp_quant, csi_ref_exp_quant, csi_ref_exp_f32, NULL}, /* CSINN_OP_EXP */
    {csi_ref_expand_dims_quant, csi_ref_expand_dims_quant, csi_ref_expand_dims_quant, csi_ref_expand_dims_quant, csi_ref_expand_dims_quant, csi_ref_expand_dims_quant, csi_ref_expand_dims_quant, csi_ref_expand_dims_f32, NULL}, /* CSINN_OP_EXPAND_DIMS */
    {csi_ref_expm1_quant, csi_ref_expm1_quant, csi_ref_expm1_quant, csi_ref_expm1_quant, csi_ref_expm1_quant, csi_ref_expm1_quant, csi_ref_expm1_quant, csi_ref_expm1_f32, NULL}, /* CSINN_OP_EXPM1 */
    {csi_ref_flatten, csi_ref_flatten, csi_ref_flatten, csi_ref_flatten, csi_ref_flatten, csi_ref_flatten, csi_ref_flatten, csi_ref_flatten, csi_ref_flatten}, /* CSINN_OP_FLATTEN */
    {csi_ref_floor_divide_quant, csi_ref_floor_divide_quant, csi_ref_floor_divide_quant, csi_ref_floor_divide_quant, csi_ref_floor_divide_quant, csi_ref_floor_divide_quant, csi_ref_floor_divide_quant, csi_ref_floor_divide_f32, NULL}, /* CSINN_OP_FLOOR_DIVIDE */
    {csi_ref_floor_mod_quant, csi_ref_floor_mod_quant, csi_ref_floor_mod_quant, csi_ref_floor_mod_quant, csi_ref_floor_mod_quant, csi_ref_floor_mod_quant, csi_ref_floor_mod_quant, csi_ref_floor_mod_f32, NULL}, /* CSINN_OP_FLOOR_MOD */
    {csi_ref_floor_quant, csi_ref_floor_quant, csi_ref_floor_quant, csi_ref_floor_quant, csi_ref_floor_quant, csi_ref_floor_quant, csi_ref_floor_quant, csi_ref_floor_f32, NULL}, /* CSINN_OP_FLOOR */
    {csi_ref_fullyconnected_quant, csi_ref_fullyconnected_quant, NULL, NULL, NULL, NULL, NULL, csi_ref_fullyconnected_f32, NULL}, /* CSINN_OP_FULLYCONNECTED */
    {csi_ref_gather_nd_quant, csi_ref_gather_nd_quant, csi_ref_gather_nd_quant, csi_ref_gather_nd_quant, csi_ref_gather_nd_quant, csi_ref_gather_nd_quant, csi_ref_gather_nd_quant, csi_ref_gather_nd_f32, NULL}, /* CSINN_OP_GATHER_ND */
    {csi_ref_gather_quant, csi_ref_gather_quant, csi_ref_gather_quant, csi_ref_gather_quant, csi_ref_gather_quant, csi_ref_gather_quant, csi_ref_gather_quant, csi_ref_gather_f32, NULL}, /* CSINN_OP_GATHER */
    {csi_ref_global_averagepool_quant, csi_ref_global_averagepool_quant, csi_ref_global_averagepool_quant, csi_ref_global_averagepool_quant, csi_ref_global_averagepool_quant, csi_ref_global_averagepool_quant, csi_ref_global_averagepool_quant, csi_ref_global_averagepool_f32, NULL}, /* CSINN_OP_GLOBAL_AVGPOOL2D */
    {csi_ref_global_maxpool_quant, csi_ref_global_maxpool_quant, csi_ref_global_maxpool_quant, csi_ref_global_maxpool_quant, csi_ref_global_maxpool_quant, csi_ref_global_maxpool_quant, csi_ref_global_maxpool_quant, csi_ref_global_maxpool_f32, NULL}, /* CSINN_OP_GLOBAL_MAXPOOL2D */
    {csi_ref_greater_equal_quant, csi_ref_greater_equal_quant, csi_ref_greater_equal_quant, csi_ref_greater_equal_quant, csi_ref_greater_equal_quant, csi_ref_greater_equal_quant, csi_ref_greater_equal_quant, csi_ref_greater_equal_f32, NULL}, /* CSINN_OP_GREATHER_EQUAL */
    {csi_ref_greater_quant, csi_ref_greater_quant, csi_ref_greater_quant, csi_ref_greater_quant, csi_ref_greater_quant, csi_ref_greater_quant, csi_ref_greater_quant, csi_ref_greater_f32, NULL}, /* CSINN_OP_GREATHER */
    {csi_ref_hard_sigmoid_quant, csi_ref_hard_sigmoid_quant, csi_ref_hard_sigmoid_quant, csi_ref_hard_sigmoid_quant, csi_ref_hard_sigmoid_quant, csi_ref_hard_sigmoid_quant, csi_ref_hard_sigmoid_quant, csi_ref_hard_sigmoid_f32, NULL}, /* CSINN_OP_HARD_SIGMOID */
    {csi_ref_im2col_quant, csi_ref_im2col_quant, csi_ref_im2col_quant, csi_ref_im2col_quant, csi_ref_im2col_quant, csi_ref_im2col_quant, csi_ref_im2col_quant, csi_ref_im2col_f32, NULL}, /* CSINN_OP_IM2COL */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, csi_ref_isnan_bool_f32, NULL}, /* CSINN_OP_ISNAN */
    {csi_ref_l2_normalization_quant, csi_ref_l2_normalization_quant, csi_ref_l2_normalization_quant, csi_ref_l2_normalization_quant, csi_ref_l2_normalization_quant, csi_ref_l2_normalization_quant, csi_ref_l2_normalization_quant, csi_ref_l2_normalization_f32, NULL}, /* CSINN_OP_L2N */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, csi_ref_l2pool_f32, NULL}, /* CSINN_OP_L2POOL2D */
    {csi_ref_leaky_relu_quant, csi_ref_leaky_relu_quant, csi_ref_leaky_relu_quant, csi_ref_leaky_relu_quant, csi_ref_leaky_relu_quant, csi_ref_leaky_relu_quant, csi_ref_leaky_relu_quant, csi_ref_leaky_relu_f32, NULL}, /* CSINN_OP_LEAKY_RELU */
    {csi_ref_less_equal_quant, csi_ref_less_equal_quant, csi_ref_less_equal_quant, csi_ref_less_equal_quant, csi_ref_less_equal_quant, csi_ref_less_equal_quant, csi_ref_less_equal_quant, csi_ref_less_equal_f32, NULL}, /* CSINN_OP_LESS_EQUAL */
    {csi_ref_less_quant, csi_ref_less_quant, csi_ref_less_quant, csi_ref_less_quant, csi_ref_less_quant, csi_ref_less_quant, csi_ref_less_quant, csi_ref_less_f32, NULL}, /* CSINN_OP_LESS */
    {csi_ref_log_softmax_quant, csi_ref_log_softmax_quant, csi_ref_log_softmax_quant, csi_ref_log_softmax_quant, csi_ref_log_softmax_quant, csi_ref_log_softmax_quant, csi_ref_log_softmax_quant, csi_ref_log_softmax_f32, NULL}, /* CSINN_OP_LOG_SOFTMAX */
    {csi_ref_log_quant, csi_ref_log_quant, csi_ref_log_quant, csi_ref_log_quant, csi_ref_log_quant, csi_ref_log_quant, csi_ref_log_quant, csi_ref_log_f32, NULL}, /* CSINN_OP_LOG */
    {csi_ref_log1p_quant, csi_ref_log1p_quant, csi_ref_log1p_quant, csi_ref_log1p_quant, csi_ref_log1p_quant, csi_ref_log1p_quant, csi_ref_log1p_quant, csi_ref_log1p_f32, NULL}, /* CSINN_OP_LOG1P */
    {csi_ref_logical_and_quant, csi_ref_logical_and_quant, csi_ref_logical_and_quant, csi_ref_logical_and_quant, csi_ref_logical_and_quant, csi_ref_logical_and_quant, csi_ref_logical_and_quant, csi_ref_logical_and_f32, NULL}, /* CSINN_OP_LOGICAL_AND */
    {csi_ref_logical_not_quant, csi_ref_logical_not_quant, csi_ref_logical_not_quant, csi_ref_logical_not_quant, csi_ref_logical_not_quant, csi_ref_logical_not_quant, csi_ref_logical_not_quant, csi_ref_logical_not_f32, NULL}, /* CSINN_OP_LOGICAL_NOT */
    {csi_ref_logical_or_quant, csi_ref_logical_or_quant, csi_ref_logical_or_quant, csi_ref_logical_or_quant, csi_ref_logical_or_quant, csi_ref_logical_or_quant, csi_ref_logical_or_quant, csi_ref_logical_or_f32, NULL}, /* CSINN_OP_LOGICAL_OR */
    {csi_ref_logical_xor_quant, csi_ref_logical_xor_quant, csi_ref_logical_xor_quant, csi_ref_logical_xor_quant, csi_ref_logical_xor_quant, csi_ref_logical_xor_quant, csi_ref_logical_xor_quant, csi_ref_logical_xor_f32, NULL}, /* CSINN_OP_LOGICAL_XOR */
    {csi_ref_lrn_quant, csi_ref_lrn_quant, csi_ref_lrn_quant, csi_ref_lrn_quant, csi_ref_lrn_quant, csi_ref_lrn_quant, csi_ref_lrn_quant, csi_ref_lrn_f32, NULL}, /* CSINN_OP_LRN */
    {csi_ref_matmul_quant, csi_ref_matmul_quant, csi_ref_matmul_quant, csi_ref_matmul_quant, csi_ref_matmul_quant, csi_ref_matmul_quant, csi_ref_matmul_quant, csi_ref_matmul_f32, NULL}, /* CSINN_OP_MATMUL */
    {csi_ref_max_stride_quant, csi_ref_max_stride_quant, csi_ref_max_stride_quant, csi_ref_max_stride_quant, csi_ref_max_stride_quant, csi_ref_max_stride_quant, csi_ref_max_stride_quant, csi_ref_max_stride_f32, NULL}, /* CSINN_OP_MAX */
    {csi_ref_maximum_quant, csi_ref_maximum_quant, csi_ref_maximum_quant, csi_ref_maximum_quant, csi_ref_maximum_quant, csi_ref_maximum_quant, csi_ref_maximum_quant, csi_ref_maximum_f32, NULL}, /* CSINN_OP_MAXINUM */
    {csi_ref_maxpool_quant, csi_ref_maxpool_quant, csi_ref_maxpool_quant, csi_ref_maxpool_quant, csi_ref_maxpool_quant, csi_ref_maxpool_quant, csi_ref_maxpool_quant, csi_ref_maxpool_f32, NULL}, /* CSINN_OP_MAXPOOL2D */
    {csi_ref_maxpool2d_locat_quant, csi_ref_maxpool2d_locat_quant, csi_ref_maxpool2d_locat_quant, csi_ref_maxpool2d_locat_quant, csi_ref_maxpool2d_locat_quant, csi_ref_maxpool2d_locat_quant, csi_ref_maxpool2d_locat_quant, csi_ref_maxpool2d_locat_f32, NULL}, /* CSINN_OP_MAXPOOL2D_LOCAT */
    {csi_ref_maxpool3d_quant, csi_ref_maxpool3d_quant, csi_ref_maxpool3d_quant, csi_ref_maxpool3d_quant, csi_ref_maxpool3d_quant, csi_ref_maxpool3d_quant, csi_ref_maxpool3d_quant, csi_ref_maxpool3d_f32, NULL}, /* CSINN_OP_MAXPOOL3D */
    {csi_ref_mean_stride_quant, csi_ref_mean_stride_quant, csi_ref_mean_stride_quant, csi_ref_mean_stride_quant, csi_ref_mean_stride_quant, csi_ref_mean_stride_quant, csi_ref_mean_stride_quant, NULL, NULL}, /* CSINN_OP_MEAN */
    {csi_ref_mean_stride_quant, csi_ref_mean_stride_quant, csi_ref_mean_stride_quant, csi_ref_mean_stride_quant, csi_ref_mean_stride_quant, csi_ref_mean_stride_quant, csi_ref_mean_stride_quant, csi_ref_mean_stride_f32, NULL}, /* CSINN_OP_MEAN_STRIDE */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_MIN */
    {csi_ref_min_stride_quant, csi_ref_min_stride_quant, csi_ref_min_stride_quant, csi_ref_min_stride_quant, csi_ref_min_stride_quant, csi_ref_min_stride_quant, csi_ref_min_stride_quant, csi_ref_min_stride_f32, NULL}, /* CSINN_OP_MIN_STRIDE */
    {csi_ref_minimum_quant, csi_ref_minimum_quant, csi_ref_minimum_quant, csi_ref_minimum_quant, csi_ref_minimum_quant, csi_ref_minimum_quant, csi_ref_minimum_quant, csi_ref_minimum_f32, NULL}, /* CSINN_OP_MINIMUM */
    {csi_ref_mod_quant, csi_ref_mod_quant, csi_ref_mod_quant, csi_ref_mod_quant, csi_ref_mod_quant, csi_ref_mod_quant, csi_ref_mod_quant, csi_ref_mod_f32, NULL}, /* CSINN_OP_MOD */
    {csi_ref_mul_quant, csi_ref_mul_quant, csi_ref_mul_quant, csi_ref_mul_quant, csi_ref_mul_quant, csi_ref_mul_quant, csi_ref_mul_quant, csi_ref_mul_f32, NULL}, /* CSINN_OP_MUL */
    {csi_ref_ndarray_size_u8, csi_ref_ndarray_size_i8, NULL, NULL, NULL, csi_ref_ndarray_size_i32, NULL, csi_ref_ndarray_size_f32, NULL}, /* CSINN_OP_NDARRAY_SIZE */
    {csi_ref_negative_quant, csi_ref_negative_quant, csi_ref_negative_quant, csi_ref_negative_quant, csi_ref_negative_quant, csi_ref_negative_quant, csi_ref_negative_quant, csi_ref_negative_f32, NULL}, /* CSINN_OP_NEGATIIVE */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, csi_ref_non_max_suppression_std, NULL}, /* CSINN_OP_NON_MAX_SUPPRESSION */
    {csi_ref_not_equal_quant, csi_ref_not_equal_quant, csi_ref_not_equal_quant, csi_ref_not_equal_quant, csi_ref_not_equal_quant, csi_ref_not_equal_quant, csi_ref_not_equal_quant, csi_ref_not_equal_f32, NULL}, /* CSINN_OP_NOT_EQUAL */
    {csi_ref_not_u8, csi_ref_not_i8, NULL, NULL, csi_ref_not_u32, NULL, NULL, NULL, NULL}, /* CSINN_OP_NOT */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_ONE_HOT */
    {csi_ref_or_u8, csi_ref_or_i8, NULL, NULL, csi_ref_or_u32, NULL, NULL, NULL, NULL}, /* CSINN_OP_OR */
    {csi_ref_pad_quant, csi_ref_pad_quant, csi_ref_pad_quant, csi_ref_pad_quant, csi_ref_pad_quant, csi_ref_pad_quant, csi_ref_pad_quant, csi_ref_pad_f32, NULL}, /* CSINN_OP_PAD */
    {csi_ref_power_quant, csi_ref_power_quant, csi_ref_power_quant, csi_ref_power_quant, csi_ref_power_quant, csi_ref_power_quant, csi_ref_power_quant, csi_ref_power_f32, NULL}, /* CSINN_OP_POWER */
    {csi_ref_prelu_quant, csi_ref_prelu_quant, csi_ref_prelu_quant, csi_ref_prelu_quant, csi_ref_prelu_quant, csi_ref_prelu_quant, csi_ref_prelu_quant, csi_ref_prelu_f32, NULL}, /* CSINN_OP_PRELU */
    {csi_ref_prod_stride_quant, csi_ref_prod_stride_quant, csi_ref_prod_stride_quant, csi_ref_prod_stride_quant, csi_ref_prod_stride_quant, csi_ref_prod_stride_quant, csi_ref_prod_stride_quant, csi_ref_prod_stride_f32, NULL}, /* CSINN_OP_PROD */
    {csi_ref_proposal_quant, csi_ref_proposal_quant, csi_ref_proposal_quant, csi_ref_proposal_quant, csi_ref_proposal_quant, csi_ref_proposal_quant, csi_ref_proposal_quant, csi_ref_proposal_f32, NULL}, /* CSINN_OP_PROPOSAL */
    {csi_ref_psroipooling_quant, csi_ref_psroipooling_quant, csi_ref_psroipooling_quant, csi_ref_psroipooling_quant, csi_ref_psroipooling_quant, csi_ref_psroipooling_quant, csi_ref_psroipooling_quant, csi_ref_psroipooling_f32, NULL}, /* CSINN_OP_PSROIPOOLING */
    {csi_ref_reduce_logsumexp_quant, csi_ref_reduce_logsumexp_quant, csi_ref_reduce_logsumexp_quant, csi_ref_reduce_logsumexp_quant, csi_ref_reduce_logsumexp_quant, csi_ref_reduce_logsumexp_quant, csi_ref_reduce_logsumexp_quant, csi_ref_reduce_logsumexp_f32, NULL}, /* CSINN_OP_REDUCE_LOGSUMEXP */
    {csi_ref_reduce_max_quant, csi_ref_reduce_max_quant, csi_ref_reduce_max_quant, csi_ref_reduce_max_quant, csi_ref_reduce_max_quant, csi_ref_reduce_max_quant, csi_ref_reduce_max_quant, csi_ref_reduce_max_f32, NULL}, /* CSINN_OP_REDUCE_MAX */
    {csi_ref_reduce_mean_quant, csi_ref_reduce_mean_quant, csi_ref_reduce_mean_quant, csi_ref_reduce_mean_quant, csi_ref_reduce_mean_quant, csi_ref_reduce_mean_quant, csi_ref_reduce_mean_quant, csi_ref_reduce_mean_f32, NULL}, /* CSINN_OP_REDUCE_MEAN */
    {csi_ref_reduce_min_quant, csi_ref_reduce_min_quant, csi_ref_reduce_min_quant, csi_ref_reduce_min_quant, csi_ref_reduce_min_quant, csi_ref_reduce_min_quant, csi_ref_reduce_min_quant, csi_ref_reduce_min_f32, NULL}, /* CSINN_OP_REDUCE_MIN */
    {csi_ref_reduce_prod_quant, csi_ref_reduce_prod_quant, csi_ref_reduce_prod_quant, csi_ref_reduce_prod_quant, csi_ref_reduce_prod_quant, csi_ref_reduce_prod_quant, csi_ref_reduce_prod_quant, csi_ref_reduce_prod_f32, NULL}, /* CSINN_OP_REDUCE_PROD */
    {csi_ref_reduce_sum_quant, csi_ref_reduce_sum_quant, csi_ref_reduce_sum_quant, csi_ref_reduce_sum_quant, csi_ref_reduce_sum_quant, csi_ref_reduce_sum_quant, csi_ref_reduce_sum_quant, csi_ref_reduce_sum_f32, NULL}, /* CSINN_OP_REDUCE_SUM */
    {csi_ref_relu_quant, csi_ref_relu_quant, csi_ref_relu_quant, csi_ref_relu_quant, csi_ref_relu_quant, csi_ref_relu_quant, csi_ref_relu_quant, csi_ref_relu_f32, NULL}, /* CSINN_OP_RELU */
    {csi_ref_relu1_quant, csi_ref_relu1_quant, csi_ref_relu1_quant, csi_ref_relu1_quant, csi_ref_relu1_quant, csi_ref_relu1_quant, csi_ref_relu1_quant, csi_ref_relu1_f32, NULL}, /* CSINN_OP_RELU1 */
    {csi_ref_relu6_quant, csi_ref_relu6_quant, csi_ref_relu6_quant, csi_ref_relu6_quant, csi_ref_relu6_quant, csi_ref_relu6_quant, csi_ref_relu6_quant, csi_ref_relu6_f32, NULL}, /* CSINN_OP_RELU6 */
    {csi_ref_relun_quant, csi_ref_relun_quant, csi_ref_relun_quant, csi_ref_relun_quant, csi_ref_relun_quant, csi_ref_relun_quant, csi_ref_relun_quant, csi_ref_relun_f32, NULL}, /* CSINN_OP_RELUN */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_REORG */
    {csi_ref_reshape, csi_ref_reshape, csi_ref_reshape, csi_ref_reshape, csi_ref_reshape, csi_ref_reshape, csi_ref_reshape, csi_ref_reshape, csi_ref_reshape}, /* CSINN_OP_RESHAPE */
    {csi_ref_resize_quant, csi_ref_resize_quant, csi_ref_resize_quant, csi_ref_resize_quant, csi_ref_resize_quant, csi_ref_resize_quant, csi_ref_resize_quant, csi_ref_resize_f32, NULL}, /* CSINN_OP_RESIZE */
    {csi_ref_reverse_quant, csi_ref_reverse_quant, csi_ref_reverse_quant, csi_ref_reverse_quant, csi_ref_reverse_quant, csi_ref_reverse_quant, csi_ref_reverse_quant, csi_ref_reverse_f32, NULL}, /* CSINN_OP_REVERSE */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, csi_ref_roi_align_f32, NULL}, /* CSINN_OP_ROIALIGN */
    {csi_ref_roipool_quant, csi_ref_roipool_quant, csi_ref_roipool_quant, csi_ref_roipool_quant, csi_ref_roipool_quant, csi_ref_roipool_quant, csi_ref_roipool_quant, csi_ref_roipool_f32, NULL}, /* CSINN_OP_ROIPOOL */
    {csi_ref_round_quant, csi_ref_round_quant, csi_ref_round_quant, csi_ref_round_quant, csi_ref_round_quant, csi_ref_round_quant, csi_ref_round_quant, csi_ref_round_f32, NULL}, /* CSINN_OP_ROUND */
    {csi_ref_rsqrt_quant, csi_ref_rsqrt_quant, csi_ref_rsqrt_quant, csi_ref_rsqrt_quant, csi_ref_rsqrt_quant, csi_ref_rsqrt_quant, csi_ref_rsqrt_quant, csi_ref_rsqrt_f32, NULL}, /* CSINN_OP_RSQRT */
    {csi_ref_scatter_nd_quant, csi_ref_scatter_nd_quant, NULL, NULL, NULL, NULL, NULL, csi_ref_scatter_nd_f32, NULL}, /* CSINN_OP_SCATTER_ND */
    {csi_ref_segment_max_quant, csi_ref_segment_max_quant, csi_ref_segment_max_quant, csi_ref_segment_max_quant, csi_ref_segment_max_quant, csi_ref_segment_max_quant, csi_ref_segment_max_quant, csi_ref_segment_max_f32, NULL}, /* CSINN_OP_SEGMENT_MAX */
    {csi_ref_unsorted_segment_max_quant, csi_ref_unsorted_segment_max_quant, csi_ref_unsorted_segment_max_quant, csi_ref_unsorted_segment_max_quant, csi_ref_unsorted_segment_max_quant, csi_ref_unsorted_segment_max_quant, csi_ref_unsorted_segment_max_quant, csi_ref_unsorted_segment_max_f32, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MAX */
    {csi_ref_segment_mean_quant, csi_ref_segment_mean_quant, csi_ref_segment_mean_quant, csi_ref_segment_mean_quant, csi_ref_segment_mean_quant, csi_ref_segment_mean_quant, csi_ref_segment_mean_quant, csi_ref_segment_mean_f32, NULL}, /* CSINN_OP_SEGMENT_MEAN */
    {csi_ref_unsorted_segment_mean_quant, csi_ref_unsorted_segment_mean_quant, csi_ref_unsorted_segment_mean_quant, csi_ref_unsorted_segment_mean_quant, csi_ref_unsorted_segment_mean_quant, csi_ref_unsorted_segment_mean_quant, csi_ref_unsorted_segment_mean_quant, csi_ref_unsorted_segment_mean_f32, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MEAN */
    {csi_ref_segment_min_quant, csi_ref_segment_min_quant, csi_ref_segment_min_quant, csi_ref_segment_min_quant, csi_ref_segment_min_quant, csi_ref_segment_min_quant, csi_ref_segment_min_quant, csi_ref_segment_min_f32, NULL}, /* CSINN_OP_SEGMENT_MIN */
    {csi_ref_unsorted_segment_min_quant, csi_ref_unsorted_segment_min_quant, csi_ref_unsorted_segment_min_quant, csi_ref_unsorted_segment_min_quant, csi_ref_unsorted_segment_min_quant, csi_ref_unsorted_segment_min_quant, csi_ref_unsorted_segment_min_quant, csi_ref_unsorted_segment_min_f32, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_MIN */
    {csi_ref_segment_prod_quant, csi_ref_segment_prod_quant, csi_ref_segment_prod_quant, csi_ref_segment_prod_quant, csi_ref_segment_prod_quant, csi_ref_segment_prod_quant, csi_ref_segment_prod_quant, csi_ref_segment_prod_f32, NULL}, /* CSINN_OP_SEGMENT_PROD */
    {csi_ref_unsorted_segment_prod_quant, csi_ref_unsorted_segment_prod_quant, csi_ref_unsorted_segment_prod_quant, csi_ref_unsorted_segment_prod_quant, csi_ref_unsorted_segment_prod_quant, csi_ref_unsorted_segment_prod_quant, csi_ref_unsorted_segment_prod_quant, csi_ref_unsorted_segment_prod_f32, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_PROD */
    {csi_ref_segment_sum_quant, csi_ref_segment_sum_quant, csi_ref_segment_sum_quant, csi_ref_segment_sum_quant, csi_ref_segment_sum_quant, csi_ref_segment_sum_quant, csi_ref_segment_sum_quant, csi_ref_segment_sum_f32, NULL}, /* CSINN_OP_SEGMENT_SUM */
    {csi_ref_unsorted_segment_sum_quant, csi_ref_unsorted_segment_sum_quant, csi_ref_unsorted_segment_sum_quant, csi_ref_unsorted_segment_sum_quant, csi_ref_unsorted_segment_sum_quant, csi_ref_unsorted_segment_sum_quant, csi_ref_unsorted_segment_sum_quant, csi_ref_unsorted_segment_sum_f32, NULL}, /* CSINN_OP_UNSORTED_SEGMENT_SUM */
    {csi_ref_select_u8, csi_ref_select_i8, NULL, NULL, NULL, NULL, NULL, csi_ref_select_f32, NULL}, /* CSINN_OP_SELECT */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_SEQUENCE_MASK */
    {csi_ref_shape_u8, csi_ref_shape_i8, NULL, NULL, NULL, csi_ref_shape_i32, NULL, NULL, NULL}, /* CSINN_OP_SHAPE */
    {csi_ref_shuffle_channel_quant, csi_ref_shuffle_channel_quant, csi_ref_shuffle_channel_quant, csi_ref_shuffle_channel_quant, csi_ref_shuffle_channel_quant, csi_ref_shuffle_channel_quant, csi_ref_shuffle_channel_quant, csi_ref_shuffle_channel_f32, NULL}, /* CSINN_OP_SHUFFLE_CHANNEL */
    {csi_ref_sigmoid_quant, csi_ref_sigmoid_quant, csi_ref_sigmoid_quant, csi_ref_sigmoid_quant, csi_ref_sigmoid_quant, csi_ref_sigmoid_quant, csi_ref_sigmoid_quant, csi_ref_sigmoid_f32, NULL}, /* CSINN_OP_SIGMOID */
    {csi_ref_sign_quant, csi_ref_sign_quant, csi_ref_sign_quant, csi_ref_sign_quant, csi_ref_sign_quant, csi_ref_sign_quant, csi_ref_sign_quant, csi_ref_sign_f32, NULL}, /* CSINN_OP_SIGN */
    {csi_ref_sin_quant, csi_ref_sin_quant, csi_ref_sin_quant, csi_ref_sin_quant, csi_ref_sin_quant, csi_ref_sin_quant, csi_ref_sin_quant, csi_ref_sin_f32, NULL}, /* CSINN_OP_SIN */
    {csi_ref_sinh_quant, csi_ref_sinh_quant, csi_ref_sinh_quant, csi_ref_sinh_quant, csi_ref_sinh_quant, csi_ref_sinh_quant, csi_ref_sinh_quant, csi_ref_sinh_f32, NULL}, /* CSINN_OP_SINH */
    {csi_ref_slice_quant, csi_ref_slice_quant, csi_ref_slice_quant, csi_ref_slice_quant, csi_ref_slice_quant, csi_ref_slice_quant, csi_ref_slice_quant, csi_ref_slice_f32, NULL}, /* CSINN_OP_SLICE */
    {csi_ref_softmax_quant, csi_ref_softmax_quant, csi_ref_softmax_quant, csi_ref_softmax_quant, csi_ref_softmax_quant, csi_ref_softmax_quant, csi_ref_softmax_quant, csi_ref_softmax_f32, NULL}, /* CSINN_OP_SOFTMAX */
    {csi_ref_softplus_quant, csi_ref_softplus_quant, csi_ref_softplus_quant, csi_ref_softplus_quant, csi_ref_softplus_quant, csi_ref_softplus_quant, csi_ref_softplus_quant, csi_ref_softplus_f32, NULL}, /* CSINN_OP_SOFTPLUS */
    {csi_ref_softrelu_quant, csi_ref_softrelu_quant, csi_ref_softrelu_quant, csi_ref_softrelu_quant, csi_ref_softrelu_quant, csi_ref_softrelu_quant, csi_ref_softrelu_quant, csi_ref_softrelu_f32, NULL}, /* CSINN_OP_SOFTRELU */
    {csi_ref_softsign_quant, csi_ref_softsign_quant, csi_ref_softsign_quant, csi_ref_softsign_quant, csi_ref_softsign_quant, csi_ref_softsign_quant, csi_ref_softsign_quant, csi_ref_softsign_f32, NULL}, /* CSINN_OP_SOFTSIGN */
    {csi_ref_space_to_batch_quant, csi_ref_space_to_batch_quant, csi_ref_space_to_batch_quant, csi_ref_space_to_batch_quant, csi_ref_space_to_batch_quant, csi_ref_space_to_batch_quant, csi_ref_space_to_batch_quant, csi_ref_space_to_batch_f32, NULL}, /* CSINN_OP_SPACE_TO_BATCH */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_SPACE_TO_BATCH_ND */
    {csi_ref_space_to_depth_quant, csi_ref_space_to_depth_quant, csi_ref_space_to_depth_quant, csi_ref_space_to_depth_quant, csi_ref_space_to_depth_quant, csi_ref_space_to_depth_quant, csi_ref_space_to_depth_quant, csi_ref_space_to_depth_f32, NULL}, /* CSINN_OP_SPACE_TO_DEPTH */
    {csi_ref_split, csi_ref_split, csi_ref_split, csi_ref_split, csi_ref_split, csi_ref_split, csi_ref_split, csi_ref_split, csi_ref_split}, /* CSINN_OP_SPLIT */
    {csi_ref_sqrt_quant, csi_ref_sqrt_quant, csi_ref_sqrt_quant, csi_ref_sqrt_quant, csi_ref_sqrt_quant, csi_ref_sqrt_quant, csi_ref_sqrt_quant, csi_ref_sqrt_f32, NULL}, /* CSINN_OP_SQRT */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, csi_ref_square_f32, NULL}, /* CSINN_OP_SQUARE */
    {csi_ref_squeeze, csi_ref_squeeze, csi_ref_squeeze, csi_ref_squeeze, csi_ref_squeeze, csi_ref_squeeze, csi_ref_squeeze, csi_ref_squeeze, csi_ref_squeeze}, /* CSINN_OP_SQUEEZE */
    {csi_ref_stack_quant, csi_ref_stack_quant, csi_ref_stack_quant, csi_ref_stack_quant, csi_ref_stack_quant, csi_ref_stack_quant, csi_ref_stack_quant, csi_ref_stack_f32, NULL}, /* CSINN_OP_STACK */
    {csi_ref_strided_slice_quant, csi_ref_strided_slice_quant, csi_ref_strided_slice_quant, csi_ref_strided_slice_quant, csi_ref_strided_slice_quant, csi_ref_strided_slice_quant, csi_ref_strided_slice_quant, csi_ref_strided_slice_f32, NULL}, /* CSINN_OP_STRIDED_SLICE */
    {csi_ref_sub_quant, csi_ref_sub_quant, csi_ref_sub_quant, csi_ref_sub_quant, csi_ref_sub_quant, csi_ref_sub_quant, csi_ref_sub_quant, csi_ref_sub_f32, NULL}, /* CSINN_OP_SUB */
    {csi_ref_sum_stride_quant, csi_ref_sum_stride_quant, csi_ref_sum_stride_quant, csi_ref_sum_stride_quant, csi_ref_sum_stride_quant, csi_ref_sum_stride_quant, csi_ref_sum_stride_quant, csi_ref_sum_stride_f32, NULL}, /* CSINN_OP_SUM */
    {csi_ref_tan_quant, csi_ref_tan_quant, csi_ref_tan_quant, csi_ref_tan_quant, csi_ref_tan_quant, csi_ref_tan_quant, csi_ref_tan_quant, csi_ref_tan_f32, NULL}, /* CSINN_OP_TAN */
    {csi_ref_tanh_quant, csi_ref_tanh_quant, csi_ref_tanh_quant, csi_ref_tanh_quant, csi_ref_tanh_quant, csi_ref_tanh_quant, csi_ref_tanh_quant, csi_ref_tanh_f32, csi_ref_tanh_f64}, /* CSINN_OP_TANH */
    {csi_ref_threshold_relu_quant, csi_ref_threshold_relu_quant, csi_ref_threshold_relu_quant, csi_ref_threshold_relu_quant, csi_ref_threshold_relu_quant, csi_ref_threshold_relu_quant, csi_ref_threshold_relu_quant, csi_ref_threshold_relu_f32, NULL}, /* CSINN_OP_THRESHOLD_RELU */
    {csi_ref_tile_quant, csi_ref_tile_quant, csi_ref_tile_quant, csi_ref_tile_quant, csi_ref_tile_quant, csi_ref_tile_quant, csi_ref_tile_quant, csi_ref_tile_f32, NULL}, /* CSINN_OP_TILE */
    {csi_ref_topk_quant, csi_ref_topk_quant, csi_ref_topk_quant, csi_ref_topk_quant, csi_ref_topk_quant, csi_ref_topk_quant, csi_ref_topk_quant, csi_ref_topk_f32, NULL}, /* CSINN_OP_TOPK */
    {csi_ref_transpose, csi_ref_transpose, NULL, NULL, NULL, NULL, NULL, csi_ref_transpose, NULL}, /* CSINN_OP_TRANSPOSE */
    {csi_ref_trunc_quant, csi_ref_trunc_quant, csi_ref_trunc_quant, csi_ref_trunc_quant, csi_ref_trunc_quant, csi_ref_trunc_quant, csi_ref_trunc_quant, csi_ref_trunc_f32, NULL}, /* CSINN_OP_TRUNC */
    {csi_ref_unpooling_quant, csi_ref_unpooling_quant, csi_ref_unpooling_quant, csi_ref_unpooling_quant, csi_ref_unpooling_quant, csi_ref_unpooling_quant, csi_ref_unpooling_quant, csi_ref_unpooling_f32, NULL}, /* CSINN_OP_UNPOOLING */
    {csi_ref_unstack_qunat, csi_ref_unstack_qunat, csi_ref_unstack_qunat, csi_ref_unstack_qunat, csi_ref_unstack_qunat, csi_ref_unstack_qunat, csi_ref_unstack_qunat, csi_ref_unstack_f32, NULL}, /* CSINN_OP_UNSTACK */
    {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}, /* CSINN_OP_WHERE */
    {csi_ref_xor_u8, csi_ref_xor_i8, NULL, NULL, csi_ref_xor_u32, NULL, NULL, NULL, NULL}, /* CSINN_OP_XOR */
    {csi_ref_yuv_rgb_scale_quant, csi_ref_yuv_rgb_scale_quant, csi_ref_yuv_rgb_scale_quant, csi_ref_yuv_rgb_scale_quant, csi_ref_yuv_rgb_scale_quant, csi_ref_yuv_rgb_scale_quant, csi_ref_yuv_rgb_scale_quant, csi_ref_yuv_rgb_scale_f32, NULL}, /* CSINN_OP_YUV_RGB_SCALE */
};

void *csi_bc_map_ref(int op, int dtype)
{
    return csi_ref_bc_map_table[op][dtype];
}
