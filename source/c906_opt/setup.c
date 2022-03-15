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

#include "csi_c906.h"

static struct csi_bc_op_list csi_nn_c906_init_bc_op_list;
static struct csi_bc_op_list csi_nn_c906_func_bc_op_list;

int csi_nn_c906_register_op_init(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *bc)
{
    struct csi_bc_op_list *list_end = csi_bc_list_end(&csi_nn_c906_init_bc_op_list);
    struct csi_bc_op_list *next = csi_mem_alloc(sizeof(struct csi_bc_op_list));
    next->bc = bc;
    next->dtype = dtype;
    next->op_name = op_name;
    list_end->next = next;
    return CSINN_TRUE;
}

int csi_nn_c906_register_op(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *bc)
{
    struct csi_bc_op_list *list_end = csi_bc_list_end(&csi_nn_c906_func_bc_op_list);
    struct csi_bc_op_list *next = csi_mem_alloc(sizeof(struct csi_bc_op_list));
    next->bc = bc;
    next->dtype = dtype;
    next->op_name = op_name;
    list_end->next = next;
    return CSINN_TRUE;
}

static inline void register_op_init_all(enum csinn_op_enum op_name, void *bc)
{
    csi_nn_c906_register_op_init(CSINN_DTYPE_FLOAT16, op_name, bc);
    csi_nn_c906_register_op_init(CSINN_DTYPE_FLOAT32, op_name, bc);
}

void __attribute__((weak)) csi_nn_c906_bc_init_reg()
{
    register_op_init_all(CSINN_OP_CONV2D, csi_c906_conv2d_init);
    register_op_init_all(CSINN_OP_GROUP_CONV2D, csi_c906_conv2d_init);
    register_op_init_all(CSINN_OP_CONV1D, csi_c906_conv1d_init);
    register_op_init_all(CSINN_OP_MAXPOOL2D, csi_c906_maxpool2d_init);
    register_op_init_all(CSINN_OP_AVGPOOL2D, csi_c906_avgpool2d_init);
    register_op_init_all(CSINN_OP_DEPTHWISE_CONV2D, csi_c906_depthwise_conv2d_init);
    register_op_init_all(CSINN_OP_FULLYCONNECTED, csi_c906_fullyconnected_init);
    register_op_init_all(CSINN_OP_CACHE_MATMUL, csi_c906_cache_matmul_init);
    register_op_init_all(CSINN_OP_DIV, csi_c906_div_init);
    register_op_init_all(CSINN_OP_CACHE_CONV1D, csi_c906_cache_conv1d_init);
}

void *csi_init_map_c906(int op, int dtype)
{
    static int has_reg;
    if (has_reg == 0) {
        csi_nn_c906_bc_init_reg();
        has_reg = 1;
    }
    void *ret = csi_bc_list_match(&csi_nn_c906_init_bc_op_list, dtype, op);
    if (ret == NULL) {
        csi_debug_info("no c906 init\n");
    }
    return ret;
}

void __attribute__((weak)) csi_nn_c906_bc_reg()
{
    /* float16 */
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ABS, csi_c906_abs_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ACOS, csi_ref_acos_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ACOSH, csi_ref_acosh_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ADD, csi_c906_add_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_AND, csi_ref_and_i8);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ARANGE, csi_ref_arange_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ARGMAX, csi_ref_argmax_stride_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ARGMIN, csi_ref_argmin_stride_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ASIN, csi_ref_asin_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ASINH, csi_ref_asinh_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ATAN, csi_ref_atan_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ATANH, csi_ref_atanh_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_AVGPOOL2D, csi_ref_avgpool2d_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_AVGPOOL3D, csi_ref_avgpool3d_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_BN, csi_ref_batch_normalization_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_BATCH_TO_SPACE,
                            csi_ref_batch_to_space_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_BROADCOST, csi_ref_broadcast_to_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CACHE_MATMUL, csi_c906_cache_matmul_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CACHE_CONV1D, csi_c906_cache_conv1d_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CEIL, csi_ref_ceil_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CLIP, csi_c906_clip_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONCAT, csi_c906_concat_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV1D, csi_ref_conv1d_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, csi_ref_conv2d_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D_RELU, csi_ref_conv2d_relu_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D_RELU6, csi_ref_conv2d_relu6_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D,
                            csi_ref_depthwise_conv2d_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D_RELU,
                            csi_ref_depthwise_conv2d_relu_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D_RELU6,
                            csi_ref_depthwise_conv2d_relu6_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, csi_ref_group_conv2d_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV3D, csi_ref_conv3d_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DECONV2D, csi_ref_deconv2d_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_DECONV2D,
                            csi_ref_depthwise_deconv2d_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DECONV3D, csi_ref_deconv3d_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_COS, csi_ref_cos_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_COSH, csi_ref_cosh_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CUMPROD, csi_ref_cumprod_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CUMSUM, csi_ref_cumsum_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTH_TO_SPACE,
                            csi_ref_depth_to_space_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DIV, csi_ref_div_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ELU, csi_ref_elu_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_EQUANL, csi_ref_equal_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ERF, csi_ref_erf_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_EXP, csi_ref_exp_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_EXPAND_DIMS, csi_ref_expand_dims_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_EXPM1, csi_ref_expm1_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_FLATTEN, csi_ref_flatten);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_FLOOR_DIVIDE, csi_ref_floor_divide_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_FLOOR_MOD, csi_ref_floor_mod_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_FLOOR, csi_ref_floor_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_FSMN, csi_ref_fsmn_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED,
                            csi_c906_fullyconnected_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GATHER_ND, csi_ref_gather_nd_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GATHER, csi_c906_gather_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_AVGPOOL2D,
                            csi_ref_global_avgpool2d_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_MAXPOOL2D,
                            csi_ref_global_maxpool2d_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GREATHER_EQUAL,
                            csi_ref_greater_equal_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GREATHER, csi_ref_greater_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_HARD_SIGMOID, csi_ref_hard_sigmoid_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_IM2COL, csi_ref_im2col_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_L2N, csi_ref_l2_normalization_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LAYER_NORM, csi_c906_layer_norm_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LEAKY_RELU, csi_c906_leaky_relu_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LESS_EQUAL, csi_ref_less_equal_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LESS, csi_ref_less_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LOG_SOFTMAX, csi_ref_log_softmax_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LOG, csi_ref_log_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LOG1P, csi_ref_log1p_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LOGICAL_AND, csi_ref_logical_and_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LOGICAL_NOT, csi_ref_logical_not_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LOGICAL_OR, csi_ref_logical_or_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LOGICAL_XOR, csi_ref_logical_xor_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LRN, csi_c906_lrn_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MATMUL, csi_c906_matmul_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MAX, csi_ref_max_stride_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXIMUM, csi_ref_maximum_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL2D, csi_ref_maxpool2d_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL2D_LOCAT,
                            csi_ref_maxpool2d_locat_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL3D, csi_ref_maxpool3d_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MEAN, csi_ref_mean_stride_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MEAN_STRIDE, csi_ref_mean_stride_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MIN, csi_ref_min_stride_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MINIMUM, csi_c906_minimum_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MOD, csi_ref_mod_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MUL, csi_c906_mul_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_NDARRAY_SIZE, csi_ref_ndarray_size_i8);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_NEGATIIVE, csi_ref_negative_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_NOT_EQUAL, csi_ref_not_equal_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_NOT, csi_ref_not_i8);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_OR, csi_ref_or_i8);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_PAD, csi_ref_pad_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_POWER, csi_ref_power_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_PRELU, csi_c906_prelu_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_PROD, csi_ref_prod_stride_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_PROPOSAL, csi_ref_proposal_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_PSROIPOOLING, csi_ref_psroipooling_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_REDUCE_LOGSUMEXP,
                            csi_ref_reduce_logsumexp_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_REDUCE_MAX, csi_ref_reduce_max_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_REDUCE_MEAN, csi_ref_reduce_mean_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_REDUCE_MIN, csi_ref_reduce_min_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_REDUCE_PROD, csi_ref_reduce_prod_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_REDUCE_SUM, csi_ref_reduce_sum_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU, csi_c906_relu_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU1, csi_c906_relu1_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU6, csi_c906_relu6_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELUN, csi_ref_relun_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RESHAPE, csi_c906_reshape_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RESIZE, csi_ref_resize_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_REVERSE, csi_ref_reverse_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ROIPOOL, csi_ref_roipool_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ROUND, csi_ref_round_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RSQRT, csi_ref_rsqrt_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SCATTER_ND, csi_ref_scatter_nd_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SEGMENT_MAX, csi_ref_segment_max_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_UNSORTED_SEGMENT_MAX,
                            csi_ref_unsorted_segment_max_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SEGMENT_MEAN, csi_ref_segment_mean_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_UNSORTED_SEGMENT_MEAN,
                            csi_ref_unsorted_segment_mean_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SEGMENT_MIN, csi_ref_segment_min_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_UNSORTED_SEGMENT_MIN,
                            csi_ref_unsorted_segment_min_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SEGMENT_PROD, csi_ref_segment_prod_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_UNSORTED_SEGMENT_PROD,
                            csi_ref_unsorted_segment_prod_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SEGMENT_SUM, csi_ref_segment_sum_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_UNSORTED_SEGMENT_SUM,
                            csi_ref_unsorted_segment_sum_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SELECT, csi_ref_select_i8);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SHAPE, csi_ref_shape_i8);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SHUFFLE_CHANNEL,
                            csi_ref_shuffle_channel_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SIGMOID, csi_nn_rvv_sigmoid_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SIGN, csi_ref_sign_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SIN, csi_ref_sin_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SINH, csi_ref_sinh_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SLICE, csi_ref_slice_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SOFTMAX, csi_nn_rvv_softmax_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SOFTPLUS, csi_ref_softplus_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SOFTRELU, csi_ref_softrelu_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SOFTSIGN, csi_ref_softsign_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SPACE_TO_BATCH,
                            csi_ref_space_to_batch_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SPACE_TO_DEPTH,
                            csi_ref_space_to_depth_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SPLIT, csi_c906_split_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SQRT, csi_ref_sqrt_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SQUEEZE, csi_ref_squeeze);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_STACK, csi_ref_stack_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_STRIDED_SLICE,
                            csi_ref_strided_slice_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SUB, csi_c906_sub_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SUM, csi_c906_sum_stride_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_TAN, csi_ref_tan_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_TANH, csi_ref_tanh_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_THRESHOLD_RELU,
                            csi_ref_threshold_relu_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_TILE, csi_ref_tile_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_TOPK, csi_ref_topk_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_TRUNC, csi_ref_trunc_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_TRANSPOSE, csi_c906_transpose_fp16);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_UNPOOLING, csi_ref_unpooling_quant);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_UNSTACK, csi_ref_unstack_qunat);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_XOR, csi_ref_xor_i8);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT16, CSINN_OP_YUV_RGB_SCALE,
                            csi_ref_yuv_rgb_scale_quant);

    /* float32 */
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ABS, csi_c906_abs_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ACOS, csi_ref_acos_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ACOSH, csi_ref_acosh_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ADD, csi_c906_add_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ARANGE, csi_ref_arange_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ARGMAX, csi_ref_argmax_stride_i32_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ARGMIN, csi_ref_argmin_stride_i32_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ASIN, csi_ref_asin_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ASINH, csi_ref_asinh_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ATAN, csi_ref_atan_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ATANH, csi_ref_atanh_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_AVGPOOL2D, csi_ref_avgpool2d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_AVGPOOL3D, csi_ref_avgpool3d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_BN, csi_ref_batch_normalization_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_BATCH_TO_SPACE,
                            csi_ref_batch_to_space_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_BROADCOST, csi_ref_broadcast_to_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CACHE_MATMUL, csi_ref_cache_matmul_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CACHE_CONV1D, csi_ref_cache_conv1d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CEIL, csi_ref_ceil_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CLIP, csi_c906_clip_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONCAT, csi_c906_concat_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV1D, csi_ref_conv1d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, csi_ref_conv2d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D_RELU, csi_ref_conv2d_relu_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTHWISE_CONV2D,
                            csi_ref_depthwise_conv2d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, csi_ref_group_conv2d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV3D, csi_ref_conv3d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DECONV2D, csi_ref_deconv2d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTHWISE_DECONV2D,
                            csi_ref_depthwise_deconv2d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DECONV3D, csi_ref_deconv3d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_COS, csi_ref_cos_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_COSH, csi_ref_cosh_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CUMPROD, csi_ref_cumprod_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CUMSUM, csi_ref_cumsum_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTH_TO_SPACE,
                            csi_ref_depth_to_space_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DIV, csi_ref_div_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ELU, csi_ref_elu_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_EQUANL, csi_ref_equal_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ERF, csi_ref_erf_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_EXP, csi_ref_exp_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_EXPAND_DIMS, csi_ref_expand_dims_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_EXPM1, csi_ref_expm1_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_FLATTEN, csi_ref_flatten);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_FLOOR_DIVIDE, csi_ref_floor_divide_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_FLOOR_MOD, csi_ref_floor_mod_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_FLOOR, csi_ref_floor_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_FSMN, csi_ref_fsmn_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_FULLYCONNECTED,
                            csi_c906_fullyconnected_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GATHER_ND, csi_ref_gather_nd_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GATHER, csi_ref_gather_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_AVGPOOL2D,
                            csi_c906_global_avgpool2d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_MAXPOOL2D,
                            csi_c906_global_maxpool2d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GREATHER_EQUAL,
                            csi_ref_greater_equal_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GREATHER, csi_ref_greater_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_HARD_SIGMOID, csi_ref_hard_sigmoid_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_IM2COL, csi_ref_im2col_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_L2N, csi_ref_l2_normalization_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_L2POOL2D, csi_ref_l2pool_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LAYER_NORM, csi_ref_layer_norm_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LEAKY_RELU, csi_c906_leaky_relu_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LESS_EQUAL, csi_ref_less_equal_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LESS, csi_ref_less_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LOG_SOFTMAX, csi_ref_log_softmax_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LOG, csi_ref_log_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LOG1P, csi_ref_log1p_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LOGICAL_AND, csi_ref_logical_and_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LOGICAL_NOT, csi_ref_logical_not_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LOGICAL_OR, csi_ref_logical_or_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LOGICAL_XOR, csi_ref_logical_xor_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LRN, csi_ref_lrn_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MATMUL, csi_ref_matmul_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MAX, csi_ref_max_stride_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXIMUM, csi_ref_maximum_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXPOOL2D, csi_ref_maxpool2d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXPOOL2D_LOCAT,
                            csi_ref_maxpool2d_locat_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXPOOL3D, csi_ref_maxpool3d_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MEAN, csi_ref_mean_stride_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MEAN_STRIDE, csi_ref_mean_stride_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MINIMUM, csi_c906_minimum_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MOD, csi_ref_mod_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MUL, csi_c906_mul_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_NDARRAY_SIZE, csi_ref_ndarray_size_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_NEGATIIVE, csi_ref_negative_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_NOT_EQUAL, csi_ref_not_equal_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_PAD, csi_ref_pad_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_POWER, csi_ref_power_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_PRELU, csi_c906_prelu_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_PROD, csi_ref_prod_stride_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_PROPOSAL, csi_ref_proposal_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_PSROIPOOLING, csi_ref_psroipooling_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_REDUCE_LOGSUMEXP,
                            csi_ref_reduce_logsumexp_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_REDUCE_MAX, csi_ref_reduce_max_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_REDUCE_MEAN, csi_ref_reduce_mean_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_REDUCE_MIN, csi_ref_reduce_min_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_REDUCE_PROD, csi_ref_reduce_prod_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_REDUCE_SUM, csi_ref_reduce_sum_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU, csi_c906_relu_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU1, csi_c906_relu1_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU6, csi_c906_relu6_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELUN, csi_ref_relun_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RESHAPE, csi_ref_reshape);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RESIZE, csi_ref_resize_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_REVERSE, csi_ref_reverse_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ROIALIGN, csi_ref_roi_align_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ROIPOOL, csi_ref_roipool_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ROUND, csi_ref_round_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RSQRT, csi_ref_rsqrt_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SCATTER_ND, csi_ref_scatter_nd_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SEGMENT_MAX, csi_ref_segment_max_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_UNSORTED_SEGMENT_MAX,
                            csi_ref_unsorted_segment_max_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SEGMENT_MEAN, csi_ref_segment_mean_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_UNSORTED_SEGMENT_MEAN,
                            csi_ref_unsorted_segment_mean_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SEGMENT_MIN, csi_ref_segment_min_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_UNSORTED_SEGMENT_MIN,
                            csi_ref_unsorted_segment_min_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SEGMENT_PROD, csi_ref_segment_prod_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_UNSORTED_SEGMENT_PROD,
                            csi_ref_unsorted_segment_prod_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SEGMENT_SUM, csi_ref_segment_sum_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_UNSORTED_SEGMENT_SUM,
                            csi_ref_unsorted_segment_sum_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SELECT, csi_ref_select_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SHUFFLE_CHANNEL,
                            csi_ref_shuffle_channel_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SIGMOID, csi_ref_sigmoid_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SIGN, csi_ref_sign_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SIN, csi_ref_sin_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SINH, csi_ref_sinh_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SLICE, csi_ref_slice_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SOFTMAX, csi_ref_softmax_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SOFTPLUS, csi_ref_softplus_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SOFTRELU, csi_ref_softrelu_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SOFTSIGN, csi_ref_softsign_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SPACE_TO_BATCH,
                            csi_ref_space_to_batch_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SPACE_TO_DEPTH,
                            csi_ref_space_to_depth_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SPLIT, csi_c906_split_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SQRT, csi_ref_sqrt_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SQUEEZE, csi_ref_square_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_STACK, csi_ref_stack_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_STRIDED_SLICE, csi_ref_strided_slice_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SUB, csi_c906_sub_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SUM, csi_ref_sum_stride_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_TAN, csi_ref_tan_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_TANH, csi_ref_tanh_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_THRESHOLD_RELU,
                            csi_ref_threshold_relu_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_TILE, csi_ref_tile_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_TOPK, csi_ref_topk_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_TRUNC, csi_ref_trunc_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_TRANSPOSE, csi_ref_transpose);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_UNPOOLING, csi_ref_unpooling_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_UNSTACK, csi_ref_unstack_f32);
    csi_nn_c906_register_op(CSINN_DTYPE_FLOAT32, CSINN_OP_YUV_RGB_SCALE, csi_ref_yuv_rgb_scale_f32);

    /* int8 */
    csi_nn_c906_register_op(CSINN_DTYPE_INT8, CSINN_OP_CONCAT, csi_nn_rvv_concat_int8);
    csi_nn_c906_register_op(CSINN_DTYPE_INT8, CSINN_OP_MUL, csi_nn_rvv_mul_int8);
    csi_nn_c906_register_op(CSINN_DTYPE_INT8, CSINN_OP_RELU, csi_nn_rvv_relu_int8);
    csi_nn_c906_register_op(CSINN_DTYPE_INT8, CSINN_OP_RESHAPE, csi_ref_reshape);
    csi_nn_c906_register_op(CSINN_DTYPE_INT8, CSINN_OP_SUM, csi_nn_rvv_sum_stride_int8);
    csi_nn_c906_register_op(CSINN_DTYPE_INT8, CSINN_OP_SOFTMAX, csi_ref_softmax_quant);
}

void *csi_bc_map_c906(int op, int dtype) {
    static int has_reg;
    if (has_reg == 0) {
        csi_nn_c906_bc_reg();
        has_reg = 1;
    }
    void *ret = csi_bc_list_match(&csi_nn_c906_func_bc_op_list, dtype, op);
    if (ret == NULL) {
        csi_debug_info("cannot find c906 func\n");
    }
    return ret;
}
