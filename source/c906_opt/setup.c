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

#include "shl_c906.h"

static struct shl_cb_op_list shl_c906_cb_op_list;

int shl_c906_reg_op(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *init, void *exec)
{
    struct shl_cb_op_list *list_end = shl_cb_list_end(&shl_c906_cb_op_list);
    struct shl_cb_op_list *next = shl_mem_alloc(sizeof(struct shl_cb_op_list));
    next->cb = shl_mem_alloc(sizeof(struct csinn_callback));
    next->cb->init = init;
    next->cb->exec = exec;
    next->dtype = dtype;
    next->op_name = op_name;
    list_end->next = next;
    return CSINN_TRUE;
}

int shl_c906_reg_op_est(enum csinn_dtype_enum dtype, enum csinn_op_enum op_name, void *est)
{
    struct csinn_callback *cb = shl_cb_list_match(&shl_c906_cb_op_list, dtype, op_name);
    if (cb == NULL) {
        shl_debug_info("%s: cannot find c906 est\n", __func__);
    } else {
        cb->est = est;
    }

    return CSINN_TRUE;
}

struct csinn_callback *shl_cb_map_rvv(int op, int dtype);
struct csinn_callback *shl_cb_map_c906(int op, int dtype)
{
    struct csinn_callback *cb = shl_cb_list_match(&shl_c906_cb_op_list, dtype, op);
    if (cb == NULL) {
        cb = shl_cb_map_rvv(op, dtype);
    }
    return cb;
}

void __attribute__((weak)) shl_target_init_c906()
{
    shl_register_runtime_callback(CSINN_C906, NULL);
    shl_register_op_callback(CSINN_C906, shl_cb_map_c906);

    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_c906_conv2d_init, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_c906_conv2d_init, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, shl_c906_conv2d_init, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, shl_c906_conv2d_init, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV1D, shl_c906_conv1d_init, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV1D, shl_c906_conv1d_init, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL2D, shl_c906_maxpool2d_init, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXPOOL2D, shl_c906_maxpool2d_init, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_AVGPOOL2D, shl_c906_avgpool2d_init, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_AVGPOOL2D, shl_c906_avgpool2d_init, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D, shl_c906_depthwise_conv2d_init,
                    NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTHWISE_CONV2D, shl_c906_depthwise_conv2d_init,
                    NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED, shl_c906_fullyconnected_init,
                    NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_FULLYCONNECTED, shl_c906_fullyconnected_init,
                    NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_DIV, shl_c906_div_init, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_DIV, shl_c906_div_init, NULL);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ABS, NULL, shl_c906_abs_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_ADD, NULL, shl_c906_add_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CACHE_MATMUL, shl_c906_cache_matmul_init,
                    shl_c906_cache_matmul_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CACHE_CONV1D, shl_c906_cache_conv1d_init,
                    shl_c906_cache_conv1d_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CLIP, NULL, shl_c906_clip_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_CONCAT, NULL, shl_c906_concat_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_AVGPOOL2D, NULL,
                    shl_c906_global_avgpool2d_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_MAXPOOL2D, NULL,
                    shl_c906_global_maxpool2d_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_GATHER, NULL, shl_c906_gather_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LAYER_NORM, NULL, shl_c906_layer_norm_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LEAKY_RELU, NULL, shl_c906_leaky_relu_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_LRN, NULL, shl_c906_lrn_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MATMUL, NULL, shl_c906_matmul_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MINIMUM, NULL, shl_c906_minimum_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_MUL, NULL, shl_c906_mul_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_PRELU, NULL, shl_c906_prelu_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU, NULL, shl_c906_relu_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU1, NULL, shl_c906_relu1_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU6, NULL, shl_c906_relu6_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_RESHAPE, NULL, shl_c906_reshape_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SPLIT, NULL, shl_c906_split_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SUB, NULL, shl_c906_sub_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_SUM, NULL, shl_c906_sum_stride_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT16, CSINN_OP_TRANSPOSE, NULL, shl_c906_transpose_fp16);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ABS, NULL, shl_c906_abs_f32);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_ADD, NULL, shl_c906_add_f32);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CLIP, NULL, shl_c906_clip_f32);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_CONCAT, NULL, shl_c906_concat_f32);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_AVGPOOL2D, NULL,
                    shl_c906_global_avgpool2d_f32);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_MAXPOOL2D, NULL,
                    shl_c906_global_maxpool2d_f32);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_LEAKY_RELU, NULL, shl_c906_leaky_relu_f32);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MINIMUM, NULL, shl_c906_minimum_f32);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_MUL, NULL, shl_c906_mul_f32);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_PRELU, NULL, shl_c906_prelu_f32);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU, NULL, shl_c906_relu_f32);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU1, NULL, shl_c906_relu1_f32);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU6, NULL, shl_c906_relu6_f32);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SPLIT, NULL, shl_c906_split_f32);
    shl_c906_reg_op(CSINN_DTYPE_FLOAT32, CSINN_OP_SUB, NULL, shl_c906_sub_f32);

#ifdef SHL_BUILD_GREF
    shl_register_runtime_callback(CSINN_C906, shl_gref_runtime_callback);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV2D, shl_gref_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV2D, shl_gref_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_GROUP_CONV2D, shl_gref_group_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_GROUP_CONV2D, shl_gref_group_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CONV1D, shl_gref_conv1d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_CONV1D, shl_gref_conv1d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_MAXPOOL2D, shl_gref_maxpool2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_MAXPOOL2D, shl_gref_maxpool2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_AVGPOOL2D, shl_gref_avgpool2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_AVGPOOL2D, shl_gref_avgpool2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_DEPTHWISE_CONV2D, shl_gref_depthwise_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_DEPTHWISE_CONV2D, shl_gref_depthwise_conv2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_FULLYCONNECTED, shl_gref_fullyconnected);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_FULLYCONNECTED, shl_gref_fullyconnected);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_DIV, shl_gref_div);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_DIV, shl_gref_div);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_ABS, shl_gref_abs);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_ADD, shl_gref_add);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CACHE_MATMUL, shl_gref_cache_matmul);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CACHE_CONV1D, shl_gref_cache_conv1d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CLIP, shl_gref_clip);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_CONCAT, shl_gref_concat);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_AVGPOOL2D, shl_gref_global_avgpool2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_GLOBAL_MAXPOOL2D, shl_gref_global_maxpool2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_GATHER, shl_gref_gather);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_LAYER_NORM, shl_gref_layer_norm);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_LEAKY_RELU, shl_gref_leaky_relu);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_LRN, shl_gref_lrn);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_MATMUL, shl_gref_matmul);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_MINIMUM, shl_gref_minimum);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_MUL, shl_gref_mul);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_PRELU, shl_gref_prelu);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU, shl_gref_relu);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU1, shl_gref_relu1);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_RELU6, shl_gref_relu6);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_RESHAPE, shl_gref_reshape);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_SPLIT, shl_gref_split);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_SUB, shl_gref_sub);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_SUM, shl_gref_sum);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT16, CSINN_OP_TRANSPOSE, shl_gref_transpose);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_ABS, shl_gref_abs);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_ADD, shl_gref_add);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_CLIP, shl_gref_clip);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_CONCAT, shl_gref_concat);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_AVGPOOL2D, shl_gref_global_avgpool2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_GLOBAL_MAXPOOL2D, shl_gref_global_maxpool2d);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_LEAKY_RELU, shl_gref_leaky_relu);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_MINIMUM, shl_gref_minimum);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_MUL, shl_gref_mul);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_PRELU, shl_gref_prelu);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU, shl_gref_relu);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU1, shl_gref_relu1);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_RELU6, shl_gref_relu6);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_SPLIT, shl_gref_split);
    shl_c906_reg_op_est(CSINN_DTYPE_FLOAT32, CSINN_OP_SUB, shl_gref_sub);
#endif
}
