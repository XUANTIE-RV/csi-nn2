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

#include "c906/c906.h"
#include "c906/perf.h"

static struct shl_function_map shl_c906_kernel_map[] = {
    {shl_c906_abs_f32, "shl_c906_abs_f32"},
    {shl_c906_add_f32, "shl_c906_add_f32"},
    {shl_c906_sub_f32, "shl_c906_sub_f32"},
    {shl_c906_mul_f32, "shl_c906_mul_f32"},
    {shl_c906_minimum_f32, "shl_c906_minimum_f32"},
    {shl_c906_broadcast_to_f32, "shl_c906_broadcast_to_f32"},
    {shl_c906_clip_f32, "shl_c906_clip_f32"},
    {shl_c906_concat_f32, "shl_c906_concat_f32"},
    {shl_c906_split_f32, "shl_c906_split_f32"},
    {shl_c906_pad_f32, "shl_c906_pad_f32"},
    {shl_c906_prelu_f32, "shl_c906_prelu_f32"},
    {shl_c906_relu_f32, "shl_c906_relu_f32"},
    {shl_c906_relu1_f32, "shl_c906_relu1_f32"},
    {shl_c906_relu6_f32, "shl_c906_relu6_f32"},
    {shl_c906_leaky_relu_f32, "shl_c906_leaky_relu_f32"},
    {shl_c906_global_maxpool2d_f32, "shl_c906_global_maxpool2d_f32"},
    {shl_c906_reorder_kernel, "shl_c906_reorder_kernel"},
    {shl_c906_reorder_input_1, "shl_c906_reorder_input_1"},
    {shl_c906_sgemm_kernel_f32, "shl_c906_sgemm_kernel_f32"},
    {shl_c906_conv1x1s1_sgemm_transform_kernel, "shl_c906_conv1x1s1_sgemm_transform_kernel"},
    {shl_c906_conv_im2col_sgemm_transform_kernel, "shl_c906_conv_im2col_sgemm_transform_kernel"},
    {shl_c906_conv3x3s1_winograd64_transform_kernel_pack4,
     "shl_c906_conv3x3s1_winograd64_transform_kernel_pack4"},
    {shl_c906_conv1x1s1_sgemm, "shl_c906_conv1x1s1_sgemm"},
    {shl_c906_conv1x1s1_sgemm_fuse_relu, "shl_c906_conv1x1s1_sgemm_fuse_relu"},
    {shl_c906_conv_im2col_sgemm, "shl_c906_conv_im2col_sgemm"},
    {shl_c906_conv_im2col_sgemm_fuse_relu, "shl_c906_conv_im2col_sgemm_fuse_relu"},
    {shl_c906_conv3x3s1_winograd64_pack4, "shl_c906_conv3x3s1_winograd64_pack4"},
    {shl_c906_dwconv3x3s1, "shl_c906_dwconv3x3s1"},
    {shl_c906_dwconv3x3s2, "shl_c906_dwconv3x3s2"},
    {shl_c906_dwconv5x5s1, "shl_c906_dwconv5x5s1"},
    {shl_c906_dwconv5x5s2, "shl_c906_dwconv5x5s2"},
    {shl_c906_dwconv3x3s1_pack4, "shl_c906_dwconv3x3s1_pack4"},
    {shl_c906_dwconv3x3s2_pack4, "shl_c906_dwconv3x3s2_pack4"},
    {shl_c906_dwconv3x3s1_fuse_relu, "shl_c906_dwconv3x3s1_fuse_relu"},
    {shl_c906_dwconv3x3s2_fuse_relu, "shl_c906_dwconv3x3s2_fuse_relu"},
    {shl_c906_dwconv5x5s1_fuse_relu, "shl_c906_dwconv5x5s1_fuse_relu"},
    {shl_c906_dwconv5x5s2_fuse_relu, "shl_c906_dwconv5x5s2_fuse_relu"},
    {shl_c906_dwconv3x3s1_pack4_fuse_relu, "shl_c906_dwconv3x3s1_pack4_fuse_relu"},
    {shl_c906_dwconv3x3s2_pack4_fuse_relu, "shl_c906_dwconv3x3s2_pack4_fuse_relu"},
    {shl_c906_dwconv2d_s1_pad0_fp16, "shl_c906_dwconv2d_s1_pad0_fp16"},
    {shl_c906_add_fp16, "shl_c906_add_fp16"},
    {shl_c906_sub_fp16, "shl_c906_sub_fp16"},
    {shl_c906_mul_fp16, "shl_c906_mul_fp16"},
    {shl_c906_minimum_fp16, "shl_c906_minimum_fp16"},
    {shl_c906_global_avgpool2d_fp16, "shl_c906_global_avgpool2d_fp16"},
    {shl_c906_global_maxpool2d_fp16, "shl_c906_global_maxpool2d_fp16"},
    {shl_c906_pad_fp16, "shl_c906_pad_fp16"},
    {shl_c906_relu_fp16, "shl_c906_relu_fp16"},
    {shl_c906_relu1_fp16, "shl_c906_relu1_fp16"},
    {shl_c906_relu6_fp16, "shl_c906_relu6_fp16"},
    {shl_c906_prelu_fp16, "shl_c906_prelu_fp16"},
    {shl_c906_leaky_relu_fp16, "shl_c906_leaky_relu_fp16"},
    {shl_c906_abs_fp16, "shl_c906_abs_fp16"},
    {shl_c906_clip_fp16, "shl_c906_clip_fp16"},
    {shl_c906_concat_fp16, "shl_c906_concat_fp16"},
    {shl_c906_split_fp16, "shl_c906_split_fp16"},
    {shl_c906_fullyconnected_pack16_fp16, "shl_c906_fullyconnected_pack16_fp16"},
    {shl_c906_fullyconnected_pack16_output16_fp16, "shl_c906_fullyconnected_pack16_output16_fp16"},
    {shl_c906_reorder_weight_n16_fp16, "shl_c906_reorder_weight_n16_fp16"},
    {shl_c906_reorder_kernel_fp16, "shl_c906_reorder_kernel_fp16"},
    {shl_c906_reorder_input_fp16_1, "shl_c906_reorder_input_fp16_1"},
    {shl_c906_sgemm_kernel_fp16, "shl_c906_sgemm_kernel_fp16"},
    // {shl_c906_sgemm_kernel_fp16_1, "shl_c906_sgemm_kernel_fp16_1"},
    {shl_c906_conv1x1s1_sgemm_transform_kernel_fp16,
     "shl_c906_conv1x1s1_sgemm_transform_kernel_fp16"},
    {shl_c906_conv1x1s1_sgemm_transform_kernel_fp16_w_int8,
     "shl_c906_conv1x1s1_sgemm_transform_kernel_fp16_w_int8"},
    {shl_c906_conv_im2col_sgemm_transform_kernel_fp16,
     "shl_c906_conv_im2col_sgemm_transform_kernel_fp16"},
    {shl_c906_conv_im2col_sgemm_transform_kernel_fp16_w_int8,
     "shl_c906_conv_im2col_sgemm_transform_kernel_fp16_w_int8"},
    {shl_c906_conv3x3s1_winograd64_transform_kernel_pack8_fp16,
     "shl_c906_conv3x3s1_winograd64_transform_kernel_pack8_fp16"},
    {shl_c906_conv1x1s1_sgemm_fp16, "shl_c906_conv1x1s1_sgemm_fp16"},
    {shl_c906_conv_im2col_sgemm_fp16, "shl_c906_conv_im2col_sgemm_fp16"},
    {shl_c906_conv3x3s1_winograd64_pack8_fp16, "shl_c906_conv3x3s1_winograd64_pack8_fp16"},
    {shl_c906_dwconv3x3s1_fp16, "shl_c906_dwconv3x3s1_fp16"},
    {shl_c906_dwconv3x3s2_fp16, "shl_c906_dwconv3x3s2_fp16"},
    {shl_c906_dwconv3x3s1_pack8_fp16, "shl_c906_dwconv3x3s1_pack8_fp16"},
    {shl_c906_dwconv3x3s2_pack8_fp16, "shl_c906_dwconv3x3s2_pack8_fp16"},
    // {shl_c906_matmul_fp32, "shl_c906_matmul_fp32"},
    {shl_c906_cache_matmul_fp16, "shl_c906_cache_matmul_fp16"},
    {shl_c906_matmul_fp16, "shl_c906_matmul_fp16"},
    {shl_c906_reshape_fp16, "shl_c906_reshape_fp16"},
    {shl_c906_cache_conv1d_fp16, "shl_c906_cache_conv1d_fp16"},
    {shl_c906_lrn_fp16, "shl_c906_lrn_fp16"},
    {shl_c906_reduce_sum_fp16, "shl_c906_reduce_sum_fp16"},
    {NULL, NULL}};

char *shl_rvv_get_kernel_name(void *exec);

char *shl_c906_get_kernel_name(void *exec)
{
    char *name = shl_find_function_name(shl_c906_kernel_map, exec);
    if (name == NULL) {
        name = shl_rvv_get_kernel_name(exec);
    }
    return name;
}

int shl_c906_conv2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv2d_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_depthwise_conv2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv2d_params *params,
                                   struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_conv1d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv1d_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_depthwise_conv1d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_conv1d_params *params,
                                   struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_fullyconnected_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *weights, struct csinn_tensor *bias,
                                 struct csinn_fc_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_maxpool2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_pool_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_avgpool2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                            struct csinn_pool_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_div_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params,
                      struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_abs_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_siso_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_add_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params,
                      struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_clip_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_clip_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_concat_perf(struct csinn_tensor **input, struct csinn_tensor *output,
                         struct csinn_clip_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_global_avgpool2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_pool_params *params,
                                   struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_global_maxpool2d_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_pool_params *params,
                                   struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_leaky_relu_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_relu_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_lrn_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_lrn_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_matmul_perf(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                         struct csinn_tensor *output, struct csinn_matmul_params *params,
                         struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_minimum_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                          struct csinn_tensor *output, struct csinn_diso_params *params,
                          struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_mul_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params,
                      struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_prelu_perf(struct csinn_tensor *input, struct csinn_tensor *alpha,
                        struct csinn_tensor *output, struct csinn_prelu_params *params,
                        struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_relu_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_relu1_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_relu_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_relu6_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_relu_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_split_perf(struct csinn_tensor *input, struct csinn_tensor **output,
                        struct csinn_split_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_sub_perf(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params,
                      struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_reshape_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_reshape_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}

int shl_c906_reduce_sum_perf(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_reduce_params *params, struct csinn_perf_info *perf_info)
{
    perf_info->kernel_name = shl_c906_get_kernel_name(params->base.cb->exec);
    return CSINN_TRUE;
}