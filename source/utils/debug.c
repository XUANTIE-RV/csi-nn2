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

#ifndef SHL_BUILD_RTOS
#include <dirent.h>
#endif
#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>

#include "shl_debug.h"
#include "shl_ref.h"

int shl_debug_level = SHL_DEBUG_LEVEL_WARNING;

int shl_debug_get_level() { return shl_debug_level; }

void shl_debug_set_level(int level) { shl_debug_level = level; }
#ifdef SHL_DEBUG
void shl_debug_debug(const char *format, ...)
{
    if (shl_debug_get_level() <= SHL_DEBUG_LEVEL_DEBUG) {
        va_list arg;
        va_start(arg, format);
#ifdef SHL_BUILD_RTOS
        printf(format, arg);
#else
        vfprintf(stdout, format, arg);
#endif
        va_end(arg);
    }
}

void shl_debug_info(const char *format, ...)
{
    if (shl_debug_get_level() <= SHL_DEBUG_LEVEL_INFO) {
        va_list arg;
        va_start(arg, format);
#ifdef SHL_BUILD_RTOS
        printf(format, arg);
#else
        vfprintf(stdout, format, arg);
#endif
        va_end(arg);
    }
}

void shl_debug_warning(const char *format, ...)
{
    if (shl_debug_get_level() <= SHL_DEBUG_LEVEL_WARNING) {
        va_list arg;
        va_start(arg, format);
#ifdef SHL_BUILD_RTOS
        printf(format, arg);
#else
        vfprintf(stdout, format, arg);
#endif
        va_end(arg);
    }
}

void shl_debug_error(const char *format, ...)
{
    if (shl_debug_get_level() <= SHL_DEBUG_LEVEL_ERROR) {
        va_list arg;
        va_start(arg, format);
#ifdef SHL_BUILD_RTOS
        printf(format, arg);
#else
        vfprintf(stdout, format, arg);
#endif
        va_end(arg);
    }
}

void shl_debug_fatal(const char *format, ...)
{
    if (shl_debug_get_level() <= SHL_DEBUG_LEVEL_FATAL) {
        va_list arg;
        va_start(arg, format);
#ifdef SHL_BUILD_RTOS
        printf(format, arg);
#else
        vfprintf(stdout, format, arg);
#endif
        va_end(arg);
    }
}

static int shl_debug_print_list_int(int32_t *list, int len, char *name)
{
    shl_debug_info("%s", name);
    for (int i = 0; i < len; i++) {
        if (i == 0) {
            shl_debug_info("[");
        }
        shl_debug_info("%4d", list[i]);
        if (i == (len - 1)) {
            shl_debug_info("]");
        } else {
            shl_debug_info(",");
        }
    }
    return CSINN_TRUE;
}

static int shl_debug_print_list_float(float *list, int len, char *name)
{
    shl_debug_info("%s", name);
    for (int i = 0; i < len; i++) {
        if (i == 0) {
            shl_debug_info("[");
        }
        shl_debug_info("%f", list[i]);
        if (i == (len - 1)) {
            shl_debug_info("]");
        } else {
            shl_debug_info(",");
        }
    }
    return CSINN_TRUE;
}

int shl_debug_print_tensor(struct csinn_tensor *t)
{
    shl_debug_info("%s(", t->name);
    shl_debug_print_list_int(t->dim, t->dim_count, "");
    shl_debug_info(", ");

    /* FIX ME : channel quantize for input and output tensor ??? */
    if (t->quant_channel != 0) {
        shl_debug_info("max=%f, min=%f,", t->qinfo->max, t->qinfo->min);
        shl_debug_info("scale=%f, zp=%d", t->qinfo->scale, t->qinfo->zero_point);
    }

    shl_debug_info("), ");
    return CSINN_TRUE;
}

int shl_debug_print_params_base(struct csinn_params_base *base)
{
    shl_debug_info("%s(", base->name);
    if (base->layout == CSINN_LAYOUT_NCHW) {
        shl_debug_info("NCHW, ");
    } else if (base->layout == CSINN_LAYOUT_NHWC) {
        shl_debug_info("NHWC, ");
    }
    /* TODO : params.base.API ? */

    return CSINN_TRUE;
}

int shl_debug_print_siso_base(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_params_base *base, const char *name)
{
    shl_debug_info("%s = %s(", output->name, name);
    shl_debug_print_tensor(input);
    shl_debug_print_params_base(base);
    return CSINN_TRUE;
}

int shl_debug_print_diso_base(struct csinn_tensor *input0, struct csinn_tensor *input1,
                              struct csinn_tensor *output, struct csinn_params_base *base,
                              const char *name)
{
    shl_debug_info("%s = %s(", output->name, name);
    shl_debug_print_tensor(input0);
    shl_debug_print_tensor(input1);
    shl_debug_print_params_base(base);
    return CSINN_TRUE;
}

int shl_debug_print_sidcso_base(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                struct csinn_params_base *base, const char *name)
{
    shl_debug_info("%s = %s(", output->name, name);
    shl_debug_print_tensor(input);
    shl_debug_print_tensor(kernel);
    shl_debug_print_tensor(bias);
    shl_debug_print_params_base(base);
    return CSINN_TRUE;
}

int shl_siso_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_siso_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_diso_debug_info(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_diso_params *params,
                        const char *name)
{
    shl_debug_print_diso_base(input0, input1, output, &(params->base), name);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_conv1d_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv1d_params *params, const char *name)
{
    shl_debug_print_sidcso_base(input, output, kernel, bias, &(params->base), name);
    return CSINN_TRUE;
}

int shl_conv2d_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv2d_params *params, const char *name)
{
    shl_debug_print_sidcso_base(input, output, kernel, bias, &(params->base), name);
    shl_debug_info("pad=[%d,%d,%d,%d], stride=[%d,%d], dilation=[%d,%d])", params->pad_top,
                   params->pad_down, params->pad_left, params->pad_right, params->stride_height,
                   params->stride_width, params->dilation_height, params->dilation_width);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_fullyconnected_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_tensor *weights, struct csinn_tensor *bias,
                                  struct csinn_fc_params *params, const char *name)
{
    shl_debug_print_sidcso_base(input, output, weights, bias, &(params->base), name);
    shl_debug_info("units=%d", params->units);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_layer_norm_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *gamma, struct csinn_tensor *beta,
                              struct csinn_layer_norm_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    return CSINN_TRUE;
}

int shl_relu_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_relu_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("clip_min=0.0, clip_max=%f", params->n);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_conv3d_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                          struct csinn_conv3d_params *params, const char *name)
{
    shl_debug_print_sidcso_base(input, output, kernel, bias, &(params->base), name);
    shl_debug_info("pad=[%d,%d,%d,%d,%d,%d], stride=[%d,%d,%d], dilation=[%d,%d,%d]",
                   params->pad_front, params->pad_back, params->pad_top, params->pad_down,
                   params->pad_left, params->pad_right, params->stride_depth, params->stride_height,
                   params->stride_width, params->dilation_depth, params->dilation_height,
                   params->dilation_width);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_arange_debug_info(struct csinn_tensor *output, struct csinn_arange_params *params,
                          const char *name)
{
    shl_debug_info("%s = %s()\n", output->name, name);
    shl_debug_info("start=%f, stop=%f, step=%f", params->start, params->stop, params->step);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_pool_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_pool_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("pad=[%d,%d,%d,%d,%d,%d], stride=[%d,%d,%d], filter=[%d,%d,%d]",
                   params->pad_front, params->pad_back, params->pad_top, params->pad_down,
                   params->pad_left, params->pad_right, params->stride_depth, params->stride_height,
                   params->stride_width, params->filter_depth, params->filter_height,
                   params->filter_width);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_pad_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_pad_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("pad_value=%f, pad_mode=%d, ", params->pad_value, params->pad_mode);
    shl_debug_print_list_int(params->pad_before, params->pad_num, "pad_before=");
    shl_debug_info(", ");
    shl_debug_print_list_int(params->pad_after, params->pad_num, "pad_after=");
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_crop_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_crop_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("axis=%d, ", params->axis);
    shl_debug_print_list_int(params->offset, input->dim_count - params->axis, "offset=");
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_roi_pool_debug_info(struct csinn_tensor *data, struct csinn_tensor *rois,
                            struct csinn_tensor *output, struct csinn_roi_pool_params *params,
                            const char *name)
{
    shl_debug_print_siso_base(data, output, &(params->base), name);
    shl_debug_info("pooled_h=%d, pooled_w=%d, spatial_scale=%f", params->pooled_size_h,
                   params->pooled_size_w, params->spatial_scale);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_bn_debug_info(struct csinn_tensor *input, struct csinn_tensor *mean,
                      struct csinn_tensor *variance, struct csinn_tensor *gamma,
                      struct csinn_tensor *beta, struct csinn_tensor *output,
                      struct csinn_bn_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("epsilon=%f", params->epsilon);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_batch_to_space_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_batch_to_space_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("block_size=%d, crop=[%d,%d,%d,%d]", params->block_size, params->crop_top,
                   params->crop_bottom, params->crop_left, params->crop_right);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_batch_to_space_nd_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_batch_to_space_nd_params *params,
                                     const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_print_list_int(params->block_shape, params->spatial_dim_cnt, "block_shape=");
    shl_debug_print_list_int(params->crops, 2 * params->spatial_dim_cnt, "crops=");
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_depth_to_space_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_depth_to_space_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("block_size=%d\n", params->block_size);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_space_to_depth_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_space_to_depth_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("block_size=%d", params->block_size);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_space_to_batch_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_space_to_batch_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("block_size=%d, pad=[%d,%d,%d,%d]", params->block_size, params->pad_top,
                   params->pad_bottom, params->pad_left, params->pad_right);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_space_to_batch_nd_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_space_to_batch_nd_params *params,
                                     const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_print_list_int(params->block_shape, params->spatial_dim_cnt, "block_shape=");
    shl_debug_print_list_int(params->paddings, 2 * params->spatial_dim_cnt, "paddings=");
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_broadcast_to_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_broadcast_to_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_print_list_int(params->shape, params->shape_count, "shape=");
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_reduce_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_reduce_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("keepdim=%d, ", params->keepdims);
    shl_debug_print_list_int(params->axis, params->axis_count, "axis=");
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_cache_matmul_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *weight, struct csinn_tensor *bias,
                                struct csinn_cache_matmul_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    return CSINN_TRUE;
}

int shl_cache_conv1d_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_tensor *weight, struct csinn_tensor *bias,
                                struct csinn_cache_conv1d_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    return CSINN_TRUE;
}

int shl_clip_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_clip_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("min_value=%f, max_value=%f", params->min_value, params->max_value);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_col2im_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_col2im_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("pad_h=%d, pad_w=%d, stride_h=%d, stride_w=%d", params->pad_h, params->pad_w,
                   params->stride_h, params->stride_w);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_concat_debug_info(struct csinn_tensor **input, struct csinn_tensor *output,
                          struct csinn_concat_params *params, const char *name)
{
    shl_debug_info("%s = %s(", output->name, name);
    for (int i = 0; i < params->inputs_count; i++) {
        shl_debug_print_tensor(input[i]);
    }
    shl_debug_print_params_base(&(params->base));
    shl_debug_info("input_count=%d, axis=%d", params->inputs_count, params->axis);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_cumprod_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_cumprod_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("axis=%d, exclusive=%d", params->axis, params->exclusive);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_cumsum_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_cumsum_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("axis=%d, exclusive=%d", params->axis, params->exclusive);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_expand_dims_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                               struct csinn_expand_dims_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("axis=%d", params->axis);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_flatten_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_flatten_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_fsmn_debug_info(struct csinn_tensor *frame, struct csinn_tensor *l_filter,
                        struct csinn_tensor *r_filter, struct csinn_tensor *frame_sequence,
                        struct csinn_tensor *frame_counter, struct csinn_tensor *output,
                        struct csinn_fsmn_params *params, const char *name)
{
    shl_debug_info("%s = %s(", output->name, name);
    shl_debug_print_tensor(frame);
    shl_debug_print_tensor(l_filter);
    shl_debug_print_tensor(r_filter);
    shl_debug_print_tensor(frame_sequence);
    shl_debug_print_tensor(frame_counter);
    shl_debug_print_params_base(&(params->base));
    shl_debug_info("l_order=%d, r_order=%d, l_stride=%d, r_stride=%d, unavailable_frames=%d)",
                   params->l_order, params->r_order, params->l_stride, params->r_stride,
                   params->unavailable_frames);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_gather_nd_debug_info(struct csinn_tensor *input, struct csinn_tensor *indices,
                             struct csinn_tensor *output, struct csinn_gather_nd_params *params,
                             const char *name)
{
    shl_debug_print_diso_base(input, indices, output, &(params->base), name);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_gather_debug_info(struct csinn_tensor *input, struct csinn_tensor *indices,
                          struct csinn_tensor *output, struct csinn_gather_params *params,
                          const char *name)
{
    shl_debug_print_diso_base(input, indices, output, &(params->base), name);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_hard_sigmoid_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_sigmoid_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_im2col_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_im2col_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("pad=[%d,%d,%d,%d], stride=[%d,%d], kernel_size=[%d,%d]", params->pad_top,
                   params->pad_down, params->pad_left, params->pad_right, params->stride_h,
                   params->stride_w, params->kernel_h, params->kernel_w);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_l2n_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_l2n_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("spsilon=%f", params->epsilon);
    shl_debug_print_list_int(params->axis, params->n, "axis=");
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_softmax_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_softmax_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("axis=%d", params->axis);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_lrn_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_lrn_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("range=%d, bias=%f, alpha=%f, beta=%f", params->range, params->bias,
                   params->alpha, params->beta);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_matmul_debug_info(struct csinn_tensor *mat0, struct csinn_tensor *mat1,
                          struct csinn_tensor *output, struct csinn_matmul_params *params,
                          const char *name)
{
    shl_debug_print_diso_base(mat0, mat1, output, &(params->base), name);
    shl_debug_info("trans_a=%d, trans_b=%d", params->trans_a, params->trans_b);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_ndarray_size_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                struct csinn_ndarray_size_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_nms_debug_info(struct csinn_tensor *input0, struct csinn_tensor *input1,
                       struct csinn_tensor *output, struct csinn_non_max_suppression_params *params,
                       const char *name)
{
    shl_debug_print_diso_base(input0, input1, output, &(params->base), name);
    shl_debug_info("max_output_size=%d, iou_threshold=%f", params->max_output_size,
                   params->iou_threshold);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_one_hot_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_one_hot_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("on_value=%f, off_value=%f, depth=%d, axis=%d", params->f_on_value,
                   params->f_off_value, params->depth, params->axis);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_prelu_debug_info(struct csinn_tensor *input0, struct csinn_tensor *input1,
                         struct csinn_tensor *output, struct csinn_prelu_params *params,
                         const char *name)
{
    shl_debug_print_diso_base(input0, input1, output, &(params->base), name);
    shl_debug_info("axis=%d", params->axis);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_proposal_debug_info(struct csinn_tensor *cls_prob, struct csinn_tensor *bbox_pred,
                            struct csinn_tensor *im_info, struct csinn_tensor *output,
                            struct csinn_proposal_params *params, const char *name)
{
    shl_debug_print_siso_base(cls_prob, output, &(params->base), name);
    shl_debug_print_list_float(params->scales, params->scales_num, "scales=");
    shl_debug_info(", ");
    shl_debug_print_list_float(params->ratios, params->ratios_num, "ratios=");
    shl_debug_info(
        ", feature_stride=%d, threshold=%f, rpn_pre_nms_top_n=%d, rpn_post_nms_top_n=%d, "
        "rpn_min_size=%d, iou_loss=%d",
        params->feature_stride, params->threshold, params->rpn_pre_nms_top_n,
        params->rpn_post_nms_top_n, params->rpn_min_size, params->iou_loss);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_psroipooling_debug_info(struct csinn_tensor *data, struct csinn_tensor *rois,
                                struct csinn_tensor *output,
                                struct csinn_psroipooling_params *params, const char *name)
{
    shl_debug_print_siso_base(data, output, &(params->base), name);
    shl_debug_info("output_dim=%d, group_size=%d, spatial_scale=%f", params->output_dim,
                   params->group_size, params->spatial_scale);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_reorg_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_reorg_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("stride=%d", params->stride);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_reshape_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reshape_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_print_list_int(params->shape, params->shape_num, "shape=");
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_resize_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_resize_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("resize_mode=%d, align_corners=%d", params->resize_mode, params->align_corners);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_reverse_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_reverse_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("axis=%d", params->axis);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_roi_align_debug_info(struct csinn_tensor *data, struct csinn_tensor *rois,
                             struct csinn_tensor *output, struct csinn_roi_align_params *params,
                             const char *name)
{
    shl_debug_print_siso_base(data, output, &(params->base), name);
    shl_debug_info("pooled_h=%d, pool_w=%d, spatial_scale=%f, sample_ratio=%d",
                   params->pooled_size_h, params->pooled_size_w, params->spatial_scale,
                   params->sample_ratio);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_scatter_nd_debug_info(struct csinn_tensor *input, struct csinn_tensor *indices,
                              struct csinn_tensor *updates, struct csinn_tensor *output,
                              struct csinn_scatter_nd_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_segment_debug_info(struct csinn_tensor *input0, struct csinn_tensor *input1,
                           struct csinn_tensor *output, struct csinn_segment_params *params,
                           const char *name)
{
    shl_debug_print_diso_base(input0, input1, output, &(params->base), name);
    shl_debug_info("segment_nums=%d, unsorted=%d", params->num_segments, params->unsorted);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_select_debug_info(struct csinn_tensor *condition, struct csinn_tensor *input0,
                          struct csinn_tensor *input1, struct csinn_tensor *output,
                          struct csinn_select_params *params, const char *name)
{
    shl_debug_print_diso_base(input0, input1, output, &(params->base), name);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_sequence_mask_debug_info(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                 struct csinn_tensor *output,
                                 struct csinn_sequence_mask_params *params, const char *name)
{
    shl_debug_print_diso_base(input0, input1, output, &(params->base), name);
    shl_debug_info("mask_value=%f, axis=%d", params->mask_value, params->axis);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_shape_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_shape_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_shuffle_channel_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_shuffle_channel_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("group=%d", params->group);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_sigmoid_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_sigmoid_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_slice_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_slice_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_print_list_int(params->begin, params->slice_num, "begin=");
    shl_debug_info(", ");
    shl_debug_print_list_int(params->end, params->slice_num, "end=");
    shl_debug_info(", ");
    shl_debug_print_list_int(params->strides, params->slice_num, "strides=");
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_split_debug_info(struct csinn_tensor *input, struct csinn_tensor **output,
                         struct csinn_split_params *params, const char *name)
{
    shl_debug_info("%s-%s = %s(", output[0]->name, output[params->output_num - 1]->name, name);
    shl_debug_print_tensor(input);
    shl_debug_print_params_base(&(params->base));
    shl_debug_info("axis=%d, ", params->axis);
    if (params->split_index != NULL) {
        shl_debug_print_list_int(params->split_index, params->output_num, "split_index=");
    }
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_squeeze_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                           struct csinn_squeeze_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_print_list_int(params->axis, params->axis_num, "axis=");
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_stack_debug_info(struct csinn_tensor **input, struct csinn_tensor *output,
                         struct csinn_stack_params *params, const char *name)
{
    shl_debug_info("%s = %s(", output->name, name);
    for (int i = 0; i < params->inputs_count; i++) {
        shl_debug_print_tensor(input[i]);
    }
    shl_debug_print_params_base(&(params->base));
    shl_debug_info("input_count=%d, axis=%d", params->inputs_count, params->axis);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_strided_slice_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_strided_slice_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_print_list_int(params->begin, params->slice_count, "begin=");
    shl_debug_info(", ");
    shl_debug_print_list_int(params->end, params->slice_count, "end=");
    shl_debug_info(", ");
    shl_debug_print_list_int(params->stride, params->slice_count, "stride=");
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_tile_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_tile_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_print_list_int(params->reps, params->reps_num, "reps=");
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_topk_debug_info(struct csinn_tensor *input0, struct csinn_tensor *input1,
                        struct csinn_tensor *output, struct csinn_topk_params *params,
                        const char *name)
{
    shl_debug_print_diso_base(input0, input1, output, &(params->base), name);
    shl_debug_info("k=%d", params->k);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_transpose_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_transpose_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_print_list_int(params->permute, params->permute_num, "permute=");
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_unpooling_debug_info(struct csinn_tensor *input, struct csinn_tensor *mask,
                             struct csinn_tensor *output, struct csinn_unpooling_params *params,
                             const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("scale_h=%d, scale_w=%d, pad_out_h=%d, pad_out_w=%d", params->scale_height,
                   params->scale_width, params->pad_out_height, params->pad_out_width);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_unstack_debug_info(struct csinn_tensor *input, struct csinn_tensor **output,
                           struct csinn_unstack_params *params, const char *name)
{
    shl_debug_info("%s-%s = %s(", output[0]->name, output[params->outputs_count - 1]->name, name);
    shl_debug_print_tensor(input);
    shl_debug_print_params_base(&(params->base));
    shl_debug_info("outputs_count=%d, axis=%d", params->outputs_count, params->axis);
    return CSINN_TRUE;
}

int shl_where_debug_info(struct csinn_tensor *condition, struct csinn_tensor *x,
                         struct csinn_tensor *y, struct csinn_tensor *output,
                         struct csinn_where_params *params, const char *name)
{
    shl_debug_print_siso_base(x, output, &(params->base), name);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_where_softmax_debug_info(struct csinn_tensor *condition, struct csinn_tensor *y,
                                 struct csinn_tensor *output, struct csinn_where_softmax_params *params,
                                 const char *name)
{
    shl_debug_print_diso_base(condition, y, output, &(params->base), name);
    shl_debug_info("axis=%d", params->axis);
    shl_debug_info(")\n");
    return CSINN_TRUE;
}

int shl_cast_debug_info(struct csinn_tensor *input, struct csinn_tensor *output,
                        struct csinn_cast_params *params, const char *name)
{
    shl_debug_print_siso_base(input, output, &(params->base), name);
    shl_debug_info("dtype=%d", params->dtype);
    return CSINN_TRUE;
}

int shl_debug_callback_unset(char *func_name)
{
    shl_debug_info("callback function unset: %s\n", func_name);
    return CSINN_CALLBACK_UNSET;
}

int shl_debug_dump_data(struct csinn_tensor *input, char *filename)
{
    float *data = input->data;
    int size = csinn_tensor_size(input);
    int i = 0;
    FILE *fp = fopen(filename, "w+");
    for (i = 0; i < size; i++) {
        if (i == size - 1) {
            fprintf(fp, "%f", data[i]);
        } else {
            fprintf(fp, "%f\n", data[i]);
        }
    }
    fclose(fp);
    return CSINN_TRUE;
}

// TODO:complete string pointer table
char *op_strings[] = {
    [CSINN_OP_ABS] = "abs",
    [CSINN_OP_ADD] = "add",
    [CSINN_OP_MUL] = "mul",
    [CSINN_OP_AVGPOOL2D] = "avgpool2d",
    [CSINN_OP_CONCAT] = "concat",
    [CSINN_OP_CONV2D] = "conv2d",
    [CSINN_OP_CONV2D_RELU] = "conv2d_relu",
    [CSINN_OP_GROUP_CONV2D] = "group_conv2d",
    [CSINN_OP_GROUP_CONV2D_RELU6] = "group_conv2d_relu",
    [CSINN_OP_DEPTHWISE_CONV2D] = "dwconv2d",
    [CSINN_OP_DEPTHWISE_CONV2D_RELU] = "dwconv2d_relu",
    [CSINN_OP_DATA_CONVERT] = "data_convert",
    [CSINN_OP_FULLYCONNECTED] = "fullyconnected",
    [CSINN_OP_GLOBAL_AVGPOOL2D] = "global_avgpool2d",
    [CSINN_OP_LEAKY_RELU] = "leaky_relu",
    [CSINN_OP_MAXPOOL2D] = "maxpool2d",
    [CSINN_OP_RELU] = "relu",
    [CSINN_OP_RELU1] = "relu1",
    [CSINN_OP_RELU6] = "relu6",
    [CSINN_OP_RESHAPE] = "reshape",
    [CSINN_OP_RESIZE] = "resize",
    [CSINN_OP_TRANSPOSE] = "transpose",
    [CSINN_OP_SOFTMAX] = "softmax",
    [CSINN_OP_YUV_RGB_SCALE] = "yuv_rgb_scale",
    [CSINN_OP_SUB] = "sub",
    [CSINN_OP_MATMUL] = "matmul",
    [CSINN_OP_SPLIT] = "split",
    [CSINN_OP_SIGMOID] = "sigmoid",
    [CSINN_OP_PAD] = "pad",
    [CSINN_OP_STRIDED_SLICE] = "strided_slice",
    [CSINN_OP_MEAN] = "mean",
    [CSINN_OP_DIV] = "div",
    [CSINN_OP_POWER] = "power",
    [CSINN_OP_SQRT] = "sqrt",
    [CSINN_OP_GATHER] = "gather",
    [CSINN_OP_LAYER_NORM] = "layer_norm",
    [CSINN_OP_WHERE] = "where",
    [CSINN_OP_CONV1D] = "conv1d",
    [CSINN_OP_GROUP_CONV1D] = "group_conv1d",
    [CSINN_OP_DEPTHWISE_CONV1D] = "dwconv1d",
    [CSINN_OP_CLIP] = "clip",
    [CSINN_OP_WHERE_SOFTMAX] = "where_softmax",
    [CSINN_OP_ERF] = "erf",
    [CSINN_OP_CAST] = "cast",
};

// #define FREQ 50  // FPGA: 50MHz
int shl_benchmark_layer(struct shl_node *n, uint64_t start_time, uint64_t end_time, int layer_idx)
{
    struct shl_node *node = NULL;
    if (n->type == CSINN_SUBGRAPH) {
        // subgraph that holds some independent ops.
        struct shl_ref_graph *sgraph = n->data;
        // FIXME(@chenf): use the first node in subgraph.
        node = sgraph->layer[0];
    } else {
        // single cpu op
        node = n;
    }
    char *op_name = op_strings[node->type];
    float time_ms = (end_time - start_time) / 1000000.0f;
    shl_debug_info("[%3d]: %-16s %7.2fms  ^*^:", layer_idx, op_name, time_ms);

    struct csinn_tensor *in0 = (struct csinn_tensor *)node->in[0]->data;
    struct csinn_tensor *out0 = (struct csinn_tensor *)node->out[0]->data;
    // print first input node and first output node dim
    shl_debug_print_list_int(in0->dim, in0->dim_count, "");
    shl_debug_info(" ==> ");
    shl_debug_print_list_int(out0->dim, out0->dim_count, "");

    // conv/dwconv/deconv/fc/pool ...
    if ((node->type >= CSINN_OP_CONV2D && node->type <= CSINN_OP_CONV2D_CHANNEL_RELU6) ||
        (node->type >= CSINN_OP_GROUP_CONV2D && node->type <= CSINN_OP_GROUP_CONV2D_CHANNEL_RELU)) {
        struct csinn_tensor *in1 = (struct csinn_tensor *)node->in[1]->data;
        struct csinn_conv2d_params *params = node->data;
        int32_t k_h, k_w, in_c = 0;
        if (in1->layout == CSINN_LAYOUT_OIHW) {
            k_h = in1->dim[2];
            k_w = in1->dim[3];
            in_c = in1->dim[1];
        } else if (in1->layout == CSINN_LAYOUT_OHWI) {
            k_h = in1->dim[1];
            k_w = in1->dim[2];
            in_c = in1->dim[3];
        } else {
            shl_debug_info(" unsupport kernel layout ");
            k_h = 0;
            k_w = 0;
            in_c = 0;
        }
        float cacls = out0->dim[1] * out0->dim[2] * out0->dim[3] * 0.000001f * in_c * k_h * k_w * 2;
        shl_debug_info(" | k: %dx%d |", k_h, k_w);
        shl_debug_info(" s: %dx%d |", params->stride_height, params->stride_width);
        shl_debug_info(" p: %d %d %d %d | ", params->pad_top, params->pad_left, params->pad_down,
                       params->pad_right);
        shl_debug_info(" MOPS:%6.2f (%7.4fGOPS)", cacls, cacls / time_ms);
    } else if (node->type >= CSINN_OP_DEPTHWISE_CONV2D &&
               node->type <= CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6) {
        struct csinn_tensor *in1 = (struct csinn_tensor *)node->in[1]->data;
        struct csinn_conv2d_params *params = node->data;
        int32_t k_h, k_w = 0;
        if (in1->layout == CSINN_LAYOUT_O1HW) {
            k_h = in1->dim[2];
            k_w = in1->dim[3];
        } else if (in1->layout == CSINN_LAYOUT_1HWO) {
            k_h = in1->dim[1];
            k_w = in1->dim[2];
        } else {
            shl_debug_info(" unsupport kernel layout ");
            k_h = 0;
            k_w = 0;
        }
        float cacls = out0->dim[1] * out0->dim[2] * out0->dim[3] * 0.000001f * k_h * k_w * 2;
        shl_debug_info(" | k: %dx%d |", k_h, k_w);
        shl_debug_info(" s: %dx%d |", params->stride_height, params->stride_width);
        shl_debug_info(" p: %d %d %d %d | ", params->pad_top, params->pad_left, params->pad_down,
                       params->pad_right);
        shl_debug_info(" MOPS:%6.2f (%7.4fGOPS)", cacls, cacls / time_ms);
    } else if (node->type == CSINN_OP_AVGPOOL2D || node->type == CSINN_OP_MAXPOOL2D) {
        struct csinn_pool_params *params = node->data;
        shl_debug_info(" | k: %dx%d |", params->filter_height, params->filter_width);
        shl_debug_info(" s: %dx%d |", params->stride_height, params->stride_width);
        shl_debug_info(" p: %d %d %d %d | ", params->pad_top, params->pad_left, params->pad_down,
                       params->pad_right);
    } else if (node->type == CSINN_OP_FULLYCONNECTED) {
        float cacls = in0->dim[0] * in0->dim[1] * out0->dim[1] * 0.000001f * 2;
        shl_debug_info(" MOPS:%6.2f (%7.4fGOPS)", cacls, cacls / time_ms);
    } else if (node->type == CSINN_OP_MATMUL) {
        struct csinn_tensor *in1 = (struct csinn_tensor *)node->in[1]->data;
        struct csinn_matmul_params *params = node->data;
        int dim_m = in0->dim[in0->dim_count - (params->trans_a ? 1 : 2)];
        int dim_k = in0->dim[in0->dim_count - (params->trans_a ? 2 : 1)];
        int dim_n = in1->dim[in1->dim_count - (params->trans_b ? 2 : 1)];
        float cacls = dim_n * 0.000001f * 2;
        for (int i = 0; i < in0->dim_count; i++) {
            cacls *= in0->dim[i];
        }
        shl_debug_info(" | m,k,n: %d,%d,%d | ", dim_m, dim_k, dim_n);
        shl_debug_info(" MOPS:%6.2f (%7.4fGOPS)", cacls, cacls / time_ms);
    }
    shl_debug_info("\n");
    fflush(stdout);
    return CSINN_TRUE;
}

static uint32_t shl_debug_shape2string(uint32_t *shape, uint32_t dim_count, char *buf,
                                       uint32_t buf_sz)
{
    if (NULL == shape || NULL == buf || dim_count == 0 || buf_sz == 0) {
        return 0;
    }
    uint32_t count = 0;
    for (int i = 0; i < dim_count; i++) {
        if (count >= buf_sz) {
            break;
        }
        count += snprintf(&buf[count], buf_sz - count, "%d_", shape[i]);
    }
    buf[count - 1] = 0;
    return count;
}

static char *shl_debug_filter_invalid_char(char *src)
{
    char *dst = shl_mem_alloc(1024);
    const char INVALID_CHAR[] = "/!#$%^&*()-+<>?;\\ ";
    int k = 0;
    for (int i = 0; src[i] != '\0'; i++) {
        bool flag = true;
        for (int j = 0; INVALID_CHAR[j] != '\0'; j++) {
            if (src[i] == INVALID_CHAR[j]) {
                flag = false;
                break;
            }
        }
        dst[k++] = flag ? src[i] : '_';
    }
    dst[k] = '\0';
    return dst;
}

int __attribute__((weak)) shl_dump_output_tensor(struct shl_node *node)
{
#ifndef SHL_BUILD_RTOS
    const char TENSOR_DUMP_DIR[] = "shl_dump";
    DIR *dir = opendir(TENSOR_DUMP_DIR);
    if (dir) {
        closedir(dir);
    } else {
        mkdir(TENSOR_DUMP_DIR, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    }
    int output_num = 1;
    struct shl_node **output_node;
    int is_cpu_node = 1;
    if (node->type == CSINN_SUBGRAPH) {
        struct shl_ref_graph *sgraph = node->data;
        output_num = sgraph->output_num;
        output_node = sgraph->output;
        is_cpu_node = 0;
    } else {
        output_num = node->out_num;
        output_node = node->out;
    }
    for (int i = 0; i < output_num; i++) {
        struct csinn_tensor *output = (struct csinn_tensor *)output_node[i]->data;
        char shape[128] = {0};
        shl_debug_shape2string(output->dim, output->dim_count, shape, 128);
        char *output_name = shl_debug_filter_invalid_char(output->name);
        char filename[1024] = {0};
        snprintf(filename, 1024, "%s/%s_%s.txt", TENSOR_DUMP_DIR, output_name, shape);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
        shl_debug_dump_data(foutput, filename);
        shl_ref_tensor_transform_free_f32(foutput);
        shl_mem_free(output_name);
    }
    if (node->type == CSINN_OP_CONV2D || node->type == CSINN_OP_DEPTHWISE_CONV2D ||
        node->type == CSINN_OP_FULLYCONNECTED && is_cpu_node) {
        // dump output
        struct csinn_tensor *kernel_node = node->in[1]->data;
        char shape[128] = {0};
        shl_debug_shape2string(kernel_node->dim, kernel_node->dim_count, shape, 128);
        char *kernel_name = shl_debug_filter_invalid_char(kernel_node->name);
        char filename[1024] = {0};
        snprintf(filename, 1024, "%s/%s_%s.txt", TENSOR_DUMP_DIR, kernel_name, shape);
        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(kernel_node);
        shl_debug_dump_data(foutput, filename);
        shl_ref_tensor_transform_free_f32(foutput);
        shl_mem_free(kernel_name);

        // dump input
        struct csinn_tensor *input_node = node->in[0]->data;
        char input_shape[128] = {0};
        shl_debug_shape2string(input_node->dim, input_node->dim_count, shape, 128);
        char *input_name = shl_debug_filter_invalid_char(input_node->name);
        char input_filename[1024] = {0};
        snprintf(filename, 1024, "%s/%s_%s_input.txt", TENSOR_DUMP_DIR, input_name, shape);
        struct csinn_tensor *finput = shl_ref_tensor_transform_f32(input_node);
        shl_debug_dump_data(finput, filename);
        shl_ref_tensor_transform_free_f32(finput);
        shl_mem_free(input_name);
    }
#endif
    return CSINN_TRUE;
}
#endif
