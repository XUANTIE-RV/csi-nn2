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

#include <stdarg.h>
#include <stdio.h>
#include "csi_nn.h"
#include "csi_node.h"

int csi_debug_level = CSI_DEBUG_LEVEL_WARNING;

int csi_debug_get_level()
{
    return csi_debug_level;
}

void csi_debug_set_level(int level)
{
    csi_debug_level = level;
}
#ifdef CSI_DEBUG
void csi_debug_debug(const char *format, ...)
{
    if (csi_debug_get_level() <= CSI_DEBUG_LEVEL_DEBUG) {
        va_list arg;
        va_start(arg, format);
        vfprintf(stdout, format, arg);
        va_end(arg);
    }
}

void csi_debug_info(const char *format, ...)
{
    if (csi_debug_get_level() <= CSI_DEBUG_LEVEL_INFO) {
        va_list arg;
        va_start(arg, format);
        vfprintf(stdout, format, arg);
        va_end(arg);
    }
}

void csi_debug_warning(const char *format, ...)
{
    if (csi_debug_get_level() <= CSI_DEBUG_LEVEL_WARNING) {
        va_list arg;
        va_start(arg, format);
        vfprintf(stdout, format, arg);
        va_end(arg);
    }
}

void csi_debug_error(const char *format, ...)
{
    if (csi_debug_get_level() <= CSI_DEBUG_LEVEL_ERROR) {
        va_list arg;
        va_start(arg, format);
        vfprintf(stdout, format, arg);
        va_end(arg);
    }
}

void csi_debug_fatal(const char *format, ...)
{
    if (csi_debug_get_level() <= CSI_DEBUG_LEVEL_FATAL) {
        va_list arg;
        va_start(arg, format);
        vfprintf(stdout, format, arg);
        va_end(arg);
    }
}

static int csi_debug_print_list_int(int32_t *list, int len, char *name)
{
    csi_debug_info("%s", name);
    for (int i = 0; i < len; i++) {
        if (i == 0) {
            csi_debug_info("[");
        }
        csi_debug_info("%4d", list[i]);
        if (i == (len - 1)) {
            csi_debug_info("]");
        } else {
            csi_debug_info(",");
        }
    }
    return CSINN_TRUE;
}

static int csi_debug_print_list_float(float *list, int len, char *name)
{
    csi_debug_info("%s", name);
    for (int i = 0; i < len; i++) {
        if (i == 0) {
            csi_debug_info("[");
        }
        csi_debug_info("%f", list[i]);
        if (i == (len - 1)) {
            csi_debug_info("]");
        } else {
            csi_debug_info(",");
        }
    }
    return CSINN_TRUE;
}

int csi_debug_print_tensor(struct csi_tensor *t)
{
    csi_debug_info("%s(", t->name);
    csi_debug_print_list_int(t->dim, t->dim_count, "");
    csi_debug_info(", ");

    /* FIX ME : channel quantize for input and output tensor ??? */
    if (t->quant_channel != 0) {
        csi_debug_info("max=%f, min=%f", t->qinfo->max, t->qinfo->min);
    }

    csi_debug_info("), ");
    return CSINN_TRUE;
}

int csi_debug_print_params_base(struct csi_params_base *base)
{
    csi_debug_info("%s(", base->name);
    if (base->layout == CSINN_LAYOUT_NCHW) {
        csi_debug_info("NCHW, ");
    } else if (base->layout == CSINN_LAYOUT_NHWC) {
        csi_debug_info("NHWC, ");
    }
    /* TODO : params.base.API ? */

    return CSINN_TRUE;
}

int csi_debug_print_siso_base(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct csi_params_base *base,
                              const char *name)
{
    csi_debug_info("%s = %s(", output->name, name);
    csi_debug_print_tensor(input);
    csi_debug_print_params_base(base);
    return CSINN_TRUE;
}

int csi_debug_print_diso_base(struct csi_tensor *input0,
                              struct csi_tensor *input1,
                              struct csi_tensor *output,
                              struct csi_params_base *base,
                              const char *name)
{
    csi_debug_info("%s = %s(", output->name, name);
    csi_debug_print_tensor(input0);
    csi_debug_print_tensor(input1);
    csi_debug_print_params_base(base);
    return CSINN_TRUE;
}

int csi_debug_print_sidcso_base(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct csi_tensor *kernel,
                                struct csi_tensor *bias,
                                struct csi_params_base *base,
                                const char *name)
{
    csi_debug_info("%s = %s(", output->name, name);
    csi_debug_print_tensor(input);
    csi_debug_print_tensor(kernel);
    csi_debug_print_tensor(bias);
    csi_debug_print_params_base(base);
    return CSINN_TRUE;
}

int csi_siso_debug_info(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params,
                        const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_diso_debug_info(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params,
                        const char *name)
{
    csi_debug_print_diso_base(input0, input1, output, &(params->base), name);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_conv1d_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct csi_tensor *kernel,
                          struct csi_tensor *bias,
                          struct conv1d_params *params,
                          const char *name)
{
    csi_debug_print_sidcso_base(input, output, kernel, bias, &(params->base), name);
    return CSINN_TRUE;
}

int csi_conv2d_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct csi_tensor *kernel,
                          struct csi_tensor *bias,
                          struct conv2d_params *params,
                          const char *name)
{
    csi_debug_print_sidcso_base(input, output, kernel, bias, &(params->base), name);
    csi_debug_info("pad=[%d,%d,%d,%d], stride=[%d,%d], dilation=[%d,%d])",
        params->pad_top, params->pad_down, params->pad_left, params->pad_right,
        params->stride_height, params->stride_width,
        params->dilation_height, params->dilation_width);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_fullyconnected_debug_info(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct csi_tensor *weights,
                                  struct csi_tensor *bias,
                                  struct fc_params *params,
                                  const char *name)
{
    csi_debug_print_sidcso_base(input, output, weights, bias, &(params->base), name);
    csi_debug_info("units=%d", params->units);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_layer_norm_debug_info(struct csi_tensor *input,
                              struct csi_tensor *output,
                              struct csi_tensor *gamma,
                              struct csi_tensor *beta,
                              struct layer_norm_params *params,
                              const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    return CSINN_TRUE;
}

int csi_relu_debug_info(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct relu_params *params,
                        const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("clip_min=0.0, clip_max=%f", params->n);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_conv3d_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct csi_tensor *kernel,
                          struct csi_tensor *bias,
                          struct conv3d_params *params,
                          const char *name)
{
    csi_debug_print_sidcso_base(input, output, kernel, bias, &(params->base), name);
    csi_debug_info("pad=[%d,%d,%d,%d,%d,%d], stride=[%d,%d,%d], dilation=[%d,%d,%d]",
        params->pad_front, params->pad_back, params->pad_top, params->pad_down, params->pad_left, params->pad_right,
        params->stride_depth, params->stride_height, params->stride_width,
        params->dilation_depth, params->dilation_height, params->dilation_width);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_arange_debug_info(struct csi_tensor *output,
                          struct arange_params *params,
                          const char *name)
{
    csi_debug_info("%s = %s()\n", output->name, name);
    csi_debug_info("start=%f, stop=%f, step=%f",params->start, params->stop, params->step);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_pool_debug_info(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct pool_params *params,
                        const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("pad=[%d,%d,%d,%d,%d,%d], stride=[%d,%d,%d], filter=[%d,%d,%d]",
        params->pad_front, params->pad_back, params->pad_top, params->pad_down, params->pad_left, params->pad_right,
        params->stride_depth, params->stride_height, params->stride_width,
        params->filter_depth, params->filter_height, params->filter_width);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_pad_debug_info(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct pad_params *params,
                       const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("pad_value=%f, pad_mode=%d, ", params->pad_value, params->pad_mode);
    csi_debug_print_list_int(params->pad_before, params->pad_num, "pad_before=");
    csi_debug_info(", ");
    csi_debug_print_list_int(params->pad_after, params->pad_num, "pad_after=");
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_crop_debug_info(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct crop_params *params,
                        const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("axis=%d, ", params->axis);
    csi_debug_print_list_int(params->offset, input->dim_count - params->axis, "offset=");
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_roi_pool_debug_info(struct csi_tensor *data,
                            struct csi_tensor *rois,
                            struct csi_tensor *output,
                            struct roi_pool_params *params,
                            const char *name)
{
    csi_debug_print_siso_base(data, output, &(params->base), name);
    csi_debug_info("pooled_h=%d, pooled_w=%d, spatial_scale=%f",
        params->pooled_size_h, params->pooled_size_w, params->spatial_scale);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_bn_debug_info(struct csi_tensor *input,
                      struct csi_tensor *mean,
                      struct csi_tensor *variance,
                      struct csi_tensor *gamma,
                      struct csi_tensor *beta,
                      struct csi_tensor *output,
                      struct bn_params *params,
                      const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("epsilon=%f", params->epsilon);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_batch_to_space_debug_info(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct batch_to_space_params *params,
                                  const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("block_size=%d, crop=[%d,%d,%d,%d]", params->block_size,
        params->crop_top, params->crop_bottom, params->crop_left, params->crop_right);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_batch_to_space_nd_debug_info(struct csi_tensor *input,
                                     struct csi_tensor *output,
                                     struct batch_to_space_nd_params *params,
                                     const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_print_list_int(params->block_shape, params->spatial_dim_cnt, "block_shape=");
    csi_debug_print_list_int(params->crops, 2 * params->spatial_dim_cnt, "crops=");
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_depth_to_space_debug_info(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct depth_to_space_params *params,
                                  const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("block_size=%d\n", params->block_size);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_space_to_depth_debug_info(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct space_to_depth_params *params,
                                  const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("block_size=%d", params->block_size);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_space_to_batch_debug_info(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct space_to_batch_params *params,
                                  const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("block_size=%d, pad=[%d,%d,%d,%d]", params->block_size,
        params->pad_top, params->pad_bottom, params->pad_left, params->pad_right);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_space_to_batch_nd_debug_info(struct csi_tensor *input,
                                     struct csi_tensor *output,
                                     struct space_to_batch_nd_params *params,
                                     const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_print_list_int(params->block_shape, params->spatial_dim_cnt, "block_shape=");
    csi_debug_print_list_int(params->paddings, 2 * params->spatial_dim_cnt, "paddings=");
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_broadcast_to_debug_info(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct broadcast_to_params *params,
                                const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_print_list_int(params->shape, params->shape_count, "shape=");
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_reduce_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct reduce_params *params,
                          const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("keepdim=%d, ", params->keepdims);
    csi_debug_print_list_int(params->axis, params->axis_count, "axis=");
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_cache_matmul_debug_info(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *weight, struct csi_tensor *bias,
                                struct cache_matmul_params *params, const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    return CSINN_TRUE;
}

int csi_cache_conv1d_debug_info(struct csi_tensor *input, struct csi_tensor *output,
                                struct csi_tensor *weight, struct csi_tensor *bias,
                                struct cache_conv1d_params *params, const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    return CSINN_TRUE;
}

int csi_clip_debug_info(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct clip_params *params,
                        const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("min_value=%f, max_value=%f", params->min_value, params->max_value);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_col2im_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct col2im_params *params,
                          const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("pad_h=%d, pad_w=%d, stride_h=%d, stride_w=%d",
        params->pad_h, params->pad_w, params->stride_h, params->stride_w);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_concat_debug_info(struct csi_tensor **input,
                          struct csi_tensor *output,
                          struct concat_params *params,
                          const char *name)
{
    csi_debug_info("%s = %s(", output->name, name);
    for (int i = 0; i < params->inputs_count; i++) {
        csi_debug_print_tensor(input[i]);
    }
    csi_debug_print_params_base(&(params->base));
    csi_debug_info("input_count=%d, axis=%d", params->inputs_count, params->axis);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_cumprod_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct cumprod_params *params,
                           const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("axis=%d, exclusive=%d", params->axis, params->exclusive);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_cumsum_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct cumsum_params *params,
                          const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("axis=%d, exclusive=%d", params->axis, params->exclusive);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_expand_dims_debug_info(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct expand_dims_params *params,
                               const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("axis=%d", params->axis);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_flatten_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct flatten_params *params,
                           const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_fsmn_debug_info(struct csi_tensor *frame,
                        struct csi_tensor *l_filter,
                        struct csi_tensor *r_filter,
                        struct csi_tensor *frame_sequence,
                        struct csi_tensor *frame_counter,
                        struct csi_tensor *output,
                        struct fsmn_params *params,
                        const char *name)
{
    csi_debug_info("%s = %s(", output->name, name);
    csi_debug_print_tensor(frame);
    csi_debug_print_tensor(l_filter);
    csi_debug_print_tensor(r_filter);
    csi_debug_print_tensor(frame_sequence);
    csi_debug_print_tensor(frame_counter);
    csi_debug_print_params_base(&(params->base));
    csi_debug_info("l_order=%d, r_order=%d, l_stride=%d, r_stride=%d, unavailable_frames=%d)",
        params->l_order, params->r_order, params->l_stride, params->r_stride,
        params->unavailable_frames);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_gather_nd_debug_info(struct csi_tensor *input,
                             struct csi_tensor *indices,
                             struct csi_tensor *output,
                             struct gather_nd_params *params,
                             const char *name)
{
    csi_debug_print_diso_base(input, indices, output, &(params->base), name);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_gather_debug_info(struct csi_tensor *input,
                          struct csi_tensor *indices,
                          struct csi_tensor *output,
                          struct gather_params *params,
                          const char *name)
{
    csi_debug_print_diso_base(input, indices, output, &(params->base), name);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_hard_sigmoid_debug_info(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct sigmoid_params *params,
                                const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_im2col_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct im2col_params *params,
                          const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("pad=[%d,%d,%d,%d], stride=[%d,%d], kernel_size=[%d,%d]",
    params->pad_top, params->pad_down, params->pad_left, params->pad_right,
    params->stride_h, params->stride_w, params->kernel_h, params->kernel_w);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_l2n_debug_info(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct l2n_params *params,
                       const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("spsilon=%f", params->epsilon);
    csi_debug_print_list_int(params->axis, params->n, "axis=");
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_softmax_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct softmax_params *params,
                           const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("axis=%d", params->axis);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_lrn_debug_info(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct lrn_params *params,
                       const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("range=%d, bias=%f, alpha=%f, beta=%f", params->range, params->bias, params->alpha, params->beta);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_matmul_debug_info(struct csi_tensor *mat0,
                          struct csi_tensor *mat1,
                          struct csi_tensor *output,
                          struct matmul_params *params,
                          const char *name)
{
    csi_debug_print_diso_base(mat0, mat1, output, &(params->base), name);
    csi_debug_info("trans_a=%d, trans_b=%d", params->trans_a, params->trans_b);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_ndarray_size_debug_info(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct ndarray_size_params *params,
                                const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_nms_debug_info(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct non_max_suppression_params *params,
                       const char *name)
{
    csi_debug_print_diso_base(input0, input1, output, &(params->base), name);
    csi_debug_info("max_output_size=%d, iou_threshold=%f", params->max_output_size, params->iou_threshold);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_one_hot_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct one_hot_params *params,
                           const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("on_value=%f, off_value=%f, depth=%d, axis=%d", params->f_on_value, params->f_off_value, params->depth, params->axis);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_prelu_debug_info(struct csi_tensor *input0,
                         struct csi_tensor *input1,
                         struct csi_tensor *output,
                         struct prelu_params *params,
                         const char *name)
{
    csi_debug_print_diso_base(input0, input1, output, &(params->base), name);
    csi_debug_info("axis=%d", params->axis);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_proposal_debug_info(struct csi_tensor *cls_prob,
                            struct csi_tensor *bbox_pred,
                            struct csi_tensor *im_info,
                            struct csi_tensor *output,
                            struct proposal_params *params,
                            const char *name)
{
    csi_debug_print_siso_base(cls_prob, output, &(params->base), name);
    csi_debug_print_list_float(params->scales, params->scales_num, "scales=");
    csi_debug_info(", ");
    csi_debug_print_list_float(params->ratios, params->ratios_num, "ratios=");
    csi_debug_info(", feature_stride=%d, threshold=%f, rpn_pre_nms_top_n=%d, rpn_post_nms_top_n=%d, rpn_min_size=%d, iou_loss=%d",
        params->feature_stride, params->threshold, params->rpn_pre_nms_top_n, params->rpn_post_nms_top_n, params->rpn_min_size, params->iou_loss);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_psroipooling_debug_info(struct csi_tensor *data,
                                struct csi_tensor *rois,
                                struct csi_tensor *output,
                                struct psroipooling_params *params,
                                const char *name)
{
    csi_debug_print_siso_base(data, output, &(params->base), name);
    csi_debug_info("output_dim=%d, group_size=%d, spatial_scale=%f",
        params->output_dim, params->group_size, params->spatial_scale);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_reorg_debug_info(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct reorg_params *params,
                         const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("stride=%d", params->stride);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_reshape_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct reshape_params *params,
                           const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_print_list_int(params->shape, params->shape_num, "shape=");
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_resize_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct resize_params *params,
                          const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("resize_mode=%d, align_corners=%d", params->resize_mode, params->align_corners);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_reverse_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct reverse_params *params,
                           const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("axis=%d", params->axis);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_roi_align_debug_info(struct csi_tensor *data,
                             struct csi_tensor *rois,
                             struct csi_tensor *output,
                             struct roi_align_params *params,
                             const char *name)
{
    csi_debug_print_siso_base(data, output, &(params->base), name);
    csi_debug_info("pooled_h=%d, pool_w=%d, spatial_scale=%f, sample_ratio=%d",
        params->pooled_size_h, params->pooled_size_w, params->spatial_scale, params->sample_ratio);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_scatter_nd_debug_info(struct csi_tensor *input,
                              struct csi_tensor *indices,
                              struct csi_tensor *updates,
                              struct csi_tensor *output,
                              struct scatter_nd_params *params,
                              const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_segment_debug_info(struct csi_tensor *input0,
                           struct csi_tensor *input1,
                           struct csi_tensor *output,
                           struct segment_params *params,
                           const char *name)
{
    csi_debug_print_diso_base(input0, input1, output, &(params->base), name);
    csi_debug_info("segment_nums=%d, unsorted=%d", params->num_segments, params->unsorted);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_select_debug_info(struct csi_tensor *condition,
                          struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct select_params *params,
                          const char *name)
{
    csi_debug_print_diso_base(input0, input1, output, &(params->base), name);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_sequence_mask_debug_info(struct csi_tensor *input0,
                                 struct csi_tensor *input1,
                                 struct csi_tensor *output,
                                 struct sequence_mask_params *params,
                                 const char *name)
{
    csi_debug_print_diso_base(input0, input1, output, &(params->base), name);
    csi_debug_info("mask_value=%f, axis=%d", params->mask_value, params->axis);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_shape_debug_info(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct shape_params *params,
                         const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_shuffle_channel_debug_info(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct shuffle_channel_params *params,
                                   const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("group=%d", params->group);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_sigmoid_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct sigmoid_params *params,
                           const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_slice_debug_info(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct slice_params *params,
                         const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_print_list_int(params->begin, params->slice_num, "begin=");
    csi_debug_info(", ");
    csi_debug_print_list_int(params->end, params->slice_num, "end=");
    csi_debug_info(", ");
    csi_debug_print_list_int(params->strides, params->slice_num, "strides=");
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_split_debug_info(struct csi_tensor *input,
                         struct csi_tensor **output,
                         struct split_params *params,
                         const char *name)
{
    csi_debug_info("%s-%s = %s(", output[0]->name, output[params->output_num - 1]->name, name);
    csi_debug_print_tensor(input);
    csi_debug_print_params_base(&(params->base));
    csi_debug_info("axis=%d, ", params->axis);
    csi_debug_print_list_int(params->split_index, params->output_num, "split_index=");
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_squeeze_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct squeeze_params *params,
                           const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_print_list_int(params->axis, params->axis_num, "axis=");
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_stack_debug_info(struct csi_tensor **input,
                         struct csi_tensor *output,
                         struct stack_params *params,
                         const char *name)
{
    csi_debug_info("%s = %s(", output->name, name);
    for (int i = 0; i < params->inputs_count; i++) {
        csi_debug_print_tensor(input[i]);
    }
    csi_debug_print_params_base(&(params->base));
    csi_debug_info("input_count=%d, axis=%d", params->inputs_count, params->axis);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_strided_slice_debug_info(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct strided_slice_params *params,
                                 const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_print_list_int(params->begin, params->slice_count, "begin=");
    csi_debug_info(", ");
    csi_debug_print_list_int(params->end, params->slice_count, "end=");
    csi_debug_info(", ");
    csi_debug_print_list_int(params->stride, params->slice_count, "stride=");
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_tile_debug_info(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct tile_params *params,
                        const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_print_list_int(params->reps, params->reps_num, "reps=");
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_topk_debug_info(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct topk_params *params,
                        const char *name)
{
    csi_debug_print_diso_base(input0, input1, output, &(params->base), name);
    csi_debug_info("k=%d", params->k);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_transpose_debug_info(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct transpose_params *params,
                             const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_print_list_int(params->permute, params->permute_num, "permute=");
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_unpooling_debug_info(struct csi_tensor *input,
                             struct csi_tensor *mask,
                             struct csi_tensor *output,
                             struct unpooling_params *params,
                             const char *name)
{
    csi_debug_print_siso_base(input, output, &(params->base), name);
    csi_debug_info("scale_h=%d, scale_w=%d, pad_out_h=%d, pad_out_w=%d",
        params->scale_height, params->scale_width, params->pad_out_height , params->pad_out_width);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_unstack_debug_info(struct csi_tensor *input,
                           struct csi_tensor **output,
                           struct unstack_params *params,
                           const char *name)
{
    csi_debug_info("%s-%s = %s(", output[0]->name, output[params->outputs_count - 1]->name, name);
    csi_debug_print_tensor(input);
    csi_debug_print_params_base(&(params->base));
    csi_debug_info("outputs_count=%d, axis=%d", params->outputs_count, params->axis);
    return CSINN_TRUE;
}

int csi_where_debug_info(struct csi_tensor *condition,
                         struct csi_tensor *x,
                         struct csi_tensor *y,
                         struct csi_tensor *output,
                         struct where_params *params,
                         const char *name)
{
    csi_debug_print_siso_base(x, output, &(params->base), name);
    csi_debug_info(")\n");
    return CSINN_TRUE;
}

int csi_debug_callback_unset(char *func_name)
{
    csi_debug_info("callback function unset: %s\n", func_name);
    return CSINN_CALLBACK_UNSET;
}

int csi_debug_dump_data(struct csi_tensor *input, char *filename)
{
    float *data = input->data;
    int size = csi_tensor_size(input);
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
    [CSINN_OP_AVGPOOL2D] = "avgpool2d",
    [CSINN_OP_CONCAT] = "concat",
    [CSINN_OP_CONV2D] = "conv2d",
    [CSINN_OP_CONV2D_RELU] = "conv2d_relu",
    [CSINN_OP_DEPTHWISE_CONV2D] = "dwconv2d",
    [CSINN_OP_DEPTHWISE_CONV2D_RELU] = "dwconv2d_relu",
    [CSINN_OP_FULLYCONNECTED] = "fullyconnected",
    [CSINN_OP_GLOBAL_AVGPOOL2D] = "global_avgpool2d",
    [CSINN_OP_LEAKY_RELU] = "leaky_relu",
    [CSINN_OP_MAXPOOL2D] = "maxpool2d",
    [CSINN_OP_RELU] = "relu",
    [CSINN_OP_RESHAPE] = "reshape",
    [CSINN_OP_SOFTMAX] = "softmax",
    [CSINN_OP_YUV_RGB_SCALE] = "yuv_rgb_scale",
};

#define FREQ 30  // FPGA: 30MHz
// TODO: support NHWC layout too
int csi_benchmark_layer(struct csi_node *node, uint64_t start_time, uint64_t end_time,
                        int layer_idx)
{
    char *op_name = op_strings[node->type];
    csi_debug_info("[%3d]: %-18s %6.2lfms  ^*^ feature_map:", layer_idx, op_name,
                   (end_time - start_time) / 1000000.0f);

    struct csi_tensor *in0 = (struct csi_tensor *)node->in[0]->data;
    struct csi_tensor *out0 = (struct csi_tensor *)node->out[0]->data;
    // print first input node and first output node dim
    csi_debug_print_list_int(in0->dim, in0->dim_count, "");
    csi_debug_info(" ==> ");
    csi_debug_print_list_int(out0->dim, out0->dim_count, "");
    // print kernel dim
    if (node->type >= CSINN_OP_CONV1D && node->type <= CSINN_OP_CONV3D) {
        struct csi_tensor *in1 = (struct csi_tensor *)node->in[1]->data;
        int64_t cacls = out0->dim[1] * out0->dim[2] * out0->dim[3] * in0->dim[1] * in1->dim[2] *
                        in1->dim[3] * 2;
        if (node->type >= CSINN_OP_DEPTHWISE_CONV2D &&
            node->type <= CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6) {
            cacls = out0->dim[1] * out0->dim[2] * out0->dim[3] * in1->dim[2] * in1->dim[3] * 2;
        }
        csi_debug_info("  (%2.4lfGOPS)", cacls / ((end_time - start_time) * 30 / 1000.0f));
        csi_debug_info("   kernel:");
        csi_debug_print_list_int(in1->dim, in1->dim_count, "");
    }
    csi_debug_info("\n");
    return CSINN_TRUE;
}

#endif
