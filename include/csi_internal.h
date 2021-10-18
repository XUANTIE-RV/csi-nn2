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
#ifndef _CSI_INTERNAL_H
#define _CSI_INTERNAL_H

/* data type */
enum
{
    CSINN_DTYPE_UINT8   = 0x0,
    CSINN_DTYPE_INT8    = 0x1,
    CSINN_DTYPE_UINT16  = 0x2,
    CSINN_DTYPE_INT16   = 0x3,
    CSINN_DTYPE_UINT32  = 0x4,
    CSINN_DTYPE_INT32   = 0x5,
    CSINN_DTYPE_FLOAT16 = 0x6,
    CSINN_DTYPE_FLOAT32 = 0x7,
    CSINN_DTYPE_FLOAT64 = 0x8,
};

/* pad mode */
enum
{
    CSINN_PAD_CONSTANT = 0x0, /* pads with constant_value pad_value */
    CSINN_PAD_EDGE = 0x1,     /* pads using the edge values of the input array */
    CSINN_PAD_REFLECT = 0x2,  /* pads by reflecting values with respect to the edge */
};

/* resize mode */
enum
{
    CSINN_RESIZE_BILINEAR = 0x0,
    CSINN_RESIZE_NEAREST_NEIGHBOR = 0x1,
    CSINN_RESIZE_NEAREST_BICUBIC = 0x2,
};

enum
{
    CSINN_NCHW = 0x0,
    CSINN_NHWC = 0x1,
    CSINN_NCDHW = 0x2,
    CSINN_NDHWC = 0x3,
};

enum
{
    CSINN_UNSUPPORT_LAYOUT = -3,
    CSINN_UNSUPPORT_DTYPE = -2,
    CSINN_CALLBACK_UNSET = -1,
    CSINN_FALSE = 0,
    CSINN_TRUE = 1,
};

#define MAX_DIM 8
struct csi_tensor
{
    void *data;
    int32_t dtype;
    int32_t dim[MAX_DIM];
    int32_t dim_count;
    int32_t zero_point;
    float scale;
    int32_t offset;
    int32_t multiplier;
    int32_t shift;
    int32_t layout;
    void *t_private;
} __attribute__((packed));

struct conv2d_params
{
    int (*bc)();
    int32_t layout;
    int32_t group;
    int32_t stride_height;
    int32_t stride_width;
    int32_t pad_top;
    int32_t pad_left;
    int32_t pad_down;
    int32_t pad_right;
    int32_t dilation_height;
    int32_t dilation_width;
};

struct conv3d_params
{
    int (*bc)();
    int32_t api;
    int32_t layout;
    int32_t group;
    int32_t stride_depth;
    int32_t stride_height;
    int32_t stride_width;
    int32_t pad_top;
    int32_t pad_left;
    int32_t pad_down;
    int32_t pad_right;
    int32_t pad_front;
    int32_t pad_back;
    int32_t dilation_depth;
    int32_t dilation_height;
    int32_t dilation_width;
    int32_t out_pad_depth;
    int32_t out_pad_height;
    int32_t out_pad_width;
};

struct fc_params
{
    int (*bc)();
    int32_t layout;
};

struct pool_params
{
    int (*bc)();
    int32_t layout;
    int32_t pool_type;
    int32_t filter_height;
    int32_t filter_width;
    int32_t filter_depth;
    int32_t stride_height;
    int32_t stride_width;
    int32_t stride_depth;
    int32_t pad_top;
    int32_t pad_left;
    int32_t pad_down;
    int32_t pad_right;
    int32_t pad_front;
    int32_t pad_back;
};

struct unpooling_params
{
    int (*bc)();
    int32_t layout;
    int32_t scale_height;
    int32_t scale_width;
    int32_t pad_out_height;
    int32_t pad_out_width;
};

struct roi_align_params
{
    int (*bc)();
    int32_t layout;
    int32_t pooled_size_h;
    int32_t pooled_size_w;
    float spatial_scale;
    int32_t spatial_scale_multiplier;
    int32_t spatial_scale_shift;
    int32_t sample_ratio;
};

struct roi_pool_params
{
    int (*bc)();
    int32_t layout;
    int32_t pooled_size_h;
    int32_t pooled_size_w;
    float spatial_scale;
    int32_t spatial_scale_multiplier;
    int32_t spatial_scale_shift;
};

struct siso_params
{
    int (*bc)();
    int32_t layout;
};

struct sigmoid_params
{
    int (*bc)();
    int32_t layout;
};

struct relu_params
{
    int (*bc)();
    int32_t layout;

    /* n / alpha / threshold */
    float n;
    int32_t n_multiplier;
    int32_t n_shift;
};

struct prelu_params
{
    int (*bc)();
    int32_t layout;
    int32_t axis;
};

struct softmax_params
{
    int (*bc)();
    int32_t layout;
    int32_t axis;
};

struct bn_params
{
    int (*bc)();
    int32_t layout;

    float epsilon;
    int32_t epsilon_multiplier;
    int32_t epsilon_shift;
};

struct l2n_params
{
    int (*bc)();
    int32_t layout;

    float epsilon;
    int32_t epsilon_multiplier;
    int32_t epsilon_shift;
    int32_t *axis;
    int32_t n;
};

struct lrn_params
{
    int (*bc)();
    int32_t layout;

    int32_t range;
    double bias;
    int32_t bias_multiplier;
    int32_t bias_shift;
    double alpha;
    int32_t alpha_multiplier;
    int32_t alpha_shift;
    double beta;
    int32_t beta_multiplier;
    int32_t beta_shift;
};

struct matmul_params
{
    int (*bc)();
    int32_t layout;

    bool trans_a;
    bool trans_b;
};

struct diso_params
{
    int (*bc)();
    int32_t layout;
};

struct select_params
{
    int (*bc)();
    int32_t layout;
};

struct pad_params
{
    int (*bc)();
    int32_t layout;

    int32_t *pad_before;
    int32_t *pad_after;
    float pad_value;
    int32_t pad_mode;
};

struct resize_params
{
    int (*bc)();
    int32_t layout;

    int32_t resize_mode;
    bool align_corners;
};

struct concat_params
{
    int (*bc)();
    int32_t layout;

    int32_t inputs_count;
    int32_t axis;
};

struct proposal_params
{
    int (*bc)();
    int32_t layout;

    float *scales;
    int32_t *scale_multipliers;
    int32_t *scale_shifts;
    int32_t scales_num;
    float *ratios;
    int32_t *ratio_multipliers;
    int32_t *ratio_shifts;
    int32_t ratios_num;
    int32_t feature_stride;
    float threshold;
    int32_t threshold_multiplier;
    int32_t threshold_shift;
    int rpn_pre_nms_top_n;
    int rpn_post_nms_top_n;
    int rpn_min_size;
    bool iou_loss;
};

struct psroipooling_params
{
    int (*bc)();
    int32_t layout;

    int32_t output_dim;
    int32_t group_size;
    float spatial_scale;
    int32_t spatial_scale_multiplier;
    int32_t spatial_scale_shift;
};

struct transpose_params
{
    int (*bc)();
    int32_t layout;

    int32_t *permute;
};

struct reshape_params
{
    int (*bc)();
    int32_t layout;
};

struct shape_params
{
    int (*bc)();
    int32_t layout;
};

struct expand_dims_params
{
    int (*bc)();
    int32_t layout;
    int32_t axis;
};

struct reverse_params
{
    int (*bc)();
    int32_t layout;

    int32_t axis;
};

struct flatten_params
{
    int (*bc)();
    int32_t layout;
};

struct crop_params
{
    int (*bc)();
    int32_t layout;

    int32_t axis;
    int32_t *offset;
};

struct slice_params
{
    int (*bc)();
    int32_t layout;

    int32_t *begin;
    int32_t *end;
    int32_t *strides;
};

struct split_params
{
    int (*bc)();
    int32_t layout;

    int32_t *split_index;
    int32_t output_num;
    int32_t axis;
};

struct stack_params
{
    int (*bc)();
    int32_t layout;

    int32_t inputs_count;
    int32_t axis;
};

struct tile_params
{
    int (*bc)();
    int32_t layout;

    int32_t *reps;
    int32_t reps_num;
};

struct arange_params
{
    int (*bc)();
    int32_t layout;

    float start;
    int32_t start_multiplier;
    int32_t start_shift;
    float stop;
    int32_t stop_multiplier;
    int32_t stop_shift;
    float step;
    int32_t step_multiplier;
    int32_t step_shift;
};

struct where_params
{
    int (*bc)();
    int32_t layout;
};

struct unstack_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t outputs_count;
    int32_t axis;
};

struct take_params
{
    int (*bc)();
    int32_t layout;

    int32_t axis;
    const char *mode;
};

struct gather_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t *indices;
    int32_t indices_count;
};
struct gather_nd_params
{
    int (*bc)();
    int32_t layout;
};

struct squeeze_params
{
    int (*bc)();
    int32_t layout;
};

struct ndarray_size_params
{
    int (*bc)();
    int32_t layout;
};

struct space_to_batch_params
{
    int (*bc)();
    int32_t layout;
    int32_t pad_top;
    int32_t pad_bottom;
    int32_t pad_left;
    int32_t pad_right;
    int32_t block_size;
};

struct batch_to_space_params
{
    int (*bc)();
    int32_t layout;
    int32_t crop_top;
    int32_t crop_bottom;
    int32_t crop_left;
    int32_t crop_right;
    int32_t block_size;
};

struct space_to_depth_params
{
    int (*bc)();
    int32_t layout;
    int32_t block_size;
};

struct depth_to_space_params
{
    int (*bc)();
    int32_t layout;
    int32_t block_size;
};

struct one_hot_params
{
    int (*bc)();
    int32_t layout;
    float f_on_value;
    float f_off_value;
    int32_t on_value;
    int32_t off_value;
    int32_t depth;
    int32_t axis;
};

struct sequence_mask_params
{
    int (*bc)();
    int32_t layout;
    float mask_value;
    int32_t mask_value_multiplier;
    int32_t mask_value_shift;
    int32_t axis;
};

struct im2col_params
{
    int (*bc)();
    int32_t layout;
    int32_t pad_h;
    int32_t pad_w;
    int32_t stride_h;
    int32_t stride_w;
};

struct col2im_params
{
    int (*bc)();
    int32_t layout;
    int32_t pad_h;
    int32_t pad_w;
    int32_t stride_h;
    int32_t stride_w;
};

struct reduce_params
{
    int (*bc)();
    int32_t layout;

    int32_t *out_strides;
    int32_t *out_extents;
    int32_t n;
    int32_t *inner_strides;
    int32_t *inner_extents;
    int32_t m;

    int32_t *axis;
    int32_t axis_count;
    bool keepdims;
};

struct reorg_params
{
    int (*bc)();
    int32_t layout;

    int32_t stride;
};

struct segment_params
{
    int (*bc)();
    int32_t layout;
    int32_t num_segments;
    bool unsorted;
};

struct cumsum_params
{
    int (*bc)();
    int32_t layout;
    int32_t axis;
    bool exclusive;
};

struct cumprod_params
{
    int (*bc)();
    int32_t layout;
    int32_t axis;
    bool exclusive;
};

struct broadcast_to_params
{
    int (*bc)();
    int32_t layout;
    int32_t *shape;
    int32_t shape_count;
};

struct clip_params
{
    int (*bc)();
    int32_t api;
    int32_t layout;
    float min_value;
    float max_value;
};

struct strided_slice_params
{
    int (*bc)();
    int32_t api;
    int32_t *begin;
    int32_t *end;
    int32_t *stride;
    int32_t slice_count;
};

#endif
