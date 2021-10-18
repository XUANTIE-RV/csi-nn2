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
    CSINN_DTYPE_UINT8 = 0,
    CSINN_DTYPE_INT8,
    CSINN_DTYPE_UINT16,
    CSINN_DTYPE_INT16,
    CSINN_DTYPE_UINT32,
    CSINN_DTYPE_INT32,
    CSINN_DTYPE_FLOAT16,
    CSINN_DTYPE_FLOAT32,
    CSINN_DTYPE_FLOAT64,
    CSINN_DTYPE_SIZE,
};

/* API type */
enum
{
    CSINN_REF = 0,
    CSINN_C860,
    CSINN_C906,
    CSINN_C910,
    CSINN_ANOLE,
    CSINN_TX510,
    CSINN_LIGHT,
    CSINN_TVMGEN,
    CSINN_API_SIZE,
};

/* op and utils */
enum
{
    CSINN_OP_ABS = 0,
    CSINN_OP_ACOS,
    CSINN_OP_ACOSH,
    CSINN_OP_ADD,
    CSINN_OP_ALL,
    CSINN_OP_AND,
    CSINN_OP_ANY,
    CSINN_OP_ARANGE,
    CSINN_OP_ARGMAX,
    CSINN_OP_ARGMIN,
    CSINN_OP_ASIN,
    CSINN_OP_ASINH,
    CSINN_OP_ATAN,
    CSINN_OP_ATANH,
    CSINN_OP_AVGPOOL2D,
    CSINN_OP_AVGPOOL3D,
    CSINN_OP_BN,
    CSINN_OP_BATCH_TO_SPACE,
    CSINN_OP_BROADCOST,
    CSINN_OP_CEIL,
    CSINN_OP_CLIP,
    CSINN_OP_COL2IM,
    CSINN_OP_CONCAT,
    CSINN_OP_CONV2D,
    CSINN_OP_CONV2D_RELU,
    CSINN_OP_CONV2D_RELU6,
    CSINN_OP_CONV2D_CHANNEL,
    CSINN_OP_CONV2D_CHANNEL_RELU,
    CSINN_OP_CONV2D_CHANNEL_RELU6,
    CSINN_OP_DEPTHWISE_CONV2D,
    CSINN_OP_DEPTHWISE_CONV2D_RELU,
    CSINN_OP_DEPTHWISE_CONV2D_RELU6,
    CSINN_OP_DEPTHWISE_CONV2D_CHANNEL,
    CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU,
    CSINN_OP_DEPTHWISE_CONV2D_CHANNEL_RELU6,
    CSINN_OP_GROUP_CONV2D,
    CSINN_OP_GROUP_CONV2D_RELU,
    CSINN_OP_GROUP_CONV2D_CHANNEL,
    CSINN_OP_GROUP_CONV2D_CHANNEL_RELU,
    CSINN_OP_CONV3D,
    CSINN_OP_COS,
    CSINN_OP_COSH,
    CSINN_OP_CUMPROD,
    CSINN_OP_CUMSUM,
    CSINN_OP_DECONV2D,
    CSINN_OP_DEPTHWISE_DECONV2D,
    CSINN_OP_DECONV3D,
    CSINN_OP_DEPTH_TO_SPACE,
    CSINN_OP_DIV,
    CSINN_OP_ELU,
    CSINN_OP_EQUANL,
    CSINN_OP_ERF,
    CSINN_OP_EXP,
    CSINN_OP_EXPAND_DIMS,
    CSINN_OP_EXPM1,
    CSINN_OP_FLATTEN,
    CSINN_OP_FLOOR_DIVIDE,
    CSINN_OP_FLOOR_MOD,
    CSINN_OP_FLOOR,
    CSINN_OP_FULLYCONNECTED,
    CSINN_OP_GATHER_ND,
    CSINN_OP_GATHER,
    CSINN_OP_GLOBAL_AVGPOOL2D,
    CSINN_OP_GLOBAL_MAXPOOL2D,
    CSINN_OP_GREATHER_EQUAL,
    CSINN_OP_GREATHER,
    CSINN_OP_HARD_SIGMOID,
    CSINN_OP_IM2COL,
    CSINN_OP_ISNAN,
    CSINN_OP_L2N,
    CSINN_OP_L2POOL2D,
    CSINN_OP_LEAKY_RELU,
    CSINN_OP_LESS_EQUAL,
    CSINN_OP_LESS,
    CSINN_OP_LOG_SOFTMAX,
    CSINN_OP_LOG,
    CSINN_OP_LOG1P,
    CSINN_OP_LOGICAL_AND,
    CSINN_OP_LOGICAL_NOT,
    CSINN_OP_LOGICAL_OR,
    CSINN_OP_LOGICAL_XOR,
    CSINN_OP_LRN,
    CSINN_OP_MATMUL,
    CSINN_OP_MAX,
    CSINN_OP_MAXINUM,
    CSINN_OP_MAXPOOL2D,
    CSINN_OP_MAXPOOL2D_LOCAT,
    CSINN_OP_MAXPOOL3D,
    CSINN_OP_MEAN,
    CSINN_OP_MEAN_STRIDE,
    CSINN_OP_MIN,
    CSINN_OP_MIN_STRIDE,
    CSINN_OP_MINIMUM,
    CSINN_OP_MOD,
    CSINN_OP_MUL,
    CSINN_OP_NDARRAY_SIZE,
    CSINN_OP_NEGATIIVE,
    CSINN_OP_NON_MAX_SUPPRESSION,
    CSINN_OP_NOT_EQUAL,
    CSINN_OP_NOT,
    CSINN_OP_ONE_HOT,
    CSINN_OP_OR,
    CSINN_OP_PAD,
    CSINN_OP_POWER,
    CSINN_OP_PRELU,
    CSINN_OP_PROD,
    CSINN_OP_PROPOSAL,
    CSINN_OP_PSROIPOOLING,
    CSINN_OP_REDUCE_LOGSUMEXP,
    CSINN_OP_REDUCE_MAX,
    CSINN_OP_REDUCE_MEAN,
    CSINN_OP_REDUCE_MIN,
    CSINN_OP_REDUCE_PROD,
    CSINN_OP_REDUCE_SUM,
    CSINN_OP_RELU,
    CSINN_OP_RELU1,
    CSINN_OP_RELU6,
    CSINN_OP_RELUN,
    CSINN_OP_REORG,
    CSINN_OP_RESHAPE,
    CSINN_OP_RESIZE,
    CSINN_OP_REVERSE,
    CSINN_OP_ROIALIGN,
    CSINN_OP_ROIPOOL,
    CSINN_OP_ROUND,
    CSINN_OP_RSQRT,
    CSINN_OP_SEGMENT_MAX,
    CSINN_OP_UNSORTED_SEGMENT_MAX,
    CSINN_OP_SEGMENT_MEAN,
    CSINN_OP_UNSORTED_SEGMENT_MEAN,
    CSINN_OP_SEGMENT_MIN,
    CSINN_OP_UNSORTED_SEGMENT_MIN,
    CSINN_OP_SEGMENT_PROD,
    CSINN_OP_UNSORTED_SEGMENT_PROD,
    CSINN_OP_SEGMENT_SUM,
    CSINN_OP_UNSORTED_SEGMENT_SUM,
    CSINN_OP_SELECT,
    CSINN_OP_SEQUENCE_MASK,
    CSINN_OP_SHAPE,
    CSINN_OP_SHUFFLE_CHANNEL,
    CSINN_OP_SIGMOID,
    CSINN_OP_SIGN,
    CSINN_OP_SIN,
    CSINN_OP_SINH,
    CSINN_OP_SLICE,
    CSINN_OP_SOFTMAX,
    CSINN_OP_SOFTPLUS,
    CSINN_OP_SOFTRELU,
    CSINN_OP_SOFTSIGN,
    CSINN_OP_SPACE_TO_BATCH,
    CSINN_OP_SPACE_TO_DEPTH,
    CSINN_OP_SPLIT,
    CSINN_OP_SQRT,
    CSINN_OP_SQUARE,
    CSINN_OP_SQUEEZE,
    CSINN_OP_STACK,
    CSINN_OP_STRIDED_SLICE,
    CSINN_OP_SUB,
    CSINN_OP_SUM,
    CSINN_OP_TAN,
    CSINN_OP_TANH,
    CSINN_OP_THRESHOLD_RELU,
    CSINN_OP_TILE,
    CSINN_OP_TOPK,
    CSINN_OP_TRANSPOSE,
    CSINN_OP_TRUNC,
    CSINN_OP_UNPOOLING,
    CSINN_OP_UNSTACK,
    CSINN_OP_WHERE,
    CSINN_OP_XOR,
    CSINN_OP_YUV_RGB_SCALE,

    /* utils functions */
    CSINN_SESSION_INIT,
    CSINN_SESSION_DEINIT,
    CSINN_SESSION_SETUP,
    CSINN_SESSION_RUN,
    CSINN_UPDATE_INPUT,
    CSINN_SET_INPUT_NUMBER,
    CSINN_SET_OUTPUT_NUMBER,
    CSINN_GET_INPUT_NUMBER,
    CSINN_GET_OUTPUT_NUMBER,
    CSINN_SET_INPUT,
    CSINN_SET_OUTPUT,
    CSINN_GET_INPUT,
    CSINN_GET_OUTPUT,
    CSINN_OP_SIZE,
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
    char *name;
    int32_t zero_point;
    float scale;
    int32_t multiplier;
    int32_t shift;
    int32_t layout;
    float min;
    float max;
    struct csi_session *sess;
} __attribute__((packed));

struct conv2d_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t group;
    int32_t stride_height;
    int32_t stride_width;
    int32_t pad_top;
    int32_t pad_left;
    int32_t pad_down;
    int32_t pad_right;
    int32_t dilation_height;
    int32_t dilation_width;
    char *name;
    float *wscales;
    int32_t *wzps;
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
    int32_t api;
};

struct pool_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
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
    int32_t api;
    int32_t scale_height;
    int32_t scale_width;
    int32_t pad_out_height;
    int32_t pad_out_width;
};

struct roi_align_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
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
    int32_t api;
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
    int32_t api;
};

struct sigmoid_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
};

struct relu_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    char *name;

    /* n / alpha / threshold */
    float n;
    int32_t n_multiplier;
    int32_t n_shift;
};

struct prelu_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t axis;
};

struct softmax_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t axis;
};

struct bn_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    float epsilon;
    int32_t epsilon_multiplier;
    int32_t epsilon_shift;
};

struct l2n_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
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
    int32_t api;
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
    int32_t api;
    bool trans_a;
    bool trans_b;
};

struct diso_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
};

struct select_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
};

struct pad_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t *pad_before;
    int32_t *pad_after;
    float pad_value;
    int32_t pad_mode;
};

struct resize_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t resize_mode;
    bool align_corners;
};

struct concat_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t inputs_count;
    int32_t axis;
};

struct proposal_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
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
    int32_t api;
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
    int32_t api;
    int32_t *permute;
};

struct reshape_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
};

struct shape_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
};

struct expand_dims_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t axis;
};

struct reverse_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t axis;
};

struct flatten_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
};

struct crop_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t axis;
    int32_t *offset;
};

struct slice_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t *begin;
    int32_t *end;
    int32_t *strides;
};

struct split_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t *split_index;
    int32_t output_num;
    int32_t axis;
};

struct stack_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t inputs_count;
    int32_t axis;
};

struct tile_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t *reps;
    int32_t reps_num;
};

struct arange_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
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
    int32_t api;
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
    int32_t api;
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
    int32_t api;
};

struct squeeze_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
};

struct ndarray_size_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
};

struct space_to_batch_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
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
    int32_t api;
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
    int32_t api;
    int32_t block_size;
};

struct depth_to_space_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t block_size;
};

struct one_hot_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
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
    int32_t api;
    float mask_value;
    int32_t mask_value_multiplier;
    int32_t mask_value_shift;
    int32_t axis;
};

struct im2col_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t pad_top;
    int32_t pad_down;
    int32_t pad_left;
    int32_t pad_right;
    int32_t stride_h;
    int32_t stride_w;
    int32_t kernel_h;
    int32_t kernel_w;
};

struct col2im_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t pad_h;
    int32_t pad_w;
    int32_t stride_h;
    int32_t stride_w;
};

struct reduce_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
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
    int32_t api;
    int32_t stride;
};

struct segment_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t num_segments;
    bool unsorted;
};

struct cumsum_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t axis;
    bool exclusive;
};

struct cumprod_params
{
    int (*bc)();
    int32_t layout;
    int32_t api;
    int32_t axis;
    bool exclusive;
};

struct broadcast_to_params
{
    int (*bc)();
    int32_t api;
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

struct shuffle_channel_params
{
    int (*bc)();
    int32_t api;
    int32_t layout;
    int32_t group;
};

struct topk_params
{
    int (*bc)();
    int32_t api;
    int32_t layout;
    int32_t k;
};

struct non_max_suppression_params
{
    int (*bc)();
    int32_t api;
    int32_t layout;
    int32_t max_output_size;
    float iou_threshold;
    // float score_threshold;
};

#endif
