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

/* CSI-NN2 version 1.10.x */
#ifndef _CSI_DEBUG_H
#define _CSI_DEBUG_H

enum csinn_debug_enum {
    CSI_DEBUG_LEVEL_INFO = -1,
    CSI_DEBUG_LEVEL_WARNING,
    CSI_DEBUG_LEVEL_ERROR,
};

#ifdef CSI_DEBUG
#define CSI_DEBUG_CALL(func) func
void csi_debug_info(const char *format, ...);
void csi_debug_warning(const char *format, ...);
void csi_debug_error(const char *format, ...);
int csi_debug_callback_unset();
#else
#define CSI_DEBUG_CALL(func)
inline void csi_debug_info(const char *format, ...) {}
inline void csi_debug_warning(const char *format, ...) {}
inline void csi_debug_error(const char *format, ...) {}
inline int csi_debug_callback_unset() {return CSINN_CALLBACK_UNSET;}
#endif

int csi_debug_get_level();
void csi_debug_set_level(int level);

int csi_conv2d_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct csi_tensor *kernel,
                          struct csi_tensor *bias,
                          struct conv2d_params *params,
                          const char *name);

int csi_conv3d_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct csi_tensor *kernel,
                          struct csi_tensor *bias,
                          struct conv3d_params *params,
                          const char *name);

int csi_fsmn_debug_info(struct csi_tensor *frame,
                        struct csi_tensor *l_filter,
                        struct csi_tensor *r_filter,
                        struct csi_tensor *frame_sequence,
                        struct csi_tensor *frame_counter,
                        struct csi_tensor *output,
                        struct fsmn_params *params,
                        const char *name);

int csi_siso_debug_info(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct siso_params *params,
                        const char *name);

int csi_diso_debug_info(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct diso_params *params,
                        const char *name);

int csi_relu_debug_info(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct relu_params *params,
                        const char *name);

int csi_arange_debug_info(struct csi_tensor *output,
                          struct arange_params *params,
                          const char *name);

int csi_pool_debug_info(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct pool_params *params,
                        const char *name);

int csi_pad_debug_info(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct pad_params *params,
                        const char *name);

int csi_crop_debug_info(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct crop_params *params,
                        const char *name);

int csi_roi_pool_debug_info(struct csi_tensor *data,
                            struct csi_tensor *rois,
                            struct csi_tensor *output,
                            struct roi_pool_params *params,
                            const char *name);

int csi_bn_debug_info(struct csi_tensor *input,
                      struct csi_tensor *mean,
                      struct csi_tensor *variance,
                      struct csi_tensor *gamma,
                      struct csi_tensor *beta,
                      struct csi_tensor *output,
                      struct bn_params *params,
                      const char *name);

int csi_batch_to_space_debug_info(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct batch_to_space_params *params,
                                  const char *name);

int csi_batch_to_space_nd_debug_info(struct csi_tensor *input,
                                     struct csi_tensor *output,
                                     struct batch_to_space_nd_params *params,
                                     const char *name);

int csi_depth_to_space_debug_info(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct depth_to_space_params *params,
                                  const char *name);

int csi_space_to_depth_debug_info(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct space_to_depth_params *params,
                                  const char *name);

int csi_space_to_batch_debug_info(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct space_to_batch_params *params,
                                  const char *name);

int csi_space_to_batch_nd_debug_info(struct csi_tensor *input,
                                     struct csi_tensor *output,
                                     struct space_to_batch_nd_params *params,
                                     const char *name);

int csi_broadcast_to_debug_info(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct broadcast_to_params *params,
                                const char *name);

int csi_reduce_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct reduce_params *params,
                          const char *name);

int csi_clip_debug_info(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct clip_params *params,
                        const char *name);

int csi_col2im_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct col2im_params *params,
                          const char *name);

int csi_concat_debug_info(struct csi_tensor **input,
                          struct csi_tensor *output,
                          struct concat_params *params,
                          const char *name);

int csi_cumprod_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct cumprod_params *params,
                           const char *name);

int csi_cumsum_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct cumsum_params *params,
                          const char *name);

int csi_expand_dims_debug_info(struct csi_tensor *input,
                               struct csi_tensor *output,
                               struct expand_dims_params *params,
                               const char *name);

int csi_flatten_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct flatten_params *params,
                           const char *name);

int csi_fullyconnected_debug_info(struct csi_tensor *input,
                                  struct csi_tensor *output,
                                  struct csi_tensor *weights,
                                  struct csi_tensor *bias,
                                  struct fc_params *params,
                                  const char *name);

int csi_gather_nd_debug_info(struct csi_tensor *input,
                             struct csi_tensor *indices,
                             struct csi_tensor *output,
                             struct gather_nd_params *params,
                             const char *name);

int csi_gather_debug_info(struct csi_tensor *input,
                          struct csi_tensor *indices,
                          struct csi_tensor *output,
                          struct gather_params *params,
                          const char *name);

int csi_hard_sigmoid_debug_info(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct sigmoid_params *params,
                                const char *name);

int csi_im2col_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct im2col_params *params,
                          const char *name);

int csi_l2n_debug_info(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct l2n_params *params,
                       const char *name);

int csi_softmax_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct softmax_params *params,
                           const char *name);

int csi_lrn_debug_info(struct csi_tensor *input,
                       struct csi_tensor *output,
                       struct lrn_params *params,
                       const char *name);

int csi_matmul_debug_info(struct csi_tensor *mat0,
                          struct csi_tensor *mat1,
                          struct csi_tensor *output,
                          struct matmul_params *params,
                          const char *name);

int csi_ndarray_size_debug_info(struct csi_tensor *input,
                                struct csi_tensor *output,
                                struct ndarray_size_params *params,
                                const char *name);

int csi_nms_debug_info(struct csi_tensor *input0,
                       struct csi_tensor *input1,
                       struct csi_tensor *output,
                       struct non_max_suppression_params *params,
                       const char *name);

int csi_one_hot_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct one_hot_params *params,
                           const char *name);

int csi_prelu_debug_info(struct csi_tensor *input0,
                         struct csi_tensor *input1,
                         struct csi_tensor *output,
                         struct prelu_params *params,
                         const char *name);

int csi_proposal_debug_info(struct csi_tensor *cls_prob,
                            struct csi_tensor *bbox_pred,
                            struct csi_tensor *im_info,
                            struct csi_tensor *output,
                            struct proposal_params *params,
                            const char *name);

int csi_psroipooling_debug_info(struct csi_tensor *data,
                                struct csi_tensor *rois,
                                struct csi_tensor *output,
                                struct psroipooling_params *params,
                                const char *name);

int csi_reorg_debug_info(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct reorg_params *params,
                         const char *name);

int csi_reshape_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct reshape_params *params,
                           const char *name);

int csi_resize_debug_info(struct csi_tensor *input,
                          struct csi_tensor *output,
                          struct resize_params *params,
                          const char *name);

int csi_reverse_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct reverse_params *params,
                           const char *name);

int csi_roi_align_debug_info(struct csi_tensor *data,
                             struct csi_tensor *rois,
                             struct csi_tensor *output,
                             struct roi_align_params *params,
                             const char *name);

int csi_scatter_nd_debug_info(struct csi_tensor *input,
                              struct csi_tensor *indices,
                              struct csi_tensor *updates,
                              struct csi_tensor *output,
                              struct scatter_nd_params *params,
                              const char *name);

int csi_segment_debug_info(struct csi_tensor *input0,
                           struct csi_tensor *input1,
                           struct csi_tensor *output,
                           struct segment_params *params,
                           const char *name);

int csi_select_debug_info(struct csi_tensor *condition,
                          struct csi_tensor *input0,
                          struct csi_tensor *input1,
                          struct csi_tensor *output,
                          struct select_params *params,
                          const char *name);

int csi_sequence_mask_debug_info(struct csi_tensor *input0,
                                 struct csi_tensor *input1,
                                 struct csi_tensor *output,
                                 struct sequence_mask_params *params,
                                 const char *name);

int csi_shape_debug_info(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct shape_params *params,
                         const char *name);

int csi_shuffle_channel_debug_info(struct csi_tensor *input,
                                   struct csi_tensor *output,
                                   struct shuffle_channel_params *params,
                                   const char *name);

int csi_sigmoid_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct sigmoid_params *params,
                           const char *name);

int csi_slice_debug_info(struct csi_tensor *input,
                         struct csi_tensor *output,
                         struct slice_params *params,
                         const char *name);

int csi_split_debug_info(struct csi_tensor *input,
                         struct csi_tensor **output,
                         struct split_params *params,
                         const char *name);

int csi_squeeze_debug_info(struct csi_tensor *input,
                           struct csi_tensor *output,
                           struct squeeze_params *params,
                           const char *name);

int csi_stack_debug_info(struct csi_tensor **input,
                         struct csi_tensor *output,
                         struct stack_params *params,
                         const char *name);

int csi_strided_slice_debug_info(struct csi_tensor *input,
                                 struct csi_tensor *output,
                                 struct strided_slice_params *params,
                                 const char *name);

int csi_tile_debug_info(struct csi_tensor *input,
                        struct csi_tensor *output,
                        struct tile_params *params,
                        const char *name);

int csi_topk_debug_info(struct csi_tensor *input0,
                        struct csi_tensor *input1,
                        struct csi_tensor *output,
                        struct topk_params *params,
                        const char *name);

int csi_transpose_debug_info(struct csi_tensor *input,
                             struct csi_tensor *output,
                             struct transpose_params *params,
                             const char *name);

int csi_unpooling_debug_info(struct csi_tensor *input,
                             struct csi_tensor *mask,
                             struct csi_tensor *output,
                             struct unpooling_params *params,
                             const char *name);

int csi_unstack_debug_info(struct csi_tensor *input,
                           struct csi_tensor **output,
                           struct unstack_params *params,
                           const char *name);

int csi_where_debug_info(struct csi_tensor *condition,
                         struct csi_tensor *x,
                         struct csi_tensor *y,
                         struct csi_tensor *output,
                         struct where_params *params,
                         const char *name);

#endif
