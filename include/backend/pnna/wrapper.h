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

#ifndef INCLUDE_SHL_PNNA_WRAPPER_H_
#define INCLUDE_SHL_PNNA_WRAPPER_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
int shl_pnna_session_init_internal(struct shl_pnna_target_data *td);
int shl_pnna_session_deinit_internal(struct shl_pnna_target_data *td);
int shl_pnna_session_setup_internal(struct shl_pnna_target_data *td);
int shl_pnna_session_create_network_binary(struct csinn_session *sess,
                                           struct shl_pnna_target_data *td);
int shl_pnna_session_run_internal(struct csinn_session *sess, int input_num, int output_num);
void shl_pnna_load_binary_model_internal(void *addr, size_t size, struct shl_pnna_target_data *td);
int shl_pnna_create_tensor_internal(struct csinn_tensor *t, struct shl_pnna_target_data *td);
int shl_pnna_set_output_internal(int index, struct csinn_tensor *t,
                                 struct shl_pnna_target_data *td);
int shl_pnna_update_input_internal(int index, void *buffer, struct csinn_session *sess);
int shl_pnna_update_output_internal(int index, void *buffer, struct csinn_session *sess);
int shl_pnna_get_output_internal(int index, struct csinn_tensor *output,
                                 struct shl_pnna_target_data *td);
void shl_pnna_set_input_strides_internal(struct shl_pnna_target_data *td, int byte_size,
                                         int input_fix_h, int input_fix_w);
int shl_pnna_create_io_memory(struct csinn_session *sess);

/* internal op */
int shl_pnna_create_argmax_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_reduce_params *params,
                                    struct shl_pnna_target_data *td);
int shl_pnna_create_avgpool_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_pool_params *params,
                                     struct shl_pnna_target_data *td);
int shl_pnna_create_batch_to_space_nd_internal(struct csinn_tensor *input,
                                               struct csinn_tensor *output,
                                               struct csinn_batch_to_space_nd_params *params,
                                               struct shl_pnna_target_data *td);
int shl_pnna_create_concat_internal(struct csinn_tensor **input, struct csinn_tensor *output,
                                    struct csinn_concat_params *params,
                                    struct shl_pnna_target_data *td);
int shl_pnna_create_conv2d_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                    struct csinn_conv2d_params *params,
                                    struct shl_pnna_target_data *td);
int shl_pnna_create_deconv2d_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                      struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                      struct csinn_conv2d_params *params,
                                      struct shl_pnna_target_data *td);
int shl_pnna_create_dense_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                   struct csinn_fc_params *params, struct shl_pnna_target_data *td);
int shl_pnna_create_depth_to_space_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_depth_to_space_params *params,
                                            struct shl_pnna_target_data *td);
int shl_pnna_create_depthwise_conv2d_internal(
    struct csinn_tensor *input, struct csinn_tensor *output, struct csinn_tensor *kernel,
    struct csinn_tensor *bias, struct csinn_conv2d_params *params, struct shl_pnna_target_data *td);
int shl_pnna_create_diso_internal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                  struct csinn_tensor *output, int op,
                                  struct shl_pnna_target_data *td);
int shl_pnna_create_flatten_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_flatten_params *params,
                                     struct shl_pnna_target_data *td);
int shl_pnna_create_group_conv2d_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                          struct csinn_conv2d_params *params,
                                          struct shl_pnna_target_data *td);
int shl_pnna_create_global_avgpool_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_pool_params *params,
                                            struct shl_pnna_target_data *td);
int shl_pnna_create_global_maxpool_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_pool_params *params,
                                            struct shl_pnna_target_data *td);
int shl_pnna_create_leaky_relu_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                        struct csinn_relu_params *params,
                                        struct shl_pnna_target_data *td);
int shl_pnna_create_lrn_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_lrn_params *params, struct shl_pnna_target_data *td);
int shl_pnna_create_mean_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                  struct csinn_reduce_params *params,
                                  struct shl_pnna_target_data *td);
int shl_pnna_create_maxpool_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_pool_params *params,
                                     struct shl_pnna_target_data *td);
int shl_pnna_create_maxpool2d_locat_internal(struct csinn_tensor *data, struct csinn_tensor *output,
                                             struct csinn_pool_params *params,
                                             struct shl_pnna_target_data *td);
int shl_pnna_create_pad_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_pad_params *params, struct shl_pnna_target_data *td);
int shl_pnna_create_prelu_internal(struct csinn_tensor *input, struct csinn_tensor *alpha,
                                   struct csinn_tensor *output, struct csinn_prelu_params *params,
                                   struct shl_pnna_target_data *td);
int shl_pnna_create_proposal_internal(struct csinn_tensor *cls_prob, struct csinn_tensor *bbox_pred,
                                      struct csinn_tensor *im_info, struct csinn_tensor *output,
                                      struct csinn_proposal_params *params,
                                      struct shl_pnna_target_data *td);
int shl_pnna_create_relu1_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_relu_params *params,
                                   struct shl_pnna_target_data *td);
int shl_pnna_create_relu6_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                   struct csinn_relu_params *params,
                                   struct shl_pnna_target_data *td);
int shl_pnna_create_reshape_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_reshape_params *params,
                                     struct shl_pnna_target_data *td);
int shl_pnna_create_resize_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_resize_params *params,
                                    struct shl_pnna_target_data *td);
int shl_pnna_create_roipool_internal(struct csinn_tensor *data, struct csinn_tensor *rois,
                                     struct csinn_tensor *output,
                                     struct csinn_roi_pool_params *params,
                                     struct shl_pnna_target_data *td);
int shl_pnna_create_siso_internal(struct csinn_tensor *input, struct csinn_tensor *output, int op,
                                  struct shl_pnna_target_data *td);
int shl_pnna_create_softmax_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_softmax_params *params,
                                     struct shl_pnna_target_data *td);
int shl_pnna_create_space_to_depth_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_space_to_depth_params *params,
                                            struct shl_pnna_target_data *td);
int shl_pnna_create_space_to_batch_nd_internal(struct csinn_tensor *input,
                                               struct csinn_tensor *output,
                                               struct csinn_space_to_batch_nd_params *params,
                                               struct shl_pnna_target_data *td);
int shl_pnna_create_split_internal(struct csinn_tensor *input, struct csinn_tensor **output,
                                   struct csinn_split_params *params,
                                   struct shl_pnna_target_data *td);
int shl_pnna_create_squeeze_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                     struct csinn_squeeze_params *params,
                                     struct shl_pnna_target_data *td);
int shl_pnna_create_strided_slice_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                           struct csinn_strided_slice_params *params,
                                           struct shl_pnna_target_data *td);
int shl_pnna_create_cus_strided_slice_internal(struct csinn_tensor *input,
                                               struct csinn_tensor *output,
                                               struct csinn_strided_slice_params *params,
                                               struct shl_pnna_target_data *td);
int shl_pnna_create_transpose_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_transpose_params *params,
                                       struct shl_pnna_target_data *td);
int shl_pnna_create_unpooling_internal(struct csinn_tensor *input, struct csinn_tensor *mask,
                                       struct csinn_tensor *output,
                                       struct csinn_unpooling_params *params,
                                       struct shl_pnna_target_data *td);
int shl_pnna_create_matmul_internal(struct csinn_tensor *input0, struct csinn_tensor *input1,
                                    struct csinn_tensor *output, struct csinn_matmul_params *params,
                                    struct shl_pnna_target_data *td);
int shl_pnna_create_data_convert_internal(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_siso_params *params,
                                          struct shl_pnna_target_data *td);
#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SHL_PNNA_WRAPPER_H_
