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

#ifndef _CSI_NN_PNNA_WRAPPER_H
#define _CSI_NN_PNNA_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif
int csi_pnna_session_init_internal(struct csi_pnna_target_data *td);
int csi_pnna_session_deinit_internal(struct csi_pnna_target_data *td);
int csi_pnna_session_setup_internal(struct csi_pnna_target_data *td);
int csi_pnna_session_create_network_binary(char *path, struct csi_pnna_target_data *td);
int csi_pnna_session_run_internal(struct csi_session *sess, int input_num, int output_num);
void csi_pnna_load_binary_model_internal(char *path, struct csi_pnna_target_data *td);
int csi_pnna_create_tensor_internal(struct csi_tensor *t, struct csi_pnna_target_data *td);
int csi_pnna_set_output_internal(int index, struct csi_tensor *t, struct csi_pnna_target_data *td);
int csi_pnna_get_output_internal(int index, struct csi_tensor *output,
                                 struct csi_pnna_target_data *td);
void csi_pnna_set_input_strides_internal(struct csi_pnna_target_data *td, int byte_size,
                                         int input_fix_h, int input_fix_w);

/* internal op */
int csi_pnna_create_argmax_internal(struct csi_tensor *input, struct csi_tensor *output,
                                    struct reduce_params *params, struct csi_pnna_target_data *td);
int csi_pnna_create_avgpool_internal(struct csi_tensor *input, struct csi_tensor *output,
                                     struct pool_params *params, struct csi_pnna_target_data *td);
int csi_pnna_create_batch_to_space_nd_internal(struct csi_tensor *input, struct csi_tensor *output,
                                               struct batch_to_space_nd_params *params,
                                               struct csi_pnna_target_data *td);
int csi_pnna_create_concat_internal(struct csi_tensor **input, struct csi_tensor *output,
                                    struct concat_params *params, struct csi_pnna_target_data *td);
int csi_pnna_create_conv2d_internal(struct csi_tensor *input, struct csi_tensor *output,
                                    struct csi_tensor *kernel, struct csi_tensor *bias,
                                    struct conv2d_params *params, struct csi_pnna_target_data *td);
int csi_pnna_create_deconv2d_internal(struct csi_tensor *input, struct csi_tensor *output,
                                      struct csi_tensor *kernel, struct csi_tensor *bias,
                                      struct conv2d_params *params,
                                      struct csi_pnna_target_data *td);
int csi_pnna_create_dense_internal(struct csi_tensor *input, struct csi_tensor *output,
                                   struct csi_tensor *kernel, struct csi_tensor *bias,
                                   struct fc_params *params, struct csi_pnna_target_data *td);
int csi_pnna_create_depth_to_space_internal(struct csi_tensor *input, struct csi_tensor *output,
                                            struct depth_to_space_params *params,
                                            struct csi_pnna_target_data *td);
int csi_pnna_create_depthwise_conv2d_internal(struct csi_tensor *input, struct csi_tensor *output,
                                              struct csi_tensor *kernel, struct csi_tensor *bias,
                                              struct conv2d_params *params,
                                              struct csi_pnna_target_data *td);
int csi_pnna_create_diso_internal(struct csi_tensor *input0, struct csi_tensor *input1,
                                  struct csi_tensor *output, int op,
                                  struct csi_pnna_target_data *td);
int csi_pnna_create_flatten_internal(struct csi_tensor *input, struct csi_tensor *output,
                                     struct flatten_params *params,
                                     struct csi_pnna_target_data *td);
int csi_pnna_create_group_conv2d_internal(struct csi_tensor *input, struct csi_tensor *output,
                                          struct csi_tensor *kernel, struct csi_tensor *bias,
                                          struct conv2d_params *params,
                                          struct csi_pnna_target_data *td);
int csi_pnna_create_global_avgpool_internal(struct csi_tensor *input, struct csi_tensor *output,
                                            struct pool_params *params,
                                            struct csi_pnna_target_data *td);
int csi_pnna_create_global_maxpool_internal(struct csi_tensor *input, struct csi_tensor *output,
                                            struct pool_params *params,
                                            struct csi_pnna_target_data *td);
int csi_pnna_create_leaky_relu_internal(struct csi_tensor *input, struct csi_tensor *output,
                                        struct relu_params *params,
                                        struct csi_pnna_target_data *td);
int csi_pnna_create_lrn_internal(struct csi_tensor *input, struct csi_tensor *output,
                                 struct lrn_params *params, struct csi_pnna_target_data *td);
int csi_pnna_create_mean_internal(struct csi_tensor *input, struct csi_tensor *output,
                                  struct reduce_params *params, struct csi_pnna_target_data *td);
int csi_pnna_create_maxpool_internal(struct csi_tensor *input, struct csi_tensor *output,
                                     struct pool_params *params, struct csi_pnna_target_data *td);
int csi_pnna_create_maxpool2d_locat_internal(struct csi_tensor *data, struct csi_tensor *output,
                                             struct pool_params *params,
                                             struct csi_pnna_target_data *td);
int csi_pnna_create_pad_internal(struct csi_tensor *input, struct csi_tensor *output,
                                 struct pad_params *params, struct csi_pnna_target_data *td);
int csi_pnna_create_prelu_internal(struct csi_tensor *input, struct csi_tensor *alpha,
                                   struct csi_tensor *output, struct prelu_params *params,
                                   struct csi_pnna_target_data *td);
int csi_pnna_create_proposal_internal(struct csi_tensor *cls_prob, struct csi_tensor *bbox_pred,
                                      struct csi_tensor *im_info, struct csi_tensor *output,
                                      struct proposal_params *params,
                                      struct csi_pnna_target_data *td);
int csi_pnna_create_relu1_internal(struct csi_tensor *input, struct csi_tensor *output,
                                   struct relu_params *params, struct csi_pnna_target_data *td);
int csi_pnna_create_relu6_internal(struct csi_tensor *input, struct csi_tensor *output,
                                   struct relu_params *params, struct csi_pnna_target_data *td);
int csi_pnna_create_reshape_internal(struct csi_tensor *input, struct csi_tensor *output,
                                     struct reshape_params *params,
                                     struct csi_pnna_target_data *td);
int csi_pnna_create_resize_internal(struct csi_tensor *input, struct csi_tensor *output,
                                    struct resize_params *params, struct csi_pnna_target_data *td);
int csi_pnna_create_roipool_internal(struct csi_tensor *data, struct csi_tensor *rois,
                                     struct csi_tensor *output, struct roi_pool_params *params,
                                     struct csi_pnna_target_data *td);
int csi_pnna_create_siso_internal(struct csi_tensor *input, struct csi_tensor *output, int op,
                                  struct csi_pnna_target_data *td);
int csi_pnna_create_softmax_internal(struct csi_tensor *input, struct csi_tensor *output,
                                     struct softmax_params *params,
                                     struct csi_pnna_target_data *td);
int csi_pnna_create_space_to_depth_internal(struct csi_tensor *input, struct csi_tensor *output,
                                            struct space_to_depth_params *params,
                                            struct csi_pnna_target_data *td);
int csi_pnna_create_space_to_batch_nd_internal(struct csi_tensor *input, struct csi_tensor *output,
                                               struct space_to_batch_nd_params *params,
                                               struct csi_pnna_target_data *td);
int csi_pnna_create_split_internal(struct csi_tensor *input, struct csi_tensor **output,
                                   struct split_params *params, struct csi_pnna_target_data *td);
int csi_pnna_create_squeeze_internal(struct csi_tensor *input, struct csi_tensor *output,
                                     struct squeeze_params *params,
                                     struct csi_pnna_target_data *td);
int csi_pnna_create_strided_slice_internal(struct csi_tensor *input, struct csi_tensor *output,
                                           struct strided_slice_params *params,
                                           struct csi_pnna_target_data *td);
int csi_pnna_create_cus_strided_slice_internal(struct csi_tensor *input, struct csi_tensor *output,
                                               struct strided_slice_params *params,
                                               struct csi_pnna_target_data *td);
int csi_pnna_create_transpose_internal(struct csi_tensor *input, struct csi_tensor *output,
                                       struct transpose_params *params,
                                       struct csi_pnna_target_data *td);
int csi_pnna_create_unpooling_internal(struct csi_tensor *input, struct csi_tensor *mask,
                                       struct csi_tensor *output, struct unpooling_params *params,
                                       struct csi_pnna_target_data *td);
#ifdef __cplusplus
}
#endif

#endif
