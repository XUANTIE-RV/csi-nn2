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

#ifndef _CSI_NN_PNNA_WRAPPER_H
#define _CSI_NN_PNNA_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif
int csi_pnna_session_init_internal(struct csi_pnna_target_data *td);
int csi_pnna_session_deinit_internal(struct csi_pnna_target_data *td);
int csi_pnna_session_setup_internal(struct csi_pnna_target_data *td);
int csi_pnna_session_run_internal(struct csi_session *sess, int input_num, int output_num);
void csi_pnna_load_binary_model_internal(char *path, struct csi_pnna_target_data *td);
int csi_pnna_create_val(struct csi_tensor *t);
int csi_pnna_create_const_val(struct csi_tensor *t);
int csi_pnna_create_conv2d(struct conv2d_params *params, struct csi_tensor *kernel,
                           struct csi_tensor *output, int channel, bool use_bias, bool depthwise);
int csi_pnna_create_deconv2d(struct conv2d_params *params, struct csi_tensor *kernel,
                           struct csi_tensor *output, int channel, bool use_bias);
int csi_pnna_create_fc(struct fc_params *params, struct csi_tensor *output, bool use_bias);
int csi_pnna_create_reshape(struct reshape_params *params, struct csi_tensor *output);
int csi_pnna_create_siso_op(char *op, char *op_name, struct csi_tensor *t);
int csi_pnna_create_elemwise(char *op, char *op_name, struct csi_tensor *t);
int csi_pnna_create_softmax(struct softmax_params *params, struct csi_tensor *t);
int csi_pnna_create_resize(struct resize_params *params, struct csi_tensor *t);
int csi_pnna_create_pool(char *op, struct pool_params *params, struct csi_tensor *t);
int csi_pnna_create_clip(char *op, struct relu_params *params, struct csi_tensor *t);
int csi_pnna_create_leaky_relu(struct relu_params *params, struct csi_tensor *t);
int csi_pnna_create_prelu(struct prelu_params *params, struct csi_tensor *t);
int csi_pnna_create_squeeze(struct squeeze_params *params, struct csi_tensor *t, int *axis, int axis_num);
int csi_pnna_create_space_to_depth(struct space_to_depth_params *params, struct csi_tensor *t);
int csi_pnna_create_depth_to_space(struct depth_to_space_params *params, struct csi_tensor *t);
int csi_pnna_create_batch_to_space_nd(struct batch_to_space_nd_params *params, struct csi_tensor *output);
int csi_pnna_create_space_to_batch_nd(struct space_to_batch_nd_params *params, struct csi_tensor *output);
int csi_pnna_create_concat(struct concat_params *params, struct csi_tensor *output);
int csi_pnna_create_transpose(struct transpose_params *params, struct csi_tensor *output);
int csi_pnna_create_pad(struct pad_params *params, struct csi_tensor *t);
int csi_pnna_create_batch_normalization(struct bn_params *params, struct csi_tensor *output,
                                        bool use_gamma, bool use_beta);
int csi_pnna_create_l2_normalization(struct l2n_params *params, struct csi_tensor *output);
int csi_pnna_create_lrn(struct lrn_params *params, struct csi_tensor *output);
int csi_pnna_create_reduce(char *op, struct reduce_params *params, struct csi_tensor *t, int keepdims);
int csi_pnna_create_crop(struct crop_params *params, struct csi_tensor *t);
int csi_pnna_create_split(struct split_params *params, struct csi_tensor *t);

int csi_pnna_set_graph_attrs(void *attrs, struct csi_quant_info *qinfo, char *name);
void *csi_pnna_create_node_entry(void *nodes_vec, int node, int index, int version);
void csi_pnna_set_node_input(void *nodes_vec, int node, void *node_entry);
void csi_pnna_set_output_internal(void *graph, void *node_entry);
int csi_pnna_get_output_internal(int index, struct csi_tensor *output,
                                 struct csi_pnna_target_data* td);
#ifdef __cplusplus
}
#endif

#endif
