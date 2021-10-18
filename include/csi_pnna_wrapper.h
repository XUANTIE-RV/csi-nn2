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

#ifndef _CSI_NN_PNNA_WRAPPER_H
#define _CSI_NN_PNNA_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif
int csi_pnna_session_init_internal(struct csi_pnna_target_data *td, void *params_buf);
int csi_pnna_session_deinit_internal(struct csi_pnna_target_data *td);
int csi_pnna_session_setup_internal(struct csi_pnna_target_data *td);
int csi_pnna_session_run_internal(struct csi_pnna_target_data* td, void *input_buf, int input_num);
int csi_pnna_create_val(struct csi_tensor *t);
int csi_pnna_create_const_val(struct csi_tensor *t);
int csi_pnna_create_conv2d(struct conv2d_params *params, struct csi_tensor *kernel,
		           struct csi_tensor *output, int channel, bool use_bias);
int csi_pnna_create_relu(struct relu_params *params, struct csi_tensor *t);
int csi_pnna_set_graph_attrs(void *attrs, double max, double min, char *name);
void csi_pnna_set_node_input(void *nodes_vec, int node, int input, int index);
void csi_pnna_set_output_internal(void *graph, void *nodes_vec, int output, int index);
#ifdef __cplusplus
}
#endif

#endif
