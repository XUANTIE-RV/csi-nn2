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

#ifndef INCLUDE_CSI_UTILS_H_
#define INCLUDE_CSI_UTILS_H_

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if (!defined CSI_BUILD_RTOS)
#include <omp.h>
#endif
#include "csi_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* misc */
void csi_get_top5(float *buf, uint32_t size, float *prob, uint32_t *cls);
void csi_show_top5(struct csi_tensor *output, struct csi_session *sess);
uint64_t csi_get_timespec();
void csi_print_time_interval(uint64_t start, uint64_t end, const char *msg);
void csi_statistical_mean_std(float *data, int sz);
void csi_quantize_multiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift);

/* tensor */
int csi_tensor_size(struct csi_tensor *tensor);
int csi_tensor_byte_size(struct csi_tensor *tensor);
struct csi_tensor *csi_alloc_tensor(struct csi_session *session);
void csi_free_tensor(struct csi_tensor *tensor);
void csi_realloc_quant_info(struct csi_tensor *tensor, int quant_info_num);
void csi_tensor_copy(struct csi_tensor *dest, struct csi_tensor *src);
int csi_tensor_data_convert(struct csi_tensor *dest, struct csi_tensor *src);

/* op parameters */
void *csi_alloc_params(int params_size, struct csi_session *session);
void csi_free_params(void *params);

/* session */
struct csi_session *csi_alloc_session();
void csi_free_session(struct csi_session *session);
void csi_session_init(struct csi_session *session);
void csi_session_deinit(struct csi_session *session);
int csi_session_setup(struct csi_session *session);
int csi_session_run(struct csi_session *session);
int csi_load_binary_model(char *path, struct csi_session *session);

/* input/output */
void csi_set_input_number(int number, struct csi_session *sess);
void csi_set_output_number(int number, struct csi_session *sess);
int csi_get_input_number(struct csi_session *sess);
int csi_get_output_number(struct csi_session *sess);
int csi_set_input(int index, struct csi_tensor *input, struct csi_session *sess);
int csi_set_output(int index, struct csi_tensor *output, struct csi_session *sess);
int csi_get_input(int index, struct csi_tensor *input, struct csi_session *sess);
int csi_get_output(int index, struct csi_tensor *output, struct csi_session *sess);
int csi_update_input(int index, struct csi_tensor *input, struct csi_session *sess);
int csi_update_output(int index, struct csi_tensor *output, struct csi_session *sess);
int csi_set_tensor_entry(struct csi_tensor *tensor, struct csi_session *sess);

/*
 * model setup and run
 */
void csi_nn_init(struct csi_tensor *input, struct csi_tensor *output);

void csi_nn_setup(void *td);

void csi_nn_run(void *td);

void csi_nn_postprocess(void *td);

void csi_nn_deinit(struct csi_tensor *input, struct csi_tensor *output);

void *csi_nn_presetup(int input, int output);
void *csi_bc_map(int api, int rmode, int op, int dtype);
void *csi_init_map(int api, int op, int dtype);

struct csi_bc_op_list *csi_bc_list_end(struct csi_bc_op_list *list);
void *csi_bc_list_match(struct csi_bc_op_list *list, enum csinn_dtype_enum dtype,
                        enum csinn_op_enum op_name);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_CSI_UTILS_H_
