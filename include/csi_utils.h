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

#ifndef _CSI_NN_UTIL_H
#define _CSI_NN_UTIL_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <omp.h>

#define CSINN_MAX_INPUT 4
#define CSINN_MAX_OUTPUT 8
struct csi_session {
    int32_t base_dtype;
    int32_t base_layout;
    int32_t base_api;
    int32_t input_num;
    int32_t output_num;
    struct csi_tensor *input[CSINN_MAX_INPUT];
    struct csi_tensor *output[CSINN_MAX_OUTPUT];
    void *td;
};

int32_t csi_max_internal_s32(int32_t a, int32_t b);
int32_t csi_min_internal_s32(int32_t a, int32_t b);
uint8_t csi_saturate_u8(int32_t input);

int32_t csi_get_index(int32_t *dim, int32_t index0, int32_t index1, int32_t index2, int32_t index3);
int32_t csi_get_index_5(int32_t *dim, int32_t index0, int32_t index1, int32_t index2, int32_t index3, int32_t index4);
int32_t csi_get_index_6(int32_t *dim, int32_t index0, int32_t index1, int32_t index2, int32_t index3, int32_t index4, int32_t index5);

float csi_get_scale(int32_t multiplier, int32_t shift);
int32_t csi_dequantize_u8(uint8_t input, int32_t offset, int32_t multiplier, int32_t shift);
uint8_t csi_quantize_u8(int32_t input, int32_t offset, int32_t multiplier, int32_t shift);
int8_t csi_quantize_i8(int32_t input, int32_t offset, int32_t multiplier, int32_t shift);
float csi_dequantize_u8_to_f32(uint8_t input, int32_t offset, int32_t multiplier, int32_t shift);
float csi_dequantize_i8_to_f32(int8_t input, int32_t offset, int32_t multiplier, int32_t shift);
void csi_dequantize_f32_c860(uint8_t *input, float * output, int32_t offset, int32_t multiplier, int32_t shift, int32_t length);
uint8_t csi_quantize_f32_to_u8(float input, int32_t offset, int32_t multiplier, int32_t shift);
int8_t csi_quantize_f32_to_i8(float input, int32_t offset, int32_t multiplier, int32_t shift);
uint8_t csi_requantize_u8(uint8_t input, int32_t input_offset, int32_t input_multiplier,
                          int32_t input_shift, int32_t output_offset, int32_t output_multiplier,
                          int32_t output_shift);
uint8_t csi_quantize_channel_u8(int32_t data, struct csi_tensor* input, struct csi_tensor* output, float wscale);
float uint8_to_float_channel(uint8_t i, float scale, int32_t zero_point);
float uint8_to_float(uint8_t i, struct csi_tensor *t);
float int8_to_float(int8_t i, struct csi_tensor *t);
uint8_t float_to_uint8(float i, struct csi_tensor *t);
int8_t float_to_int8(float i, struct csi_tensor *t);
int64_t conv_out_u8(int64_t res, struct csi_tensor *input, struct csi_tensor *output, struct csi_tensor *kernel);
int64_t conv_out_i8(int64_t res, struct csi_tensor *input, struct csi_tensor *output, struct csi_tensor *kernel);
int64_t conv_relu6_out_u8(int64_t res, struct csi_tensor *input, struct csi_tensor *output, struct csi_tensor *kernel);
int64_t conv_relu6_out_i8(int64_t res, struct csi_tensor *input, struct csi_tensor *output, struct csi_tensor *kernel);
int64_t conv_channel_out_u8(int64_t res, struct csi_tensor *input, struct csi_tensor *output, float kscale);
int64_t conv_channel_relu6_u8(int64_t res, struct csi_tensor *input, struct csi_tensor *output, float kscale);
struct csi_tensor *csi_nchw_to_nhwc_8(struct csi_tensor *t);
void csi_nhwc_to_nchw_8(struct csi_tensor *nt, struct csi_tensor *t);
struct csi_tensor *csi_deconv_kernel_nchw_to_nhwc_u8(struct csi_tensor *t, int32_t permute[4]);
struct csi_tensor *csi_nchw_to_nhwc_f32(struct csi_tensor *t);
void csi_nhwc_to_nchw_f32(struct csi_tensor *nt, struct csi_tensor *t);
void csi_get_top5(float *buf, uint32_t size, float *prob, uint32_t *cls);
uint64_t csi_get_timespec();

struct csi_tensor *csi_nchw_to_nhwc_u8_new(struct csi_tensor *t, int32_t permute[4]);
int32_t get_reduction_index(int32_t k, const int32_t *strides,
                            const int32_t *extents, int32_t n);

struct csi_tensor *csi_alloc_tensor(struct csi_session *session);
struct csi_session *csi_alloc_session();
void csi_free_session(struct csi_session *session);
void csi_session_init(struct csi_session *session);
void csi_session_deinit(struct csi_session *session);
int csi_session_setup(struct csi_session *session);
int csi_session_run(struct csi_session *session);
void csi_set_input_number(int number, struct csi_session *sess);
void csi_set_output_number(int number, struct csi_session *sess);
int csi_get_input_number(struct csi_session *sess);
int csi_get_output_number(struct csi_session *sess);
int csi_set_input(int index, struct csi_tensor *input, struct csi_session *sess);
int csi_set_output(int index, struct csi_tensor *output, struct csi_session *sess);
int csi_get_input(int index, struct csi_tensor *input, struct csi_session *sess);
int csi_get_output(int index, struct csi_tensor *output, struct csi_session *sess);
int csi_update_input(int index, struct csi_tensor *input, struct csi_session *sess);

/*
 * model setup and run
 */

void csi_nn_init(struct csi_tensor *input,
                 struct csi_tensor *output);

void csi_nn_setup(void *td);

void csi_nn_run(void *td);

void csi_nn_postprocess(void *td);

void csi_nn_deinit(struct csi_tensor *input,
                   struct csi_tensor *output);

void *csi_nn_presetup(int input, int output);
void *csi_bc_map(int api, int op, int dtype);
#endif

