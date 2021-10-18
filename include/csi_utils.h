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

struct __target_data {
    void *graph;
    int32_t input_num;
    int32_t output_num;
    int32_t layer_num;
    struct csi_tensor *input;
    struct layer_item *net;
    int32_t *output_index;
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
float csi_dequantize_f32(uint8_t input, int32_t offset, int32_t multiplier, int32_t shift);
uint8_t csi_quantize_f32(float input, int32_t offset, int32_t multiplier, int32_t shift);
uint8_t csi_requantize_u8(uint8_t input, int32_t input_offset, int32_t input_multiplier,
                          int32_t input_shift, int32_t output_offset, int32_t output_multiplier,
                          int32_t output_shift);

struct csi_tensor *csi_nchw_to_nhwc_u8(struct csi_tensor *t);
void csi_nhwc_to_nchw_u8(struct csi_tensor *nt, struct csi_tensor *t);
struct csi_tensor *csi_deconv_kernel_nchw_to_nhwc_u8(struct csi_tensor *t, int32_t permute[4]);
struct csi_tensor *csi_nchw_to_nhwc_f32(struct csi_tensor *t);
void csi_nhwc_to_nchw_f32(struct csi_tensor *nt, struct csi_tensor *t);
#define MAX_INPUT_INDEX 8
#define MAX_OUTPUT_INDEX 8
struct layer_item {
    int32_t input_index[MAX_INPUT_INDEX];
    int32_t output_index[MAX_OUTPUT_INDEX];
    int32_t input_num;
    int32_t output_num;
};

struct csi_tensor *csi_nchw_to_nhwc_u8_new(struct csi_tensor *t, int32_t permute[4]);
int32_t get_reduction_index(int32_t k, const int32_t *strides,
                         const int32_t *extents, int32_t n);


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

#endif

