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

/* CSI-NN2 version 1.8.x */

#ifndef TEST_UTILS_H
#define TEST_UTILS_H


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "csi_nn.h"
#include "csi_ref.h"

int *read_input_data_f32(char *path);
float compute_kl(float *p, float *q, uint32_t size);
float compute_cs(float *a, float *b, uint32_t size);
void result_verify_int32(int *reference, int *output, int *input, float gap, int size, bool save);
void result_verify_f32(float *reference, float *output, float *input, float gap, int size, bool save);
void result_verify_bool(bool *reference, bool *output, float *input, float gap, int size, bool save);
void result_verify_8(float *reference, struct csi_tensor *output, int8_t *input, float gap, int size, bool save);
void get_scale_and_zp(float max_value, float min_value, float *scale, int *zp);
void get_scale_and_zp_i8(float max_value, float min_value, float *scale, int *zp);
void quantize_multiplier(double double_multiplier, int32_t* quantized_multiplier, int* shift);
void find_min_max(float *input, float *max_value, float *min_value, int size);
void get_quant_info(struct csi_tensor *tensor);

extern void init_testsuite(const char* testname);
extern int  done_testing(void);

#endif	/* TEST_UTILS_H */
