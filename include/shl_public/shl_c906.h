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

#ifndef INCLUDE_SHL_C906_H_
#define INCLUDE_SHL_C906_H_

#include "csi_nn.h"

void shl_c906_reset_fcsr();
int shl_c906_get_fcsr();

/* hardware performance */
struct shl_c906_hpm {
    size_t inst;
    size_t cycle;
    size_t l1_icache_access;
    size_t l1_icache_miss;
    size_t store_inst;
    size_t l1_dcache_raccess;
    size_t l1_dcache_rmiss;
    size_t l1_dcache_waccess;
    size_t l1_dcache_wmiss;
};

uint64_t shl_c906_get_inst();
uint64_t shl_c906_get_cycle();
uint64_t shl_c906_get_l1_icache_access();
uint64_t shl_c906_get_l1_icache_miss();
uint64_t shl_c906_get_cb_miss();
uint64_t shl_c906_get_cb_inst();
uint64_t shl_c906_get_store_inst();
uint64_t shl_c906_get_l1_dcache_raccess();
uint64_t shl_c906_get_l1_dcache_rmiss();
uint64_t shl_c906_get_l1_dcache_waccess();
uint64_t shl_c906_get_l1_dcache_wmiss();

struct shl_c906_hpm shl_c906_get_hw_perf();

int shl_c906_reduce_sum_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                             struct csinn_reduce_params *params);

void shl_c906_u8_to_f32(const uint8_t *input, float *output, int32_t offset, float *scale,
                        uint32_t length);
void shl_c906_i8_to_f32(const int8_t *input, float *output, int32_t offset, float *scale,
                        uint32_t length);
void shl_c906_f32_to_u8(const float *input, uint8_t *output, int32_t offset, float *scale,
                        uint32_t length);
void shl_c906_f32_to_i8(const float *input, int8_t *output, int32_t offset, float *scale,
                        uint32_t length);

#endif  // INCLUDE_SHL_C906_H_
