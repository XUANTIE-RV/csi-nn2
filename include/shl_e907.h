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

#ifndef INCLUDE_SHL_E907_H_
#define INCLUDE_SHL_E907_H_

#if __riscv_dsp
#include <riscv-dsp.h>
#endif  //__riscv_dsp

#include "csi_nn.h"
#include "shl_gref.h"
#include "shl_ref.h"

int shl_e907_fullyconnected_init(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *weights, struct csinn_tensor *bias,
                                 struct csinn_fc_params *params);

int shl_e907_concat_int8(struct csinn_tensor **input, struct csinn_tensor *output,
                         struct csinn_concat_params *params);
int shl_e907_fullyconnected_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                 struct csinn_tensor *weights, struct csinn_tensor *bias,
                                 struct csinn_fc_params *params);
int shl_e907_mul_int8(struct csinn_tensor *input0, struct csinn_tensor *input1,
                      struct csinn_tensor *output, struct csinn_diso_params *params);
int shl_e907_softmax_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                          struct csinn_softmax_params *params);
int shl_e907_relu_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                       struct csinn_relu_params *params);
int shl_e907_sum_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                      struct csinn_reduce_params *params);
int shl_e907_conv2d_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                         struct csinn_conv2d_params *params);

int shl_rvp_get_xlenb();
void shl_rvp_int8_to_int16(int8_t *src, int16_t *dst, size_t len);
void shl_rvp_int16_to_int32(int16_t *src, int32_t *dst, size_t len);
void shl_rvp_int32_to_int16(int32_t *src, int16_t *dst, size_t len);
void shl_rvp_int16_to_int8(int16_t *src, int8_t *dst, size_t len);
intXLEN_t shl_rvp_int16_to_xlen(int16_t val);
intXLEN_t shl_rvp_int32_to_xlen(int32_t val);
void shl_rvp_requantize(int32_t *src, int32_t multiplier, int32_t shift, int channel_size);
void shl_rvp_saturated_int8(int32_t *src, int8_t *dst, int32_t out_zp, int size);

static inline int32_t shl_rvp_mulh(int32_t rs1, int32_t rs2)
{
    int ret = 0;
    asm volatile("mulh %0, %1, %2" : "=r"(ret) : "r"(rs1), "r"(rs2));
    return ret;
}

static inline int8_t shl_rvp_clip_i8(int32_t val)
{
    if (val > 127) {
        return 127;
    } else if (val < -128) {
        return -128;
    } else {
        return (int8_t)val;
    }
}

#endif  // INCLUDE_SHL_E907_H_
