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

/* SHL version 2.1.x */

#include "shl_e907.h"

int shl_rvp_get_xlenb()
{
#if __riscv_xlen == 64
    return 8;
#elif __riscv_xlen == 32
    return 4;
#endif
}

void shl_rvp_int8_to_int16(int8_t *src, int16_t *dst, size_t len)
{
    for (int i = 0; i < len; i++) {
        dst[i] = (int16_t)src[i];
    }
}

void shl_rvp_int16_to_int32(int16_t *src, int32_t *dst, size_t len)
{
    for (int i = 0; i < len; i++) {
        dst[i] = (int32_t)src[i];
    }
}

void shl_rvp_int32_to_int16(int32_t *src, int16_t *dst, size_t len)
{
    for (int i = 0; i < len; i++) {
        dst[i] = (int16_t)src[i];
    }
}

void shl_rvp_int16_to_int8(int16_t *src, int8_t *dst, size_t len)
{
    for (int i = 0; i < len; i++) {
        dst[i] = (int8_t)src[i];
    }
}

intXLEN_t shl_rvp_int16_to_xlen(int16_t val)
{
#if __riscv_xlen == 64
    int16_t tmp[4] = {val, val, val, val};
#elif __riscv_xlen == 32
    int16_t tmp[2] = {val, val};
#endif
    intXLEN_t *res = (intXLEN_t *)tmp;
    return *res;
}

intXLEN_t shl_rvp_int32_to_xlen(int32_t val)
{
#if __riscv_xlen == 64
    int32_t tmp[2] = {val, val};
#elif __riscv_xlen == 32
    int32_t tmp[1] = {val};
#endif
    intXLEN_t *res = (intXLEN_t *)tmp;
    return *res;
}

/********************* for int8 quantization *********************/
// 再量化 int32 -> int8
void shl_rvp_requantize(int32_t *src, int32_t multiplier, int32_t shift, int channel_size)
{
#if __riscv_xlen == 64
    int64_t multiplier_32x2 = shl_rvp_int32_to_xlen(multiplier);
    int i = 0;
    for (; i + 1 < channel_size; i += 2) {
        int64_t *src_i32x2 = src + i;
        int64_t res = src_i32x2[0];
        if (shift < 0) {
            res = __rv__sra32_u(res, -shift - 1);
        } else {
            res = __rv__sll32(res, shift + 1);
        }
        res = __rv__smmul_u(res, multiplier_32x2);
        src_i32x2[0] = res;
    }
    for (; i < channel_size; i++) {
        int32_t res = src[i];
        res = shl_rvp_mulh(res, multiplier);
        if (shift < 0) {
            res >>= -shift - 1;
        } else {
            res <<= shift + 1;
        }
        src[i] = res;
    }
#elif __riscv_xlen == 32
    for (int i = 0; i < channel_size; i++) {
        int32_t res = src[i];
        // FIXME: precision error
        if (shift < 0) {
            res >>= -shift - 1;
        } else {
            res <<= shift + 1;
        }
        res = __rv__smmul_u(res, multiplier);
        src[i] = res;
    }
#endif
}

// add output_zeropoint
void shl_rvp_saturated_int8(int32_t *src, int8_t *dst, int32_t out_zp, int size)
{
    int i = 0;
#if __riscv_xlen == 64
    int64_t out_zp_32x2 = shl_rvp_int32_to_xlen(out_zp);
    for (; i + 1 < size; i += 2) {
        int64_t *src_i32x2 = src + i;
        int64_t tmp = __rv__add32(src_i32x2[0], out_zp_32x2);
        int64_t res = __rv__sclip32(tmp, 7);
        int32_t *res1 = (int32_t *)(&res);
        dst[i] = (int8_t)res1[0];
        dst[i + 1] = (int8_t)res1[1];
    }
#endif
    for (; i < size; i++) {
        int32_t res = src[i] + out_zp;
        dst[i] = shl_rvp_clip_i8(res);
    }
}
