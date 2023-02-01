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

int32_t shl_rvp_int8_to_int8x4(int8_t val)
{
    int8_t tmp[4] = {val, val, val, val};
    int32_t *res = (int32_t *)tmp;
    return *res;
}

/********************* for int8 quantization *********************/
// 再量化 int32 -> int8
void shl_rvp_requantize(int32_t *src, int32_t multiplier, int32_t shift, int channel_size)
{
#if __riscv_xlen == 32
    for (int i = 0; i < channel_size; i++) {
        intXLEN_t tmp = src[i];
        // FIXME: precision error
        if (shift < 0) {
            tmp >>= -shift - 1;
        } else {
            tmp <<= shift + 1;
        }
        intXLEN_t res = __rv__smmul_u(tmp, multiplier);
        src[i] = res;
    }
#endif
}

// add output_zeropint
void shl_rvp_saturated_int8(int32_t *src, int8_t *dst, int32_t out_zp, int size)
{
#if __riscv_xlen == 32
    for (int i = 0; i < size; i++) {
        intXLEN_t tmp = src[i] + out_zp;
        intXLEN_t res = __rv__sclip32(tmp, 7);
        dst[i] = (int8_t)res;
    }
#endif
}
