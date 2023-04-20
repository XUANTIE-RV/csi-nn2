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

/******************************************************************************
 * @file     shl_i805_reshape_8.S
 * @brief    uint8 reshape/memcpy layer function.
 * @version  V1.0
 * @date     9. Jul 2021
 ******************************************************************************/

/*
    void shl_i805_reshape_opt_u8(uint8_t * input_data,
                                 uint8_t * output_data
                                 int32_t size)

    Algorithm works as follows:
        (1) memcpy, may have been optimized by complier
        (2) loop 64 elements copy --> 16 elements --> tail elements

    register definition:
        a0: input_data pointer
        a1: output_data pointer
        a2: tensor size
*/

    .file           "shl_i805_reshape_8.S"
    .section        .text.shl_i805_reshape_opt_u8,"ax",@progbits
    .align          2
    .global         shl_i805_reshape_opt_u8
    .type           shl_i805_reshape_opt_u8, @function


shl_i805_reshape_opt_u8:
    lsri            t0, a2, 6           // t0 = size / 64
    bez             t0, .TAIL_64

.LOOP_64:
    vldmu.8         vr0-vr3, (a0)
    vstmu.8         vr0-vr3, (a1)
    bnezad          t0, .LOOP_64        // (t0-1) > 0 ?

.TAIL_64:
    andi            t0, a2, 63
    lsri            t1, t0, 4           // t1 = t0 >> 4
    bez             t1, .TAIL_15

.LOOP_16:
    vldmu.8         vr0-vr0, (a0)
    vstmu.8         vr0-vr0, (a1)
    bnezad          t1, .LOOP_16

.TAIL_15:
    andi            t1, t0, 15
    bez             t1, .END

    vldx.8          vr0, (a0), t1
    vstx.8          vr0, (a1), t1

.END:
    rts
    .size           shl_i805_reshape_opt_u8, .-shl_i805_reshape_opt_u8
