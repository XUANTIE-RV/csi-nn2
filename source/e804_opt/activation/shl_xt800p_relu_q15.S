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

/******************************************************************************
 * @file     shl_xt800p_relu_q15.S
 * @brief    Q15 version of ReLU.
 * @version  V1.0
 * @date     01. June 2018
 ******************************************************************************/

/*
 *void shl_xt800p_relu_q15(q15_t * data,
 *                   uint16_t size)
 */

    .file           "shl_xt800p_relu_q15.S"
    .section        .text.shl_xt800p_relu_q15,"ax",@progbits
    .align          2
    .global         shl_xt800p_relu_q15
    .type           shl_xt800p_relu_q15, @function

shl_xt800p_relu_q15:
    movi            t9, 0
    mov             t8, a0
    lsri            t7, a1, 3
    bez             t7, .L1

.L0:
    pldbi.d         t0, (a0)
    pldbi.d         t2, (a0)
    pmax.s16        t0, t0, t9
    pmax.s16        t1, t1, t9
    pmax.s16        t2, t2, t9
    pmax.s16        t3, t3, t9

    st.w            t0, (t8, 0x0)
    st.w            t1, (t8, 0x4)
    st.w            t2, (t8, 0x8)
    st.w            t3, (t8, 0xc)
    addi            t8, t8, 16

    bnezad          t7, .L0

.L1:
    andi            t7, a1, 7
    bez             t7, .L3

.L4:
    ldbi.hs         t0, (a0)
    max.s32         t0, t0, t9
    stbi.h          t0, (t8)

    bnezad          t7, .L4

.L3:
    rts
    .size           shl_xt800p_relu_q15, .-shl_xt800p_relu_q15
.weak csky_dsp2_relu_q15
.set  csky_dsp2_relu_q15, shl_xt800p_relu_q15
