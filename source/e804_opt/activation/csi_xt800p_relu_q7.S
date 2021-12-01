/*
 * Copyright (C) 2016-2019 C-SKY Limited. All rights reserved.
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
 * @file     csi_xt800p_relu_q7.S
 * @brief    Q15 version of ReLU.
 * @version  V1.0
 * @date     01. June 2018
 ******************************************************************************/

/*
 *void csi_xt800p_relu_q7(q7_t * data,
 *                   uint8_t size)
 */

    .file           "csi_xt800p_relu_q7.S"
    .section        .text.csi_xt800p_relu_q7,"ax",@progbits
    .align          2
    .global         csi_xt800p_relu_q7
    .type           csi_xt800p_relu_q7, @function

csi_xt800p_relu_q7:
    movi            t9, 0
    mov             t8, a0
    lsri            t7, a1, 4
    bez             t7, .L1

.L0:
    pldbi.d         t0, (a0)
    pldbi.d         t2, (a0)
    pmax.s8         t0, t0, t9
    pmax.s8         t1, t1, t9
    pmax.s8         t2, t2, t9
    pmax.s8         t3, t3, t9

    st.w            t0, (t8, 0x0)
    st.w            t1, (t8, 0x4)
    st.w            t2, (t8, 0x8)
    st.w            t3, (t8, 0xc)
    addi            t8, t8, 16

    bnezad          t7, .L0

.L1:
    andi            t7, a1, 31
    lsri            t6, t7, 2
    bez             t6, .L3

.L2:
    ldbi.w          t0, (a0)
    pmax.s8         t0, t0, t9
    stbi.w          t0, (t8)

    bnezad          t6, .L2

.L3:
    andi            t6, t7, 3
    bez             t6, .L5

.L4:
    ld.bs           t0, (a0)
    max.s32         t0, t0, t9
    stbi.b          t0, (a0)

    bnezad          t6, .L4

.L5:
    rts
    .size           csi_xt800p_relu_q7, .-csi_xt800p_relu_q7
.weak csi_relu_q7
.set  csi_relu_q7, csi_xt800p_relu_q7
.weak csky_dsp2_relu_q7
.set  csky_dsp2_relu_q7, csi_xt800p_relu_q7
