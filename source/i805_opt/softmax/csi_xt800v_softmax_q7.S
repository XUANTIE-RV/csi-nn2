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
 * @file     csi_xt800v_softmax_q7.S
 * @brief    Pooling functions implementations.
 * @version  V1.0
 * @date     04. June 2018
 ******************************************************************************/

/*
 * void csi_xt800v_softmax_q7(const q7_t * vec_in,
 *                      const uint16_t dim_vec,
 *                      q7_t * p_out)
 */

    .file           "csi_xt800v_softmax_q7.S"
    .section        .text.csi_xt800v_softmax_q7,"ax",@progbits
    .align          2
    .global         csi_xt800v_softmax_q7
    .type           csi_xt800v_softmax_q7, @function

csi_xt800v_softmax_q7:
    push            l0, l1, l2
    subi            sp, sp, 32
    vstm.8          vr8-vr9, (sp)
    vmovi.8         vr7, 0x80           // init max value
    vmovi.8         vr9, 0x1
    vmovi.8         vr6, 0x8
    mov             l0, a0

    lsri            a3, a1, 6
    bez             a3, .L1

.L0:
    vldmu.8         vr0-vr3, (a0)
    vmax.s8         vr7, vr7, vr0       // max
    vmax.s8         vr7, vr7, vr1
    vmax.s8         vr7, vr7, vr2
    vmax.s8         vr7, vr7, vr3

    bnezad          a3, .L0

.L1:
    andi            t0, a1, 63
    lsri            a3, t0, 4
    bez             a3, .L3

.L2:
    vldmu.8         vr0-vr0, (a0)
    vmax.s8         vr7, vr7, vr0

    bnezad          a3, .L2

.L3:
    vpmax.s8        vr7, vr7, vr7
    vpmax.s8        vr7, vr7, vr7
    vpmax.s8        vr7, vr7, vr7
    vpmax.s8        vr7, vr7, vr7
    vmfvr.s8        t1, vr7, 0

.L4:
    andi            a3, t0, 15
    bez             a3, .L6

.L5:
    ld.bs           t0, (a0, 0x0)
    cmplt           t1, t0
    movt            t1, t0
    addi            a0, a0, 1

    bnezad          a3, .L5

.L6:
    vdupg.8         vr7, t1             // the max value
    vsub.s8         vr7, vr7, vr6       // base = base - 8
    mov             a0, l0

    vmovi.8         vr6, 0              // sum = 0
    lsri            a3, a1, 6
    bez             a3, .L8

.L7:
    vldmu.8         vr0-vr3, (a0)
    vcmplt.s8       vr8, vr7, vr0
    vsub.s8         vr0, vr0, vr7
    vclip.u8        vr0, vr0, 5         // shift
    vshl.u8.s       vr4, vr9, vr0
    vand.8          vr4, vr4, vr8

    vcmplt.s8       vr8, vr7, vr1
    vsub.s8         vr1, vr1, vr7
    vclip.u8        vr1, vr1, 5         // shift
    vshl.u8.s       vr5, vr9, vr1
    vand.8          vr5, vr5, vr8
    vadd.u8.e       vr4, vr4, vr5
    vadd.s16.e      vr4, vr4, vr5
    vadd.s32.s      vr6, vr6, vr4       // sum
    vadd.s32.s      vr6, vr6, vr5

    vcmplt.s8       vr8, vr7, vr2
    vsub.s8         vr2, vr2, vr7
    vclip.u8        vr2, vr2, 5         // shift
    vshl.u8.s       vr4, vr9, vr2
    vand.8          vr4, vr4, vr8

    vcmplt.s8       vr8, vr7, vr3
    vsub.s8         vr3, vr3, vr7
    vclip.u8        vr3, vr3, 5         // shift
    vshl.u8.s       vr5, vr9, vr3
    vand.8          vr5, vr5, vr8
    vadd.u8.e       vr4, vr4, vr5
    vadd.s16.e      vr4, vr4, vr5
    vadd.s32.s      vr6, vr6, vr4       // sum
    vadd.s32.s      vr6, vr6, vr5

    bnezad          a3, .L7

.L8:
    andi            t0, a1, 63
    lsri            a3, t0, 4
    bez             a3, .L10

.L9:
    vldmu.8         vr0-vr0, (a0)
    vcmplt.s8       vr8, vr7, vr0
    vsub.s8         vr0, vr0, vr7
    vclip.u8        vr0, vr0, 5         // shift
    vshl.u8.s       vr4, vr9, vr0
    vand.8          vr4, vr4, vr8
    vmov.u8.e       vr0, vr4
    vadd.s16.e      vr0, vr0, vr1
    vadd.s32.s      vr6, vr6, vr0
    vadd.s32.s      vr6, vr6, vr1

    bnezad          a3, .L9

.L10:
    vpadd.s32.s     vr0, vr6, vr6
    vpadd.s32.s     vr1, vr0, vr0

    andi            a3, t0, 15
    bez             a3, .L12

.L11:
    vldu.8.1        vr0, (a0)
    vcmplt.s8       vr8, vr7, vr0
    vsub.s8         vr0, vr0, vr7
    vclip.u8        vr0, vr0, 5         // shift
    vshl.u8.s       vr4, vr9, vr0
    vand.8          vr4, vr4, vr8
    vmov.u8.e       vr4, vr4
    vadd.s16.e      vr4, vr4, vr5
    vadd.s32.s      vr1, vr1, vr4

    bnezad          a3, .L11

.L12:
    vmfvr.s32       l1, vr1, 0
    movi            l2, 1
    lsli            l2, l2, 20
    divs            l1, l2, l1
    vdupg.32        vr6, l1

    vmovi.8         vr9, 13
    mov             a0, l0
    lsri            t0, a1, 4
    bez             t0, .L14

.L13:
    vldmu.8         vr0-vr0, (a0)
    vcmplt.s8       vr8, vr7, vr0
    vsub.s8.s       vr0, vr7, vr0
    vadd.s8.s       vr0, vr0, vr9
    vclip.u8        vr0, vr0, 5         // shift value
    vmov.u8.e       vr0, vr0
    vmov.u16.e      vr2, vr0
    vshr.s32        vr2, vr6, vr2
    vshr.s32        vr3, vr6, vr3
    vmov.u16.e      vr4, vr1
    vshr.s32        vr4, vr6, vr4
    vshr.s32        vr5, vr6, vr5
    vmov.s32.sl     vr2, vr2, vr3
    vmov.s32.sl     vr3, vr4, vr5
    vmov.s16.sl     vr2, vr2, vr3
    vand.8          vr0, vr8, vr2
    vstmu.8         vr0-vr0, (a2)

    bnezad          t0, .L13

.L14:
    andi            t0, a1, 15
    bez             t0, .L18

.L17:
    vldu.8.1        vr0, (a0)
    vcmplt.s8       vr4, vr7, vr0
    vsub.s8.s       vr0, vr7, vr0
    vadd.s8.s       vr0, vr0, vr9
    vclip.u8        vr0, vr0, 5
    vmov.u8.e       vr0, vr0
    vmov.u16.e      vr0, vr0
    vshr.s32        vr0, vr6, vr0
    vclip.s32       vr0, vr0, 8
    vand.8          vr0, vr0, vr4
    vstu.8.1        vr0, (a2)

    bnezad          t0, .L17

.L18:
    vldmu.8         vr8-vr9, (sp)
    pop             l0, l1, l2
    .size           csi_xt800v_softmax_q7, .-csi_xt800v_softmax_q7
.weak csi_softmax_q7
.set  csi_softmax_q7, csi_xt800v_softmax_q7
.weak csky_vdsp2_softmax_q7
.set  csky_vdsp2_softmax_q7, csi_xt800v_softmax_q7
