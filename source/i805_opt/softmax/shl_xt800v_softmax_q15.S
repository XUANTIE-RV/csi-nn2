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
 * @file     shl_xt800v_softmax_q15.S
 * @brief    Pooling functions implementations.
 * @version  V1.0
 * @date     01. June 2018
 ******************************************************************************/

/*
 * void shl_xt800v_softmax_q15(const q15_t * vec_in,
 *                       const uint16_t dim_vec,
 *                       q15_t * p_out)
 */

    .file           "shl_xt800v_softmax_q15.S"
    .section        .text.shl_xt800v_softmax_q15,"ax",@progbits
    .align          2
    .global         shl_xt800v_softmax_q15
    .type           shl_xt800v_softmax_q15, @function

shl_xt800v_softmax_q15:
    push            l0, l1, l2
    subi            sp, sp, 64
    vstm.8          vr8-vr11, (sp)
    subi            sp, sp, 48
    vstm.8          vr12-vr14, (sp)
    vlrw.32.4       vr7, 0x80008000_80008000_80008000_80008000
    vmovi.u16       vr9, 0x1, 0
    vmovi.u16       vr6, 0x10, 0
    mov             l0, a0

    lsri            a3, a1, 5
    bez             a3, .L1

.L0:
    vldmu.16        vr0-vr3, (a0)
    vmax.s16        vr7, vr7, vr0       // max
    vmax.s16        vr7, vr7, vr1
    vmax.s16        vr7, vr7, vr2
    vmax.s16        vr7, vr7, vr3

    bnezad          a3, .L0

.L1:
    andi            t0, a1, 31
    lsri            a3, t0, 3
    bez             a3, .L3

.L2:
    vldmu.16        vr0-vr0, (a0)
    vmax.s16        vr7, vr7, vr0

    bnezad          a3, .L2

.L3:
    vpmax.s16       vr7, vr7, vr7
    vpmax.s16       vr7, vr7, vr7
    vpmax.s16       vr7, vr7, vr7
    vmfvr.s16       t1, vr7, 0

.L4:
    andi            a3, t0, 7
    bez             a3, .L6

.L5:
    ld.hs           t0, (a0, 0x0)
    cmplt           t1, t0
    movt            t1, t0
    addi            a0, a0, 2

    bnezad          a3, .L5

.L6:
    vdupg.16        vr7, t1             // the max value
    vsub.s16        vr7, vr7, vr6       // base = base - 16
    mov             a0, l0

    vmovi.8         vr5, 0              // sum = 0
    vmovi.8         vr6, 0
    lsri            a3, a1, 5
    bez             a3, .L8

.L7:
    vldmu.16        vr0-vr3, (a0)
    vcmplt.s16      vr8, vr7, vr0
    vcmplt.s16      vr10, vr7, vr1
    vcmplt.s16      vr11, vr7, vr2
    vcmplt.s16      vr12, vr7, vr3

    vsub.s16        vr0, vr0, vr7
    vsub.s16        vr1, vr1, vr7
    vsub.s16        vr2, vr2, vr7
    vsub.s16        vr3, vr3, vr7

    vclip.u16       vr0, vr0, 5         // shift
    vclip.u16       vr1, vr1, 5         // shift
    vclip.u16       vr2, vr2, 5         // shift
    vclip.u16       vr3, vr3, 5         // shift

    vshl.u16.s      vr4, vr9, vr0
    vand.16         vr4, vr4, vr8
    vadd.u16.x      vr5, vr5, vr4

    vshl.u16.s      vr4, vr9, vr1
    vand.16         vr4, vr4, vr10
    vadd.u16.x      vr5, vr5, vr4

    vshl.u16.s      vr4, vr9, vr2
    vand.16         vr4, vr4, vr11
    vadd.u16.x      vr5, vr5, vr4

    vshl.u16.s      vr4, vr9, vr3
    vand.16         vr4, vr4, vr12
    vadd.u16.x      vr5, vr5, vr4

    bnezad          a3, .L7

.L8:
    andi            t0, a1, 31
    lsri            a3, t0, 3
    bez             a3, .L10

.L9:
    vldmu.16        vr0-vr0, (a0)
    vcmplt.s16      vr8, vr7, vr0
    vsub.s16        vr0, vr0, vr7
    vclip.u16       vr0, vr0, 5         // shift
    vshl.u16.s      vr4, vr9, vr0
    vand.16         vr4, vr4, vr8
    vadd.u16.x      vr5, vr5, vr4

    bnezad          a3, .L9

.L10:
    vadd.s32.s      vr5, vr5, vr6
    vpadd.s32.s     vr0, vr5, vr5
    vpadd.s32.s     vr1, vr0, vr0

    andi            a3, t0, 7
    bez             a3, .L12

.L11:
    vldu.16.1       vr0, (a0)
    vcmplt.s16      vr8, vr7, vr0
    vsub.s16        vr0, vr0, vr7
    vclip.u16       vr0, vr0, 5         // shift
    vshl.u16.s      vr4, vr9, vr0
    vand.16         vr4, vr4, vr8
    vadd.u16.x      vr1, vr1, vr4

    bnezad          a3, .L11

.L12:
    vmfvr.s32       l1, vr1, 0
    movi            l2, 0
    movi            t0, 0
    movi            t1, 1
    divsl           l1, t0, l1
    vdupg.32        vr6, l1

    vmovi.u16       vr5, 17, 0
    mov             a0, l0
    lsri            t0, a1, 5
    bez             t0, .L14

.L13:
    vldmu.16        vr0-vr3, (a0)
    vcmplt.s16      vr4, vr7, vr0
    vcmplt.s16      vr10, vr7, vr1
    vcmplt.s16      vr11, vr7, vr2
    vcmplt.s16      vr12, vr7, vr3

    vsub.s16.s      vr0, vr7, vr0
    vsub.s16.s      vr1, vr7, vr1
    vsub.s16.s      vr2, vr7, vr2
    vsub.s16.s      vr3, vr7, vr3

    vadd.s16.s      vr0, vr0, vr5
    vadd.s16.s      vr1, vr1, vr5
    vadd.s16.s      vr2, vr2, vr5
    vadd.s16.s      vr3, vr3, vr5

    vclip.u16       vr0, vr0, 5
    vclip.u16       vr1, vr1, 5
    vclip.u16       vr2, vr2, 5
    vclip.u16       vr3, vr3, 5

    vmov.u16.e      vr8, vr0
    vmov.u16.e      vr13, vr1
    vshr.s32        vr8, vr6, vr8
    vshr.s32        vr9, vr6, vr9
    vshr.s32        vr13, vr6, vr13
    vshr.s32        vr14, vr6, vr14

    vmov.s32.sl     vr8, vr8, vr9
    vand.16         vr0, vr8, vr4
    vmov.s32.sl     vr13, vr13, vr14
    vand.16         vr1, vr13, vr10

    vmov.u16.e      vr8, vr2
    vmov.u16.e      vr13, vr3
    vshr.s32        vr8, vr6, vr8
    vshr.s32        vr9, vr6, vr9
    vshr.s32        vr13, vr6, vr13
    vshr.s32        vr14, vr6, vr14

    vmov.s32.sl     vr8, vr8, vr9
    vand.16         vr2, vr8, vr11
    vmov.s32.sl     vr13, vr13, vr14
    vand.16         vr3, vr13, vr12

    vstmu.16        vr0-vr3, (a2)

    bnezad          t0, .L13

.L14:
    andi            t1, a1, 31
    lsri            t0, t1, 3
    bez             t0, .L16

.L15:
    vldmu.16        vr0-vr0, (a0)
    vcmplt.s16      vr4, vr7, vr0
    vsub.s16.s      vr0, vr7, vr0
    vadd.s16.s      vr0, vr0, vr5
    vclip.u16       vr0, vr0, 5
    vmov.u16.e      vr8, vr0
    vshr.s32        vr8, vr6, vr8
    vshr.s32        vr9, vr6, vr9
    vmov.s32.sl     vr8, vr8, vr9
    vand.16         vr0, vr8, vr4
    vstmu.16        vr0-vr0, (a2)

    bnezad          t0, .L15

.L16:
    andi            t0, t1, 7
    bez             t0, .L18

.L17:
    vldu.16.1       vr0, (a0)
    vcmplt.s16      vr4, vr7, vr0
    vsub.s16.s      vr0, vr7, vr0
    vadd.s16.s      vr0, vr0, vr5
    vclip.u16       vr0, vr0, 5
    vmov.u16.e      vr8, vr0
    vshr.s32        vr8, vr6, vr8
    vmov.s32.sl     vr8, vr8, vr8
    vand.16         vr0, vr8, vr4
    vstu.16.1       vr0, (a2)

    bnezad          t0, .L17

.L18:
    vldmu.8         vr12-vr14, (sp)
    vldmu.8         vr8-vr11, (sp)
    pop             l0, l1, l2
    .size           shl_xt800v_softmax_q15, .-shl_xt800v_softmax_q15

.weak csky_vdsp2_softmax_q15
.set  csky_vdsp2_softmax_q15, shl_xt800v_softmax_q15
