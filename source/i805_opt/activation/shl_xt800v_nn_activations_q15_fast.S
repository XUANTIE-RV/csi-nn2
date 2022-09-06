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

/******************************************************************************
 * @file     shl_xt800v_nn_activations_q15_fast.S
 * @brief    Q15 neural network activation function using direct table look-up.
 * @version  V1.0
 * @date     01. June 2018
 ******************************************************************************/
.import sigmoidTable_q15
.import tanhTable_q15

/*
 *void shl_xt800v_nn_activations_direct_q15(q15_t * data,
 *                                   uint16_t size,
 *                                   uint16_t int_width,
 *                                   shl_xt800v_nn_activation_type type)
 */

    .file           "shl_xt800v_nn_activations_q15.S"
    .section        .text.shl_xt800v_nn_activations_direct_q15,"ax",@progbits
    .align          2
    .global         shl_xt800v_nn_activations_direct_q15
    .type           shl_xt800v_nn_activations_direct_q15, @function

shl_xt800v_nn_activations_direct_q15:
    push            l0, l1, l2, l3
    subi            sp, sp, 32
    vstm.8          vr8-vr9, (sp)
    vlrw.32.4       vr7, 0x7ff07ff_7ff07ff_7ff07ff_7ff07ff
    vdupg.16        vr0, a2
    vshr.s16        vr7, vr7, vr0       // bit_mask
    movi            t0, 11
    subu            a2, t0, a2
    vdupg.16        vr6, a2             // shift_size
    vmovi.u16       vr8, 0x1, 0
    vadd.u16        vr9, vr7, vr8       // full_frac
    lrw             l0, sigmoidTable_q15
    lrw             l1, tanhTable_q15

    cmplti          a3, 1
    movt            l2, l0
    movf            l2, l1

    mov             l0, a0
    mov             l1, a0

    lsri            t0, a1, 3
    bez             t0, .L1

.L0:
    vldmu.16        vr0-vr0, (l0)       // in
    vand.16         vr1, vr0, vr7       // frc = in & bit_mask
    vshr.s16        vr0, vr0, vr6       // in >> shift_size
    vclip.s8        vr2, vr0, 8         // value
    vmov.16.l       vr2, vr2, vr2
    vmov.u8.e       vr2, vr2
    vshli.u16       vr2, vr2, 1
    vadd.s16        vr4, vr0, vr8
    vlde.16         vr0, (l2, vr2)
    vclip.s8        vr3, vr4, 8         // value2
    vmov.16.l       vr3, vr3, vr3
    vmov.u8.e       vr3, vr3
    vshli.u16       vr3, vr3, 1
    vlde.16         vr4, (l2, vr3)

    vsub.u16        vr5, vr9, vr1       // full_frac - frac
    vmul.s16.e      vr2, vr5, vr0
    vmula.s16.e     vr2, vr4, vr1

    vshr.s32        vr2, vr2, vr6
    vshr.s32        vr3, vr3, vr6

    vmov.32.l       vr2, vr2, vr3
    vstmu.16        vr2-vr2, (l1)

    bnezad          t0, .L0

.L1:
    andi            t0, a1, 7
    bez             t0, .L3

.L2:
    vldx.16         vr0, (l0), t0       // in
    vand.16         vr1, vr0, vr7       // frc = in & bit_mask
    vshr.s16        vr0, vr0, vr6       // in >> shift_size
    vclip.s8        vr2, vr0, 8         // value
    vmov.16.l       vr2, vr2, vr2
    vmov.u8.e       vr2, vr2
    vshli.u16       vr2, vr2, 1
    vadd.s16        vr4, vr0, vr8
    vlde.16         vr0, (l2, vr2)
    vclip.s8        vr3, vr4, 8         // value2
    vmov.16.l       vr3, vr3, vr3
    vmov.u8.e       vr3, vr3
    vshli.u16       vr3, vr3, 1
    vlde.16         vr4, (l2, vr3)

    vsub.s16.s      vr5, vr9, vr1       // full_frac - frac
    vmul.s16.e      vr2, vr5, vr0
    vmula.s16.e     vr2, vr4, vr1

    vshr.s32        vr2, vr2, vr6
    vshr.s32        vr3, vr3, vr6

    vmov.32.l       vr2, vr2, vr3
    vstx.16         vr2, (l1), t0

.L3:
    vldmu.8         vr8-vr9, (sp)
    pop             l0, l1, l2, l3
    .size           shl_xt800v_nn_activations_direct_q15, .-shl_xt800v_nn_activations_direct_q15

.weak csky_vdsp2_nn_activations_direct_q15
.set  csky_vdsp2_nn_activations_direct_q15, shl_xt800v_nn_activations_direct_q15
