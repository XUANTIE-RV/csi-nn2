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
 * @file     csi_xt800v_nn_activations_q15.S
 * @brief    Q15 neural network activation function using direct table look-up.
 * @version  V1.0
 * @date     01. June 2018
 ******************************************************************************/
.import sigmoidTable_q15
.import tanhTable_q15

/*
 *void csi_xt800v_nn_activations_direct_q15(q15_t * data,
 *                                   uint16_t size,
 *                                   uint16_t int_width,
 *                                   csi_xt800v_nn_activation_type type)
 */

    .file           "csi_xt800v_nn_activations_q15.S"
    .section        .text.csi_xt800v_nn_activations_direct_q15,"ax",@progbits
    .align          2
    .global         csi_xt800v_nn_activations_direct_q15
    .type           csi_xt800v_nn_activations_direct_q15, @function

csi_xt800v_nn_activations_direct_q15:
    push            l0, l1, l2, l3
    subi            sp, sp, 64
    vstm.8          vr8-vr11, (sp)
    subi            sp, sp, 64
    vstm.8          vr12-vr15, (sp)
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
    vdupg.32        vr10, l2
    vdupg.32        vr11, l2

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
    vadd.u16.x      vr4, vr10, vr2
    vadd.s16        vr0, vr0, vr8
    vclip.s8        vr2, vr0, 8         // value2
    vmov.16.l       vr2, vr2, vr2
    vmov.u8.e       vr2, vr2
    vshli.u16       vr2, vr2, 1
    vadd.u16.x      vr2, vr10, vr2
    vmfvr.s32       l3, vr4, 0
    vldu.16.1       vr12, (l3)
    vmfvr.s32       l3, vr4, 1
    vldu.16.1       vr13, (l3)
    vmfvr.s32       l3, vr4, 2
    vldu.16.1       vr14, (l3)
    vmfvr.s32       l3, vr4, 3
    vldu.16.1       vr15, (l3)
    vpkg.16.4       vr0, vr12, 0, 0
    vmfvr.s32       l3, vr5, 0
    vldu.16.1       vr12, (l3)
    vmfvr.s32       l3, vr5, 1
    vldu.16.1       vr13, (l3)
    vmfvr.s32       l3, vr5, 2
    vldu.16.1       vr14, (l3)
    vmfvr.s32       l3, vr5, 3
    vldu.16.1       vr15, (l3)
    vpkg.16.4       vr0, vr12, 0, 4     // value

    vmfvr.s32       l3, vr2, 0
    vldu.16.1       vr12, (l3)
    vmfvr.s32       l3, vr2, 1
    vldu.16.1       vr13, (l3)
    vmfvr.s32       l3, vr2, 2
    vldu.16.1       vr14, (l3)
    vmfvr.s32       l3, vr2, 3
    vldu.16.1       vr15, (l3)
    vpkg.16.4       vr4, vr12, 0, 0
    vmfvr.s32       l3, vr3, 0
    vldu.16.1       vr12, (l3)
    vmfvr.s32       l3, vr3, 1
    vldu.16.1       vr13, (l3)
    vmfvr.s32       l3, vr3, 2
    vldu.16.1       vr14, (l3)
    vmfvr.s32       l3, vr3, 3
    vldu.16.1       vr15, (l3)
    vpkg.16.4       vr4, vr12, 0, 4     // value2

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
    vshli.u16       vr2, vr2, 1
    vmov.16.l       vr2, vr2, vr2
    vmov.u8.e       vr2, vr2
    vadd.u16.x      vr4, vr10, vr2
    vadd.s16        vr0, vr0, vr8
    vclip.s8        vr2, vr0, 8         // value2
    vmov.16.l       vr2, vr2, vr2
    vmov.u8.e       vr2, vr2
    vshli.u16       vr2, vr2, 1
    vadd.u16.x      vr2, vr10, vr2
    vmfvr.s32       l3, vr4, 0
    vldu.16.1       vr12, (l3)
    vmfvr.s32       l3, vr4, 1
    vldu.16.1       vr13, (l3)
    vmfvr.s32       l3, vr4, 2
    vldu.16.1       vr14, (l3)
    vmfvr.s32       l3, vr4, 3
    vldu.16.1       vr15, (l3)
    vpkg.16.4       vr0, vr12, 0, 0
    vmfvr.s32       l3, vr5, 0
    vldu.16.1       vr12, (l3)
    vmfvr.s32       l3, vr5, 1
    vldu.16.1       vr13, (l3)
    vmfvr.s32       l3, vr5, 2
    vldu.16.1       vr14, (l3)
    vmfvr.s32       l3, vr5, 3
    vldu.16.1       vr15, (l3)
    vpkg.16.4       vr0, vr12, 0, 4     // value

    vmfvr.s32       l3, vr2, 0
    vldu.16.1       vr12, (l3)
    vmfvr.s32       l3, vr2, 1
    vldu.16.1       vr13, (l3)
    vmfvr.s32       l3, vr2, 2
    vldu.16.1       vr14, (l3)
    vmfvr.s32       l3, vr2, 3
    vldu.16.1       vr15, (l3)
    vpkg.16.4       vr4, vr12, 0, 0
    vmfvr.s32       l3, vr3, 0
    vldu.16.1       vr12, (l3)
    vmfvr.s32       l3, vr3, 1
    vldu.16.1       vr13, (l3)
    vmfvr.s32       l3, vr3, 2
    vldu.16.1       vr14, (l3)
    vmfvr.s32       l3, vr3, 3
    vldu.16.1       vr15, (l3)
    vpkg.16.4       vr4, vr12, 0, 4     // value2

    vsub.s16.s      vr5, vr9, vr1       // full_frac - frac
    vmul.s16.e      vr2, vr5, vr0
    vmula.s16.e     vr2, vr4, vr1

    vshr.s32        vr2, vr2, vr6
    vshr.s32        vr3, vr3, vr6

    vmov.32.l       vr2, vr2, vr3
    vstx.16         vr2, (l1), t0

    bnezad          t0, .L2

.L3:
    vldmu.8         vr12-vr15, (sp)
    vldmu.8         vr8-vr11, (sp)
    pop             l0, l1, l2, l3
    .size           csi_xt800v_nn_activations_direct_q15, .-csi_xt800v_nn_activations_direct_q15
