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
 * @file     csi_xt800p_nn_activations_q15.S
 * @brief    Q15 neural network activation function using direct table look-up.
 * @version  V1.0
 * @date     01. June 2018
 ******************************************************************************/
.import sigmoidTable_q15
.import tanhTable_q15

/*
 *void csi_xt800p_nn_activations_direct_q15(q15_t * data,
 *                                   uint16_t size,
 *                                   uint16_t int_width,
 *                                   csi_xt800p_nn_activation_type type)
 */

    .file           "csi_xt800p_nn_activations_q15.S"
    .section        .text.csi_xt800p_nn_activations_direct_q15,"ax",@progbits
    .align          2
    .global         csi_xt800p_nn_activations_direct_q15
    .type           csi_xt800p_nn_activations_direct_q15, @function

csi_xt800p_nn_activations_direct_q15:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9
    lrw             l0, sigmoidTable_q15
    lrw             l1, tanhTable_q15
    cmplti          a3, 1
    movt            l2, l0
    movf            l2, l1

    mov             l0, a0
    lrw             l1, 0x7ff07ff
    movi            l3, 11
    subu            l3, l3, a2          // shift_size
    plsr.u16        l1, l1, a2          // bit_mask
    lrw             a3, 0x10001
    padd.16         l4, l1, a3          // full_frac


    lsri            t9, a1, 2
    bez             t9, .L1

.L0:
    pldbi.d         t0, (a0)            // in
    pasr.s16        t2, t0, l3          // in >> shift_size
    pasr.s16        t3, t1, l3
    and             t0, t0, l1          // frac = in & bit_mask
    and             t1, t1, l1
    padd.16         t4, t2, a3
    padd.16         t5, t3, a3
    pclipi.s16      t2, t2, 8           // __SSAT(8)
    pclipi.s16      t3, t3, 8
    pclipi.s16      t4, t4, 8
    pclipi.s16      t5, t5, 8

    zextb           t6, t2
    xtrb1           t7, t2
    zextb           l5, t4
    xtrb1           l6, t4
    ldr.hs          t6, (l2, t6 << 1)   // value
    ldr.hs          t7, (l2, t7 << 1)
    ldr.hs          l5, (l2, l5 << 1)   // value2
    ldr.hs          l6, (l2, l6 << 1)

    mulll.s16       l8, l5, t0          // value2 * frac
    psub.16         l7, l4, t0          // full_frac - frac
    mulall.s16.s    l8, l7, t6
    mulhl.s16       l9, t0, l6          // value2 * frac
    mulahl.s16.s    l9, l7, t7
    asr             l8, l8, l3          // >> shift_size
    asr             l9, l9, l3
    pkgll           l8, l8, l9
    stbi.w          l8, (l0)

    zextb           t6, t3
    xtrb1           t7, t3
    zextb           l5, t5
    xtrb1           l6, t5
    ldr.hs          t6, (l2, t6 << 1)   // value
    ldr.hs          t7, (l2, t7 << 1)
    ldr.hs          l5, (l2, l5 << 1)   // value2
    ldr.hs          l6, (l2, l6 << 1)

    mulll.s16       l8, l5, t1          // value2 * frac
    psub.16         l7, l4, t1          // full_frac - frac
    mulall.s16.s    l8, l7, t6
    mulhl.s16       l9, t1, l6          // value2 * frac
    mulahl.s16.s    l9, l7, t7
    asr             l8, l8, l3          // >> shift_size
    asr             l9, l9, l3
    pkgll           l8, l8, l9
    stbi.w          l8, (l0)

    bnezad          t9, .L0

.L1:
    andi            t9, a1, 3
    bez             t9, .L3

.L2:
    ldbi.h          t0, (a0)            // in
    pasr.s16        t2, t0, l3          // in >> shift_size
    and             t0, t0, l1          // frac = in & bit_mask
    padd.16         t4, t2, a3
    pclipi.s16      t2, t2, 8           // __SSAT(8)
    pclipi.s16      t4, t4, 8

    zextb           t6, t2
    zextb           l5, t4
    ldr.hs          t6, (l2, t6 << 1)   // value
    ldr.hs          l5, (l2, l5 << 1)   // value2

    mulll.s16       l8, l5, t0          // value2 * frac
    psub.16         l7, l4, t0          // full_frac - frac
    mulall.s16.s    l8, l7, t6
    asr             l8, l8, l3          // >> shift_size
    stbi.h          l8, (l0)

    bnezad          t9, .L2

.L3:
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9
    .size           csi_xt800p_nn_activations_direct_q15, .-csi_xt800p_nn_activations_direct_q15
.weak csi_nn_activations_direct_q15
.set  csi_nn_activations_direct_q15, csi_xt800p_nn_activations_direct_q15
.weak csky_dsp2_nn_activations_direct_q15
.set  csky_dsp2_nn_activations_direct_q15, csi_xt800p_nn_activations_direct_q15
