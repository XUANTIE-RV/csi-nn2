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
 * @file     csi_xt800p_nn_activations_q7.S
 * @brief    Q7 neural network activation function using direct table look-up.
 * @version  V1.0
 * @date     05. June 2018
 ******************************************************************************/

.import sigmoidTable_q7
.import tanhTable_q7
/*
 *void csi_xt800p_nn_activations_direct_q7(q7_t * data,
 *                                   uint16_t size,
 *                                   uint16_t int_width,
 *                                   csi_xt800p_nn_activation_type type)
 */

    .file           "csi_xt800p_nn_activations_q7.S"
    .section        .text.csi_xt800p_nn_activations_direct_q7,"ax",@progbits
    .align          2
    .global         csi_xt800p_nn_activations_direct_q7
    .type           csi_xt800p_nn_activations_direct_q7, @function

csi_xt800p_nn_activations_direct_q7:
    push            l0, l1, l2, l3, l4, l5, l6, l7
    movi            l0, 3               // shift_size = 3 - int_width
    subu            t2, l0, a2
    lrw             l0, sigmoidTable_q7
    lrw             l1, tanhTable_q7
    movi            t1, 0
    mov             l3, a0

    cmplti          a3, 1               // swith(type)
    movt            l2, l0
    movf            l2, l1

    lsri            t0, a1, 3
    bez             t0, .L1

.L0:
    pldbi.d         t4, (a0)
    pext.u8.e       t6, t4
    pext.u8.e       t8, t5
    plsr.u16        t6, t6, t2          // int >> shift_size
    plsr.u16        t7, t7, t2
    plsr.u16        t8, t8, t2
    plsr.u16        t9, t9, t2
    pkgll           l0, t6, t1
    pkghh           l1, t6, t1
    pkgll           t4, t7, t1
    pkghh           t5, t7, t1
    ldr.b           l4, (l2, l0 << 0)        // table lookup
    ldr.b           l5, (l2, l1 << 0)
    ldr.b           l6, (l2, t4 << 0)
    ldr.b           l7, (l2, t5 << 0)
    pkgll           l4, l4, l5
    pkgll           l5, l6, l7
    narl            l4, l4, l5
    stbi.w          l4, (l3)

    pkgll           l0, t8, t1
    pkghh           l1, t8, t1
    pkgll           t4, t9, t1
    pkghh           t5, t9, t1
    ldr.b           l4, (l2, l0 << 0)        // table lookup
    ldr.b           l5, (l2, l1 << 0)
    ldr.b           l6, (l2, t4 << 0)
    ldr.b           l7, (l2, t5 << 0)
    pkgll           l4, l4, l5
    pkgll           l5, l6, l7
    narl            l4, l4, l5
    stbi.w          l4, (l3)

    bnezad          t0, .L0

.L1:
    andi            t0, a1, 7
    bez             t0, .L3

.L2:
    ldbi.b          t4, (a0)
    pext.u8.e       t6, t4
    plsr.u16        t6, t6, t2          // int >> shift_size
    pkgll           l0, t6, t1
    ldr.b           l4, (l2, l0 << 0)        // table lookup
    stbi.b          l4, (l3)

    bnezad          t0, .L2

.L3:
    pop             l0, l1, l2, l3, l4, l5, l6, l7
    .size           csi_xt800p_nn_activations_direct_q7, .-csi_xt800p_nn_activations_direct_q7
.weak csi_nn_activations_direct_q7
.set  csi_nn_activations_direct_q7, csi_xt800p_nn_activations_direct_q7
.weak csky_dsp2_nn_activations_direct_q7
.set  csky_dsp2_nn_activations_direct_q7, csi_xt800p_nn_activations_direct_q7
