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
 * @file     csi_xt800v_nn_activations_q7_fast.S
 * @brief    Q7 neural network activation function using direct table look-up.
 * @version  V1.0
 * @date     05. June 2018
 ******************************************************************************/

.import sigmoidTable_q7
.import tanhTable_q7
/*
 *void csi_xt800v_nn_activations_direct_q7(q7_t * data,
 *                                   uint16_t size,
 *                                   uint16_t int_width,
 *                                   csi_xt800v_nn_activation_type type)
 */

    .file           "csi_xt800v_nn_activations_q7.S"
    .section        .text.csi_xt800v_nn_activations_direct_q7,"ax",@progbits
    .align          2
    .global         csi_xt800v_nn_activations_direct_q7
    .type           csi_xt800v_nn_activations_direct_q7, @function

csi_xt800v_nn_activations_direct_q7:
    push            l0, l1, l2, l3
    movi            l0, 3               // shift_size = 3 - int_width
    subu            l0, l0, a2
    vdupg.8         vr7, l0
    lrw             l0, sigmoidTable_q7
    lrw             l1, tanhTable_q7

    cmplti          a3, 1               // swith(type)
    movt            l2, l0
    movf            l2, l1

    lsri            t0, a1, 4
    bez             t0, .L1

.L0:
    vldm.8          vr0-vr0, (a0)
    vshr.u8         vr0, vr0, vr7       // int >> shift_size
    vlde.8          vr2, (l2, vr0)

    vstmu.8         vr2-vr2, (a0)

    bnezad          t0, .L0

.L1:
    andi            t0, a1, 15
    bez             t0, .L2

    vldx.8          vr0, (a0), t0
    vshr.u8         vr0, vr0, vr7       // int >> shift_size
    vlde.8          vr2, (l2, vr0)
    vstx.8          vr2, (a0), t0

.L2:
    pop             l0, l1, l2, l3
    .size           csi_xt800v_nn_activations_direct_q7, .-csi_xt800v_nn_activations_direct_q7
.weak csi_nn_activations_direct_q7
.set  csi_nn_activations_direct_q7, csi_xt800v_nn_activations_direct_q7
.weak csky_vdsp2_nn_activations_direct_q7
.set  csky_vdsp2_nn_activations_direct_q7, csi_xt800v_nn_activations_direct_q7
