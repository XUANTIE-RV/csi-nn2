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
 * @file     csi_xt800v_relu_q15.S
 * @brief    Q15 version of ReLU.
 * @version  V1.0
 * @date     01. June 2018
 ******************************************************************************/

/*
 *void csi_xt800v_relu_q15(q15_t * data,
 *                   uint16_t size)
 */

    .file           "csi_xt800v_relu_q15.S"
    .section        .text.csi_xt800v_relu_q15,"ax",@progbits
    .align          2
    .global         csi_xt800v_relu_q15
    .type           csi_xt800v_relu_q15, @function

csi_xt800v_relu_q15:
    vmovi.8         vr7, 0
    lsri            t0, a1, 5
    bez             t0, .L1

.L0:
    vldm.16         vr0-vr3, (a0)
    vmax.s16        vr0, vr0, vr7
    vmax.s16        vr1, vr1, vr7
    vmax.s16        vr2, vr2, vr7
    vmax.s16        vr3, vr3, vr7

    vstmu.16        vr0-vr3, (a0)

    bnezad          t0, .L0

.L1:
    andi            t1, a1, 31
    lsri            t0, t1, 3
    bez             t0, .L3

.L2:
    vldm.16         vr0-vr0, (a0)
    vmax.s16        vr0, vr0, vr7
    vstmu.16        vr0-vr0, (a0)

    bnezad          t0, .L2


.L3:
    andi            t0, t1, 7
    bez             t0, .L4
    vldx.16         vr0, (a0), t0
    vmax.s16        vr0, vr0, vr7
    vstx.16         vr0, (a0), t0

.L4:
    rts
    .size           csi_xt800v_relu_q15, .-csi_xt800v_relu_q15
.weak csi_relu_q15
.set  csi_relu_q15, csi_xt800v_relu_q15
.weak csky_vdsp2_relu_q15
.set  csky_vdsp2_relu_q15, csi_xt800v_relu_q15
