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
 * @file     shl_xt800v_relu_q7.S
 * @brief    Q15 version of ReLU.
 * @version  V1.0
 * @date     01. June 2018
 ******************************************************************************/

/*
 *void shl_xt800v_relu_q7(q7_t * data,
 *                   uint8_t size)
 */

    .file           "shl_xt800v_relu_q7.S"
    .section        .text.shl_xt800v_relu_q7,"ax",@progbits
    .align          2
    .global         shl_xt800v_relu_q7
    .type           shl_xt800v_relu_q7, @function

shl_xt800v_relu_q7:
    vmovi.8         vr7, 0
    lsri            t0, a1, 6
    bez             t0, .L1

.L0:
    vldm.8          vr0-vr3, (a0)
    vmax.s8         vr0, vr0, vr7
    vmax.s8         vr1, vr1, vr7
    vmax.s8         vr2, vr2, vr7
    vmax.s8         vr3, vr3, vr7

    vstmu.8         vr0-vr3, (a0)

    bnezad          t0, .L0

.L1:
    andi            t1, a1, 63
    lsri            t0, t1, 4
    bez             t0, .L3

.L2:
    vldm.8          vr0-vr0, (a0)
    vmax.s8         vr0, vr0, vr7
    vstmu.8         vr0-vr0, (a0)

    bnezad          t0, .L2


.L3:
    andi            t0, t1, 15
    bez             t0, .L4
    vldx.8          vr0, (a0), t0
    vmax.s8         vr0, vr0, vr7
    vstx.8          vr0, (a0), t0

.L4:
    rts
    .size           shl_xt800v_relu_q7, .-shl_xt800v_relu_q7
.weak csinn_relu_q7
.set  csinn_relu_q7, shl_xt800v_relu_q7
.weak csky_vdsp2_relu_q7
.set  csky_vdsp2_relu_q7, shl_xt800v_relu_q7
