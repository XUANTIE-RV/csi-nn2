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
 * @file     shl_xt800v_fully_connected_q7.S
 * @brief    Q7 basic fully-connected layer function.
 * @version  V1.0
 * @date     19. Mar 2018
 ******************************************************************************/

/*
 * shl_xt800v_status
 * shl_xt800v_fully_connected_q7(const q7_t * pV,
 *                      const q7_t * pM,
 *                      const uint16_t dim_vec,
 *                      const uint16_t num_of_rows,
 *                      const uint16_t bias_shift,
 *                      const uint16_t out_shift,
 *                      const q7_t * bias,
 *                      q7_t * pOut)
 */

    .file           "shl_xt800v_fully_connected_q7.S"
    .section        .text.shl_xt800v_fully_connected_q7,"ax",@progbits
    .align          2
    .global         shl_xt800v_fully_connected_q7
    .type           shl_xt800v_fully_connected_q7, @function

shl_xt800v_fully_connected_q7:
    push            l0, l1, l2, l3, l4, l5, l6
    subi            sp, sp, 64
    vstm.8          vr8-vr11, (sp)
    subi            sp, sp, 16
    vstm.8          vr12-vr12, (sp)

    ld.h            l0, (sp, 0x6c)      // bias_shift
    vdupg.32        vr10, l0
    ld.h            l1, (sp, 0x70)      // out_shift
    vdupg.32        vr11, l1
    movi            t0, 1
    subi            l6, l1, 1
    lsl             l6, t0, l6          // round value
    vdupg.32        vr12, l6
    ld.w            l2, (sp, 0x74)      // *bias
    ld.w            l3, (sp, 0x78)      // *pOut
    mov             l1, a1

    lsri            t0, a3, 2           // rowCnt = num_of_rows >> 4u
    bez             t0, .L5

.L0:
    vldu.8.4        vr0, (l2)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr5, vr0
    vshl.s32        vr5, vr5, vr10      // sum0,  ... sum3
    vadd.s32.s      vr5, vr5, vr12      // round
    vmovi.8         vr6, 0
    vmovi.8         vr7, 0
    vmovi.8         vr8, 0
    vmovi.8         vr9, 0

    mov             l4, a0              // pA     = pV

    lsri            t1, a2, 4           // colCnt = dim_vec >> 4u
    bez             t1, .L2

.L1:
    mov             l5, a1              // pB     = pM
    vldmu.8         vr0-vr0, (l4)       // x0, ... x15
    vldmru.8        vr1-vr4, (l5), a2   // y0, ... y15
    vmulacaa.s8     vr6, vr0, vr1
    vmulacaa.s8     vr7, vr0, vr2
    vmulacaa.s8     vr8, vr0, vr3
    vmulacaa.s8     vr9, vr0, vr4

    addi            a1, a1, 16
    bnezad          t1, .L1

.L2:
    andi            t1, a2, 15          //  colCnt = dim_vec % 4u
    bez             t1, .L4

.L3:
    mov             l5, a1
    vldx.8          vr0, (l4), t1
    vldx.8          vr1, (l5), t1
    addu            l5, l5, a2
    vldx.8          vr2, (l5), t1
    addu            l5, l5, a2
    vldx.8          vr3, (l5), t1
    addu            l5, l5, a2
    vldx.8          vr4, (l5), t1

    vmulacaa.s8     vr6, vr0, vr1
    vmulacaa.s8     vr7, vr0, vr2
    vmulacaa.s8     vr8, vr0, vr3
    vmulacaa.s8     vr9, vr0, vr4

.L4:
    vpadd.s32.s     vr0, vr6, vr7
    vpadd.s32.s     vr1, vr8, vr9
    vpadd.s32.s     vr0, vr0, vr1
    vadd.s32.s      vr0, vr0, vr5
    vshr.s32        vr0, vr0, vr11
    vmov.s32.sl     vr0, vr0, vr0
    vmov.s16.sl     vr0, vr0, vr0
    vstu.8.4        vr0, (l3)

    lsli            t1, a2, 2
    addu            l1, l1, t1
    mov             a1, l1
    bnezad          t0, .L0

.L5:
    andi            t0, a3, 3          //  rowCnt = num_of_rows % 16u
    bez             t0, .L10

.L12:
    vldu.8.1        vr0, (l2)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr0, vr0
    vshl.s32        vr6, vr0, vr10

    mov             l4, a0              // pA     = pV
    mov             l5, a1

    lsri            t1, a2, 4           // colCnt = dim_vec >> 5u
    bez             t1, .L7

.L6:
    vldmu.8         vr0-vr0, (l4)
    vldmu.8         vr2-vr2, (l5)
    vmulacaa.s8     vr6, vr0, vr2

    bnezad          t1, .L6

.L7:
    andi            t1, a2, 15          // colCnt = dim_vec % 32u
    bez             t1, .L9

.L8:
    vldx.8          vr0, (l4), t1
    vldx.8          vr1, (l5), t1
    addu            l5, l5, t1
    vmulacaa.s8     vr6, vr0, vr1

.L9:
    vpadd.s32.s     vr0, vr6, vr6
    vpadd.s32.s     vr0, vr0, vr0
    vadd.s32.s      vr0, vr0, vr12      // sum
    vshr.s32        vr0, vr0, vr11
    vclip.s32       vr0, vr0, 8
    vstu.8.1        vr0, (l3)

    mov             a1, l5
    bnezad          t0, .L12

.L10:
    vldmu.8         vr12-vr12, (sp)
    vldmu.8         vr8-vr11, (sp)
    pop             l0, l1, l2, l3, l4, l5, l6
    .size           shl_xt800v_fully_connected_q7, .-shl_xt800v_fully_connected_q7

.weak csky_vdsp2_fully_connected_q7
.set  csky_vdsp2_fully_connected_q7, shl_xt800v_fully_connected_q7
