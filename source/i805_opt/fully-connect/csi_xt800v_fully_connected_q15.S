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
 * @file     csi_xt800v_fully_connected_q15.S
 * @brief    Q15 basic fully-connected layer function.
 * @version  V1.0
 * @date     31. May 2018
 ******************************************************************************/

/*
 * csi_xt800v_status
 * csi_xt800v_fully_connected_q15(const q15_t * pV,
 *                      const q15_t * pM,
 *                      const uint16_t dim_vec,
 *                      const uint16_t num_of_rows,
 *                      const uint16_t bias_shift,
 *                      const uint16_t out_shift,
 *                      const q15_t * bias,
 *                      q15_t * pOut)
 */

    .file           "csi_xt800v_fully_connected_q15.S"
    .section        .text.csi_xt800v_fully_connected_q15,"ax",@progbits
    .align          2
    .global         csi_xt800v_fully_connected_q15
    .type           csi_xt800v_fully_connected_q15, @function

csi_xt800v_fully_connected_q15:
    push            l0, l1, l2, l3, l4, l5, l6
    subi            sp, sp, 64
    vstm.8          vr8-vr11, (sp)
    subi            sp, sp, 64
    vstm.8          vr12-vr15, (sp)

    ld.h            l0, (sp, 0x9c)      // bias_shift
    vdupg.32        vr10, l0
    lsli            l0, a2, 1
    ld.h            l6, (sp, 0xa0)      // out_shift
    movi            t0, 1
    subi            t1, l6, 1
    lsl             t1, t0, t1          // round value
    vdupg.32        vr11, t1
    ld.w            l2, (sp, 0xa4)      // *bias
    ld.w            l3, (sp, 0xa8)      // *pOut
    mov             l1, a1

    lsri            t0, a3, 2           // rowCnt = num_of_rows >> 3u
    bez             t0, .L5

.L0:
    vldu.16.4       vr0, (l2)
    vmov.s16.e      vr8, vr0
    vshl.s32        vr9, vr8, vr10      // sum0,  ... sum3
    vadd.s32.s      vr9, vr9, vr11      // round
    vmovi.8         vr5, 0
    vmovi.8         vr6, 0
    vmovi.8         vr7, 0
    vmovi.8         vr8, 0
    vmovi.8         vr12, 0
    vmovi.8         vr13, 0
    vmovi.8         vr14, 0
    vmovi.8         vr15, 0

    mov             l4, a0              // pA     = pV

    lsri            t1, a2, 3           // colCnt = dim_vec >> 3u
    bez             t1, .L2

.L1:
    mov             l5, a1              // pB     = pM
    vldmu.16        vr0-vr0, (l4)       // x0, ..., x7
    vldmru.16       vr1-vr4, (l5), l0   // y00, ..., y07

    vmula.s16.e     vr5, vr0, vr1
    vmula.s16.e     vr7, vr0, vr2       // y10, ..., y17
    vmula.s16.e     vr12, vr0, vr3      // y20, ..., y27
    vmula.s16.e     vr14, vr0, vr4      // y30, ..., y37

    addi            a1, a1, 16
    bnezad          t1, .L1

.L2:
    andi            t1, a2, 7           //  colCnt = dim_vec % 8u
    bez             t1, .L4

.L3:
    mov             l5, a1
    vldx.16         vr0, (l4), t1
    vldx.16         vr1, (l5), t1
    addu            l5, l5, l0
    vldx.16         vr2, (l5), t1
    addu            l5, l5, l0
    vldx.16         vr3, (l5), t1
    addu            l5, l5, l0
    vldx.16         vr4, (l5), t1

    vmula.s16.e     vr5, vr0, vr1
    vmula.s16.e     vr7, vr0, vr2       // y10, ..., y17
    vmula.s16.e     vr12, vr0, vr3      // y20, ..., y27
    vmula.s16.e     vr14, vr0, vr4      // y30, ..., y37

.L4:
    vadd.s32.s      vr0, vr5, vr6
    vadd.s32.s      vr1, vr7, vr8
    vadd.s32.s      vr2, vr12, vr13
    vadd.s32.s      vr3, vr14, vr15
    vdupg.32        vr12, l6
    vpadd.s32.s     vr0, vr0, vr1
    vpadd.s32.s     vr1, vr2, vr3
    vpadd.s32.s     vr0, vr0, vr1
    vadd.s32.s      vr0, vr0, vr9
    vshr.s32        vr0, vr0, vr12
    vmov.s32.sl     vr0, vr0, vr0
    vstu.16.4       vr0, (l3)

    lsli            t1, a2, 3
    addu            l1, l1, t1
    mov             a1, l1
    bnezad          t0, .L0

.L5:
    andi            t0, a3, 3           //  rowCnt = num_of_rows % 8u
    bez             t0, .L10

.L12:
    vldu.16.1       vr0, (l2)
    vmov.s16.e      vr0, vr0
    vshl.s32        vr8, vr0, vr10
    vmovi.8         vr9, 0
    vdupg.32        vr15, l6

    mov             l4, a0              // pA     = pV
    mov             l5, a1

    lsri            t1, a2, 5           // colCnt = dim_vec >> 5u
    bez             t1, .L7

.L6:
    vldmu.16        vr0-vr3, (l4)
    vldmu.16        vr4-vr7, (l5)
    vmula.s16.e     vr8, vr0, vr4
    vmula.s16.e     vr8, vr1, vr5
    vmula.s16.e     vr8, vr2, vr6
    vmula.s16.e     vr8, vr3, vr7

    bnezad          t1, .L6

.L7:
    andi            t2, a2, 31          // colCnt = dim_vec % 32u
    lsri            t1, t2, 3
    bez             t1, .L8

.L11:
    vldmu.16        vr0-vr0, (l4)
    vldmu.16        vr1-vr1, (l5)
    vmula.s16.e     vr8, vr0, vr1

    bnezad          t1, .L11

.L8:
    andi            t1, t2, 7
    bez             t1, .L9
    vldx.16         vr0, (l4), t1
    vldx.16         vr1, (l5), t1
    vmula.s16.e     vr8, vr0, vr1
    ixh             l5, l5, t1

.L9:
    vadd.s32.s      vr8, vr8, vr9
    vpadd.s32.s     vr0, vr8, vr8
    vpadd.s32.s     vr0, vr0, vr0
    vadd.s32.s      vr0, vr0, vr11      // sum
    vshr.s32        vr0, vr0, vr15
    vclip.s32       vr0, vr0, 16
    vstu.16.1       vr0, (l3)

    mov             a1, l5
    bnezad          t0, .L12

.L10:
    vldmu.8         vr12-vr15, (sp)
    vldmu.8         vr8-vr11, (sp)
    pop             l0, l1, l2, l3, l4, l5, l6
    .size           csi_xt800v_fully_connected_q15, .-csi_xt800v_fully_connected_q15
.weak csi_fully_connected_q15
.set  csi_fully_connected_q15, csi_xt800v_fully_connected_q15
.weak csky_vdsp2_fully_connected_q15
.set  csky_vdsp2_fully_connected_q15, csi_xt800v_fully_connected_q15
