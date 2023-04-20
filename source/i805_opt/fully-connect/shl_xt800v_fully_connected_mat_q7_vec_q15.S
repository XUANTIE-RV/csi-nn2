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
 * @file     shl_xt800v_fully_connected_mat_q7_vec_q15.S
 * @brief    Mixed Q15-Q7 fully-connected layer function.
 * @version  V1.0
 * @date     31. May 2018
 ******************************************************************************/

/*
 * shl_xt800v_status
 * shl_xt800v_fully_connected_mat_q7_vec_q15(const q15_t * pV,
 *                      const q7_t * pM,
 *                      const uint16_t dim_vec,
 *                      const uint16_t num_of_rows,
 *                      const uint16_t bias_shift,
 *                      const uint16_t out_shift,
 *                      const q7_t * bias,
 *                      q15_t * pOut)
 */

    .file           "shl_xt800v_fully_connected_mat_q7_vec_q15.S"
    .section        .text.shl_xt800v_fully_connected_mat_q7_vec_q15,"ax",@progbits
    .align          2
    .global         shl_xt800v_fully_connected_mat_q7_vec_q15
    .type           shl_xt800v_fully_connected_mat_q7_vec_q15, @function

shl_xt800v_fully_connected_mat_q7_vec_q15:
    push            l0, l1, l2, l3, l4, l5, l6
    subi            sp, sp, 64
    vstm.8          vr8-vr11, (sp)
    subi            sp, sp, 64
    vstm.8          vr12-vr15, (sp)

    ld.h            l0, (sp, 0x9c)      // bias_shift
    vdupg.32        vr10, l0
    ld.h            l1, (sp, 0xa0)      // out_shift
    vdupg.32        vr11, l1
    movi            t0, 1
    subi            l6, l1, 1
    lsl             l6, t0, l6          // round value
    vdupg.32        vr15, l6
    ld.w            l2, (sp, 0xa4)      // *bias
    ld.w            l3, (sp, 0xa8)      // *pOut
    mov             l1, a1

    lsri            t0, a3, 4           // rowCnt = num_of_rows >> 4u
    bez             t0, .L5

.L0:
    vldmu.8         vr0-vr0, (l2)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr6, vr0
    vmov.s16.e      vr8, vr1
    vshl.s32        vr6, vr6, vr10      // sum0,  ... sum3
    vshl.s32        vr7, vr7, vr10      // sum4,  ... sum7
    vshl.s32        vr8, vr8, vr10      // sum8,  ... sum11
    vshl.s32        vr9, vr9, vr10      // sum12, ... sum15
    vadd.s32.s      vr6, vr6, vr15      // round
    vadd.s32.s      vr7, vr7, vr15
    vadd.s32.s      vr8, vr8, vr15
    vadd.s32.s      vr9, vr9, vr15

    mov             l4, a0              // pA     = pV

    lsri            t1, a2, 4           // colCnt = dim_vec >> 4u
    bez             t1, .L2

.L1:
    mov             l5, a1              // pB     = pM
    vldmu.16        vr0-vr1, (l4)       // x0, ... x15
    vldmru.8        vr12-vr12, (l5), a2 // y00, ... y015
    vmov.s8.e       vr12, vr12
    vmul.s16.e      vr2, vr0, vr12
    vmula.s16.e     vr2, vr1, vr13
    vadd.s32.s      vr2, vr2, vr3

    vldmru.8        vr12-vr12, (l5), a2 // y10, ... y115
    vmov.s8.e       vr12, vr12
    vmul.s16.e      vr3, vr0, vr12
    vmula.s16.e     vr3, vr1, vr13
    vadd.s32.s      vr3, vr3, vr4

    vldmru.8        vr12-vr12, (l5), a2 // y20, ... y215
    vmov.s8.e       vr12, vr12
    vmul.s16.e      vr4, vr0, vr12
    vmula.s16.e     vr4, vr1, vr13
    vadd.s32.s      vr4, vr4, vr5

    vldmru.8        vr12-vr12, (l5), a2 // y30, ... y315
    vmov.s8.e       vr13, vr12
    vmul.s16.e      vr12, vr0, vr13
    vmula.s16.e     vr12, vr1, vr14
    vadd.s32.s      vr5, vr12, vr13

    vpadd.s32.s     vr2, vr2, vr3
    vpadd.s32.s     vr3, vr4, vr5
    vpadd.s32.s     vr2, vr2, vr3
    vadd.s32.s      vr6, vr6, vr2       // sum0, ... sum3

    vldmru.8        vr12-vr12, (l5), a2 // y00, ... y015
    vmov.s8.e       vr12, vr12
    vmul.s16.e      vr2, vr0, vr12
    vmula.s16.e     vr2, vr1, vr13
    vadd.s32.s      vr2, vr2, vr3

    vldmru.8        vr12-vr12, (l5), a2 // y10, ... y115
    vmov.s8.e       vr12, vr12
    vmul.s16.e      vr3, vr0, vr12
    vmula.s16.e     vr3, vr1, vr13
    vadd.s32.s      vr3, vr3, vr4

    vldmru.8        vr12-vr12, (l5), a2 // y20, ... y215
    vmov.s8.e       vr12, vr12
    vmul.s16.e      vr4, vr0, vr12
    vmula.s16.e     vr4, vr1, vr13
    vadd.s32.s      vr4, vr4, vr5

    vldmru.8        vr12-vr12, (l5), a2 // y30, ... y315
    vmov.s8.e       vr13, vr12
    vmul.s16.e      vr12, vr0, vr13
    vmula.s16.e     vr12, vr1, vr14
    vadd.s32.s      vr5, vr12, vr13

    vpadd.s32.s     vr2, vr2, vr3
    vpadd.s32.s     vr3, vr4, vr5
    vpadd.s32.s     vr2, vr2, vr3
    vadd.s32.s      vr7, vr7, vr2       // sum4, ... sum7

    vldmru.8        vr12-vr12, (l5), a2 // y00, ... y015
    vmov.s8.e       vr12, vr12
    vmul.s16.e      vr2, vr0, vr12
    vmula.s16.e     vr2, vr1, vr13
    vadd.s32.s      vr2, vr2, vr3

    vldmru.8        vr12-vr12, (l5), a2 // y10, ... y115
    vmov.s8.e       vr12, vr12
    vmul.s16.e      vr3, vr0, vr12
    vmula.s16.e     vr3, vr1, vr13
    vadd.s32.s      vr3, vr3, vr4

    vldmru.8        vr12-vr12, (l5), a2 // y20, ... y215
    vmov.s8.e       vr12, vr12
    vmul.s16.e      vr4, vr0, vr12
    vmula.s16.e     vr4, vr1, vr13
    vadd.s32.s      vr4, vr4, vr5

    vldmru.8        vr12-vr12, (l5), a2 // y30, ... y315
    vmov.s8.e       vr13, vr12
    vmul.s16.e      vr12, vr0, vr13
    vmula.s16.e     vr12, vr1, vr14
    vadd.s32.s      vr5, vr12, vr13

    vpadd.s32.s     vr2, vr2, vr3
    vpadd.s32.s     vr3, vr4, vr5
    vpadd.s32.s     vr2, vr2, vr3
    vadd.s32.s      vr8, vr8, vr2       // sum8, ... sum11

    vldmru.8        vr12-vr12, (l5), a2 // y00, ... y015
    vmov.s8.e       vr12, vr12
    vmul.s16.e      vr2, vr0, vr12
    vmula.s16.e     vr2, vr1, vr13
    vadd.s32.s      vr2, vr2, vr3

    vldmru.8        vr12-vr12, (l5), a2 // y10, ... y115
    vmov.s8.e       vr12, vr12
    vmul.s16.e      vr3, vr0, vr12
    vmula.s16.e     vr3, vr1, vr13
    vadd.s32.s      vr3, vr3, vr4

    vldmru.8        vr12-vr12, (l5), a2 // y20, ... y215
    vmov.s8.e       vr12, vr12
    vmul.s16.e      vr4, vr0, vr12
    vmula.s16.e     vr4, vr1, vr13
    vadd.s32.s      vr4, vr4, vr5

    vldmru.8        vr12-vr12, (l5), a2 // y30, ... y315
    vmov.s8.e       vr13, vr12
    vmul.s16.e      vr12, vr0, vr13
    vmula.s16.e     vr12, vr1, vr14
    vadd.s32.s      vr5, vr12, vr13

    vpadd.s32.s     vr2, vr2, vr3
    vpadd.s32.s     vr3, vr4, vr5
    vpadd.s32.s     vr2, vr2, vr3
    vadd.s32.s      vr9, vr9, vr2       // sum0, ... sum3

    addi            a1, a1, 16
    bnezad          t1, .L1

.L2:
    andi            t1, a2, 15          //  colCnt = dim_vec % 15u
    bez             t1, .L4

.L3:
    mov             l5, a1
    vldu.16.1       vr0, (l4)
    vldru.8.1       vr1, (l5), a2
    vldru.8.1       vr2, (l5), a2
    vldru.8.1       vr3, (l5), a2
    vldru.8.1       vr4, (l5), a2
    vpkg.8.4        vr1, vr1, 0, 0

    vldru.8.1       vr2, (l5), a2
    vldru.8.1       vr3, (l5), a2
    vldru.8.1       vr4, (l5), a2
    vldru.8.1       vr5, (l5), a2
    vpkg.8.4        vr1, vr2, 0, 4

    vldru.8.1       vr2, (l5), a2
    vldru.8.1       vr3, (l5), a2
    vldru.8.1       vr4, (l5), a2
    vldru.8.1       vr5, (l5), a2
    vpkg.8.4        vr1, vr2, 0, 8

    vldru.8.1       vr2, (l5), a2
    vldru.8.1       vr3, (l5), a2
    vldru.8.1       vr4, (l5), a2
    vldru.8.1       vr5, (l5), a2
    vpkg.8.4        vr1, vr2, 0, 12

    vmov.s8.e       vr2, vr1
    vmuli.s16.e     vr4, vr2, vr0, 0
    vmuli.s16.e     vr2, vr3, vr0, 0

    vadd.s32.s      vr6, vr6, vr4
    vadd.s32.s      vr7, vr7, vr5
    vadd.s32.s      vr8, vr8, vr2
    vadd.s32.s      vr9, vr9, vr3

    addi            a1, a1, 1
    bnezad          t1, .L3

.L4:
    vshr.s32        vr6, vr6, vr11
    vshr.s32        vr7, vr7, vr11
    vshr.s32        vr8, vr8, vr11
    vshr.s32        vr9, vr9, vr11
    vmov.s32.sl     vr0, vr6, vr7
    vmov.s32.sl     vr1, vr8, vr9
    vstmu.16        vr0-vr1, (l3)

    lsli            t1, a2, 4
    addu            l1, l1, t1
    mov             a1, l1
    bnezad          t0, .L0

.L5:
    andi            t0, a3, 15          //  rowCnt = num_of_rows % 16u
    bez             t0, .L10
    movi            l0, 8

.L12:
    vldu.8.1        vr0, (l2)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr0, vr0
    vshl.s32        vr8, vr0, vr10
    vmovi.8         vr9, 0

    mov             l4, a0              // pA     = pV
    mov             l5, a1

    lsri            t1, a2, 5           // colCnt = dim_vec >> 5u
    bez             t1, .L7

.L6:
    vldmu.16        vr0-vr3, (l4)
    vldmu.8         vr4-vr5, (l5)
    vmov.s8.e       vr6, vr4
    vmula.s16.e     vr8, vr0, vr6
    vmula.s16.e     vr8, vr1, vr7
    vmov.s8.e       vr6, vr5
    vmula.s16.e     vr8, vr2, vr6
    vmula.s16.e     vr8, vr3, vr7

    bnezad          t1, .L6

.L7:
    andi            t2, a2, 31          // colCnt = dim_vec % 32u
    lsri            t1, t2, 3
    bez             t1, .L8

.L11:
    vldmu.16        vr0-vr0, (l4)
    vldx.8          vr1, (l5), l0
    addu            l5, l5, l0
    vmov.s8.e       vr1, vr1
    vmula.s16.e     vr8, vr0, vr1

    bnezad          t1, .L11

.L8:
    andi            t1, t2, 7
    bez             t1, .L10
    vldx.16         vr0, (l4), t1
    vldx.8          vr1, (l5), t1
    addu            l5, l5, t1
    vmov.s8.e       vr1, vr1
    vmula.s16.e     vr8, vr0, vr1

.L9:
    vadd.s32.s      vr8, vr8, vr9
    vpadd.s32.s     vr0, vr8, vr8
    vpadd.s32.s     vr0, vr0, vr0
    vadd.s32.s      vr0, vr0, vr15      // sum
    vshr.s32        vr0, vr0, vr11
    vclip.s32       vr0, vr0, 16
    vstu.16.1       vr0, (l3)

    mov             a1, l5
    bnezad          t0, .L12

.L10:
    vldmu.8         vr12-vr15, (sp)
    vldmu.8         vr8-vr11, (sp)
    pop             l0, l1, l2, l3, l4, l5, l6
    .size           shl_xt800v_fully_connected_mat_q7_vec_q15, .-shl_xt800v_fully_connected_mat_q7_vec_q15

.weak csky_vdsp2_fully_connected_mat_q7_vec_q15
.set  csky_vdsp2_fully_connected_mat_q7_vec_q15, shl_xt800v_fully_connected_mat_q7_vec_q15
