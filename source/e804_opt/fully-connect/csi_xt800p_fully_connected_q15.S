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
 * @file     csi_xt800p_fully_connected_q15.S
 * @brief    Q15 basic fully-connected layer function.
 * @version  V1.0
 * @date     31. May 2018
 ******************************************************************************/

/*
 * csi_xt800p_status
 * csi_xt800p_fully_connected_q15(const q15_t * pV,
 *                      const q15_t * pM,
 *                      const uint16_t dim_vec,
 *                      const uint16_t num_of_rows,
 *                      const uint16_t bias_shift,
 *                      const uint16_t out_shift,
 *                      const q15_t * bias,
 *                      q15_t * pOut)
 */

    .file           "csi_xt800p_fully_connected_q15.S"
    .section        .text.csi_xt800p_fully_connected_q15,"ax",@progbits
    .align          2
    .global         csi_xt800p_fully_connected_q15
    .type           csi_xt800p_fully_connected_q15, @function

csi_xt800p_fully_connected_q15:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    ld.h            l0, (sp, 0x2c)      // bias_shift
    ld.h            l1, (sp, 0x30)      // out_shift
    movi            t0, 1
    subi            t1, l1, 1
    lsl             l2, t0, t1          // round value
    ld.w            l3, (sp, 0x34)      // *bias
    ld.w            l4, (sp, 0x38)      // *pOut
    mov             l5, a1
    lsli            l6, a2, 1

    lsri            t9, a3, 2           // rowCnt = num_of_rows >> 2u
    bez             t9, .L5

.L0:
    pldbi.d         t0, (l3)            // bias
    sexth           l7, t0
    asri            l8, t0, 16
    sexth           l9, t1
    asri            lr, t1, 16
    lsl.s32.s       l7, l7, l0          // bias << bias_shift
    lsl.s32.s       l8, l8, l0
    lsl.s32.s       l9, l9, l0
    lsl.s32.s       lr, lr, l0
    addu            l7, l7, l2          // + NN_ROUND
    addu            l8, l8, l2
    addu            l9, l9, l2
    addu            lr, lr, l2

    mov             t7, a0              // pA     = pV

    lsri            t8, a2, 2           // colCnt = dim_vec >> 2u
    bez             t8, .L2

.L1:
    mov             t6, a1              // pB     = pM
    pldbi.d         t0, (t7)            // x0, ..., x3
    ldbir.w         t2, (t6), l6        // y00, ..., y03
    ldbir.w         t3, (t6), l6
    ldbir.w         t4, (t6), l6
    ldbir.w         t5, (t6), l6

    mulaca.s16.s    l7, t0, t2
    mulaca.s16.s    l8, t0, t3          // y10, ..., y13
    mulaca.s16.s    l9, t0, t4          // y20, ..., y23
    mulaca.s16.s    lr, t0, t5          // y30, ..., y33

    addi            a1, a1, 4
    mov             t6, a1
    ldbir.w         t2, (t6), l6        // y00, ..., y03
    ldbir.w         t3, (t6), l6
    ldbir.w         t4, (t6), l6
    ldbir.w         t5, (t6), l6

    mulaca.s16.s    l7, t1, t2
    mulaca.s16.s    l8, t1, t3          // y10, ..., y13
    mulaca.s16.s    l9, t1, t4          // y20, ..., y23
    mulaca.s16.s    lr, t1, t5          // y30, ..., y33

    addi            a1, a1, 4
    bnezad          t8, .L1

.L2:
    andi            t8, a2, 3           //  colCnt = dim_vec % 8u
    bez             t8, .L4

.L3:
    mov             t6, a1
    ldbi.h          t0, (t7)            // x0
    ldbir.h         t2, (t6), l6        // y00
    ldbir.h         t3, (t6), l6        // y10
    ldbir.h         t4, (t6), l6        // y20
    ldbir.h         t5, (t6), l6        // y30
    addi            a1, a1, 2

    mulaca.s16.s    l7, t0, t2
    mulaca.s16.s    l8, t0, t3          // y10
    mulaca.s16.s    l9, t0, t4          // y20
    mulaca.s16.s    lr, t0, t5          // y30

    bnezad          t8, .L3

.L4:
    asr             l7, l7, l1
    asr             l8, l8, l1
    asr             l9, l9, l1
    asr             lr, lr, l1
    clipi.s32       l7, l7, 16
    clipi.s32       l8, l8, 16
    clipi.s32       l9, l9, 16
    clipi.s32       lr, lr, 16
    pkgll           l7, l7, l8
    pkgll           l8, l9, lr
    stbi.w          l7, (l4)
    stbi.w          l8, (l4)

    lsli            t1, a2, 3
    addu            l5, l5, t1
    mov             a1, l5
    bnezad          t9, .L0

.L5:
    andi            t9, a3, 3           //  rowCnt = num_of_rows % 8u
    bez             t9, .L10
    mov             l9, a1

.L12:
    ldbi.hs         l7, (l3)
    lsl.s32.s       l7, l7, l0
    addu            l7, l7, l2

    mov             l8, a0              // pA     = pV

    lsri            t8, a2, 2           // colCnt = dim_vec >> 5u
    bez             t8, .L7

.L6:
    pldbi.d         t0, (l8)
    pldbi.d         t2, (l9)
    mulaca.s16.s    l7, t0, t2
    mulaca.s16.s    l7, t1, t3

    bnezad          t8, .L6

.L7:
    andi            t8, a2, 3          // colCnt = dim_vec % 32u
    bez             t8, .L9

.L8:
    ldbi.h          t0, (l8)
    ldbi.h          t1, (l9)
    mulaca.s16.s    l7, t0, t1

    bnezad          t8, .L8

.L9:
    asr             l7, l7, l1
    clipi.s32       l7, l7, 16
    stbi.h          l7, (l4)

    bnezad          t9, .L12

.L10:
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           csi_xt800p_fully_connected_q15, .-csi_xt800p_fully_connected_q15
.weak csi_fully_connected_q15
.set  csi_fully_connected_q15, csi_xt800p_fully_connected_q15
.weak csky_dsp2_fully_connected_q15
.set  csky_dsp2_fully_connected_q15, csi_xt800p_fully_connected_q15
