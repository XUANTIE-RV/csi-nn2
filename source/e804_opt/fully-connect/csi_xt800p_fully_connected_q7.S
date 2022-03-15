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
 * @file     csi_xt800p_fully_connected_q7.S
 * @brief    Q7 basic fully-connected layer function.
 * @version  V1.0
 * @date     19. Mar 2018
 ******************************************************************************/

/*
 * csi_xt800p_status
 * csi_xt800p_fully_connected_q7(const q7_t * pV,
 *                      const q7_t * pM,
 *                      const uint16_t dim_vec,
 *                      const uint16_t num_of_rows,
 *                      const uint16_t bias_shift,
 *                      const uint16_t out_shift,
 *                      const q7_t * bias,
 *                      q7_t * pOut)
 */

    .file           "csi_xt800p_fully_connected_q7.S"
    .section        .text.csi_xt800p_fully_connected_q7,"ax",@progbits
    .align          2
    .global         csi_xt800p_fully_connected_q7
    .type           csi_xt800p_fully_connected_q7, @function

csi_xt800p_fully_connected_q7:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    ld.h            l0, (sp, 0x2c)      // bias_shift
    ld.h            l1, (sp, 0x30)      // out_shift
    movi            t0, 1
    subi            t1, l1, 1
    lsl             l2, t0, t1          // round value
    ld.w            l3, (sp, 0x34)      // *bias
    ld.w            l4, (sp, 0x38)      // *pOut
    mov             l5, a1

    lsri            t9, a3, 2           // rowCnt = num_of_rows >> 2u
    bez             t9, .L5

.L0:
    ldbi.w          t0, (l3)            // bias
    pext.s8.e       t0, t0
    sexth           l6, t0
    asri            l7, t0, 16
    sexth           l8, t1
    asri            l9, t1, 16
    lsl.s32.s       l6, l6, l0          // bias << bias_shift
    lsl.s32.s       l7, l7, l0
    lsl.s32.s       l8, l8, l0
    lsl.s32.s       l9, l9, l0
    addu            l6, l6, l2          // + NN_ROUND
    addu            l7, l7, l2
    addu            l8, l8, l2
    addu            l9, l9, l2

    mov             t7, a0              // pA     = pV

    lsri            t8, a2, 2           // colCnt = dim_vec >> 2u
    bez             t8, .L2

.L1:
    mov             t6, a1              // pB     = pM
    ldbi.w          t0, (t7)            // x0, ..., x3
    ldbir.w         t2, (t6), a2        // y00, ..., y03
    ldbir.w         t3, (t6), a2
    ldbir.w         t4, (t6), a2
    ldbir.w         t5, (t6), a2

    mulaca.s8       t2, t0, t2
    mulaca.s8       t3, t0, t3          // y10, ..., y13
    mulaca.s8       t4, t0, t4          // y20, ..., y23
    mulaca.s8       t5, t0, t5          // y30, ..., y33

    add.s32.s       l6, l6, t2
    add.s32.s       l7, l7, t3
    add.s32.s       l8, l8, t4
    add.s32.s       l9, l9, t5

    addi            a1, a1, 4
    bnezad          t8, .L1

.L2:
    andi            t8, a2, 3           //  colCnt = dim_vec % 8u
    bez             t8, .L4

.L3:
    mov             t6, a1
    ldbi.b          t0, (t7)            // x0
    ldbir.b         t2, (t6), a2        // y00
    ldbir.b         t3, (t6), a2        // y10
    ldbir.b         t4, (t6), a2        // y20
    ldbir.b         t5, (t6), a2        // y30
    addi            a1, a1, 1

    mulaca.s8       t2, t0, t2
    mulaca.s8       t3, t0, t3          // y10
    mulaca.s8       t4, t0, t4          // y20
    mulaca.s8       t5, t0, t5          // y30

    add.s32.s       l6, l6, t2
    add.s32.s       l7, l7, t3
    add.s32.s       l8, l8, t4
    add.s32.s       l9, l9, t5

    bnezad          t8, .L3

.L4:
    asr             l6, l6, l1
    asr             l7, l7, l1
    asr             l8, l8, l1
    asr             l9, l9, l1
    clipi.s32       l6, l6, 8
    clipi.s32       l7, l7, 8
    clipi.s32       l8, l8, 8
    clipi.s32       l9, l9, 8
    pkgll           l6, l6, l7
    pkgll           l7, l8, l9
    narl            l6, l6, l7
    stbi.w          l6, (l4)

    lsli            t1, a2, 2
    addu            l5, l5, t1
    mov             a1, l5
    bnezad          t9, .L0

.L5:
    andi            t9, a3, 3           //  rowCnt = num_of_rows % 8u
    bez             t9, .L10
    mov             l9, a1

.L12:
    ldbi.bs         l7, (l3)
    lsl.s32.s       l7, l7, l0
    addu            l7, l7, l2

    mov             l8, a0              // pA     = pV

    lsri            t8, a2, 3           // colCnt = dim_vec >> 5u
    bez             t8, .L7

.L6:
    pldbi.d         t0, (l8)
    pldbi.d         t2, (l9)
    mulaca.s8       t0, t0, t2
    mulaca.s8       t1, t1, t3
    add.s32.s       l7, l7, t0
    add.s32.s       l7, l7, t1

    bnezad          t8, .L6

.L7:
    andi            t8, a2, 7          // colCnt = dim_vec % 32u
    bez             t8, .L9

.L8:
    ldbi.b          t0, (l8)
    ldbi.b          t1, (l9)
    mulaca.s8       t0, t0, t1
    add.s32.s       l7, l7, t0

    bnezad          t8, .L8

.L9:
    asr             l7, l7, l1
    clipi.s32       l7, l7, 8
    stbi.b          l7, (l4)

    bnezad          t9, .L12

.L10:
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           csi_xt800p_fully_connected_q7, .-csi_xt800p_fully_connected_q7
.weak csi_fully_connected_q7
.set  csi_fully_connected_q7, csi_xt800p_fully_connected_q7
.weak csky_dsp2_fully_connected_q7
.set  csky_dsp2_fully_connected_q7, csi_xt800p_fully_connected_q7
