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
 * @file     shl_xt800p_fully_connected_mat_q7_vec_q15.S
 * @brief    Mixed Q15-Q7 fully-connected layer function.
 * @version  V1.0
 * @date     31. May 2018
 ******************************************************************************/

/*
 * shl_xt800p_status
 * shl_xt800p_fully_connected_mat_q7_vec_q15(const q15_t * pV,
 *                      const q7_t * pM,
 *                      const uint16_t dim_vec,
 *                      const uint16_t num_of_rows,
 *                      const uint16_t bias_shift,
 *                      const uint16_t out_shift,
 *                      const q7_t * bias,
 *                      q15_t * pOut)
 */

    .file           "shl_xt800p_fully_connected_mat_q7_vec_q15.S"
    .section        .text.shl_xt800p_fully_connected_mat_q7_vec_q15,"ax",@progbits
    .align          2
    .global         shl_xt800p_fully_connected_mat_q7_vec_q15
    .type           shl_xt800p_fully_connected_mat_q7_vec_q15, @function

shl_xt800p_fully_connected_mat_q7_vec_q15:
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
    pldbi.d         t0, (t7)            // x0, ..., x3
    ldbir.w         t2, (t6), a2        // y00, ..., y03
    ldbir.w         t3, (t6), a2

    pext.s8.e       t4, t2
    mulaca.s16.s    l6, t0, t4
    mulaca.s16.s    l6, t1, t5          // y10, ..., y13

    pext.s8.e       t4, t3
    mulaca.s16.s    l7, t0, t4          // y20, ..., y23
    mulaca.s16.s    l7, t1, t5          // y30, ..., y33

    ldbir.w         t2, (t6), a2        // y00, ..., y03
    ldbir.w         t3, (t6), a2

    pext.s8.e       t4, t2
    mulaca.s16.s    l8, t0, t4
    mulaca.s16.s    l8, t1, t5          // y10, ..., y13

    pext.s8.e       t4, t3
    mulaca.s16.s    l9, t0, t4          // y20, ..., y23
    mulaca.s16.s    l9, t1, t5          // y30, ..., y33

    addi            a1, a1, 4
    bnezad          t8, .L1

.L2:
    andi            t8, a2, 3           //  colCnt = dim_vec % 8u
    bez             t8, .L4

.L3:
    mov             t6, a1
    ldbi.hs         t0, (t7)            // x0
    ldbir.bs        t2, (t6), a2        // y00
    ldbir.bs        t3, (t6), a2        // y10
    ldbir.bs        t4, (t6), a2        // y20
    ldbir.bs        t5, (t6), a2        // y30
    addi            a1, a1, 1

    mula.32.l       l6, t0, t2
    mula.32.l       l7, t0, t3          // y10
    mula.32.l       l8, t0, t4          // y20
    mula.32.l       l9, t0, t5          // y30

    bnezad          t8, .L3

.L4:
    asr             l6, l6, l1
    asr             l7, l7, l1
    asr             l8, l8, l1
    asr             l9, l9, l1
    clipi.s32       l6, l6, 16
    clipi.s32       l7, l7, 16
    clipi.s32       l8, l8, 16
    clipi.s32       l9, l9, 16
    pkgll           l6, l6, l7
    pkgll           l7, l8, l9
    stbi.w          l6, (l4)
    stbi.w          l7, (l4)

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

    lsri            t8, a2, 2           // colCnt = dim_vec >> 5u
    bez             t8, .L7

.L6:
    pldbi.d         t0, (l8)
    ldbi.w          t2, (l9)

    pext.s8.e       t4, t2
    mulaca.s16.s    l7, t0, t4
    mulaca.s16.s    l7, t1, t5

    bnezad          t8, .L6

.L7:
    andi            t8, a2, 3          // colCnt = dim_vec % 32u
    bez             t8, .L9

.L8:
    ldbi.hs         t0, (l8)
    ldbi.bs         t1, (l9)
    mula.32.l       l7, t0, t1

    bnezad          t8, .L8

.L9:
    asr             l7, l7, l1
    clipi.s32       l7, l7, 16
    stbi.h          l7, (l4)

    bnezad          t9, .L12

.L10:
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           shl_xt800p_fully_connected_mat_q7_vec_q15, .-shl_xt800p_fully_connected_mat_q7_vec_q15

.weak csky_dsp2_fully_connected_mat_q7_vec_q15
.set  csky_dsp2_fully_connected_mat_q7_vec_q15, shl_xt800p_fully_connected_mat_q7_vec_q15
