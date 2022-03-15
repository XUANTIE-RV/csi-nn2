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
 * @file     csi_xt800p_convolve_1x1_HWC_q7_fast.S
 * @brief    Fast Q7 vresion of 1x1 convolution (non-square shape).
 * @version  V1.0
 * @date     05. June 2018
 ******************************************************************************/

/*
 * void csi_xt800p_convolve_1x1_HWC_q7_fast(const q7_t * Im_in,
 *                                             const uint16_t dim_im_in_x,
 *                                             const uint16_t dim_im_in_y,
 *                                             const uint16_t ch_im_in,
 *                                             const q7_t * wt,
 *                                             const uint16_t ch_im_out,
 *                                             const q7_t * bias,
 *                                             const uint16_t bias_shift,
 *                                             const uint16_t out_shift,
 *                                             q7_t * Im_out,
 *                                             const uint16_t dim_im_out_x,
 *                                             const uint16_t dim_im_out_y,
 *                                             q15_t * bufferA)
 *
 */

    .file           "csi_xt800p_convolve_1x1_HWC_q7_fast.S"
    .section        .text.csi_xt800p_convolve_HWC_q7_fast,"ax",@progbits
    .align          2
    .global         csi_xt800p_convolve_1x1_HWC_q7_fast
    .type           csi_xt800p_convolve_1x1_HWC_q7_fast, @function

csi_xt800p_convolve_1x1_HWC_q7_fast:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    subi            sp, sp, 8
    st.w            a0, (sp, 0x0)
    st.w            a1, (sp, 0x4)
    ld.hs           l3, (sp, 0x44)      // out_shift
    movi            l5, 1
    subi            l6, l3, 1
    lsl             l5, l5, l6
    ld.hs           l1, (sp, 0x38)      // ch_im_out
    ld.hs           l6, (sp, 0x40)      // bias_shift
    ld.w            l4, (sp, 0x48)      // *im_out
    ld.w            l7, (sp, 0x54)      // *bufferA

    movi            t0, 0               // i_out_y

.L0:
    ld.hs           t9, (sp, 0x50)      // dim_im_out_y
    cmplt           t0, t9              // i_out_y < dim_im_out_y
    bf              .L18

    movi            t1, 0               // i_out_x

.L1:
    ld.hs           t9, (sp, 0x4c)      // dim_im_out_x
    cmplt           t1, t9              // i_out_x < dim_im_out_x
    bf              .L17

    ld.w            a0, (sp, 0x0)
    ld.w            a1, (sp, 0x4)
    mult            t6, t0, a1          // (i_out_y * dim_im_in_x + i_out_x)*ch_im_in
    addu            t6, t6, t1
    mult            t6, t6, a3
    addu            t6, t6, a0          // pSrc

    lsri            t2, a3, 3           // ch_im_in >> 4u
    bez             t2, .L3

.L2:
    pldbi.d         l8, (t6)
    stbi.w          l8, (l7)
    stbi.w          l9, (l7)

    bnezad          t2, .L2

.L3:
    andi            t2, a3, 7          // ch_im_in & 15u
    bez             t2, .L4

.L24:
    ldbi.b          l8, (t6)
    stbi.b          l8, (l7)

    bnezad          t2, .L24

.L4:
    ld.w            l8, (sp, 0x54)      // *bufferA
    ixh             l8, l8, a3
    cmpne           l7, l8
    bt              .L16

    ld.w            l2, (sp, 0x3c)      // bias
    ld.w            t8, (sp, 0x34)      // wt
    addu            t9, t8, a3
    lsri            t6, l1, 1           // rowCnt = ch_im_out >> 1u
    bez             t6, .L10

.L5:
    ld.w            l7, (sp, 0x54)      // *bufferA
    addu            lr, l7, a3          // *pB2 = pB + numCol_A

    ldbi.bs         t2, (l2)
    ldbi.bs         t3, (l2)
    lsl.s32.s       t2, t2, l6          // sum + bias
    lsl.s32.s       t3, t3, l6
    add.s32.s       t2, t2, l5          // + NN_ROUND
    add.s32.s       t3, t3, l5
    mov             t4, t2
    mov             t5, t3

    lsri            t7, a3, 2           // colCnt = numCol_A >> 4u
    bez             t7, .L7

.L6:
    ldbi.w          l8, (l7)            // load 4 data from col1
    ldbi.w          l9, (lr)            // load 4 data from col2
    ldbi.w          a0, (t8)            // load 4 data from kernel 1 and 2
    ldbi.w          a1, (t9)
    mulaca.s8       l0, a0, l8
    mulaca.s8       a0, a0, l9
    mulaca.s8       l8, a1, l8
    mulaca.s8       l9, a1, l9

    add.s32.s       t2, t2, l0
    add.s32.s       t3, t3, l8
    add.s32.s       t4, t4, a0
    add.s32.s       t5, t5, l9

    bnezad          t7, .L6

.L7:
    andi            t7, a3, 3           // colCnt = numCol_A & 15u
    bez             t7, .L9

.L8:
    ldbi.bs         l8, (l7)            // load 1 data from col1
    ldbi.bs         l9, (lr)            // load 1 data from col2
    ldbi.bs         a0, (t8)            // load 1 data from kernel 1 and 2
    ldbi.bs         a1, (t9)
    mula.32.l       t2, a0, l8
    mula.32.l       t3, a0, l9
    mula.32.l       t4, a1, l8
    mula.32.l       t5, a1, l9

    bnezad          t7, .L8

.L9:
    asr             t2, t2, l3
    asr             t3, t3, l3
    asr             t4, t4, l3
    asr             t5, t5, l3
    clipi.s32       t2, t2, 8
    clipi.s32       t3, t3, 8
    clipi.s32       t4, t4, 8
    clipi.s32       t5, t5, 8
    pkgll           t2, t2, t3
    pkgll           t3, t4, t5
    narl            t2, t2, t2
    narl            t3, t3, t3
    stbi.h          t2, (l4)
    addu            t4, l4, l1
    subi            t4, t4, 2
    stbi.h          t3, (t4)

    mov             t8, t9
    addu            t9, t9, a3
    bnezad          t6, .L5

.L10:
    andi            t6, l1, 1           // ch_im_out % 0x4u
    bez             t6, .L15

.L11:
    ld.w            l7, (sp, 0x54)      // *bufferA
    addu            lr, l7, a3          // *pB2 = pB + numCol_A

    ldbi.bs         t2, (l2)
    lsl.s32.s       t2, t2, l6          // sum + bias
    add.s32.s       t2, t2, l5          // + NN_ROUND
    mov             t3, t2

    lsri            t7, a3, 2           // colCnt = numCol_A >> 2u
    bez             t7, .L13

.L12:
    ldbi.w          t4, (l7)            // load 4 data from col1
    ldbi.w          t5, (lr)            // load 4 data from col2
    ldbi.w          a0, (t8)            // load 4 data from kernel 1
    mulaca.s8       t4, t4, a0
    mulaca.s8       t5, t5, a0

    add.s32.s       t2, t2, t4
    add.s32.s       t3, t3, t5

    bnezad          t7, .L12

.L13:
    andi            t7, a3, 3           // colCnt = numCol_A & 15u
    bez             t7, .L14

.L25:
    ldbi.bs         t4, (l7)            // load 4 data from col1
    ldbi.bs         t5, (lr)            // load 4 data from col2
    ldbi.bs         a0, (t8)            // load 4 data from kernel 1
    mula.32.l       t2, t4, a0
    mula.32.l       t3, t5, a0

    bnezad          t7, .L25

.L14:
    asr             t2, t2, l3
    asr             t3, t3, l3
    clipi.s32       t2, t2, 8
    clipi.s32       t3, t3, 8
    stbi.b          t2, (l4)
    addu            t4, l4, l1
    subi            t4, t4, 1
    stbi.b          t3, (t4)

    bnezad          t6, .L11

.L15:
    addu            l4, l4, l1
    ld.w            l7, (sp, 0x54)      // *bufferA

.L16:
    addi            t1, t1, 1
    br              .L1

.L17:
    addi            t0, t0, 1
    br              .L0

    /* check for left-over */
.L18:
    ld.w            l8, (sp, 0x54)      // *bufferA
    cmpne           l7, l8
    bf              .L23

    ld.w            t9, (sp, 0x34)      // wt
    ld.w            l2, (sp, 0x3c)      // bias
    movi            t1, 0

.L19:
    cmplt           t1, l1              // i < ch_im_out
    bf              .L23

    ldbi.bs         t2, (l2)
    lsl.s32.s       t2, t2, l6          // sum + bias
    add.s32.s       t2, t2, l5          // + NN_ROUND

    ld.w            l7, (sp, 0x54)      // *bufferA
    lsri            t4, a3, 2           // colCnt
    bez             t4, .L21

.L20:
    ldbi.w          l8, (l7)            // col
    ldbi.w          l9, (t9)            // kernel
    mulaca.s8       l8, l8, l9
    add.s32.s       t2, t2, l8

    bnezad          t4, .L20

.L21:
    andi            t4, a3, 3           // colCnt
    bez             t4, .L22

.L26:
    ldbi.bs         l8, (l7)            // col
    ldbi.bs         l9, (t9)            // kernel
    mula.32.l       t2, l8, l9

    bnezad          t4, .L26

.L22:
    asr             t2, t2, l3
    clipi.s32       t2, t2, 8
    stbi.b          t2, (l4)

    addi            t1, t1, 1
    br              .L19

.L23:
    addi            sp, sp, 8
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           csi_xt800p_convolve_1x1_HWC_q7_fast, .-csi_xt800p_convolve_1x1_HWC_q7_fast

.weak csi_convolve_1x1_HWC_q7_fast
.set  csi_convolve_1x1_HWC_q7_fast, csi_xt800p_convolve_1x1_HWC_q7_fast
.weak csky_dsp2_convolve_1x1_HWC_q7_fast
.set  csky_dsp2_convolve_1x1_HWC_q7_fast, csi_xt800p_convolve_1x1_HWC_q7_fast
