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
 * @file     shl_xt800p_convolve_HWC_q15_basic.S
 * @brief    Q7 vresion of convolution.
 * @version  V1.0
 * @date     19. Mar 2018
 ******************************************************************************/

/*
 * shl_xt800p_status
 * shl_xt800p_convolve_HWC_q15_basic(const q15_t * Im_in,
 *                          const uint16_t dim_im_in,
 *                          const uint16_t ch_im_in,
 *                          const q15_t * wt,
 *                          const uint16_t ch_im_out,
 *                          const uint16_t dim_kernel,
 *                          const uint16_t padding,
 *                          const uint16_t stride,
 *                          const q15_t * bias,
 *                          const uint16_t bias_shift,
 *                          const uint16_t out_shift,
 *                          q15_t * Im_out,
 *                          const uint16_t dim_im_out,
 *                          q15_t * bufferA)
 */

    .file           "shl_xt800p_convolve_HWC_q15_basic.S"
    .section        .text.shl_xt800p_convolve_HWC_q15_basic,"ax",@progbits
    .align          2
    .global         shl_xt800p_convolve_HWC_q15_basic
    .type           shl_xt800p_convolve_HWC_q15_basic, @function

shl_xt800p_convolve_HWC_q15_basic:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    subi            sp, sp, 12
    st.w            a0, (sp)
    st.w            a2, (sp, 0x4)
    st.w            a3, (sp, 0x8)
    ld.hs           l1, (sp, 0x3c)      // dim_kernel
    ld.hs           l6, (sp, 0x50)      // out_shift
    movi            l7, 1
    subi            l8, l6, 1
    lsl             l8, l7, l8          // NN_ROUND
    ld.w            l6, (sp, 0x54)      // *im_out
    ld.w            l7, (sp, 0x5c)      // *bufferA
    mult            l9, l1, l1          // ch_im_in * dim_kernel * dim_kernel
    mult            l9, l9, a2

    movi            t0, 0               // i_out_y

.L0:
    ld.hs           t9, (sp, 0x58)      // dim_im_out
    cmplt           t0, t9              // i_out_y < dim_im_out
    bf              .L16

    movi            t1, 0               // i_out_x

.L1:
    ld.hs           t9, (sp, 0x58)      // dim_im_out
    cmplt           t1, t9              // i_out_x < dim_im_out
    bf              .L15

    ld.hs           l3, (sp, 0x44)      // stride
    ld.hs           l2, (sp, 0x40)      // padding
    ld.hs           l1, (sp, 0x3c)      // dim_kernel
    mult            t2, t0, l3          // i_ker_y = i_out_y * stride
    subu            t2, t2, l2
    addu            t3, t2, l1          // i_out_y * stride - padding + dim_kernel

.L2:
    cmplt           t2, t3
    bf              .L13

    ld.hs           l3, (sp, 0x44)      // stride
    ld.hs           l2, (sp, 0x40)      // padding
    ld.hs           l1, (sp, 0x3c)      // dim_kernel
    mult            t4, t1, l3          // i_ker_x = i_out_x * stride
    subu            t4, t4, l2
    addu            t5, t4, l1          // i_out_x * stride - padding + dim_kernel

.L3:
    cmplt           t4, t5
    bf              .L12

    movi            t6, 0
    cmplt           t2, t6
    bt              .L23
    cmphs           t2, a1
    bt              .L23
    cmplt           t4, t6
    bt              .L23
    cmphs           t4, a1
    bt              .L23

.L7:                                    // else branch
    ld.w            a0, (sp, 0x0)
    ld.w            a2, (sp, 0x4)
    mult            t6, t2, a1          // (i_ker_y * dim_im_in + i_ker_x)*ch_im_in
    addu            t6, t6, t4
    mult            t6, t6, a2
    ixh             t6, a0, t6          // pSrc

    lsri            t7, a2, 2           // ch_im_in >> 2u
    bez             t7, .L9

.L8:
    pldbi.d         l0, (t6)
    stbi.w          l0, (l7)
    stbi.w          l1, (l7)

    bnezad          t7, .L8

.L9:
    andi            t7, a2, 3          // ch_im_in & 3u
    bez             t7, .L11

.L10:
    ldbi.h          l0, (t6)
    stbi.h          l0, (l7)

    bnezad          t7, .L10
    br              .L11

.L23:
    movi            l0, 0
    ld.w            a2, (sp, 0x4)
    lsri            t6, a2, 2           // ch_im_in >> 2u(if branch)
    bez             t6, .L5

.L4:
    st.w            l0, (l7, 0x0)       // 0 padding
    st.w            l0, (l7, 0x4)
    addi            l7, l7, 8

    bnezad          t6, .L4

.L5:
    andi            t6, a2, 3          // ch_im_in & 3u
    bez             t6, .L11

.L6:
    stbi.h          l0, (l7)

    bnezad          t6, .L6

.L11:
    addi            t4, t4, 1
    br              .L3

.L12:
    addi            t2, t2, 1
    br              .L2

.L13:
    ld.w            l5, (sp, 0x48)      // *bias
    ld.w            t8, (sp, 0x8)
    ixh             t9, t8, l9
    ld.hs           l0, (sp, 0x38)      // ch_im_out
    lsri            t6, l0, 1           // rowCnt = ch_im_out >> 2u
    bez             t6, .L35

.L30:
    ld.w            l7, (sp, 0x5c)      // *pB = *bufferA

    ld.hs           l4, (sp, 0x4c)      // bias_shift
    ldbi.hs         l0, (l5)
    ldbi.hs         l1, (l5)
    lsl.s32.s       l0, l0, l4          // sum0, sum1  + bias
    lsl.s32.s       l1, l1, l4
    add.s32.s       l0, l0, l8          // + NN_ROUND
    add.s32.s       l1, l1, l8

    lsri            t7, l9, 2           // colCnt = numCol_A >> 2u
    bez             t7, .L32

.L31:
    pldbi.d         a2, (l7)            // load 4 data from col1
    pldbi.d         l2, (t8)            // load 4 data from kernel 1
    ldbi.w          l4, (t9)
    ldbi.w          a0, (t9)

    mulaca.s16.s    l0, a2, l2
    mulaca.s16.s    l0, a3, l3
    mulaca.s16.s    l1, a2, l4
    mulaca.s16.s    l1, a3, a0

    bnezad          t7, .L31

.L32:
    andi            t7, l9, 3           // colCnt = numCol_A & 15u
    bez             t7, .L34

.L33:
    ldbi.h          a2, (l7)            // load 1 data from col1
    ldbi.h          l2, (t8)            // load 1 data from kernel 1
    ldbi.h          l3, (t9)            // load 1 data from kernel 2
    mulaca.s16.s    l0, a2, l2
    mulaca.s16.s    l1, a2, l3

    bnezad          t7, .L33

.L34:
    ld.hs           t7, (sp, 0x50)      // out_shift
    asr             l0, l0, t7
    asr             l1, l1, t7
    clipi.s32       l0, l0, 16
    clipi.s32       l1, l1, 16
    pkgll           l0, l0, l1
    stbi.w          l0, (l6)

    mov             t8, t9
    ixh             t9, t9, l9
    bnezad          t6, .L30

.L35:
    ld.hs           l0, (sp, 0x38)      // ch_im_out
    andi            t6, l0, 1           // ch_im_out % 0x2u
    bez             t6, .L40
    ld.hs           l4, (sp, 0x4c)      // bias_shift
    ld.hs           a0, (sp, 0x50)      // out_shift

.L36:
    ld.w            l7, (sp, 0x5c)      // *bufferA

    ldbi.hs         l0, (l5)
    lsl.s32.s       l0, l0, l4          // sum0, sum1  + bias
    add.s32.s       l0, l0, l8          // + NN_ROUND

    lsri            t7, l9, 2           // colCnt = numCol_A >> 2u
    bez             t7, .L37

.L38:
    pldbi.d         a2, (l7)            // load 4 data from col1
    pldbi.d         l2, (t8)            // load 4 data from kernel 1
    mulaca.s16.s    l0, a2, l2
    mulaca.s16.s    l0, a3, l3

    bnezad          t7, .L38

.L37:
    andi            t7, l9, 3          // colCnt = numCol_A & 15u
    bez             t7, .L39

.L41:
    ldbi.h          a2, (l7)            // load 4 data from col1
    ldbi.h          a3, (t8)            // load 4 data from kernel 1
    mulaca.s16.s    l0, a2, a3

    bnezad          t7, .L41

.L39:
    asr             l0, l0, a0
    clipi.s32       l0, l0, 16
    stbi.h          l0, (l6)

.L40:
    ld.w            l7, (sp, 0x5c)      // *bufferA

.L14:
    addi            t1, t1, 1
    br              .L1

.L15:
    addi            t0, t0, 1
    br              .L0

    /* leftover process */

.L16:
    ld.w            t7, (sp, 0x5c)      // *bufferA
    cmpne           l7, t7
    bf              .L22

    ld.w            l5, (sp, 0x48)      // *bias
    ld.hs           t6, (sp, 0x50)      // out_shift
    movi            t1, 0
    ld.w            t9, (sp, 0x8)
    ld.hs           l4, (sp, 0x4c)      // bias_shift

.L17:
    ld.hs           l0, (sp, 0x38)      // ch_im_out
    cmplt           t1, l0
    bf              .L22

    ldbi.hs         l0, (l5)
    lsl.s32.s       l0, l0, l4          // bias_shift
    addu            l0, l0, l8          // + NN_ROUND

    mov             t8, t7              // *pB = bufferA
    lsri            t4, l9, 1           // colCnt
    bez             t4, .L19

.L18:
    ldbi.w          l2, (t8)            // col
    ldbi.w          l3, (t9)            // kernel
    mulaca.s16.s    l0, l3, l2

    bnezad          t4, .L18

.L19:
    andi            t4, l9, 1          // colCnt
    bez             t4, .L21

.L20:
    ldbi.h          l2, (t8)            // col
    ldbi.h          l3, (t9)            // kernel
    mulaca.s16.s    l0, l3, l2

    bnezad          t4, .L20

.L21:
    asr             l0, l0, t6
    clipi.s32       l0, l0, 16
    stbi.h          l0, (l6)

    addi            t1, t1, 1
    br              .L17

.L22:
    addi            sp, sp, 12
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           shl_xt800p_convolve_HWC_q15_basic, .-shl_xt800p_convolve_HWC_q15_basic

.weak csky_dsp2_convolve_HWC_q15_basic
.set  csky_dsp2_convolve_HWC_q15_basic, shl_xt800p_convolve_HWC_q15_basic
