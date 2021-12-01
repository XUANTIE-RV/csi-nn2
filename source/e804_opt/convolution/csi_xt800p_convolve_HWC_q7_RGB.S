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
 * @file     csi_xt800p_convolve_HWC_q7_RGB.S
 * @brief    Q7 vresion of convolution.
 * @version  V1.0
 * @date     19. Mar 2018
 ******************************************************************************/

/*
 * csi_xt800p_status
 * csi_xt800p_convolve_HWC_q7_RGB(const q7_t * Im_in,
 *                          const uint16_t dim_im_in,
 *                          const q7_t * wt,
 *                          const uint16_t ch_im_out,
 *                          const uint16_t dim_kernel,
 *                          const uint16_t padding,
 *                          const uint16_t stride,
 *                          const q7_t * bias,
 *                          const uint16_t bias_shift,
 *                          const uint16_t out_shift,
 *                          q7_t * Im_out,
 *                          const uint16_t dim_im_out,
 *                          q15_t * bufferA)
 */

    .file           "csi_xt800p_convolve_HWC_q7_RGB.S"
    .section        .text.csi_xt800p_convolve_HWC_q7_RGB,"ax",@progbits
    .align          2
    .global         csi_xt800p_convolve_HWC_q7_RGB
    .type           csi_xt800p_convolve_HWC_q7_RGB, @function

csi_xt800p_convolve_HWC_q7_RGB:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    subi            sp, sp, 12
    st.w            a0, (sp)
    st.w            a2, (sp, 0x4)
    st.w            a3, (sp, 0x8)
    ld.hs           l1, (sp, 0x38)      // dim_kernel
    ld.hs           l6, (sp, 0x4c)      // out_shift
    movi            a2, 3
    movi            l7, 1
    subi            l8, l6, 1
    lsl             l8, l7, l8          // NN_ROUND
    ld.w            l6, (sp, 0x50)      // *im_out
    ld.w            l7, (sp, 0x58)      // *bufferA
    mult            l9, l1, l1          // ch_im_in * dim_kernel * dim_kernel
    mult            l9, l9, a2

    movi            t0, 0               // i_out_y

.L0:
    ld.hs           t9, (sp, 0x54)      // dim_im_out
    cmplt           t0, t9              // i_out_y < dim_im_out
    bf              .L16

    movi            t1, 0               // i_out_x

.L1:
    ld.hs           t9, (sp, 0x54)      // dim_im_out
    cmplt           t1, t9              // i_out_x < dim_im_out
    bf              .L15

    ld.hs           l3, (sp, 0x40)      // stride
    ld.hs           l2, (sp, 0x3c)      // padding
    ld.hs           l1, (sp, 0x38)      // dim_kernel
    mult            t2, t0, l3          // i_ker_y = i_out_y * stride
    subu            t2, t2, l2
    addu            t3, t2, l1          // i_out_y * stride - padding + dim_kernel

.L2:
    cmplt           t2, t3
    bf              .L13

    ld.hs           l3, (sp, 0x40)      // stride
    ld.hs           l2, (sp, 0x3c)      // padding
    ld.hs           l1, (sp, 0x38)      // dim_kernel
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
    movi            a2, 3
    mult            t6, t2, a1          // (i_ker_y * dim_im_in + i_ker_x)*ch_im_in
    addu            t6, t6, t4
    mult            t6, t6, a2
    addu            t6, t6, a0          // pSrc

    ldbi.h          l0, (t6)            // else branch
    ldbi.b          l1, (t6)
    stbi.h          l0, (l7)
    stbi.b          l1, (l7)

    br              .L11

.L23:
    movi            l0, 0               // if branch
    stbi.h          l0, (l7)           // 0 padding
    stbi.b          l0, (l7)

.L11:
    addi            t4, t4, 1
    br              .L3

.L12:
    addi            t2, t2, 1
    br              .L2

.L13:
    ld.w            t9, (sp, 0x58)      // *bufferA
    ixh             t9, t9, l9
    cmpne           l7, t9
    bt              .L14

    ld.w            l5, (sp, 0x44)      // *bias
    ld.w            t8, (sp, 0x4)
    addu            t9, t8, l9
    ld.hs           l0, (sp, 0x8)       // ch_im_out
    lsri            t6, l0, 1           // rowCnt = ch_im_out >> 2u
    bez             t6, .L35

.L30:
    ld.w            l7, (sp, 0x58)      // *bufferA
    addu            lr, l7, l9          // *pB2 = pB + numCol_A

    ld.hs           l4, (sp, 0x48)      // bias_shift
    ldbi.bs         l0, (l5)
    ldbi.bs         l1, (l5)
    lsl.s32.s       l0, l0, l4          // sum0, sum1  + bias
    lsl.s32.s       l1, l1, l4
    add.s32.s       l0, l0, l8          // + NN_ROUND
    add.s32.s       l1, l1, l8
    mov             l2, l0              // sum2, sum3
    mov             l3, l1

    lsri            t7, l9, 2           // colCnt = numCol_A >> 2u
    bez             t7, .L32

.L31:
    ldbi.w          a2, (l7)            // load 4 data from col1
    ldbi.w          a3, (lr)            // load 4 data from col2
    ldbi.w          l4, (t8)            // load 4 data from kernel 1
    mulaca.s8       a0, a2, l4
    mulaca.s8       l4, a3, l4
    addu            l0, l0, a0
    addu            l2, l2, l4
    ldbi.w          l4, (t9)            // load 4 data from kernel 2
    mulaca.s8       a0, a2, l4
    mulaca.s8       l4, a3, l4
    addu            l1, l1, a0
    addu            l3, l3, l4

    bnezad          t7, .L31

.L32:
    andi            t7, l9, 3          // colCnt = numCol_A & 15u
    bez             t7, .L34

.L33:
    ldbi.bs         a2, (l7)            // load 1 data from col1
    ldbi.bs         a3, (lr)            // load 1 data from col2
    ldbi.bs         l4, (t8)            // load 1 data from kernel 1
    ldbi.bs         a0, (t9)            // load 1 data from kernel 2
    mula.32.l       l0, a2, l4
    mula.32.l       l2, a3, l4
    mula.32.l       l1, a2, a0
    mula.32.l       l3, a3, a0

    bnezad          t7, .L33

.L34:
    ld.hs           t7, (sp, 0x4c)      // out_shift
    asr             l0, l0, t7
    asr             l1, l1, t7
    asr             l2, l2, t7
    asr             l3, l3, t7
    clipi.s32       l0, l0, 8
    clipi.s32       l1, l1, 8
    clipi.s32       l2, l2, 8
    clipi.s32       l3, l3, 8
    pkgll           l0, l0, l1
    pkgll           l1, l2, l3
    narl            l0, l0, l0
    narl            l1, l1, l1
    stbi.h          l0, (l6)
    ld.hs           l0, (sp, 0x8)       // ch_im_out
    addu            t7, l6, l0
    subi            t7, t7, 2
    stbi.h          l1, (t7)

    mov             t8, t9
    addu            t9, t9, l9
    bnezad          t6, .L30

.L35:
    ld.hs           l0, (sp, 0x8)       // ch_im_out
    andi            t6, l0, 1           // ch_im_out % 0x2u
    bez             t6, .L40
    ld.hs           l4, (sp, 0x48)      // bias_shift
    ld.hs           l2, (sp, 0x4c)      // out_shift

.L36:
    ld.w            l7, (sp, 0x58)      // *bufferA
    addu            lr, l7, l9          // *pB2 = pB + numCol_A

    ldbi.bs         l0, (l5)
    lsl.s32.s       l0, l0, l4          // sum0, sum1  + bias
    add.s32.s       l0, l0, l8          // + NN_ROUND
    mov             l1, l0

    lsri            t7, l9, 2           // colCnt = numCol_A >> 2u
    bez             t7, .L37

.L38:
    ldbi.w          a2, (l7)            // load 4 data from col1
    ldbi.w          a3, (lr)            // load 4 data from col2
    ldbi.w          l3, (t8)            // load 4 data from kernel 1
    mulaca.s8       a2, a2, l3
    mulaca.s8       a3, a3, l3
    addu            l0, l0, a2
    addu            l1, l1, a3

    bnezad          t7, .L38

.L37:
    andi            t7, l9, 3          // colCnt = numCol_A & 15u
    bez             t7, .L39

.L41:
    ldbi.b          a2, (l7)            // load 4 data from col1
    ldbi.b          a3, (lr)            // load 4 data from col2
    ldbi.b          l3, (t8)            // load 4 data from kernel 1
    mulaca.s8       a2, a2, l3
    mulaca.s8       a3, a3, l3
    addu            l0, l0, a2
    addu            l1, l1, a3

    bnezad          t7, .L41

.L39:
    asr             l0, l0, l2
    asr             l1, l1, l2
    clipi.s32       l0, l0, 8
    clipi.s32       l1, l1, 8
    stbi.b          l0, (l6)
    ld.hs           l0, (sp, 0x8)       // ch_im_out
    addu            t7, l6, l0
    subi            t7, t7, 1
    stbi.b          l1, (t7)

.L40:
    ld.hs           l0, (sp, 0x8)       // ch_im_out
    addu            l6, l6, l0
    ld.w            l7, (sp, 0x58)      // *bufferA

.L14:
    addi            t1, t1, 1
    br              .L1

.L15:
    addi            t0, t0, 1
    br              .L0

    /* leftover process */

.L16:
    ld.w            t7, (sp, 0x58)      // *bufferA
    cmpne           l7, t7
    bf              .L22

    ld.w            l5, (sp, 0x44)      // *bias
    ld.hs           t6, (sp, 0x4c)      // out_shift
    movi            t1, 0
    ld.w            t9, (sp, 0x4)
    ld.hs           l4, (sp, 0x48)      // bias_shift

.L17:
    ld.hs           l0, (sp, 0x8)       // ch_im_out
    cmplt           t1, l0
    bf              .L22

    ldbi.bs         l0, (l5)
    lsl.s32.s       l0, l0, l4          // bias_shift
    addu            l0, l0, l8          // + NN_ROUND

    mov             t8, t7              // *pB = bufferA
    lsri            t4, l9, 2           // colCnt
    bez             t4, .L19

.L18:
    ldbi.w          l2, (t8)            // col
    ldbi.w          l3, (t9)            // kernel
    mulaca.s8       l2, l3, l2
    addu            l0, l0, l2

    bnezad          t4, .L18

.L19:
    andi            t4, l9, 3          // colCnt
    bez             t4, .L21

.L20:
    ldbi.b          l2, (t8)            // col
    ldbi.b          l3, (t9)            // kernel
    mulaca.s8       l2, l3, l2
    addu            l0, l0, l2

    bnezad          t4, .L20

.L21:
    asr             l0, l0, t6
    clipi.s32       l0, l0, 8
    stbi.b          l0, (l6)

    addi            t1, t1, 1
    br              .L17

.L22:
    addi            sp, sp, 12
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           csi_xt800p_convolve_HWC_q7_RGB, .-csi_xt800p_convolve_HWC_q7_RGB

.weak csi_convolve_HWC_q7_RGB
.set  csi_convolve_HWC_q7_RGB, csi_xt800p_convolve_HWC_q7_RGB
.weak csky_dsp2_convolve_HWC_q7_RGB
.set  csky_dsp2_convolve_HWC_q7_RGB, csi_xt800p_convolve_HWC_q7_RGB
