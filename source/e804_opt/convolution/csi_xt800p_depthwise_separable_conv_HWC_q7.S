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
 * @file     csi_xt800p_depthwise_separable_conv_HWC_q7.S
 * @brief    Q7 depthwise separable convolution function.
 * @version  V1.0
 * @date     05. June 2018
 ******************************************************************************/

/*
 *csi_xt800p_status csi_xt800p_depthwise_separable_conv_HWC_q7(const q7_t * Im_in,
 *                                              const uint16_t dim_im_in,
 *                                              const uint16_t ch_im_in,
 *                                              const q7_t * wt,
 *                                              const uint16_t ch_im_out,
 *                                              const uint16_t dim_kernel,
 *                                              const uint16_t padding,
 *                                              const uint16_t stride,
 *                                              const q7_t * bias,
 *                                              const uint16_t bias_shift,
 *                                              const uint16_t out_shift,
 *                                              q7_t * Im_out,
 *                                              const uint16_t dim_im_out,
 *                                              q15_t * bufferA)
 */

    .file           "csi_xt800p_depthwise_separable_conv_HWC_q7.S"
    .section        .text.csi_xt800p_depthwise_separatable_conv_HWC_q7,"ax",@progbits
    .align          2
    .global         csi_xt800p_depthwise_separable_conv_HWC_q7
    .type           csi_xt800p_depthwise_separable_conv_HWC_q7, @function

csi_xt800p_depthwise_separable_conv_HWC_q7:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    subi            sp, sp, 16
    st.w            a0, (sp)
    st.w            a1, (sp, 0x4)
    st.w            a2, (sp, 0x8)
    st.w            a3, (sp, 0xc)
    ld.hs           l1, (sp, 0x40)      // dim_kernel
    ld.hs           l6, (sp, 0x54)      // out_shift
    movi            l7, 1
    subi            l8, l6, 1
    lsl             l8, l7, l8          // NN_ROUND
    ld.w            l6, (sp, 0x58)      // *im_out
    ld.w            l7, (sp, 0x60)      // *bufferA
    mult            l9, l1, l1          // ch_im_in * dim_kernel * dim_kernel

    movi            t0, 0               // i_out_y

.L0:
    ld.hs           t9, (sp, 0x5c)      // dim_im_out
    cmplt           t0, t9              // i_out_y < dim_im_out
    bf              .L16

    movi            t1, 0               // i_out_x

.L1:
    ld.hs           t9, (sp, 0x5c)      // dim_im_out
    cmplt           t1, t9              // i_out_x < dim_im_out
    bf              .L15

    ld.hs           l3, (sp, 0x48)      // stride
    ld.hs           l2, (sp, 0x44)      // padding
    ld.hs           l1, (sp, 0x40)      // dim_kernel
    mult            t2, t0, l3          // i_ker_y = i_out_y * stride
    subu            t2, t2, l2
    addu            t3, t2, l1          // i_out_y * stride - padding + dim_kernel

.L2:
    cmplt           t2, t3
    bf              .L13

    ld.hs           l3, (sp, 0x48)      // stride
    ld.hs           l2, (sp, 0x44)      // padding
    ld.hs           l1, (sp, 0x40)      // dim_kernel
    mult            t4, t1, l3          // i_ker_x = i_out_x * stride
    subu            t4, t4, l2
    addu            t5, t4, l1          // i_out_x * stride - padding + dim_kernel

.L3:
    cmplt           t4, t5
    bf              .L12

    movi            t6, 0
    ld.w            a1, (sp, 0x4)
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
    ld.w            a1, (sp, 0x4)
    ld.w            a2, (sp, 0x8)
    mult            t6, t2, a1          // (i_ker_y * dim_im_in + i_ker_x)*ch_im_in
    addu            t6, t6, t4
    mult            t6, t6, a2
    addu            t6, t6, a0          // pSrc

    lsri            t7, a2, 3           // ch_im_in >> 3u
    bez             t7, .L9

.L8:
    pldbi.d         l0, (t6)
    stbi.w          l0, (l7)
    stbi.w          l1, (l7)

    bnezad          t7, .L8

.L9:
    andi            t7, a2, 7          // ch_im_in & 7u
    bez             t7, .L11

.L10:
    ldbi.b          l0, (t6)
    stbi.b          l0, (l7)

    bnezad          t7, .L10
    br              .L11

.L23:
    movi            l0, 0
    ld.w            a2, (sp, 0x8)
    lsri            t6, a2, 3           // ch_im_in >> 4u(if branch)
    bez             t6, .L5

.L4:
    st.w            l0, (l7, 0x0)       // 0 padding
    st.w            l0, (l7, 0x4)
    addi            l7, l7, 8

    bnezad          t6, .L4

.L5:
    andi            t6, a2, 7          // ch_im_in & 7u
    bez             t6, .L11

.L6:
    stbi.b          l0, (l7)

    bnezad          t6, .L6

.L11:
    addi            t4, t4, 1
    br              .L3

.L12:
    addi            t2, t2, 1
    br              .L2

.L13:
    ld.w            a3, (sp, 0x4c)      // *bias
    ld.w            a2, (sp, 0x8)
    ld.hs           l0, (sp, 0x3c)      // ch_im_out
    ld.w            t9, (sp, 0xc)
    ld.w            lr, (sp, 0x60)      // *bufferA
    lsri            t6, l0, 2           // rowCnt = ch_im_out >> 2u
    bez             t6, .L35

.L30:
    mov             l7, lr              // *pB = *bufferA
    mov             t8, t9

    ld.hs           l4, (sp, 0x50)      // bias_shift
    ldbi.bs         l0, (a3)
    ldbi.bs         l1, (a3)
    ldbi.bs         l2, (a3)
    ldbi.bs         l3, (a3)
    lsl.s32.s       l0, l0, l4          // sum0, ..., sum3  + bias
    lsl.s32.s       l1, l1, l4
    lsl.s32.s       l2, l2, l4
    lsl.s32.s       l3, l3, l4
    add.s32.s       l0, l0, l8          // + NN_ROUND
    add.s32.s       l1, l1, l8
    add.s32.s       l2, l2, l8
    add.s32.s       l3, l3, l8

    lsri            t7, l9, 1           // colCnt = numCol_A >> 1u
    bez             t7, .L32

.L31:
    ldbir.w         a0, (l7), a2        // load 4 data from diff channels
    ldbir.w         l4, (t8), a2        // load 4 diff kernels
    pext.s8.e       a0, a0
    pext.s8.e       l4, l4
    mulall.s16.s    l0, a0, l4
    mulahh.s16.s    l1, a0, l4
    mulall.s16.s    l2, a1, l5
    mulahh.s16.s    l3, a1, l5

    ldbir.w         a0, (l7), a2
    ldbir.w         l4, (t8), a2
    pext.s8.e       a0, a0
    pext.s8.e       l4, l4
    mulall.s16.s    l0, a0, l4
    mulahh.s16.s    l1, a0, l4
    mulall.s16.s    l2, a1, l5
    mulahh.s16.s    l3, a1, l5

    bnezad          t7, .L31

.L32:
    andi            t7, l9, 1           // colCnt = numCol_A & 15u
    bez             t7, .L34

.L33:
    ldbir.w         a0, (l7), a2
    ldbir.w         l4, (t8), a2
    pext.s8.e       a0, a0
    pext.s8.e       l4, l4
    mulall.s16.s    l0, a0, l4
    mulahh.s16.s    l1, a0, l4
    mulall.s16.s    l2, a1, l5
    mulahh.s16.s    l3, a1, l5

.L34:
    ld.hs           t7, (sp, 0x54)      // out_shift
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
    narl            l0, l0, l1
    stbi.w          l0, (l6)

    addi            lr, lr, 4
    addi            t9, t9, 4
    bnezad          t6, .L30

.L35:
    ld.hs           l0, (sp, 0x3c)      // ch_im_out
    andi            t6, l0, 3           // ch_im_out % 0x4u
    bez             t6, .L40
    ld.hs           l4, (sp, 0x50)      // bias_shift
    ld.hs           l2, (sp, 0x54)      // out_shift

.L36:
    mov             l7, lr              // *pB
    mov             t8, t9              // *pA

    ldbi.bs         l0, (a3)
    lsl.s32.s       l0, l0, l4          // sum0 + bias
    add.s32.s       l0, l0, l8          // + NN_ROUND

    mov             t7, l9              // colCnt = numCol_A >> 2u
    bez             t7, .L39

.L38:
    ldbir.bs        a0, (l7), a2        // load 4 data from col1
    ldbir.bs        a1, (t8), a2        // load 4 data from kernel 1
    mula.32.l       l0, a0, a1

    bnezad          t7, .L38

.L39:
    asr             l0, l0, l2
    clipi.s32       l0, l0, 8
    stbi.b          l0, (l6)

    addi            lr, lr, 1
    addi            t9, t9, 1
    bnezad          t6, .L36

.L40:
    ld.w            l7, (sp, 0x60)      // *bufferA

.L14:
    addi            t1, t1, 1
    br              .L1

.L15:
    addi            t0, t0, 1
    br              .L0

.L16:
    addi            sp, sp, 16
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           csi_xt800p_depthwise_separable_conv_HWC_q7, .-csi_xt800p_depthwise_separable_conv_HWC_q7

.weak csi_depthwise_separable_conv_HWC_q7
.set  csi_depthwise_separable_conv_HWC_q7, csi_xt800p_depthwise_separable_conv_HWC_q7
.weak csky_dsp2_depthwise_separable_conv_HWC_q7
.set  csky_dsp2_depthwise_separable_conv_HWC_q7, csi_xt800p_depthwise_separable_conv_HWC_q7
