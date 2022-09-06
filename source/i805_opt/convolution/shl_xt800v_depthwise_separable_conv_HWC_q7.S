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
 * @file     shl_xt800v_depthwise_separable_conv_HWC_q7.S
 * @brief    Q7 depthwise separable convolution function.
 * @version  V1.0
 * @date     05. June 2018
 ******************************************************************************/

/*
 *shl_xt800v_status shl_xt800v_depthwise_separable_conv_HWC_q7(const q7_t * Im_in,
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

    .file           "shl_xt800v_depthwise_separable_conv_HWC_q7.S"
    .section        .text.shl_xt800v_depthwise_separatable_conv_HWC_q7,"ax",@progbits
    .align          2
    .global         shl_xt800v_depthwise_separable_conv_HWC_q7
    .type           shl_xt800v_depthwise_separable_conv_HWC_q7, @function

shl_xt800v_depthwise_separable_conv_HWC_q7:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    subi            sp, sp, 64
    vstm.8          vr8-vr11, (sp)
    subi            sp, sp, 64
    vstm.8          vr12-vr15, (sp)
    ld.hs           l0, (sp, 0xac)      // ch_im_out
    ld.hs           l1, (sp, 0xb0)      // dim_kernel
    ld.hs           l2, (sp, 0xb4)      // padding
    ld.hs           l3, (sp, 0xb8)      // stride
    ld.hs           l5, (sp, 0xc0)      // bias_shift
    vdupg.32        vr15, l5
    ld.hs           l6, (sp, 0xc4)      // out_shift
    movi            l7, 1
    subi            l8, l6, 1
    lsl             l7, l7, l8
    vdupg.32        vr14, l7            // NN_ROUND
    ld.w            l5, (sp, 0xc8)      // *im_out
    ld.hs           l6, (sp, 0xcc)      // dim_im_out
    ld.w            l7, (sp, 0xd0)      // *bufferB
    mult            l9, l1, l1          // dim_kernel * dim_kernel
    mov             l8, l7
    mov             lr, l8

    movi            t0, 0               // i_out_y

.L0:
    cmplt           t0, l6              // i_out_y < dim_im_out
    bf              .L16

    movi            t1, 0               // i_out_x

.L1:
    cmplt           t1, l6              // i_out_x < dim_im_out
    bf              .L15

    mult            t2, t0, l3          // i_ker_y = i_out_y * stride
    subu            t2, t2, l2
    addu            t3, t2, l1          // i_out_y * stride - padding + dim_kernel

.L2:
    cmplt           t2, t3
    bf              .L13

    mult            t4, t1, l3          // i_ker_x = i_out_x * stride
    subu            t4, t4, l2
    addu            t5, t4, l1          // i_out_x * stride - padding + dim_kernel

.L3:
    cmplt           t4, t5
    bf              .L12

    movi            t6, 0
    cmplt           t2, t6
    bt              .L10
    cmphs           t2, a1
    bt              .L10
    cmplt           t4, t6
    bt              .L10
    cmphs           t4, a1
    bt              .L10

.L7:                                    // else branch
    mult            t6, t2, a1          // (i_ker_y * dim_im_in + i_ker_x)*ch_im_in
    addu            t6, t6, t4
    mult            t6, t6, a2
    addu            t6, t6, a0          // pSrc

    lsri            t7, a2, 4           // ch_im_in >> 5u
    bez             t7, .L9

.L8:
    vldmu.8         vr0-vr0, (t6)
    vstmu.8         vr0-vr0, (l7)

    bnezad          t7, .L8

.L9:
    andi            t7, a2, 15          // ch_im_in & 15u
    bez             t7, .L11

    vldx.8          vr0, (t6), t7
    vstx.8          vr0, (l7), t7
    addu            l7, l7, t7
    br              .L11

.L10:
    vmovi.8         vr8, 0
    lsri            t6, a2, 4           // ch_im_in >> 4u(if branch)
    bez             t6, .L5

.L4:
    vstmu.8         vr8-vr8, (l7)       // 0 padding

    bnezad          t6, .L4

.L5:
    andi            t6, a2, 15          // ch_im_in & 7u
    bez             t6, .L11

.L6:
    vstx.8          vr8, (l7), t6
    addu            l7, l7, t6

.L11:
    addi            t4, t4, 1
    br              .L3

.L12:
    addi            t2, t2, 1
    br              .L2

.L13:
    ld.w            l4, (sp, 0xbc)      // *bias
    mov             t9, a3
    lsri            t6, l0, 4           // rowCnt = ch_im_out >> 4u
    bez             t6, .L35

.L30:
    mov             l7, lr              // *bufferB
    mov             t8, t9

    vldmu.8         vr0-vr0, (l4)       // 16 diff kernel a time
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr10, vr0
    vmov.s16.e      vr12, vr1
    vshl.s32.s      vr10, vr10, vr15    // sum0,  ... sum3   + bias
    vshl.s32.s      vr11, vr11, vr15    // sum4,  ... sum7   + bias
    vshl.s32.s      vr12, vr12, vr15    // sum8,  ... sum11  + bias
    vshl.s32.s      vr13, vr13, vr15    // sum12, ... sum15  + bias
    vadd.s32.s      vr10, vr10, vr14    // + NN_ROUND
    vadd.s32.s      vr11, vr11, vr14
    vadd.s32.s      vr12, vr12, vr14
    vadd.s32.s      vr13, vr13, vr14
    vmovi.8         vr6, 0
    vmovi.8         vr7, 0
    vmovi.8         vr8, 0
    vmovi.8         vr9, 0

    lsri            t7, l9, 1           // colCnt = numCol_A >> 2u
    bez             t7, .L32

.L31:
    vldmru.8        vr0-vr1, (l7), a2   // load 16 data from diff channels
    vldmru.8        vr2-vr3, (t8), a2   // load 16 diff kernels
    vmul.s8.e       vr4, vr0, vr2
    vadd.s16.x      vr6, vr6, vr4
    vadd.s16.x      vr8, vr8, vr5

    vmul.s8.e       vr4, vr1, vr3
    vadd.s16.x      vr6, vr6, vr4
    vadd.s16.x      vr8, vr8, vr5

    bnezad          t7, .L31

.L32:
    andi            t7, l9, 1           // colCnt = numCol_A & 1u
    bez             t7, .L34

.L33:
    vldmru.8        vr0-vr0, (l7), a2   // load x data from diff channels
    vldmru.8        vr1-vr1, (t8), a2   // load x kernels
    vmul.s8.e       vr2, vr0, vr1
    vadd.s16.x      vr6, vr6, vr2
    vadd.s16.x      vr8, vr8, vr3

.L34:
    ld.hs           t7, (sp, 0xc4)      // out_shift
    vdupg.32        vr0, t7
    vadd.s32.s      vr10, vr10, vr6
    vadd.s32.s      vr11, vr11, vr7
    vadd.s32.s      vr12, vr12, vr8
    vadd.s32.s      vr13, vr13, vr9
    vshr.s32        vr10, vr10, vr0
    vshr.s32        vr11, vr11, vr0
    vshr.s32        vr12, vr12, vr0
    vshr.s32        vr13, vr13, vr0
    vmov.s32.sl     vr0, vr10, vr11
    vmov.s32.sl     vr1, vr12, vr13
    vmov.s16.sl     vr0, vr0, vr1
    vstmu.8         vr0-vr0, (l5)

    addi            lr, lr, 16
    addi            t9, t9, 16
    bnezad          t6, .L30

.L35:
    andi            t6, l0, 15          // ch_im_out % 0x10u
    bez             t6, .L40

.L36:
    mov             l7, lr              // *pB
    mov             t8, t9              // *pA

    vldx.8          vr0, (l4), t6
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr10, vr0
    vmov.s16.e      vr12, vr1
    vshl.s32.s      vr10, vr10, vr15    // sum0, ... sumx + bias
    vshl.s32.s      vr11, vr11, vr15
    vshl.s32.s      vr12, vr12, vr15
    vshl.s32.s      vr13, vr13, vr15
    vadd.s32.s      vr10, vr10, vr14    // + NN_ROUND
    vadd.s32.s      vr11, vr11, vr14
    vadd.s32.s      vr12, vr12, vr14
    vadd.s32.s      vr13, vr13, vr14
    vmovi.8         vr6, 0
    vmovi.8         vr7, 0
    vmovi.8         vr8, 0
    vmovi.8         vr9, 0

    lsri            t7, l9, 1           // colCnt = numCol_A >> 1u
    bez             t7, .L37

.L38:
    vldmru.8        vr0-vr1, (l7), a2   // load 16 data from diff channels
    vldmru.8        vr2-vr3, (t8), a2   // load 16 diff kernels
    vmul.s8.e       vr4, vr0, vr2
    vadd.s16.x      vr6, vr6, vr4
    vadd.s16.x      vr8, vr8, vr5

    vmul.s8.e       vr4, vr1, vr3
    vadd.s16.x      vr6, vr6, vr4
    vadd.s16.x      vr8, vr8, vr5

    bnezad          t7, .L38

.L37:
    andi            t7, l9, 1           // colCnt = numCol_A & 1u
    bez             t7, .L39

.L41:
    vldmru.8        vr0-vr0, (l7), a2   // load x data from diff channels
    vldmru.8        vr1-vr1, (t8), a2   // load x kernels
    vmul.s8.e       vr2, vr0, vr1
    vadd.s16.x      vr6, vr6, vr2
    vadd.s16.x      vr8, vr8, vr3


.L39:
    vadd.s32.s      vr10, vr10, vr6
    vadd.s32.s      vr11, vr11, vr7
    vadd.s32.s      vr12, vr12, vr8
    vadd.s32.s      vr13, vr13, vr9
    ld.hs           t7, (sp, 0xc4)      // out_shift
    vdupg.32        vr0, t7
    vshr.s32        vr10, vr10, vr0
    vshr.s32        vr11, vr11, vr0
    vshr.s32        vr12, vr12, vr0
    vshr.s32        vr13, vr13, vr0
    vmov.s32.sl     vr0, vr10, vr11
    vmov.s32.sl     vr1, vr12, vr13
    vmov.s16.sl     vr0, vr0, vr1
    vstx.8          vr0, (l5), t6
    addu            l5, l5, t6

.L40:
    mov             l7, l8              // *bufferB
    mov             lr, l7

.L14:
    addi            t1, t1, 1
    br              .L1

.L15:
    addi            t0, t0, 1
    br              .L0

.L16:
    movi            a0, 0
    vldmu.8         vr12-vr15, (sp)
    vldmu.8         vr8-vr11, (sp)
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           shl_xt800v_depthwise_separable_conv_HWC_q7, .-shl_xt800v_depthwise_separable_conv_HWC_q7

.weak csky_vdsp2_depthwise_separable_conv_HWC_q7
.set  csky_vdsp2_depthwise_separable_conv_HWC_q7, shl_xt800v_depthwise_separable_conv_HWC_q7
