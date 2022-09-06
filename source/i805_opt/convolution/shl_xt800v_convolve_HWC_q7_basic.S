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
 * @file     shl_xt800v_convolve_HWC_q7_basic.S
 * @brief    Q7 vresion of convolution.
 * @version  V1.0
 * @date     19. Mar 2018
 ******************************************************************************/

/*
 * shl_xt800v_status
 * shl_xt800v_convolve_HWC_q7_basic(const q7_t * Im_in,
 *                          const uint16_t dim_im_in,
 *                          const uint16_t ch_im_in,
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

    .file           "shl_xt800v_convolve_HWC_q7_basic.S"
    .section        .text.shl_xt800v_convolve_HWC_q7_basic,"ax",@progbits
    .align          2
    .global         shl_xt800v_convolve_HWC_q7_basic
    .type           shl_xt800v_convolve_HWC_q7_basic, @function

shl_xt800v_convolve_HWC_q7_basic:
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
    ld.w            l7, (sp, 0xd0)      // *bufferA
    mult            l9, l1, l1          // ch_im_in * dim_kernel * dim_kernel
    mult            l9, l9, a2
    lsli            l8, l9, 1
    addu            l8, l8, l7          // bufferA + 2 * l9

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
    bt              .L23
    cmphs           t2, a1
    bt              .L23
    cmplt           t4, t6
    bt              .L23
    cmphs           t4, a1
    bt              .L23

.L7:                                    // else branch
    mult            t6, t2, a1          // (i_ker_y * dim_im_in + i_ker_x)*ch_im_in
    addu            t6, t6, t4
    mult            t6, t6, a2
    addu            t6, t6, a0          // pSrc

    lsri            t7, a2, 4           // ch_im_in >> 4u
    bez             t7, .L9

.L8:
    vldmu.8         vr0-vr0, (t6)
    vstmu.8         vr0-vr0, (l7)

    bnezad          t7, .L8

.L9:
    andi            t7, a2, 15          // ch_im_in & 15u
    bez             t7, .L11

.L10:
    vldx.8          vr0, (t6), t7
    vstx.8          vr0, (l7), t7
    addu            l7, l7, t7
    br              .L11

.L23:
    vmovi.8         vr0, 0
    lsri            t6, a2, 4           // ch_im_in >> 4u(if branch)
    bez             t6, .L5

.L4:
    vstmu.8         vr0-vr0, (l7)       // 0 padding

    bnezad          t6, .L4

.L5:
    andi            t6, a2, 15          // ch_im_in & 7u
    bez             t6, .L11

.L6:
    vstx.8          vr0, (l7), t6
    addu            l7, l7, t6

.L11:
    addi            t4, t4, 1
    br              .L3

.L12:
    addi            t2, t2, 1
    br              .L2

.L13:
    cmpne           l7, l8
    bt              .L14

    ld.w            l4, (sp, 0xbc)      // *bias
    mov             t9, a3
    lsri            t6, l0, 2           // rowCnt = ch_im_out >> 2u
    bez             t6, .L35

.L30:
    ld.w            l7, (sp, 0xd0)      // *bufferA
    addu            lr, l7, l9          // *pB2 = pB + numCol_A

    vldu.8.4        vr0, (l4)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr12, vr0
    vshl.s32.s      vr12, vr12, vr15    // sum0,  ... sum3  + bias
    vadd.s32.s      vr12, vr12, vr14    // + NN_ROUND

    vor.32          vr13, vr12, vr12    // 4 diff kernels a time, sum4, ... sum7
    vmovi.8         vr4, 0
    vmovi.8         vr5, 0
    vmovi.8         vr6, 0
    vmovi.8         vr7, 0
    vmovi.8         vr8, 0
    vmovi.8         vr9, 0
    vmovi.8         vr10, 0
    vmovi.8         vr11, 0

    lsri            t7, l9, 4           // colCnt = numCol_A >> 4u
    bez             t7, .L32

.L31:
    mov             t8, t9
    vldmu.8         vr0-vr0, (l7)       // load 16 data from col1
    vldmu.8         vr1-vr1, (lr)       // load 16 data from col2
    vldmru.8        vr2-vr3, (t8), l9   // load 16 data from kernel 1 and 2
    vmulacaa.s8     vr4, vr2, vr0
    vmulacaa.s8     vr5, vr3, vr0
    vmulacaa.s8     vr8, vr2, vr1
    vmulacaa.s8     vr9, vr3, vr1
    vldmru.8        vr2-vr3, (t8), l9   // load 16 data form kernel 3 and 4
    vmulacaa.s8     vr6, vr2, vr0
    vmulacaa.s8     vr7, vr3, vr0
    vmulacaa.s8     vr10, vr2, vr1
    vmulacaa.s8     vr11, vr3, vr1

    addi            t9, t9, 16
    bnezad          t7, .L31

.L32:
    andi            t7, l9, 15          // colCnt = numCol_A & 15u
    bez             t7, .L34

.L33:
    mov             t8, t9
    vldx.8          vr0, (l7), t7       // load x data from col1
    vldx.8          vr1, (lr), t7       // load x data from col2
    vldx.8          vr2, (t8), t7       // load x data from kernel 1
    addu            t8, t8, l9
    vldx.8          vr3, (t8), t7       // load x data from kernel 2
    addu            t8, t8, l9
    vmulacaa.s8     vr4, vr2, vr0
    vmulacaa.s8     vr5, vr3, vr0
    vmulacaa.s8     vr8, vr2, vr1
    vmulacaa.s8     vr9, vr3, vr1
    vldx.8          vr2, (t8), t7
    addu            t8, t8, l9
    vldx.8          vr3, (t8), t7
    addu            t8, t8, l9
    vmulacaa.s8     vr6, vr2, vr0
    vmulacaa.s8     vr7, vr3, vr0
    vmulacaa.s8     vr10, vr2, vr1
    vmulacaa.s8     vr11, vr3, vr1
    addu            t9, t9, t7

.L34:
    ld.hs           t7, (sp, 0xc4)      // out_shift
    vdupg.32        vr0, t7
    vpadd.s32.s     vr4, vr4, vr5
    vpadd.s32.s     vr5, vr6, vr7
    vpadd.s32.s     vr4, vr4, vr5
    vadd.s32.s      vr12, vr12, vr4
    vpadd.s32.s     vr8, vr8, vr9
    vpadd.s32.s     vr9, vr10, vr11
    vpadd.s32.s     vr8, vr8, vr9
    vadd.s32.s      vr13, vr13, vr8
    vshr.s32        vr12, vr12, vr0
    vshr.s32        vr13, vr13, vr0
    vmov.s32.sl     vr0, vr12, vr13
    vmov.s16.sl     vr0, vr0, vr0
    vexti.8         vr2, vr0, vr0, 0x2b
    vstu.8.4        vr0, (l5)
    addu            t7, l5, l0
    subi            t7, t7, 4
    vstu.8.4        vr2, (t7)

    lsli            t8, l9, 2
    addu            t9, t9, t8
    subu            t9, t9, l9
    bnezad          t6, .L30

.L35:
    andi            t6, l0, 3           // ch_im_out % 0x4u
    bez             t6, .L40
    mov             t8, t9

.L36:
    ld.w            l7, (sp, 0xd0)      // *bufferA
    addu            lr, l7, l9          // *pB2 = pB + numCol_A

    vldu.8.1        vr0, (l4)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr12, vr0
    vshl.s32.s      vr12, vr12, vr15    // sum0, sum1 + bias
    vor.32          vr13, vr12, vr12

    lsri            t7, l9, 4           // colCnt = numCol_A >> 4u
    bez             t7, .L37

.L38:
    vldmu.8         vr0-vr0, (l7)       // load 16 data from col1
    vldmu.8         vr1-vr1, (lr)       // load 16 data from col2
    vldmu.8         vr2-vr2, (t8)       // load 16 data from kernel 1
    vmulacaa.s8     vr12, vr0, vr2
    vmulacaa.s8     vr13, vr1, vr2

    bnezad          t7, .L38

.L37:
    andi            t7, l9, 15          // colCnt = numCol_A & 15u
    bez             t7, .L39

    vldx.8          vr0, (l7), t7       // load x data from col1
    vldx.8          vr1, (lr), t7       // load x data from col2
    vldx.8          vr2, (t8), t7       // load x data from kernel 1
    addu            t8, t8, t7
    vmulacaa.s8     vr12, vr0, vr2
    vmulacaa.s8     vr13, vr1, vr2

.L39:
    vpadd.s32.s     vr12, vr12, vr12
    vpadd.s32.s     vr12, vr12, vr12
    vadd.s32.s      vr12, vr12, vr14    // + NN_ROUND
    vpadd.s32.s     vr13, vr13, vr13
    vpadd.s32.s     vr13, vr13, vr13
    vadd.s32.s      vr13, vr13, vr14    // + NN_ROUND
    ld.hs           t7, (sp, 0xc4)      // out_shift
    vdupg.32        vr0, t7
    vshr.s32        vr12, vr12, vr0
    vshr.s32        vr13, vr13, vr0
    vclip.s32       vr12, vr12, 8
    vclip.s32       vr13, vr13, 8
    vstu.8.1        vr12, (l5)
    addu            t7, l5, l0
    subi            t7, t7, 1
    vstu.8.1        vr13, (t7)

    bnezad          t6, .L36

.L40:
    addu            l5, l5, l0
    ld.w            l7, (sp, 0xd0)      // *bufferA

.L14:
    addi            t1, t1, 1
    br              .L1

.L15:
    addi            t0, t0, 1
    br              .L0

.L16:
    ld.w            t7, (sp, 0xd0)      // *bufferA
    cmpne           l7, t7
    bf              .L22

    ld.w            l4, (sp, 0xbc)      // *bias
    ld.hs           t6, (sp, 0xc4)      // out_shift
    vdupg.32        vr8, t6
    movi            t1, 0
    mov             t9, a3

.L17:
    cmplt           t1, l0
    bf              .L22

    vldu.8.1        vr0, (l4)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr6, vr0
    vshl.s32.s      vr6, vr6, vr15

    mov             t8, t7              // *pB = bufferA
    lsri            t4, l9, 4           // colCnt
    bez             t4, .L19

.L18:
    vldmu.8         vr0-vr0, (t8)       // col
    vldmu.8         vr1-vr1, (t9)       // kernel
    vmulacaa.s8     vr6, vr0, vr1

    bnezad          t4, .L18

.L19:
    andi            t4, l9, 15          // colCnt
    bez             t4, .L21

.L20:
    vldx.8          vr0, (t8), t4
    vldx.8          vr1, (t9), t4
    vmulacaa.s8     vr6, vr0, vr1
    addu            t9, t9, t4

.L21:
    vpadd.s32.s     vr0, vr6, vr6
    vpadd.s32.s     vr0, vr0, vr0
    vadd.s32.s      vr0, vr0, vr14      // sum
    vshr.s32        vr0, vr0, vr8
    vclip.s32       vr0, vr0, 8
    vstu.8.1        vr0, (l5)

    addi            t1, t1, 1
    br              .L17

.L22:
    movi            a0, 0
    vldmu.8         vr12-vr15, (sp)
    vldmu.8         vr8-vr11, (sp)
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           shl_xt800v_convolve_HWC_q7_basic, .-shl_xt800v_convolve_HWC_q7_basic

.weak csky_vdsp2_convolve_HWC_q7_basic
.set  csky_vdsp2_convolve_HWC_q7_basic, shl_xt800v_convolve_HWC_q7_basic
