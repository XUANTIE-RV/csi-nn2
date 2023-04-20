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
 * @file     shl_xt800v_convolve_HWC_q15_basic.S
 * @brief    Q15 vresion of convolution.
 * @version  V1.0
 * @date     04. June 2018
 ******************************************************************************/

/*
 * shl_xt800v_status
 * shl_xt800v_convolve_HWC_q15_basic(const q15_t * Im_in,
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

    .file           "shl_xt800v_convolve_HWC_q15_basic.S"
    .section        .text.shl_xt800v_convolve_HWC_q15_basic,"ax",@progbits
    .align          2
    .global         shl_xt800v_convolve_HWC_q15_basic
    .type           shl_xt800v_convolve_HWC_q15_basic, @function

shl_xt800v_convolve_HWC_q15_basic:
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
    vdupg.32        vr14, l6
    movi            l7, 1
    subi            l6, l6, 1
    lsl             l7, l7, l6
    vdupg.32        vr12, l7            // NN_ROUND
    ld.w            l5, (sp, 0xc8)      // *im_out
    ld.hs           l6, (sp, 0xcc)      // dim_im_out
    ld.w            l7, (sp, 0xd0)      // *bufferA
    mult            l9, l1, l1          // ch_im_in * dim_kernel * dim_kernel
    mult            l9, l9, a2
    lsli            l8, l9, 2
    addu            l8, l8, l7          // bufferA + 2 * l9
    vmovi.8         vr13, 0

    movi            t0, 0               // i_out_y = 0

.L0:
    cmplt           t0, l6              // i_out_y < dim_im_out
    bf              .L16

    movi            t1, 0               // i_out_x = 0

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
    lsli            t6, t6, 1
    addu            t6, t6, a0          // pSrc

    lsri            t7, a2, 3           // ch_im_in >> 3u
    bez             t7, .L9

.L8:
    vldmu.16        vr0-vr0, (t6)       // memcpy
    vstmu.16        vr0-vr0, (l7)

    bnezad          t7, .L8

.L9:
    andi            t7, a2, 7          // ch_im_in & 7u
    bez             t7, .L11

    vldx.16         vr0, (t6), t7
    vstx.16         vr0, (l7), t7
    lsli            t7, t7, 1
    addu            l7, l7, t7
    br              .L11

.L10:
    lsri            t6, a2, 3           // ch_im_in >> 3u(if branch)
    bez             t6, .L5

.L4:
    vstmu.16        vr13-vr13, (l7)     // 0 padding

    bnezad          t6, .L4

.L5:
    andi            t6, a2, 7          // ch_im_in & 7u
    bez             t6, .L11

.L6:
    vstx.16         vr13, (l7), t6
    lsli            t6, t6, 1
    addu            l7, l7, t6

.L11:
    addi            t4, t4, 1
    br              .L3

.L12:
    addi            t2, t2, 1
    br              .L2

.L13:
    mov             lr, a3              // *pA = *wt

    ld.w            l4, (sp, 0xbc)      // *bias
    lsli            t9, l9, 1
    lsri            t6, l0, 2           // ch_im_out >> 2
    bez             t6, .L34

.L30:
    ld.w            l7, (sp, 0xd0)      // *pB = *bufferA
    vldu.16.4       vr0, (l4)
    vmov.s16.e      vr8, vr0
    vshl.s32.s      vr11, vr8, vr15
    vadd.s32.s      vr11, vr11, vr12    // sum0, ... sum3 + bias
    vmovi.8         vr3, 0
    vmovi.8         vr4, 0
    vmovi.8         vr5, 0
    vmovi.8         vr6, 0
    vmovi.8         vr7, 0
    vmovi.8         vr8, 0
    vmovi.8         vr9, 0
    vmovi.8         vr10, 0

    lsri            t7, l9, 3           // numCol_weight >> 3u
    bez             t7, .L32

.L31:
    mov             t8, lr
    vldmu.16        vr0-vr0, (l7)       // pB
    vldmru.16       vr1-vr2, (t8), t9   // load 2 kernrls
    vmula.s16.e     vr3, vr0, vr1
    vmula.s16.e     vr5, vr0, vr2
    vldmru.16       vr1-vr2, (t8), t9   // load 2 kernrls
    vmula.s16.e     vr7, vr0, vr1
    vmula.s16.e     vr9, vr0, vr2

    addi            lr, lr, 16
    bnezad          t7, .L31

.L32:
    andi            t7, l9, 7
    bez             t7, .L33

    mov             t8, lr
    vldx.16         vr0, (l7), t7
    vldx.16         vr1, (t8), t7
    addu            t8, t8, t9
    vmula.s16.e     vr3, vr0, vr1

    vldx.16         vr1, (t8), t7
    addu            t8, t8, t9
    vmula.s16.e     vr5, vr0, vr1

    vldx.16         vr1, (t8), t7
    addu            t8, t8, t9
    vmula.s16.e     vr7, vr0, vr1

    vldx.16         vr1, (t8), t7
    addu            t8, t8, t9
    vmula.s16.e     vr9, vr0, vr1
    ixh             lr, lr, t7

.L33:
    vadd.s32.s      vr0, vr3, vr4
    vadd.s32.s      vr1, vr5, vr6
    vadd.s32.s      vr2, vr7, vr8
    vadd.s32.s      vr3, vr9, vr10
    vpadd.s32.s     vr0, vr0, vr1
    vpadd.s32.s     vr1, vr2, vr3
    vpadd.s32.s     vr0, vr0, vr1
    vadd.s32.s      vr0, vr0, vr11
    vshr.s32        vr0, vr0, vr14      // sum >> out_shift
    vmov.s32.sl     vr0, vr0, vr0
    vstu.16.4       vr0, (l5)

    lsli            t8, l9, 3
    addu            lr, lr, t8
    subu            lr, lr, t9
    bnezad          t6, .L30

.L34:
    andi            t6, l0, 3
    bez             t6, .L41
    mov             t8, lr

.L35:
    ld.w            l7, (sp, 0xd0)      // *pB = *bufferA
    vldu.16.1       vr0, (l4)
    vmov.s16.e      vr10, vr0
    vshl.s32.s      vr10, vr10, vr15
    vmovi.8         vr11, 0

    lsri            t7, l9, 5
    bez             t7, .L37

.L36:
    vldmu.16        vr0-vr3, (l7)       // col
    vldmu.16        vr4-vr7, (t8)       // kernel
    vmula.s16.e     vr10, vr0, vr4
    vmula.s16.e     vr10, vr1, vr5
    vmula.s16.e     vr10, vr2, vr6
    vmula.s16.e     vr10, vr3, vr7

    bnezad          t7, .L36

.L37:
    andi            t9, l9, 31
    lsri            t7, t9, 3
    bez             t7, .L39

.L38:
    vldmu.16        vr0-vr0, (l7)
    vldmu.16        vr4-vr4, (t8)       // kernel
    vmula.s16.e     vr10, vr0, vr4

    bnezad          t7, .L38

.L39:
    andi            t7, t9, 7
    bez             t7, .L40
    vldx.16         vr0, (l7), t7
    vldx.16         vr1, (t8), t7
    ixh             t8, t8, t7
    vmula.s16.e     vr10, vr0, vr1

.L40:
    vadd.s32.s      vr0, vr10, vr11
    vpadd.s32.s     vr0, vr0, vr0
    vpadd.s32.s     vr0, vr0, vr0
    vadd.s32.s      vr0, vr0, vr12      // sum0 + bias
    vshr.s32        vr0, vr0, vr14      // sum >> out_shift
    vstu.16.1       vr0, (l5)

    bnezad          t6, .L35

.L41:
    ld.w            l7, (sp, 0xd0)      // pBuffer = im_buffer;

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
    pop             l0, l1, l2, l3, l4, l5, l6, l7,  l8, l9, lr
    .size           shl_xt800v_convolve_HWC_q15_basic, .-shl_xt800v_convolve_HWC_q15_basic

.weak csky_vdsp2_convolve_HWC_q15_basic
.set  csky_vdsp2_convolve_HWC_q15_basic, shl_xt800v_convolve_HWC_q15_basic
