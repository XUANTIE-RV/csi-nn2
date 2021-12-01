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
 * @file     csi_xt800v_convolve_1x1_HWC_q7_fast.S
 * @brief    Fast Q7 vresion of 1x1 convolution (non-square shape).
 * @version  V1.0
 * @date     05. June 2018
 ******************************************************************************/

/*
 * void csi_xt800v_convolve_1x1_HWC_q7_fast(const q7_t * Im_in,
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

    .file           "csi_xt800v_convolve_1x1_HWC_q7_fast.S"
    .section        .text.csi_xt800v_convolve_HWC_q7_fast,"ax",@progbits
    .align          2
    .global         csi_xt800v_convolve_1x1_HWC_q7_fast
    .type           csi_xt800v_convolve_1x1_HWC_q7_fast, @function

csi_xt800v_convolve_1x1_HWC_q7_fast:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    subi            sp, sp, 64
    vstm.8          vr8-vr11, (sp)
    subi            sp, sp, 64
    vstm.8          vr12-vr15, (sp)
    ld.w            l0, (sp, 0xac)      // wt
    ld.hs           l1, (sp, 0xb0)      // ch_im_out
    ld.hs           l3, (sp, 0xb8)      // bias_shift
    vdupg.32        vr15, l3
    ld.hs           l3, (sp, 0xbc)      // out_shift
    movi            l5, 1
    subi            l6, l3, 1
    lsl             l5, l5, l6
    vdupg.32        vr14, l5            // NN_ROUND
    ld.w            l4, (sp, 0xc0)      // *im_out
    ld.hs           l5, (sp, 0xc4)      // dim_im_out_x
    ld.hs           l6, (sp, 0xc8)      // dim_im_out_y
    ld.w            l7, (sp, 0xcc)      // *bufferA
    mov             l9, l7
    lsli            t9, a3, 1
    addu            l8, l7, t9          // bufferA + 2*a3

    movi            t0, 0               // i_out_y

.L0:
    cmplt           t0, l6              // i_out_y < dim_im_out_y
    bf              .L18

    movi            t1, 0               // i_out_x

.L1:
    cmplt           t1, l5              // i_out_x < dim_im_out_x
    bf              .L17

    mult            t6, t0, a1          // (i_out_y * dim_im_in_x + i_out_x)*ch_im_in
    addu            t6, t6, t1
    mult            t6, t6, a3
    addu            t6, t6, a0          // pSrc

    lsri            t2, a3, 4           // ch_im_in >> 4u
    bez             t2, .L3

.L2:
    vldmu.8         vr0-vr0, (t6)
    vstmu.8         vr0-vr0, (l7)

    bnezad          t2, .L2

.L3:
    andi            t2, a3, 15          // ch_im_in & 15u
    bez             t2, .L4

    vldx.8          vr0, (t6), t2
    vstx.8          vr0, (l7), t2
    addu            l7, l7, t2

.L4:
    cmpne           l7, l8
    bt              .L16

    ld.w            l2, (sp, 0xb4)      // bias
    mov             t9, l0
    addu            t2, l4, l1          // pOut2
    lsri            t6, l1, 2           // rowCnt = ch_im_out >> 2u
    bez             t6, .L10

.L5:
    mov             l7, l9
    addu            lr, l7, a3          // *pB2 = pB + numCol_A

    vldu.8.4        vr0, (l2)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr12, vr0
    vshl.s32.s      vr12, vr12, vr15    // sum0,  ... sum3  + bias
    vadd.s32.s      vr12, vr12, vr14    // + NN_ROUND

    vor.32          vr13, vr12, vr12    // 4 diff kernels a time, sum4, ... sum7

    lsri            t7, a3, 4           // colCnt = numCol_A >> 4u
    bez             t7, .L7

.L6:
    mov             t8, t9
    vldmu.8         vr0-vr0, (l7)       // load 16 data from col1
    vldmu.8         vr1-vr1, (lr)       // load 16 data from col2
    vldmru.8        vr2-vr3, (t8), a3   // load 16 data from kernel 1 and 2
    vmulaca.s8      vr4, vr2, vr0
    vmulaca.s8      vr5, vr3, vr0
    vmulaca.s8      vr8, vr2, vr1
    vmulaca.s8      vr9, vr3, vr1
    vldmru.8        vr2-vr3, (t8), a3   // load 16 data form kernel 3 and 4
    vmulaca.s8      vr6, vr2, vr0
    vmulaca.s8      vr7, vr3, vr0
    vmulaca.s8      vr10, vr2, vr1
    vmulaca.s8      vr11, vr3, vr1
    vpadd.s32.s     vr4, vr4, vr5
    vpadd.s32.s     vr5, vr6, vr7
    vpadd.s32.s     vr4, vr4, vr5
    vadd.s32.s      vr12, vr12, vr4     // sum0, ... sum3
    vpadd.s32.s     vr8, vr8, vr9
    vpadd.s32.s     vr9, vr10, vr11
    vpadd.s32.s     vr8, vr8, vr9
    vadd.s32.s      vr13, vr13, vr8     // sum4, ... sum7

    addi            t9, t9, 16
    bnezad          t7, .L6

.L7:
    andi            t7, a3, 15          // colCnt = numCol_A & 15u
    bez             t7, .L9

.L8:
    mov             t8, t9
    vldx.8          vr0, (l7), t7       // load x data from col1
    vldx.8          vr1, (lr), t7       // load x data from col2
    vldx.8          vr2, (t8), t7       // load x data from kernel 1
    addu            t8, t8, a3
    vldx.8          vr3, (t8), t7       // load x data from kernel 2
    addu            t8, t8, a3
    vmulaca.s8      vr4, vr2, vr0
    vmulaca.s8      vr5, vr3, vr0
    vmulaca.s8      vr8, vr2, vr1
    vmulaca.s8      vr9, vr3, vr1
    vldx.8          vr2, (t8), t7
    addu            t8, t8, a3
    vldx.8          vr3, (t8), t7
    addu            t8, t8, a3
    vmulaca.s8      vr6, vr2, vr0
    vmulaca.s8      vr7, vr3, vr0
    vmulaca.s8      vr10, vr2, vr1
    vmulaca.s8      vr11, vr3, vr1
    vpadd.s32.s     vr4, vr4, vr5
    vpadd.s32.s     vr5, vr6, vr7
    vpadd.s32.s     vr4, vr4, vr5
    vadd.s32.s      vr12, vr12, vr4     // sum0, ... sum3
    vpadd.s32.s     vr8, vr8, vr9
    vpadd.s32.s     vr9, vr10, vr11
    vpadd.s32.s     vr8, vr8, vr9
    vadd.s32.s      vr13, vr13, vr8     // sum4, ... sum7
    addu            t9, t9, t7

.L9:
    vdupg.32        vr0, l3
    vshr.s32        vr12, vr12, vr0
    vshr.s32        vr13, vr13, vr0
    vmov.s32.sl     vr0, vr12, vr13
    vmov.s16.sl     vr0, vr0, vr0
    vexti.8         vr2, vr0, vr0, 0x2b
    vstu.8.4        vr0, (l4)
    vstu.8.4        vr2, (t2)

    lsli            t8, a3, 2
    addu            t9, t9, t8
    subu            t9, t9, a3

    bnezad          t6, .L5

.L10:
    andi            t6, l1, 3           // ch_im_out % 0x4u
    bez             t6, .L15
    mov             t8, t9

.L11:
    mov             l7, l9
    addu            lr, l7, a3          // *pB2 = pB + numCol_A

    vldu.8.1        vr0, (l2)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr12, vr0
    vshl.s32.s      vr12, vr12, vr15    // sum0, sum1 + bias
    vor.32          vr13, vr12, vr12

    lsri            t7, a3, 4           // colCnt = numCol_A >> 4u
    bez             t7, .L13

.L12:
    vldmu.8         vr0-vr0, (l7)       // load 16 data from col1
    vldmu.8         vr1-vr1, (lr)       // load 16 data from col2
    vldmu.8         vr2-vr2, (t8)       // load 16 data from kernel 1
    vmulacaa.s8     vr12, vr0, vr2
    vmulacaa.s8     vr13, vr1, vr2

    bnezad          t7, .L12

.L13:
    andi            t7, a3, 15          // colCnt = numCol_A & 15u
    bez             t7, .L14

    vldx.8          vr0, (l7), t7       // load x data from col1
    vldx.8          vr1, (lr), t7       // load x data from col2
    vldx.8          vr2, (t8), t7       // load x data from kernel 1
    addu            t8, t8, t7
    vmulacaa.s8     vr12, vr0, vr2
    vmulacaa.s8     vr13, vr1, vr2

.L14:
    vpadd.s32.s     vr12, vr12, vr12
    vpadd.s32.s     vr12, vr12, vr12
    vadd.s32.s      vr12, vr12, vr14    // + NN_ROUND
    vpadd.s32.s     vr13, vr13, vr13
    vpadd.s32.s     vr13, vr13, vr13
    vadd.s32.s      vr13, vr13, vr14    // + NN_ROUND
    vdupg.32        vr0, l3
    vshr.s32        vr12, vr12, vr0
    vshr.s32        vr13, vr13, vr0
    vclip.s32       vr12, vr12, 8
    vclip.s32       vr13, vr13, 8
    vstu.8.1        vr12, (l4)
    vstu.8.1        vr13, (t2)

    bnezad          t6, .L11

.L15:
    addu            l4, l4, l1
    mov             l7, l9              // bufferA

.L16:
    addi            t1, t1, 1
    br              .L1

.L17:
    addi            t0, t0, 1
    br              .L0

    /* check for left-over */
.L18:
    cmpne           l7, l8
    bf              .L23

    ld.w            l2, (sp, 0xb4)      // bias
    vdupg.32        vr8, l3
    movi            t1, 0
    mov             t9, l0

.L19:
    cmplt           t1, l1              // i < ch_im_out
    bf              .L23

    vldu.8.1        vr0, (l2)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr6, vr0
    vshl.s32.s      vr6, vr6, vr15

    mov             l7, l9              // *pB = bufferA
    lsri            t4, a3, 4           // colCnt
    bez             t4, .L21

.L20:
    vldmu.8         vr0-vr0, (l7)       // col
    vldmu.8         vr1-vr1, (t9)       // kernel
    vmulacaa.s8     vr6, vr0, vr1

    bnezad          t4, .L20

.L21:
    andi            t4, a3, 15          // colCnt
    bez             t4, .L22

    vldx.8          vr0, (l7), t4
    vldx.8          vr1, (t9), t4
    vmulacaa.s8     vr6, vr0, vr1
    addu            t9, t9, t4

.L22:
    vpadd.s32.s     vr0, vr6, vr6
    vpadd.s32.s     vr0, vr0, vr0
    vadd.s32.s      vr0, vr0, vr14      // sum
    vshr.s32        vr0, vr0, vr8
    vclip.s32       vr0, vr0, 8
    vstu.8.1        vr0, (l4)

    addi            t1, t1, 1
    br              .L19

.L23:
    movi            a0, 0
    vldmu.8         vr12-vr15, (sp)
    vldmu.8         vr8-vr11, (sp)
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           csi_xt800v_convolve_1x1_HWC_q7_fast, .-csi_xt800v_convolve_1x1_HWC_q7_fast

.weak csi_convolve_1x1_HWC_q7_fast
.set  csi_convolve_1x1_HWC_q7_fast, csi_xt800v_convolve_1x1_HWC_q7_fast
.weak csky_vdsp2_convolve_1x1_HWC_q7_fast
.set  csky_vdsp2_convolve_1x1_HWC_q7_fast, csi_xt800v_convolve_1x1_HWC_q7_fast
