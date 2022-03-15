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
 * @file     csi_xt800v_convolve_HWC_q7_fast_nonsquare.S
 * @brief    Fast Q7 vresion of convolution (non-square shape).
 * @version  V1.0
 * @date     05. June 2018
 ******************************************************************************/

/*
 * csi_xt800v_status csi_xt800v_convolve_HWC_q7_fast_nonsquare(const q7_t * Im_in,
 *                                             const uint16_t dim_im_in_x,
 *                                             const uint16_t dim_im_in_y,
 *                                             const uint16_t ch_im_in,
 *                                             const q7_t * wt,
 *                                             const uint16_t ch_im_out,
 *                                             const uint16_t dim_kernel_x,
 *                                             const uint16_t dim_kernel_y,
 *                                             const uint16_t padding_x,
 *                                             const uint16_t padding_y,
 *                                             const uint16_t stride_x,
 *                                             const uint16_t stride_y,
 *                                             const q7_t * bias,
 *                                             const uint16_t bias_shift,
 *                                             const uint16_t out_shift,
 *                                             q7_t * Im_out,
 *                                             const uint16_t dim_im_out_x,
 *                                             const uint16_t dim_im_out_y,
 *                                             q15_t * bufferA)
 *
 */

    .file           "csi_xt800v_convolve_HWC_q7_fast_nonsquare.S"
    .section        .text.csi_xt800v_convolve_HWC_q7_fast_nonsquare,"ax",@progbits
    .align          2
    .global         csi_xt800v_convolve_HWC_q7_fast_nonsquare
    .type           csi_xt800v_convolve_HWC_q7_fast_nonsquare, @function

csi_xt800v_convolve_HWC_q7_fast_nonsquare:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    subi            sp, sp, 64
    vstm.8          vr8-vr11, (sp)
    subi            sp, sp, 64
    vstm.8          vr12-vr15, (sp)
    ld.w            l0, (sp, 0xac)      // wt
    ld.hs           l1, (sp, 0xb0)      // ch_im_out
    ld.hs           l2, (sp, 0xb4)      // dim_kernel_x
    ld.hs           l3, (sp, 0xb8)      // dim_kernel_y
    ld.hs           l5, (sp, 0xd0)      // bias_shift
    vdupg.32        vr15, l5
    ld.hs           l6, (sp, 0xd4)      // out_shift
    movi            l7, 1
    subi            l8, l6, 1
    lsl             l7, l7, l8
    vdupg.32        vr14, l7            // NN_ROUND
    ld.w            l5, (sp, 0xd8)      // *im_out
    ld.hs           l6, (sp, 0xdc)      // dim_im_out_x
    ld.hs           l7, (sp, 0xe0)      // dim_im_out_y
    ld.w            l8, (sp, 0xe4)      // *bufferA
    mult            l9, l2, l3          // ch_im_in * dim_kernel_x * dim_kernel_y
    mult            l9, l9, a3

    /* top part */
    movi            t0, 0               // i_out_y

.L0:
    ld.hs           t9, (sp, 0xc0)      // padding_y
    cmplt           t0, t9              // i_out_y < padding_y
    bf              .L41

    movi            t1, 0               // i_out_x

.L1:
    cmplt           t1, l6              // i_out_x < dim_im_out_x
    bf              .L15

    ld.hs           t8, (sp, 0xc0)      // padding_y
    ld.hs           t9, (sp, 0xc8)      // stride_y
    mult            t2, t0, t9          // i_ker_y = i_out_y * stride_y
    subu            t2, t2, t8
    addu            t3, t2, l3          // i_out_y * stride_y - padding_y + dim_kernel_y

.L2:
    cmplt           t2, t3
    bf              .L13

    ld.hs           t8, (sp, 0xbc)      // padding_x
    ld.hs           t9, (sp, 0xc4)      // stride_x
    mult            t4, t1, t9          // i_ker_x = i_out_x * stride_x
    subu            t4, t4, t8
    addu            t5, t4, l2          // i_out_x * stride - padding_x + dim_kernel_x

.L3:
    cmplt           t4, t5
    bf              .L12

    movi            t6, 0
    cmplt           t2, t6
    bt              .L10
    cmphs           t2, a2
    bt              .L10
    cmplt           t4, t6
    bt              .L10
    cmphs           t4, a1
    bt              .L10

.L7:                                    // else branch
    mult            t6, t2, a1          // (i_ker_y * dim_im_in_x + i_ker_x)*ch_im_in
    addu            t6, t6, t4
    mult            t6, t6, a3
    addu            t6, t6, a0          // pSrc

    lsri            t7, a3, 4           // ch_im_in >> 5u
    bez             t7, .L9

.L8:
    vldmu.8         vr0-vr0, (t6)
    vstmu.8         vr0-vr0, (l8)

    bnezad          t7, .L8

.L9:
    andi            t7, a3, 15          // ch_im_in & 15u
    bez             t7, .L11

    vldx.8          vr0, (t6), t7
    vstx.8          vr0, (l8), t7
    addu            l8, l8, t7
    br              .L11

.L10:
    vmovi.8         vr0, 0
    lsri            t6, a3, 4           // ch_im_in >> 4u(if branch)
    bez             t6, .L5

.L4:
    vstmu.8         vr0-vr0, (l8)       // 0 padding

    bnezad          t6, .L4

.L5:
    andi            t6, a3, 15          // ch_im_in & 7u
    bez             t6, .L11

.L6:
    vstx.8          vr0, (l8), t6
    addu            l8, l8, t6

.L11:
    addi            t4, t4, 1
    br              .L3

.L12:
    addi            t2, t2, 1
    br              .L2

.L13:
    ld.w            t9, (sp, 0xe4)      // *bufferA
    lsli            lr, l9, 1
    addu            lr, t9, lr          // bufferA + 2 * l9
    cmpne           l8, lr
    bt              .L14

    ld.w            l4, (sp, 0xcc)      // bias
    mov             t9, l0
    lsri            t6, l1, 2           // rowCnt = ch_im_out >> 2u
    bez             t6, .L35

.L30:
    ld.w            l8, (sp, 0xe4)      // *bufferA
    addu            lr, l8, l9          // *pB2 = pB + numCol_A

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
    vldmu.8         vr0-vr0, (l8)       // load 16 data from col1
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
    vldx.8          vr0, (l8), t7       // load x data from col1
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
    ld.hs           t7, (sp, 0xd4)      // out_shift
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
    vclip.s32       vr0, vr12, 8
    vclip.s32       vr2, vr13, 8
    vmov.32.l       vr0, vr0, vr0
    vmov.16.l       vr0, vr0, vr0
    vstu.8.4        vr0, (l5)
    vmov.32.l       vr2, vr2, vr2
    vmov.16.l       vr2, vr2, vr2
    addu            t7, l5, l1
    subi            t7, t7, 4
    vstu.8.4        vr2, (t7)

    lsli            t8, l9, 2
    addu            t9, t9, t8
    subu            t9, t9, l9
    bnezad          t6, .L30

.L35:
    andi            t6, l1, 3           // ch_im_out % 0x4u
    bez             t6, .L40
    mov             t8, t9

.L36:
    ld.w            l8, (sp, 0xe4)      // *bufferA
    addu            lr, l8, l9          // *pB2 = pB + numCol_A

    vldu.8.1        vr0, (l4)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr12, vr0
    vshl.s32.s      vr12, vr12, vr15    // sum0, sum1 + bias
    vor.32          vr13, vr12, vr12

    lsri            t7, l9, 4           // colCnt = numCol_A >> 4u
    bez             t7, .L37

.L38:
    vldmu.8         vr0-vr0, (l8)       // load 16 data from col1
    vldmu.8         vr1-vr1, (lr)       // load 16 data from col2
    vldmu.8         vr2-vr2, (t8)       // load 16 data from kernel 1
    vmulacaa.s8     vr12, vr0, vr2
    vmulacaa.s8     vr13, vr1, vr2

    bnezad          t7, .L38

.L37:
    andi            t7, l9, 15          // colCnt = numCol_A & 15u
    bez             t7, .L39

    vldx.8          vr0, (l8), t7       // load x data from col1
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
    ld.hs           t7, (sp, 0xd4)      // out_shift
    vdupg.32        vr0, t7
    vshr.s32        vr12, vr12, vr0
    vshr.s32        vr13, vr13, vr0
    vclip.s32       vr12, vr12, 8
    vclip.s32       vr13, vr13, 8
    vstu.8.1        vr12, (l5)
    addu            t7, l5, l1
    subi            t7, t7, 1
    vstu.8.1        vr13, (t7)

    bnezad          t6, .L36

.L40:
    addu            l5, l5, l1
    ld.w            l8, (sp, 0xe4)      // *bufferA

.L14:
    addi            t1, t1, 1
    br              .L1

.L15:
    addi            t0, t0, 1
    br              .L0

    /* middle part */
.L41:
    ld.hs           t9, (sp, 0xc0)      // padding_y
    subu            t9, l7, t9
    cmplt           t0, t9              // i_out_y < dim_in_out -padding
    bf              .L112

    /* left part */
    movi            t1, 0               // i_out_x

.L42:
    ld.hs           t9, (sp, 0xbc)      // padding_x
    cmplt           t1, t9              // i_out_x < padding_x
    bf              .L146

    ld.hs           t8, (sp, 0xc0)      // padding_y
    ld.hs           t9, (sp, 0xc8)      // stride_y
    mult            t2, t0, t9          // i_ker_y = i_out_y * stride_y
    subu            t2, t2, t8
    addu            t3, t2, l3          // i_out_y * stride_y - padding_y + dim_kernel_y

.L43:
    cmplt           t2, t3
    bf              .L54

    ld.hs           t8, (sp, 0xbc)      // padding_x
    ld.hs           t9, (sp, 0xc4)      // stride_x
    mult            t4, t1, t9          // i_ker_x = i_out_x * stride_x
    subu            t4, t4, t8
    addu            t5, t4, l2          // i_out_x * stride - padding_x + dim_kernel_x

.L44:
    cmplt           t4, t5
    bf              .L53

    movi            t6, 0
    cmplt           t4, t6
    bt              .L51
    cmphs           t4, a1
    bt              .L51

.L48:                                    // else branch
    mult            t6, t2, a1          // (i_ker_y * dim_im_in_x + i_ker_x)*ch_im_in
    addu            t6, t6, t4
    mult            t6, t6, a3
    addu            t6, t6, a0          // pSrc

    lsri            t7, a3, 4           // ch_im_in >> 5u
    bez             t7, .L50

.L49:
    vldmu.8         vr0-vr0, (t6)
    vstmu.8         vr0-vr0, (l8)

    bnezad          t7, .L49

.L50:
    andi            t7, a3, 15          // ch_im_in & 15u
    bez             t7, .L52

    vldx.8          vr0, (t6), t7
    vstx.8          vr0, (l8), t7
    addu            l8, l8, t7
    br              .L52

.L51:
    vmovi.8         vr0, 0
    lsri            t6, a3, 4           // ch_im_in >> 4u(if branch)
    bez             t6, .L46

.L45:
    vstmu.8         vr0-vr0, (l8)       // 0 padding

    bnezad          t6, .L45

.L46:
    andi            t6, a3, 15          // ch_im_in & 7u
    bez             t6, .L52

.L47:
    vstx.8          vr0, (l8), t6
    addu            l8, l8, t6

.L52:
    addi            t4, t4, 1
    br              .L44

.L53:
    addi            t2, t2, 1
    br              .L43

.L54:
    ld.w            t9, (sp, 0xe4)      // *bufferA
    lsli            lr, l9, 1
    addu            lr, t9, lr          // bufferA + 2 * l9
    cmpne           l8, lr
    bt              .L66

    ld.w            l4, (sp, 0xcc)      // bias
    mov             t9, l0
    lsri            t6, l1, 2           // rowCnt = ch_im_out >> 2u
    bez             t6, .L60

.L55:
    ld.w            l8, (sp, 0xe4)      // *bufferA
    addu            lr, l8, l9          // *pB2 = pB + numCol_A

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
    bez             t7, .L57

.L56:
    mov             t8, t9
    vldmu.8         vr0-vr0, (l8)       // load 16 data from col1
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
    bnezad          t7, .L56

.L57:
    andi            t7, l9, 15          // colCnt = numCol_A & 15u
    bez             t7, .L59

.L58:
    mov             t8, t9
    vldx.8          vr0, (l8), t7       // load x data from col1
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

.L59:
    ld.hs           t7, (sp, 0xd4)      // out_shift
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
    vclip.s32       vr0, vr12, 8
    vclip.s32       vr2, vr13, 8
    vmov.32.l       vr0, vr0, vr0
    vmov.16.l       vr0, vr0, vr0
    vstu.8.4        vr0, (l5)
    vmov.32.l       vr2, vr2, vr2
    vmov.16.l       vr2, vr2, vr2
    addu            t7, l5, l1
    subi            t7, t7, 4
    vstu.8.4        vr2, (t7)

    lsli            t8, l9, 2
    addu            t9, t8, t9
    subu            t9, t9, l9
    bnezad          t6, .L55

.L60:
    andi            t6, l1, 3           // ch_im_out % 0x4u
    bez             t6, .L65
    mov             t8, t9

.L61:
    ld.w            l8, (sp, 0xe4)      // *bufferA
    addu            lr, l8, l9          // *pB2 = pB + numCol_A

    vldu.8.1        vr0, (l4)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr12, vr0
    vshl.s32.s      vr12, vr12, vr15    // sum0, sum1 + bias
    vor.32          vr13, vr12, vr12

    lsri            t7, l9, 4           // colCnt = numCol_A >> 4u
    bez             t7, .L62

.L63:
    vldmu.8         vr0-vr0, (l8)       // load 16 data from col1
    vldmu.8         vr1-vr1, (lr)       // load 16 data from col2
    vldmu.8         vr2-vr2, (t8)       // load 16 data from kernel 1
    vmulacaa.s8     vr12, vr0, vr2
    vmulacaa.s8     vr13, vr1, vr2

    bnezad          t7, .L63

.L62:
    andi            t7, l9, 15          // colCnt = numCol_A & 15u
    bez             t7, .L64

    vldx.8          vr0, (l8), t7       // load x data from col1
    vldx.8          vr1, (lr), t7       // load x data from col2
    vldx.8          vr2, (t8), t7       // load x data from kernel 1
    addu            t8, t8, t7
    vmulacaa.s8     vr12, vr0, vr2
    vmulacaa.s8     vr13, vr1, vr2

.L64:
    vpadd.s32.s     vr12, vr12, vr12
    vpadd.s32.s     vr12, vr12, vr12
    vadd.s32.s      vr12, vr12, vr14    // + NN_ROUND
    vpadd.s32.s     vr13, vr13, vr13
    vpadd.s32.s     vr13, vr13, vr13
    vadd.s32.s      vr13, vr13, vr14    // + NN_ROUND
    ld.hs           t7, (sp, 0xd4)      // out_shift
    vdupg.32        vr0, t7
    vshr.s32        vr12, vr12, vr0
    vshr.s32        vr13, vr13, vr0
    vclip.s32       vr12, vr12, 8
    vclip.s32       vr13, vr13, 8
    vstu.8.1        vr12, (l5)
    addu            t7, l5, l1
    subi            t7, t7, 1
    vstu.8.1        vr13, (t7)

    bnezad          t6, .L61

.L65:
    addu            l5, l5, l1
    ld.w            l8, (sp, 0xe4)      // *bufferA

.L66:
    addi            t1, t1, 1
    br              .L42

.L67:
    addi            t0, t0, 1
    br              .L41

    /* mid part */

.L146:
    mult            t7, a3, l2          // ch_im_in * dim_kernel_x

.L68:
    ld.hs           t9, (sp, 0xbc)      // padding_x
    subu            t9, l6, t9
    cmplt           t1, t9              // i_out_x < dim_in_out_x -padding_x
    bf              .L86

    ld.hs           t8, (sp, 0xc0)      // padding_y
    ld.hs           t9, (sp, 0xc8)      // stride_y
    mult            t2, t0, t9          // i_ker_y = i_out_y * stride_y
    subu            t2, t2, t8
    addu            t3, t2, l3          // i_out_y * stride_y - padding_y + dim_kernel_y

.L69:
    cmplt           t2, t3
    bf              .L73

    ld.hs           t8, (sp, 0xbc)      // padding_x
    ld.hs           t9, (sp, 0xc4)      // stride_x
    mult            t4, t2, a1          // i_ker_y * dim_im_in_x
    mult            t5, t1, t9          // i_out_x * stride_x
    addu            t4, t4, t5
    subu            t4, t4, t8
    mult            t4, t4, a3
    addu            t4, t4, a0          // pSrc

    lsri            t5, t7, 4
    bez             t5, .L71

.L70:
    vldmu.8         vr0-vr0, (t4)
    vstmu.8         vr0-vr0, (l8)

    bnezad          t5, .L70

.L71:
    andi            t5, t7, 15
    bez             t5, .L72
    vldx.8          vr0, (t4), t5
    vstx.8          vr0, (l8), t5
    addu            l8, l8, t5

.L72:
    addi            t2, t2, 1
    br              .L69

.L73:
    ld.w            t9, (sp, 0xe4)      // *bufferA
    lsli            lr, l9, 1
    addu            lr, t9, lr          // bufferA + 2 * l9
    cmpne           l8, lr
    bt              .L85

    ld.w            l4, (sp, 0xcc)      // bias
    mov             t9, l0
    lsri            t5, l1, 2           // rowCnt = ch_im_out >> 2u
    bez             t5, .L79

.L74:
    ld.w            l8, (sp, 0xe4)      // *bufferA
    addu            lr, l8, l9          // *pB2 = pB + numCol_A

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

    lsri            t6, l9, 4           // colCnt = numCol_A >> 4u
    bez             t6, .L76

.L75:
    mov             t8, t9
    vldmu.8         vr0-vr0, (l8)       // load 16 data from col1
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
    bnezad          t6, .L75

.L76:
    andi            t6, l9, 15          // colCnt = numCol_A & 15u
    bez             t6, .L78

.L77:
    mov             t8, t9
    vldx.8          vr0, (l8), t6       // load x data from col1
    vldx.8          vr1, (lr), t6       // load x data from col2
    vldx.8          vr2, (t8), t6       // load x data from kernel 1
    addu            t8, t8, l9
    vldx.8          vr3, (t8), t6       // load x data from kernel 2
    addu            t8, t8, l9
    vmulacaa.s8     vr4, vr2, vr0
    vmulacaa.s8     vr5, vr3, vr0
    vmulacaa.s8     vr8, vr2, vr1
    vmulacaa.s8     vr9, vr3, vr1
    vldx.8          vr2, (t8), t6
    addu            t8, t8, l9
    vldx.8          vr3, (t8), t6
    addu            t8, t8, l9
    vmulacaa.s8     vr6, vr2, vr0
    vmulacaa.s8     vr7, vr3, vr0
    vmulacaa.s8     vr10, vr2, vr1
    vmulacaa.s8     vr11, vr3, vr1
    addu            t9, t9, t6

.L78:
    ld.hs           t6, (sp, 0xd4)      // out_shift
    vdupg.32        vr0, t6
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
    vclip.s32       vr0, vr12, 8
    vclip.s32       vr2, vr13, 8
    vmov.32.l       vr0, vr0, vr0
    vmov.16.l       vr0, vr0, vr0
    vstu.8.4        vr0, (l5)
    vmov.32.l       vr2, vr2, vr2
    vmov.16.l       vr2, vr2, vr2
    addu            t6, l5, l1
    subi            t6, t6, 4
    vstu.8.4        vr2, (t6)

    lsli            t8, l9, 2
    addu            t9, t9, t8
    subu            t9, t9, l9
    bnezad          t5, .L74

.L79:
    andi            t5, l1, 3           // ch_im_out % 0x4u
    bez             t5, .L84
    mov             t8, t9

.L80:
    ld.w            l8, (sp, 0xe4)      // *bufferA
    addu            lr, l8, l9          // *pB2 = pB + numCol_A

    vldu.8.1        vr0, (l4)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr12, vr0
    vshl.s32.s      vr12, vr12, vr15    // sum0, sum1 + bias
    vor.32          vr13, vr12, vr12

    lsri            t6, l9, 4           // colCnt = numCol_A >> 4u
    bez             t6, .L82

.L81:
    vldmu.8         vr0-vr0, (l8)       // load 16 data from col1
    vldmu.8         vr1-vr1, (lr)       // load 16 data from col2
    vldmu.8         vr2-vr2, (t8)       // load 16 data from kernel 1
    vmulacaa.s8     vr12, vr0, vr2
    vmulacaa.s8     vr13, vr1, vr2

    bnezad          t6, .L81

.L82:
    andi            t6, l9, 15          // colCnt = numCol_A & 15u
    bez             t6, .L83

    vldx.8          vr0, (l8), t6       // load x data from col1
    vldx.8          vr1, (lr), t6       // load x data from col2
    vldx.8          vr2, (t8), t6       // load x data from kernel 1
    addu            t8, t8, t6
    vmulacaa.s8     vr12, vr0, vr2
    vmulacaa.s8     vr13, vr1, vr2

.L83:
    vpadd.s32.s     vr12, vr12, vr12
    vpadd.s32.s     vr12, vr12, vr12
    vadd.s32.s      vr12, vr12, vr14    // + NN_ROUND
    vpadd.s32.s     vr13, vr13, vr13
    vpadd.s32.s     vr13, vr13, vr13
    vadd.s32.s      vr13, vr13, vr14    // + NN_ROUND
    ld.hs           t6, (sp, 0xd4)      // out_shift
    vdupg.32        vr0, t6
    vshr.s32        vr12, vr12, vr0
    vshr.s32        vr13, vr13, vr0
    vclip.s32       vr12, vr12, 8
    vclip.s32       vr13, vr13, 8
    vstu.8.1        vr12, (l5)
    addu            t6, l5, l1
    subi            t6, t6, 1
    vstu.8.1        vr13, (t6)

    bnezad          t5, .L80

.L84:
    addu            l5, l5, l1
    ld.w            l8, (sp, 0xe4)      // *bufferA

.L85:
    addi            t1, t1, 1
    br              .L68

    /* right part */
.L86:
    cmplt           t1, l6              // i_out_x < dim_im_out_x
    bf              .L111

    ld.hs           t8, (sp, 0xc0)      // padding_y
    ld.hs           t9, (sp, 0xc8)      // stride_y
    mult            t2, t0, t9          // i_ker_y = i_out_y * stride_y
    subu            t2, t2, t8
    addu            t3, t2, l3          // i_out_y * stride_y - padding_y + dim_kernel_y

.L87:
    cmplt           t2, t3
    bf              .L98

    ld.hs           t8, (sp, 0xbc)      // padding_x
    ld.hs           t9, (sp, 0xc4)      // stride_x
    mult            t4, t1, t9          // i_ker_x = i_out_x * stride_x
    subu            t4, t4, t8
    addu            t5, t4, l2          // i_out_x * stride - padding_x + dim_kernel_x

.L88:
    cmplt           t4, t5
    bf              .L97

    movi            t6, 0
    cmplt           t4, t6
    bt              .L95
    cmphs           t4, a1
    bt              .L95

.L92:                                    // else branch
    mult            t6, t2, a1          // (i_ker_y * dim_im_in_x + i_ker_x)*ch_im_in
    addu            t6, t6, t4
    mult            t6, t6, a3
    addu            t6, t6, a0          // pSrc

    lsri            t7, a3, 4           // ch_im_in >> 4u
    bez             t7, .L94

.L93:
    vldmu.8         vr0-vr0, (t6)
    vstmu.8         vr0-vr0, (l8)

    bnezad          t7, .L93

.L94:
    andi            t7, a3, 15          // ch_im_in & 15u
    bez             t7, .L96

    vldx.8          vr0, (t6), t7
    vstx.8          vr0, (l8), t7
    addu            l8, l8, t7
    br              .L96

.L95:
    vmovi.8         vr0, 0
    lsri            t6, a3, 4           // ch_im_in >> 4u(if branch)
    bez             t6, .L90

.L89:
    vstmu.8         vr0-vr0, (l8)           // 0 padding

    bnezad          t6, .L89

.L90:
    andi            t6, a3, 15          // ch_im_in & 15u
    bez             t6, .L96

.L91:
    vstx.8          vr0, (l8), t6
    addu            l8, l8, t6

.L96:
    addi            t4, t4, 1
    br              .L88

.L97:
    addi            t2, t2, 1
    br              .L87

.L98:
    ld.w            t9, (sp, 0xe4)      // *bufferA
    lsli            lr, l9, 1
    addu            lr, t9, lr          // bufferA + 2 * l9
    cmpne           l8, lr
    bt              .L110

    ld.w            l4, (sp, 0xcc)      // bias
    mov             t9, l0
    lsri            t6, l1, 2           // rowCnt = ch_im_out >> 2u
    bez             t6, .L104

.L99:
    ld.w            l8, (sp, 0xe4)      // *bufferA
    addu            lr, l8, l9          // *pB2 = pB + numCol_A

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
    bez             t7, .L101

.L100:
    mov             t8, t9
    vldmu.8         vr0-vr0, (l8)       // load 16 data from col1
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
    bnezad          t7, .L100

.L101:
    andi            t7, l9, 15          // colCnt = numCol_A & 15u
    bez             t7, .L103

.L102:
    mov             t8, t9
    vldx.8          vr0, (l8), t7       // load x data from col1
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

.L103:
    ld.hs           t7, (sp, 0xd4)      // out_shift
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
    vclip.s32       vr0, vr12, 8
    vclip.s32       vr2, vr13, 8
    vmov.32.l       vr0, vr0, vr0
    vmov.16.l       vr0, vr0, vr0
    vstu.8.4        vr0, (l5)
    vmov.32.l       vr2, vr2, vr2
    vmov.16.l       vr2, vr2, vr2
    addu            t7, l5, l1
    subi            t7, t7, 4
    vstu.8.4        vr2, (t7)

    lsli            t8, l9, 2
    addu            t9, t9, t8
    subu            t9, t9, l9
    bnezad          t6, .L99

.L104:
    andi            t6, l1, 3           // ch_im_out % 0x4u
    bez             t6, .L109
    mov             t8, t9

.L105:
    ld.w            l8, (sp, 0xe4)      // *bufferA
    addu            lr, l8, l9          // *pB2 = pB + numCol_A

    vldu.8.1        vr0, (l4)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr12, vr0
    vshl.s32.s      vr12, vr12, vr15    // sum0, sum1 + bias
    vor.32          vr13, vr12, vr12

    lsri            t7, l9, 4           // colCnt = numCol_A >> 4u
    bez             t7, .L106

.L107:
    vldmu.8         vr0-vr0, (l8)       // load 16 data from col1
    vldmu.8         vr1-vr1, (lr)       // load 16 data from col2
    vldmu.8         vr2-vr2, (t8)       // load 16 data from kernel 1
    vmulacaa.s8     vr12, vr0, vr2
    vmulacaa.s8     vr13, vr1, vr2

    bnezad          t7, .L107

.L106:
    andi            t7, l9, 15          // colCnt = numCol_A & 15u
    bez             t7, .L108

    vldx.8          vr0, (l8), t7       // load x data from col1
    vldx.8          vr1, (lr), t7       // load x data from col2
    vldx.8          vr2, (t8), t7       // load x data from kernel 1
    addu            t8, t8, t7
    vmulacaa.s8     vr12, vr0, vr2
    vmulacaa.s8     vr13, vr1, vr2

.L108:
    vpadd.s32.s     vr12, vr12, vr12
    vpadd.s32.s     vr12, vr12, vr12
    vadd.s32.s      vr12, vr12, vr14    // + NN_ROUND
    vpadd.s32.s     vr13, vr13, vr13
    vpadd.s32.s     vr13, vr13, vr13
    vadd.s32.s      vr13, vr13, vr14    // + NN_ROUND
    ld.hs           t7, (sp, 0xd4)      // out_shift
    vdupg.32        vr0, t7
    vshr.s32        vr12, vr12, vr0
    vshr.s32        vr13, vr13, vr0
    vclip.s32       vr12, vr12, 8
    vclip.s32       vr13, vr13, 8
    vstu.8.1        vr12, (l5)
    addu            t7, l5, l1
    subi            t7, t7, 1
    vstu.8.1        vr13, (t7)

    bnezad          t6, .L105

.L109:
    addu            l5, l5, l1
    ld.w            l8, (sp, 0xe4)      // *bufferA

.L110:
    addi            t1, t1, 1
    br              .L86

.L111:
    addi            t0, t0, 1
    br              .L41

    /* bottom part */
.L112:
    cmplt           t0, l7              // i_out_y < dim_im_out_y
    bf              .L139

    movi            t1, 0

.L113:
    cmplt           t1, l6              // i_out_x < dim_im_out_x
    bf              .L138

    ld.hs           t8, (sp, 0xc0)      // padding_y
    ld.hs           t9, (sp, 0xc8)      // stride_y
    mult            t2, t0, t9          // i_ker_y = i_out_y * stride_y
    subu            t2, t2, t8
    addu            t3, t2, l3          // i_out_y * stride_y - padding_y + dim_kernel_y

.L114:
    cmplt           t2, t3
    bf              .L125

    ld.hs           t8, (sp, 0xbc)      // padding_x
    ld.hs           t9, (sp, 0xc4)      // stride_x
    mult            t4, t1, t9          // i_ker_x = i_out_x * stride_x
    subu            t4, t4, t8
    addu            t5, t4, l2          // i_out_x * stride - padding_x + dim_kernel_x

.L115:
    cmplt           t4, t5
    bf              .L124

    movi            t6, 0
    cmplt           t2, t6
    bt              .L122
    cmphs           t2, a2
    bt              .L122
    cmplt           t4, t6
    bt              .L122
    cmphs           t4, a1
    bt              .L122

.L119:                                  // else branch
    mult            t6, t2, a1          // (i_ker_y * dim_im_in_x + i_ker_x)*ch_im_in
    addu            t6, t6, t4
    mult            t6, t6, a3
    addu            t6, t6, a0          // pSrc

    lsri            t7, a3, 4           // ch_im_in >> 4u
    bez             t7, .L121

.L120:
    vldmu.8         vr0-vr0, (t6)
    vstmu.8         vr0-vr0, (l8)

    bnezad          t7, .L120

.L121:
    andi            t7, a3, 15          // ch_im_in & 15u
    bez             t7, .L123

    vldx.8          vr0, (t6), t7
    vstx.8          vr0, (l8), t7
    addu            l8, l8, t7
    br              .L123

.L122:
    vmovi.8         vr0, 0
    lsri            t6, a3, 4           // ch_im_in >> 4u(if branch)
    bez             t6, .L117

.L116:
    vstmu.8         vr0-vr0, (l8)       // 0 padding

    bnezad          t6, .L116

.L117:
    andi            t6, a3, 15          // ch_im_in & 15u
    bez             t6, .L123

.L118:
    vstx.8          vr0, (l8), t6
    addu            l8, l8, t6

.L123:
    addi            t4, t4, 1
    br              .L115

.L124:
    addi            t2, t2, 1
    br              .L114

.L125:
    ld.w            t9, (sp, 0xe4)      // *bufferA
    lsli            lr, l9, 1
    addu            lr, t9, lr          // bufferA + 2 * l9
    cmpne           l8, lr
    bt              .L137

    ld.w            l4, (sp, 0xcc)      // bias
    mov             t9, l0
    lsri            t6, l1, 2           // rowCnt = ch_im_out >> 2u
    bez             t6, .L131

.L126:
    ld.w            l8, (sp, 0xe4)      // *bufferA
    addu            lr, l8, l9          // *pB2 = pB + numCol_A

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
    bez             t7, .L128

.L127:
    mov             t8, t9
    vldmu.8         vr0-vr0, (l8)       // load 16 data from col1
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
    bnezad          t7, .L127

.L128:
    andi            t7, l9, 15          // colCnt = numCol_A & 15u
    bez             t7, .L130

.L129:
    mov             t8, t9
    vldx.8          vr0, (l8), t7       // load x data from col1
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

.L130:
    ld.hs           t7, (sp, 0xd4)      // out_shift
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
    vclip.s32       vr0, vr12, 8
    vclip.s32       vr2, vr13, 8
    vmov.32.l       vr0, vr0, vr0
    vmov.16.l       vr0, vr0, vr0
    vstu.8.4        vr0, (l5)
    vmov.32.l       vr2, vr2, vr2
    vmov.16.l       vr2, vr2, vr2
    addu            t7, l5, l1
    subi            t7, t7, 4
    vstu.8.4        vr2, (t7)

    lsli            t8, l9, 2
    addu            t9, t9, t8
    subu            t9, t9, l9
    bnezad          t6, .L126

.L131:
    andi            t6, l1, 3           // ch_im_out % 0x4u
    bez             t6, .L136
    mov             t8, t9

.L132:
    ld.w            l8, (sp, 0xe4)      // *bufferA
    addu            lr, l8, l9          // *pB2 = pB + numCol_A

    vldu.8.1        vr0, (l4)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr12, vr0
    vshl.s32.s      vr12, vr12, vr15    // sum0, sum1 + bias
    vor.32          vr13, vr12, vr12

    lsri            t7, l9, 4           // colCnt = numCol_A >> 4u
    bez             t7, .L134

.L133:
    vldmu.8         vr0-vr0, (l8)       // load 16 data from col1
    vldmu.8         vr1-vr1, (lr)       // load 16 data from col2
    vldmu.8         vr2-vr2, (t8)       // load 16 data from kernel 1
    vmulacaa.s8     vr12, vr0, vr2
    vmulacaa.s8     vr13, vr1, vr2

    bnezad          t7, .L133

.L134:
    andi            t7, l9, 15          // colCnt = numCol_A & 15u
    bez             t7, .L135

    vldx.8          vr0, (l8), t7       // load x data from col1
    vldx.8          vr1, (lr), t7       // load x data from col2
    vldx.8          vr2, (t8), t7       // load x data from kernel 1
    addu            t8, t8, t7
    vmulacaa.s8     vr12, vr0, vr2
    vmulacaa.s8     vr13, vr1, vr2

.L135:
    vpadd.s32.s     vr12, vr12, vr12
    vpadd.s32.s     vr12, vr12, vr12
    vadd.s32.s      vr12, vr12, vr14    // + NN_ROUND
    vpadd.s32.s     vr13, vr13, vr13
    vpadd.s32.s     vr13, vr13, vr13
    vadd.s32.s      vr13, vr13, vr14    // + NN_ROUND
    ld.hs           t7, (sp, 0xd4)      // out_shift
    vdupg.32        vr0, t7
    vshr.s32        vr12, vr12, vr0
    vshr.s32        vr13, vr13, vr0
    vclip.s32       vr12, vr12, 8
    vclip.s32       vr13, vr13, 8
    vstu.8.1        vr12, (l5)
    addu            t7, l5, l1
    subi            t7, t7, 1
    vstu.8.1        vr13, (t7)

    bnezad          t6, .L132

.L136:
    addu            l5, l5, l1
    ld.w            l8, (sp, 0xe4)      // *bufferA

.L137:
    addi            t1, t1, 1
    br              .L113

.L138:
    addi            t0, t0, 1
    br              .L112

    /* check for left-over */
.L139:
    ld.w            t7, (sp, 0xe4)      // *bufferA
    cmpne           l8, t7
    bf              .L145

    ld.hs           t6, (sp, 0xd4)      // out_shift
    ld.w            l4, (sp, 0xcc)      // bias
    vdupg.32        vr8, t6
    movi            t1, 0
    mov             t9, l0

.L140:
    cmplt           t1, l1              // i < ch_im_out
    bf              .L145

    vldu.8.1        vr0, (l4)
    vmov.s8.e       vr0, vr0
    vmov.s16.e      vr6, vr0
    vshl.s32.s      vr6, vr6, vr15

    mov             t8, t7              // *pB = bufferA
    lsri            t4, l9, 4           // colCnt
    bez             t4, .L142

.L141:
    vldmu.8         vr0-vr0, (t8)       // col
    vldmu.8         vr1-vr1, (t9)       // kernel
    vmulacaa.s8     vr6, vr0, vr1

    bnezad          t4, .L141

.L142:
    andi            t4, l9, 15          // colCnt
    bez             t4, .L144

.L143:
    vldx.8          vr0, (t8), t4
    vldx.8          vr1, (t9), t4
    vmulacaa.s8     vr6, vr0, vr1
    addu            t9, t9, t4

.L144:
    vpadd.s32.s     vr0, vr6, vr6
    vpadd.s32.s     vr0, vr0, vr0
    vadd.s32.s      vr0, vr0, vr14
    vshr.s32        vr0, vr0, vr8
    vclip.s32       vr0, vr0, 8
    vstu.8.1        vr0, (l5)

    addi            t1, t1, 1
    br              .L140

.L145:
    movi            a0, 0
    vldmu.8         vr12-vr15, (sp)
    vldmu.8         vr8-vr11, (sp)
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           csi_xt800v_convolve_HWC_q7_fast_nonsquare, .-csi_xt800v_convolve_HWC_q7_fast_nonsquare

.weak csi_convolve_HWC_q7_fast_nonsquare
.set  csi_convolve_HWC_q7_fast_nonsquare, csi_xt800v_convolve_HWC_q7_fast_nonsquare
.weak csky_vdsp2_convolve_HWC_q7_fast_nonsquare
.set  csky_vdsp2_convolve_HWC_q7_fast_nonsquare, csi_xt800v_convolve_HWC_q7_fast_nonsquare
