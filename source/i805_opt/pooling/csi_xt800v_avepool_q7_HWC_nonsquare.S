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
 * @file     csi_xt800v_avepool_q7_HWC_nonsquare.S
 * @brief    Pooling functions implementations.
 * @version  V1.0
 * @date     31. May 2018
 ******************************************************************************/

/*
 * void csi_xt800v_avepool_q7_HWC_nonsquare(
 *    const q7_t *Im_in,           // input image
 *    const uint16_t dim_im_in_x,  // input image dimension
 *    const uint16_t dim_im_in_y,  // input image dimension
 *    const uint16_t ch_im_in,     // number of input image channels
 *    const uint16_t dim_kernel_x, // window kernel size
 *    const uint16_t dim_kernel_y, // window kernel size
 *    const uint16_t padding_x,    // padding sizes
 *    const uint16_t padding_y,    // padding sizes
 *    const uint16_t stride_x,     // stride
 *    const uint16_t stride_y,     // stride
 *    const uint16_t dim_im_out_x, // output image dimension
 *    const uint16_t dim_im_out_y, // output image dimension
 *    q7_t *bufferA,               // a buffer for local storage
 *    q7_t *Im_out,                // output feature
 *    const uint16_t out_lshift)   // output left shift (scaling)
 */

    .section        .text.csi_xt800v_avepool_q7_HWC_nonsquare,"ax",@progbits
    .align          2
    .global         csi_xt800v_avepool_q7_HWC_nonsquare
    .type           csi_xt800v_avepool_q7_HWC_nonsquare, @function

csi_xt800v_avepool_q7_HWC_nonsquare:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    ld.hs           l8, (sp, 0X2C)      // dim_kernel_x
    ld.hs           l3, (sp, 0x34)      // padding_x
    ld.hs           l2, (sp, 0x3c)      // stride_x
    ld.hs           l1, (sp, 0x44)      // dim_im_out_x
    ld.hs           l9, (sp, 0x48)      // dim_im_out_y
    ld.w            t5, (sp, 0x4c)      // bufferA
    ld.w            l0, (sp, 0x50)      // *im_out
    ld.hs           lr, (sp, 0x54)      // out_lshift
    movi            t6, 8
    movi            t9, 1
    lsli            t9, t9, 15

    /* pooling along x axis*/
    movi            t0, 0               // i_y = 0

.L30:
    cmplt           t0, a2              // i_y < dim_im_in_y
    bf              .L49

    movi            t1, 0               // i_x = 0

.L31:
    cmplt           t1, l1              // i_x < dim_im_out_x
    bf              .L48

    mult            l7, t0, a1          // i_y * dim_im_in_x
    addu            l4, l7, t1          // + i_x
    mult            l4, l4, a3          // * ch_im_in
    addu            l4, l4, a0          // + im_in

    mult            l6, t1, l2          // i_x * stride_x
    cmplt           l6, l3
    bt              .L32
    addu            l5, l6, l7
    subu            l5, l5, l3          // - padding_x
    mult            l5, l5, a3          // * ch_im_in
    addu            l5, l5, a0          // + im_in
    br              .L33

.L32:
    mov             l5, l4              // win_start

.L33:
    subu            t2, l6, l3          // - padding_x
    addu            t2, t2, l8          // + dim_kernel_x
    cmphs           t2, a1
    bf              .L34
    addu            l6, l7, a1          // + dim_im_in_x
    mult            l6, l6, a3          // * ch_im_in
    addu            l6, l6, a0          // + im_in
    br              .L35

.L34:
    addu            l6, l7, l6
    subu            l6, l6, l3          // - padding_x
    addu            l6, l6, l8          // + dim_kernrl_x
    mult            l6, l6, a3          // * ch_im_in
    addu            l6, l6, a0          // win_stop

.L35:
    mov             l7, t5
    movi            t4, 1               // count = 1
    lsri            t2, a3, 4
    bez             t2, .L37

.L36:
    vldmu.8         vr0-vr0, (l5)       // q7_to_q15
    vmov.s8.e       vr0, vr0
    vstmu.16        vr0-vr1, (l7)

    bnezad          t2, .L36

.L37:
    andi            t2, a3, 15
    bez             t2, .L39

    vldx.8          vr0, (l5), t2
    addu            l5, l5, t2
    vmov.s8.e       vr0, vr0
    cmphsi          t2, 9
    bf              .L38
    subi            t2, t2, 8
    vstmu.16        vr0-vr0, (l7)
    vstx.16         vr1, (l7), t2
    br              .L39

.L38:
    vstx.16         vr0, (l7), t2

.L39:
    cmplt           l5, l6
    bf              .L43

    mov             l7, t5
    lsri            t2, a3, 4
    bez             t2, .L41

.L40:
    vldmu.8         vr0-vr0, (l5)       // accumulate_q7_to_q15
    vldm.16         vr1-vr2, (l7)
    vadd.s8.x       vr1, vr1, vr0
    vstmu.16        vr1-vr2, (l7)

    bnezad          t2, .L40

.L41:
    andi            t2, a3, 15
    lsri            t3, t2, 3
    bez             t3, .L68

    vldx.8          vr0, (l5), t6
    addu            l5, l5, t6
    vldm.16         vr1-vr1, (l7)
    vadd.s8.x       vr1, vr1, vr0
    vstmu.16        vr1-vr1, (l7)

.L68:
    andi            t3, t2, 7
    bez             t3, .L42
    vldx.8          vr0, (l5), t3
    addu            l5, l5, t3
    vldx.16         vr1, (l7), t3
    vadd.s8.x       vr1, vr1, vr0
    vstx.16         vr1, (l7), t3

.L42:
    addi            t4, t4, 1
    br              .L39

.L43:
    divu            t2, t9, t4          // 1/count
    vdupg.16        vr7, t2
    vabs.s16.s      vr7, vr7

    mov             l7, t5
    lsri            t2, a3, 4
    bez             t2, .L45

.L44:
    vldmu.16        vr0-vr1, (l7)       // buffer_scale_back_q15_to_q7
    vrmulh.s16.rs   vr0, vr0, vr7
    vrmulh.s16.rs   vr1, vr1, vr7
    vmov.s16.sl     vr0, vr0, vr1
    vstmu.8         vr0-vr0, (l4)

    bnezad          t2, .L44

.L45:
    andi            t3, a3, 15
    lsri            t2, t3, 3
    bez             t2, .L46

    vldmu.16        vr0-vr0, (l7)
    vrmulh.s16.rs   vr0, vr0, vr7
    vmov.s16.sl     vr0, vr0, vr0
    vstx.8          vr0, (l4), t6
    addu            l4, l4, t6

.L46:
    andi            t2, t3, 7
    bez             t2, .L47
    vldx.16         vr0, (l7), t2
    vrmulh.s16.rs   vr0, vr0, vr7
    vmov.s16.sl     vr0, vr0, vr0
    vstx.8          vr0, (l4), t2

.L47:
    addi            t1, t1, 1
    br              .L31

.L48:
    addi            t0, t0, 1
    br              .L30

.L49:
    ld.hs           l8, (sp, 0x30)      // dim_kernel_y
    ld.hs           l3, (sp, 0x38)      // padding_y
    ld.hs           l2, (sp, 0x40)      // stride_y

    mult            t3, a1, a3          // dim_im_in_x * ch_im_in
    mult            t2, l1, a3          // dim_im_out_x * ch_im_in
    movi            t0, 0

.L50:
    cmplt           t0, l9              // i_y dim_im_out_y
    bf              .L67
    mult            l4, t2, t0          // target
    addu            l4, l4, l0

    mult            l7, t0, l2
    cmplt           l7, l3              // i_y * stride_y < padding_y
    bt              .L51
    subu            l5, l7, l3          // - padding_y
    mult            l5, l5, t3
    addu            l5, l5, a0
    br              .L52

.L51:
    mov             l5, a0              // row_start

.L52:
    subu            l6, l7, l3          // - padding_y
    addu            l6, l6, l8          // + dim_kernel_y
    cmphs           l6, a2
    bf              .L53
    mult            l6, t3, a2          // * dim_im_in_y
    addu            l6, l6, a0
    br              .L54

.L53:
    mult            l6, l6, t3
    addu            l6, l6, a0          // row_end

.L54:
    mov             l7, t5
    mov             t8, l5
    addu            l5, l5, t3          // update row_start
    movi            t4, 1               // count = 1
    lsri            t1, t2, 4
    bez             t1, .L56

.L55:
    vldmu.8         vr0-vr0, (t8)       // q7_to_q15
    vmov.s8.e       vr0, vr0
    vstmu.16        vr0-vr1, (l7)

    bnezad          t1, .L55

.L56:
    andi            t1, t2, 15
    bez             t1, .L58

    vldx.8          vr0, (t8), t1
    vmov.s8.e       vr0, vr0
    cmphsi          t2, 9
    bf              .L57
    subi            t1, t1, 8
    vstmu.16        vr0-vr0, (l7)
    vstx.16         vr1, (l7), t1
    br              .L58

.L57:
    vstx.16         vr0, (l7), t1

.L58:
    cmplt           l5, l6
    bf              .L62

    mov             l7, t5
    mov             t8, l5
    lsri            t1, t2, 4
    bez             t1, .L60

.L59:
    vldmu.8         vr0-vr0, (t8)       // accumulate_q7_to_q15
    vldm.16         vr1-vr2, (l7)
    vadd.s8.x       vr1, vr1, vr0
    vstmu.16        vr1-vr2, (l7)

    bnezad          t1, .L59

.L60:
    andi            t7, t2, 15
    lsri            t1, t7, 3
    bez             t1, .L69

    vldx.8          vr0, (t8), t6
    addu            t8, t8, t6
    vldm.16         vr1-vr1, (l7)
    vadd.s8.x       vr1, vr1, vr0
    vstmu.16        vr1-vr1, (l7)

.L69:
    andi            t1, t7, 7
    bez             t1, .L61
    vldx.8          vr0, (t8), t1
    vldx.16         vr1, (l7), t1
    vadd.s8.x       vr1, vr1, vr0
    vstx.16         vr1, (l7), t1

.L61:
    addu            l5, l5, t3
    addi            t4, t4, 1
    br              .L58

.L62:
    divu            t7, t9, t4          // 1/count
    vdupg.16        vr7, t7
    vabs.s16.s      vr7, vr7
    vdupg.16        vr6, lr

    mov             l7, t5
    lsri            t1, t2, 4
    bez             t1, .L64

.L63:
    vldmu.16        vr0-vr1, (l7)       // buffer_scale_back_q15_to_q7
    vshl.s16        vr0, vr0, vr6
    vshl.s16        vr1, vr1, vr6
    vrmulh.s16.rs   vr0, vr0, vr7
    vrmulh.s16.rs   vr1, vr1, vr7
    vmov.s16.sl     vr0, vr0, vr1
    vstmu.8         vr0-vr0, (l4)

    bnezad          t1, .L63

.L64:
    andi            t7, t2, 15
    lsri            t1, t7, 3
    bez             t1, .L65

    movi            t1, 8
    vldmu.16        vr0-vr0, (l7)
    vshl.s16        vr0, vr0, vr6
    vrmulh.s16.rs   vr0, vr0, vr7
    vmov.s16.sl     vr0, vr0, vr0
    vstx.8          vr0, (l4), t1
    addu            l4, l4, t1

.L65:
    andi            t1, t7, 7
    bez             t1, .L66
    vldx.16         vr0, (l7), t1
    vshl.s16        vr0, vr0, vr6
    vrmulh.s16.rs   vr0, vr0, vr7
    vmov.s16.sl     vr0, vr0, vr0
    vstx.8          vr0, (l4), t1
    addu            l4, l4, t1

.L66:
    addi            t0, t0, 1
    br              .L50

.L67:
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           csi_xt800v_avepool_q7_HWC_nonsquare, .-csi_xt800v_avepool_q7_HWC_nonsquare
.weak csi_avepool_q7_HWC_nonsquare
.set  csi_avepool_q7_HWC_nonsquare, csi_xt800v_avepool_q7_HWC_nonsquare
.weak csky_vdsp2_avepool_q7_HWC_nonsquare
.set  csky_vdsp2_avepool_q7_HWC_nonsquare, csi_xt800v_avepool_q7_HWC_nonsquare
