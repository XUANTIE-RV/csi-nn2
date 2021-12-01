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
 * @file     csi_xt800v_pool_q7_HWC.S
 * @brief    Pooling functions implementations.
 * @version  V1.0
 * @date     31. May 2018
 ******************************************************************************/

/*
 * void
 * csi_xt800v_maxpool2d_q7_HWC(q7_t * Im_in,
 *                    const uint16_t dim_im_in,
 *                    const uint16_t ch_im_in,
 *                    const uint16_t dim_kernel,
 *                    const uint16_t padding,
 *                    const uint16_t stride,
 *                    const uint16_t dim_im_out,
 *                    q7_t * bufferA,
 *                    q7_t * Im_out)
 */

    .file           "csi_xt800v_pool_HWC_q7.S"
    .section        .text.csi_xt800v_maxpool2d_q7_HWC,"ax",@progbits
    .align          2
    .global         csi_xt800v_maxpool2d_q7_HWC
    .type           csi_xt800v_maxpool2d_q7_HWC, @function

csi_xt800v_maxpool2d_q7_HWC:
    push            l0, l1, l2, l3, l4, l5, l6, l7
    ld.w            l0, (sp, 0x30)      // im_out
    ld.hs           l1, (sp, 0x28)      // dim_im_out
    ld.hs           l2, (sp, 0x24)      // stride
    ld.hs           l3, (sp, 0x20)      // padding

    /* pooling along x axis*/
    movi            t0, 0               // i_y = 0

.L0:
    cmplt           t0, a1              // i_y < dim_im_in
    bf              .L1
    movi            t1, 0               // i_x = 0

.L2:
    cmplt           t1, l1              // i_x < dim_im_out
    bf              .L15
    mult            l7, t0, a1
    addu            l4, l7, t1
    mult            l4, l4, a2
    addu            l4, l4, a0          // target

    mult            l5, t1, l2
    cmplt           l5, l3              // i_x*stride < padding
    bt              .L3
    mult            l6, t1, l2
    addu            l6, l6, l7
    subu            l6, l6, l3
    mult            l6, l6, a2
    addu            l6, l6, a0          // win_start
    br              .L4

.L3:
    mov             l6, l4

.L4:
    subu            t2, l5, l3
    addu            t2, t2, a3
    cmphs           t2, a1
    bf              .L5
    addu            l7, l7, a1
    mult            l7, l7, a2
    addu            l7, l7, a0          // win_stop
    br              .L6

.L5:
    addu            l7, l7, l5
    subu            l7, l7, l3
    addu            l7, l7, a3
    mult            l7, l7, a2
    addu            l7, l7, a0

.L6:
    mov             l5, l4
    lsri            t2, a2, 4
    bez             t2, .L8

.L7:
    vldmu.8         vr0-vr0, (l6)       // memmove
    vstmu.8         vr0-vr0, (l5)

    bnezad          t2, .L7

.L8:
    andi            t2, a2, 15
    bez             t2, .L9

    vldx.8          vr0, (l6), t2
    vstx.8          vr0, (l5), t2
    addu            l6, l6, t2

.L9:
    cmplt           l6, l7              // win_start < win_stop
    bf              .L14

.L10:
    mov             l5, l4
    lsri            t2, a2, 4
    bez             t2, .L12

.L11:
    vldm.8          vr0-vr0, (l5)       // compare_and_replace_if_larger_q7
    vldmu.8         vr1-vr1, (l6)
    vmax.s8         vr3, vr0, vr1
    vstmu.8         vr3-vr3, (l5)

    bnezad          t2, .L11

.L12:
    andi            t2, a2, 15
    bez             t2, .L13
    vldx.8          vr0, (l5), t2
    vldx.8          vr1, (l6), t2
    vmax.s8         vr3, vr0, vr1
    vstx.8          vr3, (l5), t2
    addu            l6, l6, t2
    addu            l5, l5, t2

.L13:
    br              .L9                 // for loop

.L14:
    addi            t1, t1, 1
    br              .L2

.L15:
    addi            t0, t0, 1
    br              .L0

    /* pooling along y axis*/
.L1:
    mult            t2, l1, a2          // dim_im_out * ch_im_in
    mult            t3, a1, a2          // dim_im_in * ch_im_in
    movi            t0, 0               // i_y = 0

.L16:
    cmplt           t0, l1              // i_y < dim_im_out
    bf              .L28

    mult            l4, t0, t2          // target
    addu            l4, l4, l0

    mult            l7, t0, l2
    cmplt           l7, l3              // i_y* stride -padding < 0
    bt              .L17
    subu            l5, l7, l3
    mult            l5, l5, t3
    addu            l5, l5, a0
    br              .L18

.L17:
    mov             l5, a0              // row_start

.L18:
    subu            l6, l7, l3
    addu            l6, l6, a3
    cmphs           l6, a1
    bf              .L19
    mult            l6, a1, t3
    addu            l6, l6, a0
    br              .L20

.L19:
    subu            l6, l7, l3
    addu            l6, l6, a3
    mult            l6, l6, t3          // row_end
    addu            l6, l6, a0

.L20:
    mov             l7, l4              // memmove
    mov             t4, l5
    addu            l5, l5, t3
    lsri            t1, t2, 4
    bez             t1, .L22

.L21:
    vldmu.8         vr0-vr0, (t4)
    vstmu.8         vr0-vr0, (l7)

    bnezad          t1, .L21

.L22:
    andi            t1, t2, 15
    bez             t1, .L23

    vldx.8          vr0, (t4), t1
    vstx.8          vr0, (l7), t1

.L23:
    cmplt           l5, l6
    bf              .L27

    mov             l7, l4
    mov             t4, l5
    lsri            t1, t2, 4
    bez             t1, .L25

.L24:
    vldm.8          vr0-vr0, (l7)       // compare_and_replace
    vldmu.8         vr1-vr1, (t4)
    vmax.s8         vr3, vr0, vr1
    vstmu.8         vr3-vr3, (l7)

    bnezad          t1, .L24

.L25:
    andi            t1, t2, 15
    bez             t1, .L26

    vldx.8          vr0, (l7), t1
    vldx.8          vr1, (t4), t1
    vmax.s8         vr3, vr0, vr1
    vstx.8          vr3, (l7), t1
    addu            t4, t4, t1
    addu            l7, l7, t1

.L26:
    addu            l5, l5, t3
    br              .L23

.L27:
    addi            t0, t0, 1
    br              .L16

.L28:
    pop             l0, l1, l2, l3, l4, l5, l6, l7
    .size           csi_xt800v_maxpool2d_q7_HWC, .-csi_xt800v_maxpool2d_q7_HWC

.weak csi_maxpool2d_q7_HWC
.set  csi_maxpool2d_q7_HWC, csi_xt800v_maxpool2d_q7_HWC
.weak csky_vdsp2_maxpool2d_q7_HWC
.set  csky_vdsp2_maxpool2d_q7_HWC, csi_xt800v_maxpool2d_q7_HWC

/*
 * void
 * csi_xt800v_avepool_q7_HWC(q7_t * Im_in,
 *                    const uint16_t dim_im_in,
 *                    const uint16_t ch_im_in,
 *                    const uint16_t dim_kernel,
 *                    const uint16_t padding,
 *                    const uint16_t stride,
 *                    const uint16_t dim_im_out,
 *                    q7_t * bufferA,
 *                    q7_t * Im_out)
 */

    .section        .text.csi_xt800v_avepool_q7_HWC,"ax",@progbits
    .align          2
    .global         csi_xt800v_avepool_q7_HWC
    .type           csi_xt800v_avepool_q7_HWC, @function

csi_xt800v_avepool_q7_HWC:
    push            l0, l1, l2, l3, l4, l5, l6, l7
    ld.w            l0, (sp, 0x30)      // im_out
    ld.w            t5, (sp, 0x2c)      // bufferA
    ld.hs           l1, (sp, 0x28)      // dim_im_out
    ld.hs           l2, (sp, 0x24)      // stride
    ld.hs           l3, (sp, 0x20)      // padding
    movi            t6, 8
    movi            t9, 1
    lsli            t9, t9, 15

    /* pooling along x axis*/
    movi            t0, 0               // i_y = 0

.L30:
    cmplt           t0, a1              // i_y < dim_im_in
    bf              .L49

    movi            t1, 0               // i_x = 0

.L31:
    cmplt           t1, l1              // i_x < dim_im_out
    bf              .L48

    mult            l7, t0, a1          // target
    addu            l4, l7, t1
    mult            l4, l4, a2
    addu            l4, l4, a0

    mult            l6, t1, l2
    cmplt           l6, l3
    bt              .L32
    addu            l5, l6, l7
    subu            l5, l5, l3
    mult            l5, l5, a2
    addu            l5, l5, a0
    br              .L33

.L32:
    mov             l5, l4              // win_start

.L33:
    subu            t2, l6, l3
    addu            t2, t2, a3
    cmphs           t2, a1
    bf              .L34
    addu            l6, l7, a1
    mult            l6, l6, a2
    addu            l6, l6, a0
    br              .L35

.L34:
    addu            l6, l7, l6
    subu            l6, l6, l3
    addu            l6, l6, a3
    mult            l6, l6, a2
    addu            l6, l6, a0          // win_stop

.L35:
    mov             l7, t5
    movi            t4, 1               // count = 1
    lsri            t2, a2, 4
    bez             t2, .L37

.L36:
    vldmu.8         vr0-vr0, (l5)       // q7_to_q15
    vmov.s8.e       vr0, vr0
    vstmu.16        vr0-vr1, (l7)

    bnezad          t2, .L36

.L37:
    andi            t2, a2, 15
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
    lsri            t2, a2, 4
    bez             t2, .L41

.L40:
    vldmu.8         vr0-vr0, (l5)       // accumulate_q7_to_q15
    vldm.16         vr1-vr2, (l7)
    vadd.s8.x       vr1, vr1, vr0
    vstmu.16        vr1-vr2, (l7)

    bnezad          t2, .L40

.L41:
    andi            t2, a2, 15
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
    lsri            t2, a2, 4
    bez             t2, .L45

.L44:
    vldmu.16        vr0-vr1, (l7)       // buffer_scale_back_q15_to_q7
    vrmulh.s16.rs   vr0, vr0, vr7
    vrmulh.s16.rs   vr1, vr1, vr7
    vmov.16.l       vr0, vr0, vr1
    vstmu.8         vr0-vr0, (l4)

    bnezad          t2, .L44

.L45:
    andi            t3, a2, 15
    lsri            t2, t3, 3
    bez             t2, .L46

    vldmu.16        vr0-vr0, (l7)
    vrmulh.s16.rs   vr0, vr0, vr7
    vmov.16.l       vr0, vr0, vr0
    vstx.8          vr0, (l4), t6
    addu            l4, l4, t6

.L46:
    andi            t2, t3, 7
    bez             t2, .L47
    vldx.16         vr0, (l7), t2
    vrmulh.s16.rs   vr0, vr0, vr7
    vmov.16.l       vr0, vr0, vr0
    vstx.8          vr0, (l4), t2

.L47:
    addi            t1, t1, 1
    br              .L31

.L48:
    addi            t0, t0, 1
    br              .L30

.L49:
    mult            t3, a1, a2          // dim_im_in * ch_im_in
    mult            t2, l1, a2          // dim_im_out * ch_im_in
    movi            t0, 0

.L50:
    cmplt           t0, l1
    bf              .L67
    mult            l4, t2, t0          // target
    addu            l4, l4, l0

    mult            l7, t0, l2
    cmplt           l7, l3              // i_y * stride < padding
    bt              .L51
    subu            l5, l7, l3
    mult            l5, l5, t3
    addu            l5, l5, a0
    br              .L52

.L51:
    mov             l5, a0              // row_start

.L52:
    subu            l6, l7, l3
    addu            l6, l6, a3
    cmphs           l6, a1
    bf              .L53
    mult            l6, t3, a1
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

    mov             l7, t5
    lsri            t1, t2, 4
    bez             t1, .L64

.L63:
    vldmu.16        vr0-vr1, (l7)       // buffer_scale_back_q15_to_q7
    vrmulh.s16.rs   vr0, vr0, vr7
    vrmulh.s16.rs   vr1, vr1, vr7
    vmov.16.l       vr0, vr0, vr1
    vstmu.8         vr0-vr0, (l4)

    bnezad          t1, .L63

.L64:
    andi            t7, t2, 15
    lsri            t1, t7, 3
    bez             t1, .L65

    movi            t1, 8
    vldmu.16        vr0-vr0, (l7)
    vrmulh.s16.rs   vr0, vr0, vr7
    vmov.16.l       vr0, vr0, vr0
    vstx.8          vr0, (l4), t1
    addu            l4, l4, t1

.L65:
    andi            t1, t7, 7
    bez             t1, .L66
    vldx.16         vr0, (l7), t1
    vrmulh.s16.rs   vr0, vr0, vr7
    vmov.16.l       vr0, vr0, vr0
    vstx.8          vr0, (l4), t1
    addu            l4, l4, t1

.L66:
    addi            t0, t0, 1
    br              .L50

.L67:
    pop             l0, l1, l2, l3, l4, l5, l6, l7
    .size           csi_xt800v_avepool_q7_HWC, .-csi_xt800v_avepool_q7_HWC
.weak csi_avepool_q7_HWC
.set  csi_avepool_q7_HWC, csi_xt800v_avepool_q7_HWC
.weak csky_vdsp2_avepool_q7_HWC
.set  csky_vdsp2_avepool_q7_HWC, csi_xt800v_avepool_q7_HWC
