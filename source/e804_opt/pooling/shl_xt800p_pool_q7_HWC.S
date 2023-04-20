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
 * @file     shl_xt800p_pool_q7_HWC.S
 * @brief    Pooling functions implementations.
 * @version  V1.0
 * @date     31. May 2018
 ******************************************************************************/

/*
 * void
 * shl_xt800p_maxpool2d_q7_HWC(q7_t * Im_in,
 *                    const uint16_t dim_im_in,
 *                    const uint16_t ch_im_in,
 *                    const uint16_t dim_kernel,
 *                    const uint16_t padding,
 *                    const uint16_t stride,
 *                    const uint16_t dim_im_out,
 *                    q7_t * bufferA,
 *                    q7_t * Im_out)
 */

    .file           "shl_xt800p_pool_HWC_q7.S"
    .section        .text.shl_xt800p_maxpool2d_q7_HWC,"ax",@progbits
    .align          2
    .global         shl_xt800p_maxpool2d_q7_HWC
    .type           shl_xt800p_maxpool2d_q7_HWC, @function

shl_xt800p_maxpool2d_q7_HWC:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    ld.w            l0, (sp, 0x3c)      // im_out
    ld.hs           l1, (sp, 0x34)      // dim_im_out
    ld.hs           l2, (sp, 0x30)      // stride
    ld.hs           l3, (sp, 0x2c)      // padding

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
    lsri            t2, a2, 3
    bez             t2, .L8

.L7:
    pldbi.d         l8, (l6)       // memmove
    stbi.w          l8, (l5)
    stbi.w          l9, (l5)

    bnezad          t2, .L7

.L8:
    andi            t2, a2, 7
    bez             t2, .L9

.L29:
    ldbi.b          l8, (l6)
    stbi.b          l8, (l5)

    bnezad          t2, .L29

.L9:
    cmplt           l6, l7              // win_start < win_stop
    bf              .L14

.L10:
    mov             l5, l4
    lsri            t2, a2, 3
    bez             t2, .L12

.L11:
    ld.w            t7, (l5, 0x0)       // compare_and_replace_if_larger_q7
    ld.w            t8, (l5, 0x4)       // compare_and_replace_if_larger_q7
    pldbi.d         l8, (l6)
    pmax.s8         l8, l8, t7
    pmax.s8         l9, l9, t8
    stbi.w          l8, (l5)
    stbi.w          l9, (l5)

    bnezad          t2, .L11

.L12:
    andi            t2, a2, 7
    bez             t2, .L13

.L70:
    ld.b            l8, (l5, 0x0)
    ldbi.b          l9, (l6)
    pmax.s8         l8, l8, l9
    stbi.b          l8, (l5)

    bnezad          t2, .L70

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
    lsri            t1, t2, 3
    bez             t1, .L22

.L21:
    pldbi.d         l8, (t4)
    stbi.w          l8, (l7)
    stbi.w          l9, (l7)

    bnezad          t1, .L21

.L22:
    andi            t1, t2, 7
    bez             t1, .L23

.L71:
    ldbi.b          l8, (t4)
    stbi.b          l8, (l7)

    bnezad          t1, .L71

.L23:
    cmplt           l5, l6
    bf              .L27

    mov             l7, l4
    mov             t4, l5
    lsri            t1, t2, 3
    bez             t1, .L25

.L24:
    ld.w            t7, (l7, 0x0)       // compare_and_replace
    ld.w            t8, (l7, 0x4)
    pldbi.d         l8, (t4)
    pmax.s8         l8, l8, t7
    pmax.s8         l9, l9, t8
    stbi.w          l8, (l7)
    stbi.w          l9, (l7)

    bnezad          t1, .L24

.L25:
    andi            t1, t2, 15
    bez             t1, .L26

.L72:
    ld.b            l8, (l7, 0x0)
    ldbi.b          l9, (t4)
    pmax.s8         l8, l8, l9
    stbi.b          l8, (l7)

    bnezad          t1, .L72

.L26:
    addu            l5, l5, t3
    br              .L23

.L27:
    addi            t0, t0, 1
    br              .L16

.L28:
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           shl_xt800p_maxpool2d_q7_HWC, .-shl_xt800p_maxpool2d_q7_HWC

.weak csky_dsp2_maxpool2d_q7_HWC
.set  csky_dsp2_maxpool2d_q7_HWC, shl_xt800p_maxpool2d_q7_HWC

/*
 * void
 * shl_xt800p_avepool_q7_HWC(q7_t * Im_in,
 *                    const uint16_t dim_im_in,
 *                    const uint16_t ch_im_in,
 *                    const uint16_t dim_kernel,
 *                    const uint16_t padding,
 *                    const uint16_t stride,
 *                    const uint16_t dim_im_out,
 *                    q7_t * bufferA,
 *                    q7_t * Im_out)
 */

    .section        .text.shl_xt800p_avepool_q7_HWC,"ax",@progbits
    .align          2
    .global         shl_xt800p_avepool_q7_HWC
    .type           shl_xt800p_avepool_q7_HWC, @function

shl_xt800p_avepool_q7_HWC:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    ld.w            l0, (sp, 0x3c)      // im_out
    ld.w            lr, (sp, 0x38)      // bufferA
    ld.hs           l1, (sp, 0x34)      // dim_im_out
    ld.hs           l2, (sp, 0x30)      // stride
    ld.hs           l3, (sp, 0x2c)      // padding
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
    mov             l7, lr
    movi            t4, 1               // count = 1
    lsri            t2, a2, 3
    bez             t2, .L37

.L36:
    pldbi.d         l8, (l5)            // q7_to_q15
    pext.s8.e       t5, l8
    stbi.w          t5, (l7)
    stbi.w          t6, (l7)

    pext.s8.e       t5, l9
    stbi.w          t5, (l7)
    stbi.w          t6, (l7)

    bnezad          t2, .L36

.L37:
    andi            t2, a2, 15
    bez             t2, .L39

.L38:
    ldbi.bs         l8, (l5)
    stbi.h          l8, (l7)

    bnezad          t2, .L38

.L39:
    cmplt           l5, l6
    bf              .L43

    mov             l7, lr
    lsri            t2, a2, 2
    bez             t2, .L41

.L40:
    ldbi.w          l8, (l5)            // accumulate_q7_to_q15
    ld.w            t5, (l7, 0x0)
    pext.s8.e       l8, l8
    ld.w            t6, (l7, 0x4)
    padd.s16.s      t5, t5, l8
    padd.s16.s      t6, t6, l9
    stbi.w          t5, (l7)
    stbi.w          t6, (l7)

    bnezad          t2, .L40

.L41:
    andi            t2, a2, 3
    bez             t2, .L42

.L68:
    ldbi.bs         l8, (l5)
    ld.hs           l9, (l7, 0x0)
    padd.s16.s      l8, l8, l9
    stbi.h          l8, (l7)

    bnezad          t2, .L68

.L42:
    addi            t4, t4, 1
    br              .L39

.L43:
    divu            t2, t9, t4          // 1/count
    dup.16          t5, t2, 0
    pabs.s16.s      t5, t5

    mov             l7, lr
    lsri            t2, a2, 2
    bez             t2, .L45

.L44:
    pldbi.d         l8, (l7)            // buffer_scale_back_q15_to_q7
    prmul.s16.rh    l8, l8, t5
    prmul.s16.rh    l9, l9, t5
    narl            l8, l8, l9
    stbi.w          l8, (l4)

    bnezad          t2, .L44

.L45:
    andi            t2, a2, 3
    bez             t2, .L47

.L46:
    ldbi.h          l8, (l7)
    prmul.s16.rh    l8, l8, t5
    stbi.b          l8, (l4)

.L47:
    addi            t1, t1, 1
    br              .L31

.L48:
    addi            t0, t0, 1
    br              .L30

    /*  y axis */
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
    mov             l7, lr
    mov             t8, l5
    addu            l5, l5, t3          // update row_start
    movi            t4, 1               // count = 1
    lsri            t1, t2, 3
    bez             t1, .L56

.L55:
    pldbi.d         l8, (t8)            // q7_to_q15
    pext.s8.e       t5, l8
    stbi.w          t5, (l7)
    stbi.w          t6, (l7)
    pext.s8.e       t5, l9
    stbi.w          t5, (l7)
    stbi.w          t6, (l7)

    bnezad          t1, .L55

.L56:
    andi            t1, t2, 7
    bez             t1, .L58

.L57:
    ldbi.bs         l8, (t8)
    stbi.h          l8, (l7)

    bnezad          t1, .L57

.L58:
    cmplt           l5, l6
    bf              .L62

    mov             l7, lr
    mov             t8, l5
    lsri            t1, t2, 2
    bez             t1, .L60

.L59:
    ldbi.w          l8, (t8)            // accumulate_q7_to_q15
    pext.s8.e       l8, l8
    ld.w            t5,  (l7, 0x0)
    ld.w            t6,  (l7, 0x4)
    padd.s16.s      t5, t5, l8
    padd.s16.s      t6, t6, l9
    stbi.w          t5, (l7)
    stbi.w          t6, (l7)

    bnezad          t1, .L59

.L60:
    andi            t1, t2, 3
    bez             t1, .L61

.L69:
    ldbi.bs         l8, (t8)
    ld.hs           l9, (l7, 0x0)
    padd.s16.s      l8, l8, l9
    stbi.h          l8, (l7)

    bnezad          t1, .L69

.L61:
    addu            l5, l5, t3
    addi            t4, t4, 1
    br              .L58

.L62:
    divu            t7, t9, t4          // 1/count
    dup.16          t5, t7, 0
    pabs.s16.s      t5, t5

    mov             l7, lr
    lsri            t1, t2, 2
    bez             t1, .L64

.L63:
    pldbi.d         l8, (l7)            // buffer_scale_back_q15_to_q7
    prmul.s16.rh    l8, l8, t5
    prmul.s16.rh    l9, l9, t5
    narl            l8, l8, l9
    stbi.w          l8, (l4)

    bnezad          t1, .L63

.L64:
    andi            t1, t2, 3
    bez             t1, .L66

.L65:
    ldbi.h          l8, (l7)
    prmul.s16.rh    l8, l8, t5
    stbi.b          l8, (l4)

    bnezad          t1, .L65

.L66:
    addi            t0, t0, 1
    br              .L50

.L67:
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           shl_xt800p_avepool_q7_HWC, .-shl_xt800p_avepool_q7_HWC

.weak csky_dsp2_avepool_q7_HWC
.set  csky_dsp2_avepool_q7_HWC, shl_xt800p_avepool_q7_HWC
