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
 * @file     shl_xt800p_softmax_q15.S
 * @brief    Pooling functions implementations.
 * @version  V1.0
 * @date     01. June 20116
 ******************************************************************************/

/*
 * void shl_xt800p_softmax_q15(const q15_t * vec_in,
 *                       const uint8_t dim_vec,
 *                       q15_t * p_out)
 */

    .file           "shl_xt800p_softmax_q15.S"
    .section        .text.shl_xt800p_softmax_q15,"ax",@progbits
    .align          2
    .global         shl_xt800p_softmax_q15
    .type           shl_xt800p_softmax_q15, @function

shl_xt800p_softmax_q15:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9
    lrw             t9, 0x80008000      // init max value
    mov             l0, a0
    movi            l1, 1

    lsri            a3, a1, 3
    bez             a3, .L1

.L0:
    pldbi.d         t0, (a0)
    pldbi.d         t2, (a0)
    pmax.s16        t9, t9, t0       // max
    pmax.s16        t9, t9, t1
    pmax.s16        t9, t9, t2
    pmax.s16        t9, t9, t3

    bnezad          a3, .L0

    sexth           t0, t9          // compute the max value
    asri            t1, t9, 16
    max.s32         t9, t0, t1

.L1:
    andi            a3, a1, 7
    bez             a3, .L3

.L2:
    ldbi.hs         t0, (a0)
    cmplt           t9, t0
    movt            t9, t0

    bnezad          a3, .L2

.L3:
    subi            t9, t9, 16          // base - 16
    dup.16          t9, t9, 0
    mov             a0, l0

    movi            t8, 0               // sum = 0
    lsri            a3, a1, 2
    bez             a3, .L5

.L4:
    pldbi.d         t0, (a0)
    pcmplt.s16      t6, t9, t0
    pcmplt.s16      t7, t9, t1
    psub.s16.s      t0, t0, t9
    psub.s16.s      t1, t1, t9
    pclipi.u16      t0, t0, 5
    pclipi.u16      t1, t1, 5

    zexth           t2, t0
    lsri            t3, t0, 16
    zexth           t4, t1
    lsri            t5, t1, 16
    lsl.u32.s       t2, l1, t2
    lsl.u32.s       t3, l1, t3
    lsl.u32.s       t4, l1, t4
    lsl.u32.s       t5, l1, t5

    sexth           t0, t6
    asri            t1, t6, 16
    sexth           t6, t7
    asri            t7, t7, 16
    and             t2, t2, t0
    and             t3, t3, t1
    and             t4, t4, t6
    and             t5, t5, t7

    addu            t8, t8, t2
    addu            t8, t8, t3
    addu            t8, t8, t4
    addu            t8, t8, t5

    bnezad          a3, .L4

.L5:
    andi            a3, a1, 3
    bez             a3, .L7

.L6:
    ldbi.h          t0, (a0)
    pcmplt.s16      t4, t9, t0
    psub.s16.s      t0, t0, t9
    pclipi.u16      t0, t0, 5
    zexth           t0, t0
    sexth           t4, t4
    lsl.u32.s       t0, l1, t0
    and             t0, t0, t4
    add.s32.s       t8, t8, t0

    bnezad          a3, .L6

.L7:
    movi            t0, 0
    movi            t1, 1
    sexth           l2, t9
    addi            l2, l2, 17          // 17 + base
    dup.16          l2, l2, 0
    divsl           t0, t0, t8          // output_base
    mov             t8, t0
    movi            l3, 0

    mov             a0, l0
    lsri            a3, a1, 3
    bez             a3, .L9

.L8:
    pldbi.d         t0, (a0)
    pldbi.d         t2, (a0)
    pcmplt.s16      t4, t9, t0          // if(vec[in] > base)
    pcmplt.s16      t5, t9, t1
    pcmplt.s16      t6, t9, t2
    pcmplt.s16      t7, t9, t3

    psub.s16.s      t0, l2, t0          // shift
    psub.s16.s      t1, l2, t1
    psub.s16.s      t2, l2, t2
    psub.s16.s      t3, l2, t3
    pclipi.u16      t0, t0, 5
    pclipi.u16      t1, t1, 5
    pclipi.u16      t2, t2, 5
    pclipi.u16      t3, t3, 5

    pkgll           l8, t0, l3
    pkghh           l9, t0, l3
    pkgll           t0, t1, l3
    pkghh           t1, t1, l3

    asr             l8, t8, l8
    asr             l9, t8, l9
    asr             t0, t8, t0
    asr             t1, t8, t1
    clipi.s32       l8, l8, 16
    clipi.s32       l9, l9, 16
    clipi.s32       t0, t0, 16
    clipi.s32       t1, t1, 16
    pkgll           l8, l8, l9
    pkgll           l9, t0, t1
    and             l8, l8, t4
    and             l9, l9, t5
    stbi.w          l8, (a2)
    stbi.w          l9, (a2)

    pkgll           l8, t2, l3
    pkghh           l9, t2, l3
    pkgll           t0, t3, l3
    pkghh           t1, t3, l3

    asr             l8, t8, l8
    asr             l9, t8, l9
    asr             t0, t8, t0
    asr             t1, t8, t1
    clipi.s32       l8, l8, 16
    clipi.s32       l9, l9, 16
    clipi.s32       t0, t0, 16
    clipi.s32       t1, t1, 16
    pkgll           l8, l8, l9
    pkgll           l9, t0, t1
    and             l8, l8, t6
    and             l9, l9, t7
    stbi.w          l8, (a2)
    stbi.w          l9, (a2)

    bnezad          a3, .L8

.L9:
    andi            a3, a1, 7
    bez             a3, .L11

.L10:
    ldbi.h          t0, (a0)
    pcmplt.s16      t4, t9, t0          // if(vec[in] > base)
    psub.s16.s      t0, l2, t0          // shift
    pclipi.u16      t0, t0, 5
    pkgll           l8, t0, l3
    asr             l8, t8, l8
    clipi.s32       l8, l8, 16
    and             l8, l8, t4
    stbi.h          l8, (a2)

    bnezad          a3, .L10

.L11:
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9
    .size           shl_xt800p_softmax_q15, .-shl_xt800p_softmax_q15

.weak csky_dsp2_softmax_q15
.set  csky_dsp2_softmax_q15, shl_xt800p_softmax_q15
