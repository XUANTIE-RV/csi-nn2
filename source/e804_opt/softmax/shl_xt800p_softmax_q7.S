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
 * @file     shl_xt800p_softmax_q7.S
 * @brief    Pooling functions implementations.
 * @version  V1.0
 * @date     04. June 2018
 ******************************************************************************/

/*
 * void shl_xt800p_softmax_q7(const q7_t * vec_in,
 *                      const uint16_t dim_vec,
 *                      q7_t * p_out)
 */

    .file           "shl_xt800p_softmax_q7.S"
    .section        .text.shl_xt800p_softmax_q7,"ax",@progbits
    .align          2
    .global         shl_xt800p_softmax_q7
    .type           shl_xt800p_softmax_q7, @function

shl_xt800p_softmax_q7:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9
    lrw             t9, 0x80808080      // init max value
    mov             l0, a0
    movi            l1, 1

    lsri            a3, a1, 4
    bez             a3, .L1

.L0:
    pldbi.d         t0, (a0)
    pldbi.d         t2, (a0)
    pmax.s8         t9, t9, t0       // max
    pmax.s8         t9, t9, t1
    pmax.s8         t9, t9, t2
    pmax.s8         t9, t9, t3

    bnezad          a3, .L0
    pext.s8.e       t0, t9
    pmax.s16        t9, t0, t1
    sexth           t0, t9
    asri            t1, t9, 16
    max.s32         t9, t0, t1

.L1:
    andi            a3, a1, 15
    bez             a3, .L3

.L2:
    ldbi.bs         t0, (a0)
    max.s32         t9, t9, t0

    bnezad          a3, .L2

.L3:
    subi            t9, t9, 8
    mov             a0, l0
    dup.8           t9, t9, 0

    movi            t8, 0               // sum = 0
    lsri            a3, a1, 2
    bez             a3, .L5

.L4:
    ldbi.w          t0, (a0)
    pcmplt.s8       t1, t9, t0
    psub.s8.s       t0, t0, t9
    pext.s8.e       t4, t0
    pclipi.u16      t4, t4, 5
    pclipi.u16      t5, t5, 5

    zexth           t6, t4
    lsri            t7, t4, 16
    zexth           l2, t5
    lsri            l3, t5, 16
    lsl.u32.s       t6, l1, t6
    lsl.u32.s       t7, l1, t7
    lsl.u32.s       l2, l1, l2
    lsl.u32.s       l3, l1, l3

    pext.s8.e       t2, t1
    sexth           l4, t2
    asri            l5, t2, 16
    sexth           l6, t3
    asri            l7, t3, 16

    and             t6, t6, l4
    and             t7, t7, l5
    and             l2, l2, l6
    and             l3, l3, l7

    addu            t8, t8, t6
    addu            t8, t8, t7
    addu            t8, t8, l2
    addu            t8, t8, l3

    bnezad          a3, .L4

.L5:
    andi            a3, a1, 3
    bez             a3, .L7

.L6:
    ldbi.b          t0, (a0)
    pcmplt.s8       t4, t9, t0
    psub.s8.s       t0, t0, t9
    pext.s8.e       t2, t0
    pclipi.u16      t2, t2, 5

    zexth           t2, t2
    lsl.u32.s       t2, l1, t2
    pext.s8.e       t4, t4
    sexth           t4, t4
    and             t2, t2, t4

    add.s32.s       t8, t8, t2

    bnezad          a3, .L6

.L7:
    movi            t0, 1
    lsli            t0, t0, 20          // output_base
    divs            t8, t0, t8
    pext.s8.e       l2, t9
    lrw             l3, 0x0d000d
    addu            l2, l2, l3          // 13 + base
    movi            l3, 0

    mov             a0, l0
    lsri            a3, a1, 3
    bez             a3, .L9

.L8:
    pldbi.d         t0, (a0)
    pcmplt.s8       t4, t9, t0          // if(vec[in] > base)
    pcmplt.s8       t5, t9, t1

    pext.s8.e       t2, t1              // shift
    pext.s8.e       t0, t0
    psub.s16.s      t0, l2, t0
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
    clipi.s32       l8, l8, 8
    clipi.s32       l9, l9, 8
    clipi.s32       t0, t0, 8
    clipi.s32       t1, t1, 8
    pkgll           l8, l8, l9
    pkgll           l9, t0, t1
    narl            l8, l8, l9
    and             l8, l8, t4
    stbi.w          l8, (a2)

    pkgll           l8, t2, l3
    pkghh           l9, t2, l3
    pkgll           t0, t3, l3
    pkghh           t1, t3, l3

    asr             l8, t8, l8
    asr             l9, t8, l9
    asr             t0, t8, t0
    asr             t1, t8, t1
    clipi.s32       l8, l8, 8
    clipi.s32       l9, l9, 8
    clipi.s32       t0, t0, 8
    clipi.s32       t1, t1, 8
    pkgll           l8, l8, l9
    pkgll           l9, t0, t1
    narl            l8, l8, l9
    and             l8, l8, t5
    stbi.w          l8, (a2)

    bnezad          a3, .L8

.L9:
    andi            a3, a1, 7
    bez             a3, .L11

.L10:
    ldbi.b          t0, (a0)
    pcmplt.s8       t4, t9, t0          // if(vec[in] > base)

    pext.s8.e       t0, t0
    psub.s16.s      t0, l2, t0          // shift
    pclipi.u16      t0, t0, 5
    asr             l8, t8, t0
    clipi.s32       l8, l8, 8
    and             l8, l8, t4
    stbi.b          l8, (a2)

    bnezad          a3, .L10

.L11:
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9
    .size           shl_xt800p_softmax_q7, .-shl_xt800p_softmax_q7

.weak csky_dsp2_softmax_q7
.set  csky_dsp2_softmax_q7, shl_xt800p_softmax_q7
