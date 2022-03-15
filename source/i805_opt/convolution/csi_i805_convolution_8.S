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

/* CSI-NN2 version 1.12.x */

/******************************************************************************
 * @file     csi_i805_convolution_8.S
 * @brief    uint8 basic convolution layer function.
 * @version  V1.0
 * @date     9. Jul 2021
 ******************************************************************************/
/*

    void csi_i805_conv2d_opt_u8(uint8_t * input_data,
                                uint8_t * kernel_data,
                                int32_t * bias_data,
                                uint8_t * output_data,
                                uint8_t * bufferA,
                                int32_t  input_h,
                                int32_t  input_w,
                                int32_t  input_ch,
                                int32_t  kernel_h,
                                int32_t  kernel_w,
                                int32_t  pad_h,
                                int32_t  pad_w,
                                int32_t  stride_h,
                                int32_t  stride_w,
                                int32_t  out_h,
                                int32_t  out_w,
                                int32_t  out_c,
                                int32_t  input_zero_point,
                                int32_t  weight_zero_point,
                                int32_t  output_zero_point,
                                int32_t  out_mult,
                                int32_t  out_shift);

    Algorithm works as follows:
        (1) partition im2col(one by one) + vec matrix mult
        (2) input_col[vector] * kernel[matrix]:
                1 x 4lines: 1 x [4*loop_16 + 4*tail_16]  --> loop kernel_row / 4
                1 x 1lines: 1 x [1*loop_16 + 1*tail_16]  --> loop kernel_row % 4

    constraints:
        dilation_h = dilation_w = 1

    register definition:
        t0: i_out_h
        t1: i_out_w
        t2: i_ker_h
        t4: i_ker_w
        l0: bufferA
        l1: input_ch
        l2: input_h
        l3: input_w
        l4: kernel_h
        l5: kernel_w
        l9: input_ch * kernel_h * kernel_w
        vr0: input data / output_temp
        vr1-vr4: weight data / accumulate temp for q1 * q2
        vr10: bias data
        vr5: accumulate temp for q1 * z2
        vr6-vr9: accumulate temp for q2 * z1
        vr11: input_zeropoint
        vr12: kernel_zeropoint
        vr13: output_mult
        vr14: output_shift
        vr15: constant for z1 * z2 * k

    TODO: support per-channel quantization

*/


    .file           "csi_i805_convolution_8.S"
    .section        .text.csi_i805_conv2d_opt_u8,"ax",@progbits
    .align          2
    .global         csi_i805_conv2d_opt_u8
    .type           csi_i805_conv2d_opt_u8, @function

csi_i805_conv2d_opt_u8:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    subi            sp, sp, 64
    vstm.8          vr8-vr11, (sp)
    subi            sp, sp, 64
    vstm.8          vr12-vr15, (sp)

    ld.w            t0, (sp, 0xe0)      // input_zero_point
    ld.w            t1, (sp, 0xe4)      // weight_zero_point
    ld.w            l2, (sp, 0xec)      // out_mult
    ld.w            l3, (sp, 0xf0)      // out_shift

    vdupg.8         vr11, t0
    vdupg.8         vr12, t1
    vdupg.32        vr13, l2
    vdupg.32        vr14, l3

    ld.w            l0, (sp, 0xac)      // bufferA
    ld.w            l2, (sp, 0xb0)      // input_h
    ld.w            l3, (sp, 0xb4)      // input_w
    ld.w            l1, (sp, 0xb8)      // input_ch

    ld.w            l4, (sp, 0xbc)      // kernel_h
    ld.w            l5, (sp, 0xc0)      // kernel_w
    mult            l9, l4, l5          // ker_h * ker_w
    mult            l9, l9, l1          // ker_h * ker_w * in_ch

    // z1 * z2 * k  constraints: z1 * z2 * k < int32
    mult            t2, t0, t1
    mult            t2, t2, l9          // t3 = z1 * z2 * k
    vdupg.32        vr15, t2            // v10[0..3] = z1 * z2 * k

    movi            t0, 0               // i_out_h
    mov             lr, l0              // im2col_buffer bufferA

.OUT_H:
    ld.w            t8, (sp, 0xd4)      // t8 = out_h   (out_h used for outer loop, load from stack)
    cmplt           t0, t8              // i_out_h < out_h
    bf              .BATCH_END          //

    movi            t1, 0               // i_out_w

.OUT_W:
    ld.w            t9, (sp, 0xd8)      // t9 = out_w
    cmplt           t1, t9              // i_out_w < out_w
    bf              .OUT_H_ADD          // jump to i_out_h++

    ld.w            t8, (sp, 0xc4)      // pad_h
    ld.w            t9, (sp, 0xcc)      // stride_h

    mult            t2, t0, t9          // i_out_h * stride_h
    subu            t2, t2, t8          // i_kernel_h = i_out_h * stride_h - pad_h
    addu            t3, t2, l4          // i_out_h * stride_h - pad_h + kernel_h

.KERNEL_H:
    cmplt           t2, t3              // i_ker_h < i_out_h * stride_h - pad_h + kernel_h
    bf              .IN_KER             // C flag = 0 ?

    ld.w            t8, (sp, 0xc8)      // pad_w
    ld.w            t9, (sp, 0xd0)      // stride_w

    mult            t4, t1, t9          // i_out_w * stride_w
    subu            t4, t4, t8          // i_out_w * stride_w - pad_w
    addu            t5, t4, l5          // i_out_w * stride_w - pad_w + kernel_w

.KERNEL_W:
    cmplt           t4, t5              // i_ker_w < i_out_w * stride_w - pad_w + kernel_w
    bf              .KERNEL_H_ADD

    // if (i_ker_h < 0 || i_ker_h >= input_h || i_ker_w < 0 || i_ker_w >= input_w)
    // jump to im2col_padding
    movi            t6, 0
    cmplt           t2, t6              // i_kernel_h < 0 ?
    bt              .IM2COL_PAD_PRE     // c_flag = 1 -> .IM2COL_PAD_PRE
    cmphs           t2, l2              // i_kernel_h >= input_h
    bt              .IM2COL_PAD_PRE
    cmplt           t4, t6              // i_kernel_w < 0 ?
    bt              .IM2COL_PAD_PRE
    cmphs           t4, l3              // i_kernel_w >= input_w
    bt              .IM2COL_PAD_PRE

.IM2COL_COPY_PRE:
    // else branch, prepare to copy from input_data to im2col_buf
    // copy num: input_ch --> loop_16 --> tail<16
    mult            t6, t2, l3          // i_ker_h * input_w
    addu            t6, t6, t4          // i_ker_h * input_w + i_ker_w
    mult            t6, t6, l1          // (i_ker_h * input_w + i_ker_w) * in_ch
    addu            t6, t6, a0          // (i_ker_h * input_w + i_ker_w) * in_ch + input_data

    lsri            t7, l1, 4           // in_ch >> 4u
    bez             t7, .IM2COL_COPY_TAIL

.IM2COL_COPY_16:
    vldmu.8         vr0-vr0, (t6)       // load from input_data + (i_ker_h * input_w + i_ker_w)*in_ch
    vstmu.8         vr0-vr0, (lr)       // store to im2col_buf

    bnezad          t7, .IM2COL_COPY_16

.IM2COL_COPY_TAIL:
    andi            t7, l1, 15          // in_ch & 15u
    bez             t7, .KERNEL_W_ADD   // nothing to copy,

    vldx.8          vr0, (t6), t7
    vstx.8          vr0, (lr), t7
    addu            lr, lr, t7          // update im2col_buff addr
    br              .KERNEL_W_ADD       // im2col copy done

.IM2COL_PAD_PRE:
    ld.w            t8, (sp, 0xe0)      // input_zero_point
    vdupg.8         vr0, t8
    lsri            t6, l1, 4           // in_ch >> 4u
    bez             t6, .IM2COL_PAD_TAIL

.IM2COL_PAD_16:
    vstmu.8         vr0-vr0, (lr)       // pad input_zp to im2col_buf

    bnezad          t6, .IM2COL_PAD_16

.IM2COL_PAD_TAIL:
    andi            t6, l1, 15          // in_ch & 15u
    bez             t6, .KERNEL_W_ADD   // nothing to pad

    vstx.8          vr0, (lr), t6
    addu            lr, lr, t6          // im2col pad done, im2col_buff addr had add input_ch

.KERNEL_W_ADD:
    addi            t4, t4, 1           // i_ker_w ++
    br              .KERNEL_W

.KERNEL_H_ADD:
    addi            t2, t2, 1           // i_ker_h ++
    br              .KERNEL_H


// available register: t2, t3, t4, t5, t6, t7
.IN_KER:
    // vec mult matrix_trans of 1 line input/im2col_buffer and out_ch lines kernel
    ld.w            t2, (sp, 0xdc)      // out_ch
    mov             l7, a1              // l7 = weight_data
    mov             t4, a1
    mov             l8, a2              // l8 = bias_data

    lsri            t6, t2, 2           // out_ch >> 2u
    bez             t6, .L5             // jump to out_ch_tail

.IN1_KER4:
    vldu.32.4       vr10, (l8)          // load bias and update addr

    vmovi.8         vr5, 0
    vmovi.8         vr6, 0
    vmovi.8         vr7, 0
    vmovi.8         vr8, 0
    vmovi.8         vr9, 0              // clear acc vr5-vr9

    mov             lr, l0              // im2col_buffer = bufferA

    lsri            t7, l9, 4           // in_ch * ker_h * ker_w >> 4u
    bez             t7, .L2

.IN1_KER4X16:
    mov             l6, l7
    vldmu.8         vr0-vr0, (lr)       // load im2col_buff
    vldmru.8        vr1-vr4, (l6), l9   // load weight_data 4 lines

    vmulacaa.u8     vr5, vr0, vr12      // acc(q1*z2)

    vmulacaa.u8     vr6, vr1, vr11
    vmulacaa.u8     vr7, vr2, vr11
    vmulacaa.u8     vr8, vr3, vr11
    vmulacaa.u8     vr9, vr4, vr11      // acc(q2*z1)

    vmulaca.u8      vr1, vr0, vr1
    vmulaca.u8      vr2, vr0, vr2
    vmulaca.u8      vr3, vr0, vr3
    vmulaca.u8      vr4, vr0, vr4       // acc(q1*q2)

    vpadd.s32       vr1, vr1, vr2
    vpadd.s32       vr3, vr3, vr4
    vpadd.s32       vr1, vr1, vr3       // sum[0..3]
    vadd.s32        vr10, vr10, vr1     // bias + q1*q2 temp

    addi            l7, l7, 16          // weight pointer + 16
    bnezad          t7, .IN1_KER4X16

.L2:
    andi            t7, l9, 15          // in_ch * ker_h * ker_w & 15u
    bez             t7, .OUT_4

.IN1_KER4XTAIL:
    mov             l6, l7              // weight_data point bump to line_tail
    vldx.8          vr0, (lr), t7       // load tail im2col data

    vldx.8          vr1, (l6), t7
    addu            l6, l6, l9
    vldx.8          vr2, (l6), t7
    addu            l6, l6, l9
    vldx.8          vr3, (l6), t7
    addu            l6, l6, l9
    vldx.8          vr4, (l6), t7

    vmulacaa.u8     vr5, vr0, vr12      // acc(q1*z2)

    vmulacaa.u8     vr6, vr1, vr11
    vmulacaa.u8     vr7, vr2, vr11
    vmulacaa.u8     vr8, vr3, vr11
    vmulacaa.u8     vr9, vr4, vr11      // acc(q2*z1)

    vmulaca.u8      vr1, vr0, vr1
    vmulaca.u8      vr2, vr0, vr2
    vmulaca.u8      vr3, vr0, vr3
    vmulaca.u8      vr4, vr0, vr4       // acc(q1*q2)

    vpadd.s32       vr1, vr1, vr2
    vpadd.s32       vr3, vr3, vr4
    vpadd.s32       vr1, vr1, vr3       // sum[0..3]
    vadd.s32        vr10, vr10, vr1     // bias + q1*q2 temp

.OUT_4:
    vpadd.s32.s     vr6, vr6, vr7
    vpadd.s32.s     vr8, vr8, vr9
    vpadd.s32.s     vr6, vr6, vr8       // sum(q2*z1)

    vpadd.s32.s     vr5, vr5, vr5
    vpadd.s32.s     vr5, vr5, vr5       // sum(q1*z2)

    vadd.s32        vr5, vr5, vr6       // sum(q2*z1) + sum(q1*z2)
    vadd.s32        vr10, vr10, vr15    // sum(q1*q2) + bias + z1*z2*k
    vsub.s32        vr10, vr10, vr5     // sum(q1*q2) + bias + z1*z2*k - ( sum(q2*z1) + sum(q1*z2) )

    vrmulh.s32.rs   vr0, vr10, vr13
    vshr.s32.r      vr0, vr0, vr14      // round mult scale

    ld.w            t3, (sp, 0xe8)      // z3
    vdupg.32        vr1, t3             // vr1[0..3] = z3
    vadd.s32        vr0, vr0, vr1       // add z3
    vclip.u32       vr0, vr0, 8

    vmov.u32.sl     vr0, vr0, vr0
    vmov.u16.sl     vr0, vr0, vr0
    vstu.8.4        vr0, (a3)           // store to output_data

    lsli            t7, l9, 2           // t1 = 4 * in_ch * ker_h * ker_w
    addu            t4, t4, t7
    mov             l7, t4              // weight_addr add 4 lines
    bnezad          t6, .IN1_KER4       // remain out_ch > 4

.L5:
    andi            t6, t2, 3           // out_ch_tail
    bez             t6, .OUT_W_ADD      // all out_ch done

.IN1_KERTAIL:
    vldu.32.1       vr10, (l8)

    vmovi.8         vr5, 0              // accumulate temp for q1*z
    vmovi.8         vr6, 0              // accumulate temp for q2*z1
    vmovi.8         vr7, 0              // accumulate temp for q1*q2

    mov             lr, l0              // update input_data point
    mov             l6, l7              // update weight_data point

    lsri            t7, l9, 4           // in_ch * ker_h * ker_w >> 4u
    bez             t7, .L7

.IN1_KER1X16:
    vldmu.8         vr0-vr0, (lr)
    vldmu.8         vr1-vr1, (l6)

    vmulacaa.u8     vr7, vr0, vr1       // acc(q1*q2)

    vmulacaa.u8     vr5, vr0, vr12      // acc(q1*z2)
    vmulacaa.u8     vr6, vr1, vr11      // acc(q2*z1)

    bnezad          t7, .IN1_KER1X16

.L7:
    andi            t7, l9, 15
    bez             t7, .OUT_1

.IN1_KER1XTAIL:
    vldx.8          vr0, (lr), t7
    vldx.8          vr1, (l6), t7

    vmulacaa.u8     vr7, vr0, vr1       // acc(q1*q2)

    vmulacaa.u8     vr5, vr0, vr12      // acc(q1*z2)
    vmulacaa.u8     vr6, vr1, vr11      // acc(q2*z1)

.OUT_1:
    vpadd.s32.s     vr5, vr5, vr5
    vpadd.s32.s     vr5, vr5, vr5       // vr5[0..3] = sum(q1*z2)

    vpadd.s32.s     vr6, vr6, vr6
    vpadd.s32.s     vr6, vr6, vr6       // vr6[0..3] = sum(q2*z1)

    vpadd.s32.s     vr7, vr7, vr7
    vpadd.s32.s     vr7, vr7, vr7       // vr7[0..3] = sum(q1*q2)

    vadd.s32.s      vr5, vr5, vr6       // sum(q1*z2) + sum(q2*z1)
    vadd.s32.s      vr10, vr10, vr7     // sum(q1*q2) + bias
    vadd.s32.s      vr10, vr10, vr15    // sum(q1*q2) + bias + z1*z2*k
    vsub.s32.s      vr10, vr10, vr5     // sum(q1*q2) + bias + z1*z2*k - ( sum(q1*z2) + sum(q2*z1) )

    vrmulh.s32.rs   vr0, vr10, vr13
    vshr.s32.r      vr0, vr0, vr14      // round mult scale

    ld.w            t3, (sp, 0xe8)      // z3
    vdupg.32        vr1, t3
    vadd.s32        vr0, vr0, vr1       // add output_zp
    vclip.u32       vr0, vr0, 8

    vmov.u32.sl     vr0, vr0, vr0
    vmov.u16.sl     vr0, vr0, vr0
    vstu.8.1        vr0, (a3)

    addu            l7, l7, l9          // weight_data point bump to next line
    bnezad          t6, .IN1_KERTAIL    // n_tail - 1 > 0 ?

.OUT_W_ADD:
    mov             lr, l0              // im2col_buffer = bufferA
    addi            t1, t1, 1           // i_out_w++
    br              .OUT_W

.OUT_H_ADD:
    addi            t0, t0, 1           // i_out_h++
    br              .OUT_H

.BATCH_END:
    vldmu.8         vr12-vr15, (sp)
    vldmu.8         vr8-vr11, (sp)
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           csi_i805_conv2d_opt_u8, .-csi_i805_conv2d_opt_u8
