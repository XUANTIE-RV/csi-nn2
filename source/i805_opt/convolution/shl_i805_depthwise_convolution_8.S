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

/* CSI-NN2 version 2.0.x */

/******************************************************************************
 * @file     shl_i805_depthwise_convolution_8.S
 * @brief    uint8 depthwise convolution layer function.
 * @version  V1.0
 * @date     9. Jul 2021
 ******************************************************************************/
/*

    void shl_i805_dwconv2d_opt_u8(uint8_t * input_data,
                                  uint8_t * kernel_data
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
                                  int32_t  input_zero_point,
                                  int32_t  weight_zero_point,
                                  int32_t  output_zero_point,
                                  int32_t  out_mult,
                                  int32_t  out_shift);

    Algorithm works as follows:
        (1) partition im2col(loop_4 + tail_4 buffer) + acc(matrix dot matrix) in kernel_h*kernel_w dimension
        (2) im2col_buffer_row: kernel_h * kernel_w
            im2col_buffer_col: in_ch
            kernel_data_row:   kernel_h * kernel_w
            kernel_data_col:   in_ch
        (3) 4 im2col_buffer * kernel_data = 4*in_ch output_data

    constraints:
        input_channel = output_channel  i.e. channel_mult equals 1
        dilation_h = dilation_w = 1

    register definition:
        t0:         i_out_h
        t1:         i_out_w
        t2:         i_ker_h
        t4:         i_ker_w
        vr6-vr9:    acc(q1*q2)
        vr11-vr14:  acc(q1)
        vr15:       acc(q2)

    TODO: support per-channel quantization ???

*/

    .file           "shl_i805_depthwise_convolution_8.S"
    .section        .text.shl_i805_dwconv2d_opt_u8,"ax",@progbits
    .align          2
    .global         shl_i805_dwconv2d_opt_u8
    .type           shl_i805_dwconv2d_opt_u8, @function

shl_i805_dwconv2d_opt_u8:
    push            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    subi            sp, sp, 64
    vstm.8          vr8-vr11, (sp)
    subi            sp, sp, 64
    vstm.8          vr12-vr15, (sp)

    ld.w            l0, (sp, 0xac)      // bufferA
    ld.w            l2, (sp, 0xb0)      // input_h
    ld.w            l3, (sp, 0xb4)      // input_w
    ld.w            l1, (sp, 0xb8)      // input_ch

    ld.w            l4, (sp, 0xbc)      // kernel_h
    ld.w            l5, (sp, 0xc0)      // kernel_w
    mult            l9, l4, l5          // ker_h * ker_w
    mult            l9, l9, l1          // ker_h * ker_w * in_ch

    movi            t0, 0               // i_out_h
    movi            lr, 0               // buffer_cnt

.OUT_H:
    ld.w            t8, (sp, 0xd4)      // t8 = out_h   (out_h used for outer loop, load from stack)
    cmplt           t0, t8              // i_out_h < out_h
    bf              .IN_TAIL_KER        // buffer_cnt for im2col

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
    bf              .BUFFER_CNT         // C flag = 0 ?

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
    // copy num: input_ch: loop_16 --> tail<16
    mult            t6, t2, l3          // i_ker_h * input_w
    addu            t6, t6, t4          // i_ker_h * input_w + i_ker_w
    mult            t6, t6, l1          // (i_ker_h * input_w + i_ker_w) * in_ch
    addu            t6, t6, a0          // (i_ker_h * input_w + i_ker_w) * in_ch + input_data

    lsri            t7, l1, 4           // in_ch >> 4u
    bez             t7, .IM2COL_COPY_TAIL

.IM2COL_COPY_16:
    vldmu.8         vr0-vr0, (t6)       // load from input_data + (i_ker_h * input_w + i_ker_w)*in_ch
    vstmu.8         vr0-vr0, (l0)       // store to im2col_buf

    bnezad          t7, .IM2COL_COPY_16

.IM2COL_COPY_TAIL:
    andi            t7, l1, 15          // in_ch & 15u
    bez             t7, .KERNEL_W_ADD   // nothing to copy,

    vldx.8          vr0, (t6), t7
    vstx.8          vr0, (l0), t7
    addu            l0, l0, t7          // im2col_buff addr update +input_ch
    br              .KERNEL_W_ADD       // im2col copy done

.IM2COL_PAD_PRE:
    ld.w            t8, (sp, 0xdc)      // input_zero_point
    vdupg.8         vr0, t8
    lsri            t6, l1, 4           // in_ch >> 4u
    bez             t6, .IM2COL_PAD_TAIL

.IM2COL_PAD_16:
    vstmu.8         vr0-vr0, (l0)       // pad input_zp to im2col_buf

    bnezad          t6, .IM2COL_PAD_16

.IM2COL_PAD_TAIL:
    andi            t6, l1, 15          // in_ch & 15u
    bez             t6, .KERNEL_W_ADD   // nothing to pad

    vstx.8          vr0, (l0), t6
    addu            l0, l0, t6          // im2col pad done, // im2col_buff addr uapdate +input_ch

.KERNEL_W_ADD:
    addi            t4, t4, 1           // i_ker_w ++
    br              .KERNEL_W

.KERNEL_H_ADD:
    addi            t2, t2, 1           // i_ker_h ++
    br              .KERNEL_H

.BUFFER_CNT:
    addi            lr, lr, 1           // buffer_fill_count++
    cmplti          lr, 4               // signed < 4
    bt              .OUT_W_ADD          // if branch, im2col until buffer_cnt = 4



/*
    t5: in_ch >> 2 or in_ch & 3
    t3: i_ker_hw

    t6: input_data_0
    t7: input_data_1
    t8: input_data_2
    t9: input_data_3
    l7: weight_data
    l8: bias_data
    l6: output_data
*/

// else branch (buffer_fill_count = 4)
// available register: t2, t3, t4, t5, t6, t7, t8, t9, l6, l7, l8
.IN4_KER:
    movi            lr, 0               // clear buffer_fill_count first

    lsli            t5, l9, 2           // 4 * in_ch * ker_h * ker_w
    subu            l0, l0, t5          // notice: im2col_buffer addr jump to origin addr

    mov             t6, l0              // input0_addr
    addu            t7, t6, l9          // input1_addr
    addu            t8, t7, l9          // input2_addr
    addu            t9, t8, l9          // input3_addr

    mov             l7, a1              // kernel addr
    mov             l8, a2              // bias addr
    mov             l6, a3              // output addr

    lsri            t5, l1, 2           // in_ch >> 2
    bez             t5, .CH_TAIL        // in_ch tail < 4

.IN4_KER4_PRE:
    vldu.32.4       vr10, (l8)          // load 4 int32 bias

    vmovi.8         vr6, 0
    vmovi.8         vr7, 0
    vmovi.8         vr8, 0
    vmovi.8         vr9, 0              // clear vr6-vr9 for acc q1*q2

    vmovi.8         vr11, 0
    vmovi.8         vr12, 0
    vmovi.8         vr13, 0
    vmovi.8         vr14, 0             // clear vr11-vr14 for acc q1

    vmovi.8         vr15, 0             // clear vr15 for acc q2

    mult            t3, l4, l5          // i_ker_h*w = ker_h * ker_w

.IN4_KER4:
    vldru.8.4       vr0, (l7), l1       // load kernel_data, and bump to next line
    vmov.u8.e       vr0, vr0            // -> 16bit
    vmov.u16.e      vr0, vr0            // -> 32bit

    vldru.8.4       vr1, (t6), l1
    vmov.u8.e       vr1, vr1
    vmov.u16.e      vr1, vr1            // -> 32bit

    vldru.8.4       vr2, (t7), l1
    vmov.u8.e       vr2, vr2
    vmov.u16.e      vr2, vr2            // -> 32bit

    vldru.8.4       vr3, (t8), l1
    vmov.u8.e       vr3, vr3
    vmov.u16.e      vr3, vr3            // -> 32bit

    vldru.8.4       vr4, (t9), l1
    vmov.u8.e       vr4, vr4
    vmov.u16.e      vr4, vr4            // -> 32bit

    vmula.s32       vr6, vr0, vr1
    vmula.s32       vr7, vr0, vr2
    vmula.s32       vr8, vr0, vr3
    vmula.s32       vr9, vr0, vr4       // acc q1*q2

    vadd.s32        vr11, vr11, vr1
    vadd.s32        vr12, vr12, vr2
    vadd.s32        vr13, vr13, vr3
    vadd.s32        vr14, vr14, vr4     // acc q1

    vadd.s32        vr15, vr15, vr0     // acc q2

    bnezad          t3, .IN4_KER4

.OUT_4X4:
    ld.w            t3, (sp, 0xdc)      // z1
    vdupg.32        vr5, t3

    vmul.s32        vr15, vr15, vr5     // acc(q2*z1)

    ld.w            t4, (sp, 0xe0)      // z2
    vdupg.32        vr5, t4

    vmul.s32        vr11, vr11, vr5
    vmul.s32        vr12, vr12, vr5
    vmul.s32        vr13, vr13, vr5
    vmul.s32        vr14, vr14, vr5     // acc(q1*z2)

    vadd.s32.s      vr11, vr11, vr15
    vadd.s32.s      vr12, vr12, vr15
    vadd.s32.s      vr13, vr13, vr15
    vadd.s32.s      vr14, vr14, vr15    // q1*z2 + q2*z1

    vsub.s32.s      vr6, vr6, vr11
    vsub.s32.s      vr7, vr7, vr12
    vsub.s32.s      vr8, vr8, vr13
    vsub.s32.s      vr9, vr9, vr14      // q1*q2 - (q1*z2 + q2*z1)

    mult            t2, t3, t4          // z1*z2
    mult            t2, t2, l4          // z1*z2*ker_h
    mult            t2, t2, l5          // z1*z2*ker_h*ker_w = z1*z2*k
    vdupg.32        vr3, t2

    vadd.s32.s      vr6, vr6, vr3
    vadd.s32.s      vr7, vr7, vr3
    vadd.s32.s      vr8, vr8, vr3
    vadd.s32.s      vr9, vr9, vr3       // q1*q2 - (q1*z2 + q2*z1) + z1*z2*k

    vadd.s32.s      vr6, vr6, vr10
    vadd.s32.s      vr7, vr7, vr10
    vadd.s32.s      vr8, vr8, vr10
    vadd.s32.s      vr9, vr9, vr10      // q1*q2 - (q1*z2 + q2*z1) + z1*z2*k + bias

    ld.w            t3, (sp, 0xe8)      // out_mult
    ld.w            t4, (sp, 0xec)      // out_shift
    vdupg.32        vr1, t3
    vdupg.32        vr2, t4

    vrmulh.s32.rs   vr6, vr6, vr1
    vrmulh.s32.rs   vr7, vr7, vr1
    vrmulh.s32.rs   vr8, vr8, vr1
    vrmulh.s32.rs   vr9, vr9, vr1
    vshr.s32.r      vr6, vr6, vr2
    vshr.s32.r      vr7, vr7, vr2
    vshr.s32.r      vr8, vr8, vr2
    vshr.s32.r      vr9, vr9, vr2       // round mult scale

    ld.w            t3, (sp, 0xe4)      // output_zero_point
    vdupg.32        vr0, t3

    vadd.s32        vr6, vr6, vr0
    vadd.s32        vr7, vr7, vr0
    vadd.s32        vr8, vr8, vr0
    vadd.s32        vr9, vr9, vr0       // + z3

    vclip.u32       vr6, vr6, 8
    vclip.u32       vr7, vr7, 8
    vclip.u32       vr8, vr8, 8
    vclip.u32       vr9, vr9, 8         // clip[0, 255]

    vmov.u32.sl     vr6, vr6, vr6
    vmov.u16.sl     vr6, vr6, vr6
    vmov.u32.sl     vr7, vr7, vr7
    vmov.u16.sl     vr7, vr7, vr7
    vmov.u32.sl     vr8, vr8, vr8
    vmov.u16.sl     vr8, vr8, vr8
    vmov.u32.sl     vr9, vr9, vr9
    vmov.u16.sl     vr9, vr9, vr9       // -> 8bit

    vstru.8.4       vr6, (l6), l1
    vstru.8.4       vr7, (l6), l1
    vstru.8.4       vr8, (l6), l1
    vstru.8.4       vr9, (l6), l1

    addi            t9, t8, 4
    addi            t8, t7, 4
    addi            t7, t6, 4
    subu            t6, t6, l9
    addi            t6, t6, 4           // bump input addr to first line start + 4*i_ch4

    subu            l7, l7, l9
    addi            l7, l7, 4           // bump kernel addr to first line start + 4*i_ch4

    lsli            t3, l1, 2           // 4 * in_ch
    subu            l6, l6, t3
    addi            l6, l6, 4           // // bump output addr to first line start + 4*i_ch4

    bnezad          t5, .IN4_KER4_PRE

.CH_TAIL:
    andi            t5, l1, 3           // in_ch & 3u
    bez             t5, .L11            // finish all in_ch

.IN4_CH_TAIL_PRE:
    vldx.32         vr10, (l8), t5      // load tail int32 bias

    vmovi.8         vr6, 0
    vmovi.8         vr7, 0
    vmovi.8         vr8, 0
    vmovi.8         vr9, 0              // clear vr6-vr9 for acc q1*q2

    vmovi.8         vr11, 0
    vmovi.8         vr12, 0
    vmovi.8         vr13, 0
    vmovi.8         vr14, 0             // clear vr11-vr14 for acc q1

    vmovi.8         vr15, 0             // clear vr15 for acc q2

    mult            t3, l4, l5          // i_ker_h*w = ker_h * ker_w

.IN4_CH_TAIL:
    // 最开始这里确保 input_data 和 kernel_data 在第一行末梢 tail处
    vldx.8          vr0, (l7), t5
    addu            l7, l7, l1          // kernel next line
    vmov.u8.e       vr0, vr0
    vmov.u16.e      vr0, vr0            // -> 32bit

    vldx.8          vr1, (t6), t5
    addu            t6, t6, l1          // input0 next line
    vmov.u8.e       vr1, vr1
    vmov.u16.e      vr1, vr1            // -> 32bit

    vldx.8          vr2, (t7), t5
    addu            t7, t7, l1          // input1 next line
    vmov.u8.e       vr2, vr2
    vmov.u16.e      vr2, vr2            // -> 32bit

    vldx.8          vr3, (t8), t5
    addu            t8, t8, l1          // input2 next line
    vmov.u8.e       vr3, vr3
    vmov.u16.e      vr3, vr3            // -> 32bit

    vldx.8          vr4, (t9), t5
    addu            t9, t9, l1          // input3 next line
    vmov.u8.e       vr4, vr4
    vmov.u16.e      vr4, vr4            // -> 32bit

    vmula.s32       vr6, vr0, vr1
    vmula.s32       vr7, vr0, vr2
    vmula.s32       vr8, vr0, vr3
    vmula.s32       vr9, vr0, vr4       // acc q1*q2

    vadd.s32        vr11, vr11, vr1
    vadd.s32        vr12, vr12, vr2
    vadd.s32        vr13, vr13, vr3
    vadd.s32        vr14, vr14, vr4     // acc q1

    vadd.s32        vr15, vr15, vr0     // acc q2

    bnezad          t3, .IN4_CH_TAIL

.OUT_4XTAIL:
    ld.w            t3, (sp, 0xdc)      // z1
    vdupg.32        vr5, t3

    vmul.s32        vr15, vr15, vr5     // acc(q2*z1)

    ld.w            t4, (sp, 0xe0)      // z2
    vdupg.32        vr5, t4

    vmul.s32        vr11, vr11, vr5
    vmul.s32        vr12, vr12, vr5
    vmul.s32        vr13, vr13, vr5
    vmul.s32        vr14, vr14, vr5     // acc(q1*z2)

    vadd.s32.s      vr11, vr11, vr15
    vadd.s32.s      vr12, vr12, vr15
    vadd.s32.s      vr13, vr13, vr15
    vadd.s32.s      vr14, vr14, vr15    // q1*z2 + q2*z1

    vsub.s32.s      vr6, vr6, vr11
    vsub.s32.s      vr7, vr7, vr12
    vsub.s32.s      vr8, vr8, vr13
    vsub.s32.s      vr9, vr9, vr14      // q1*q2 - (q1*z2 + q2*z1)

    mult            t2, t3, t4          // z1*z2
    mult            t2, t2, l4          // z1*z2*ker_h
    mult            t2, t2, l5          // z1*z2*ker_h*ker_w = z1*z2*k
    vdupg.32        vr3, t2

    vadd.s32.s      vr6, vr6, vr3
    vadd.s32.s      vr7, vr7, vr3
    vadd.s32.s      vr8, vr8, vr3
    vadd.s32.s      vr9, vr9, vr3       // q1*q2 - (q1*z2 + q2*z1) + z1*z2*k

    vadd.s32.s      vr6, vr6, vr10
    vadd.s32.s      vr7, vr7, vr10
    vadd.s32.s      vr8, vr8, vr10
    vadd.s32.s      vr9, vr9, vr10      // q1*q2 - (q1*z2 + q2*z1) + z1*z2*k + bias

    ld.w            t3, (sp, 0xe8)      // out_mult
    ld.w            t4, (sp, 0xec)      // out_shift
    vdupg.32        vr1, t3
    vdupg.32        vr2, t4

    vrmulh.s32.rs   vr6, vr6, vr1
    vrmulh.s32.rs   vr7, vr7, vr1
    vrmulh.s32.rs   vr8, vr8, vr1
    vrmulh.s32.rs   vr9, vr9, vr1
    vshr.s32.r      vr6, vr6, vr2
    vshr.s32.r      vr7, vr7, vr2
    vshr.s32.r      vr8, vr8, vr2
    vshr.s32.r      vr9, vr9, vr2       // round mult scale

    ld.w            t3, (sp, 0xe4)      // output_zero_point
    vdupg.32        vr0, t3

    vadd.s32        vr6, vr6, vr0
    vadd.s32        vr7, vr7, vr0
    vadd.s32        vr8, vr8, vr0
    vadd.s32        vr9, vr9, vr0       // + z3

    vclip.u32       vr6, vr6, 8
    vclip.u32       vr7, vr7, 8
    vclip.u32       vr8, vr8, 8
    vclip.u32       vr9, vr9, 8         // clip[0, 255]

    vmov.u32.sl     vr6, vr6, vr6
    vmov.u16.sl     vr6, vr6, vr6
    vmov.u32.sl     vr7, vr7, vr7
    vmov.u16.sl     vr7, vr7, vr7
    vmov.u32.sl     vr8, vr8, vr8
    vmov.u16.sl     vr8, vr8, vr8
    vmov.u32.sl     vr9, vr9, vr9
    vmov.u16.sl     vr9, vr9, vr9       // -> 8bit

    vstx.8          vr6, (l6), t5
    addu            l6, l6, l1
    vstx.8          vr7, (l6), t5
    addu            l6, l6, l1
    vstx.8          vr8, (l6), t5
    addu            l6, l6, l1
    vstx.8          vr9, (l6), t5

.L11:
    lsli            t3, l1, 2           // 4 * in_ch
    addu            a3, a3, t3          // output_addr + 4 * in_ch


.OUT_W_ADD:
    addi            t1, t1, 1           // i_out_w++
    br              .OUT_W

.OUT_H_ADD:
    addi            t0, t0, 1           // i_out_h++
    br              .OUT_H



/*
    t0: in_ch >> 2 or in_ch & 3
    t1: i_ker_hw

    l6: input_data / buffer_addr
    l7: weight_data
    l8: bias_data

    vr0: kernel_data
    vr1: input_data
    vr3: acc(q1*q2)
    vr4: acc(q1/q1*z2)
    vr5: acc(q2/q2*z1)
    vr6: z1
    vr7: z2
    vr8: z3
    vr15: z1*z2*k

*/

.IN_TAIL_KER:
    andi            lr, lr, 3           // buffer_fill_cnt & 4u
    bez             lr, .BATCH_END

    ld.w            t2, (sp, 0xdc)      // input_zero_point
    ld.w            t3, (sp, 0xe0)      // weight_zero_point
    ld.w            t4, (sp, 0xe4)      // output_zero_point

    vdupg.32        vr6, t2             // z1
    vdupg.32        vr7, t3             // z2
    vdupg.32        vr8, t4             // z3

    mult            t5, t2, t3          // z1*z2
    mult            t5, t5, l4          // z1*z2*ker_h
    mult            t5, t5, l5          // z1*z2*ker_h*ker_w = z1*z2*k
    vdupg.32        vr15, t5

    ld.w            t2, (sp, 0xe8)      // out_mult
    vdupg.32        vr13, t2
    ld.w            t2, (sp, 0xec)      // out_shift
    vdupg.32        vr14, t2

    mult            t8, l4, l5          // t8 = ker_h * ker_w
    mult            t9, t8, l1          // t9 = ker_h * ker_w * in_ch

    mult            t2, l9, lr          // buffer_cnt_tail * in_ch * ker_h * ker_w
    subu            l0, l0, t2          // notice: im2col_buffer addr jump to origin addr
    mov             l6, l0              // reset input_data addr on bufferA

.IN1_KER:
    mov             l7, a1              // reset kernel_data addr
    mov             l8, a2              // reset bias_data addr
    lsri            t0, l1, 2           // in_ch >> 2u
    bez             t0, .CH_TAIL_1

.IN1_KER4_PRE:
    vldu.32.4       vr10, (l8)

    vmovi.8         vr3, 0              // clear vr6 for acc q1*q2
    vmovi.8         vr4, 0              // clear vr11 for acc q1
    vmovi.8         vr5, 0              // clear vr for acc q2

    mov             t1, t8              // reset t1 = ker_h * ker_w

.IN1_KER4:
    vldru.8.4       vr0, (l7), l1       // load kernel_data, and bump kernel_addr to next line
    vmov.u8.e       vr0, vr0            // -> 16bit
    vmov.u16.e      vr0, vr0            // -> 32bit

    vldru.8.4       vr1, (l6), l1       // load input_data, and bump input_addr to next line
    vmov.u8.e       vr1, vr1
    vmov.u16.e      vr1, vr1            // -> 32bit

    vmula.s32       vr3, vr0, vr1       // acc q1*q2
    vadd.s32        vr4, vr4, vr1       // acc q1
    vadd.s32        vr5, vr5, vr0       // acc q2

    bnezad          t1, .IN1_KER4

.OUT_1X4:
    vmul.s32        vr4, vr4, vr7       // acc(q1*z2)
    vmul.s32        vr5, vr5, vr6       // acc(q2*z1)

    vadd.s32.s      vr4, vr4, vr5       // q1*z2 + q2*z1
    vadd.s32.s      vr3, vr3, vr15      // q1*q2 + z1*z*k
    vadd.s32.s      vr3, vr3, vr10      // q1*q2 + z1*z*k + bais
    vsub.s32.s      vr3, vr3, vr4       // q1*q2 + z1*z*k + bais - (q1*z2 + q2*z1)

    vrmulh.s32.rs   vr3, vr3, vr13
    vshr.s32.r      vr3, vr3, vr14      // round mult scale

    vadd.s32        vr3, vr3, vr8       // +z3
    vclip.u32       vr3, vr3, 8

    vmov.u32.sl     vr3, vr3, vr3
    vmov.u16.sl     vr3, vr3, vr3

    vstu.8.4        vr3, (a3)

    subu            l6, l6, l9
    addi            l6, l6, 4           // bump input addr to first line start + 4*i_ch4

    subu            l7, l7, l9
    addi            l7, l7, 4           // bump kernel addr to first line start + 4*i_ch4

    bnezad          t0, .IN1_KER4_PRE

.CH_TAIL_1:
    andi            t0, l1, 3           // in_ch & 3u
    bez             t0, .NEXT_IN1       // 一个 input 和 kernel 的所有in_ch 都处理完了, 处理下一个输入

.IN1_CH_TAIL_PRE:
    vldx.32         vr10, (l8), t0
    // addu            l8, l8, t0       // bias addr update +in_ch

    vmovi.8         vr3, 0              // clear vr6 for acc q1*q2
    vmovi.8         vr4, 0              // clear vr11 for acc q1
    vmovi.8         vr5, 0              // clear vr for acc q2

    mov             t1, t8

.IN1_CH_TAIL:
    vldx.8          vr0, (l7), t0
    addu            l7, l7, l1          // kernel next line
    vmov.u8.e       vr0, vr0
    vmov.u16.e      vr0, vr0            // -> 32bit

    vldx.8          vr1, (l6), t0
    addu            l6, l6, l1          // input next line
    vmov.u8.e       vr1, vr1
    vmov.u16.e      vr1, vr1            // -> 32bit

    vmula.s32       vr3, vr0, vr1       // acc q1*q2
    vadd.s32        vr4, vr4, vr1       // acc q1
    vadd.s32        vr5, vr5, vr0       // acc q2

    bnezad          t1, .IN1_CH_TAIL

.OUT_1XTAIL:
    vmul.s32        vr4, vr4, vr7       // acc(q1*z2)
    vmul.s32        vr5, vr5, vr6       // acc(q2*z1)

    vadd.s32.s      vr4, vr4, vr5       // q1*z2 + q2*z1
    vadd.s32.s      vr3, vr3, vr15      // q1*q2 + z1*z*k
    vadd.s32.s      vr3, vr3, vr10      // q1*q2 + z1*z*k + bais
    vsub.s32.s      vr3, vr3, vr4       // q1*q2 + z1*z*k + bais - (q1*z2 + q2*z1)

    vrmulh.s32.rs   vr3, vr3, vr13
    vshr.s32.r      vr3, vr3, vr14      // round mult scale

    vadd.s32        vr3, vr3, vr8       // +z3
    vclip.u32       vr3, vr3, 8

    vmov.u32.sl     vr3, vr3, vr3
    vmov.u16.sl     vr3, vr3, vr3

    vstx.8          vr3, (a3), t0
    addu            a3, a3, t0          // output addr updata +in_ch

    addu            l6, l6, t0          // 跳转到了下一个输入的第一行末尾

.NEXT_IN1:
    subu            l6, l6, l1          // 下一个输入起始地址
    bnezad          lr, .IN1_KER        // buffer_cnt--


.BATCH_END:
    vldmu.8         vr12-vr15, (sp)
    vldmu.8         vr8-vr11, (sp)
    pop             l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, lr
    .size           shl_i805_dwconv2d_opt_u8, .-shl_i805_dwconv2d_opt_u8
