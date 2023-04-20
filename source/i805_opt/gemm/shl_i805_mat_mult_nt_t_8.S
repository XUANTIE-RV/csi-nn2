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

/* SHL version 2.1.x */

/******************************************************************************
 * @file     shl_i805_mat_mult_nt_t_8.S
 * @brief    uint8 genenal matrix-multiplication(A * B_trans)  function.
 * @version  V1.0
 * @date     9. Jul 2021
 ******************************************************************************/


/*
    void shl_i805_mat_mult_nt_t_opt_u8(uint8_t * lhs, // input
                                       uint8_t * rhs, // kernel
                                       int32_t * bias,
                                       uint8_t * dst,
                                       int32_t  lhs_row,
                                       int32_t  lhs_col,
                                       int32_t  rhs_row,
                                       int32_t  lhs_zero_point,
                                       int32_t  rhs_zero_point,
                                       int32_t  dst_zero_point,
                                       int32_t  dst_mult,
                                       int32_t  dst_shift);

    Algorithm works as follows:
        (1) perform matrix-multiplication, lrs * rhs_trans
        (2) convolution_1x1 can be converted to mat_mult_nt_t
        (3) input layout:NHWC
            filter layout:OHWI

    register definition:
        a0: bias_data
        a1: weight_data
        a2: input_data
        a3: output_data
        l4: input_data temp
        l5: weight_data temp
        l6: bias_data temp
        vr0: input data / output_temp
        vr1-vr4: weight data / accumulate temp for q1 * q2
        vr10: bias data
        vr5: accumulate temp for q1 * z2
        vr6-vr9: accumulate temp for q2 * z1
        vr11: input_zeropoint
        vr12: kernel_zeropoint
        vr13: output_mult
        vr14: output_shift

*/

    .file           "shl_i805_mat_mult_nt_t_8.S"
    .section        .text.shl_i805_mat_mult_nt_t_opt_u8,"ax",@progbits
    .align          2
    .global         shl_i805_mat_mult_nt_t_opt_u8
    .type           shl_i805_mat_mult_nt_t_opt_u8, @function

shl_i805_mat_mult_nt_t_opt_u8:
    push            l0, l1, l2, l3, l4, l5, l6, l7
    subi            sp, sp, 64
    vstm.8          vr8-vr11, (sp)
    subi            sp, sp, 64
    vstm.8          vr12-vr15, (sp)

    ld.w            l0, (sp, 0xa0)      // lhs_row (m)
    ld.w            l1, (sp, 0xa4)      // lhs_col (k)
    ld.w            l2, (sp, 0xa8)      // rhs_row (n)
    ld.w            l3, (sp, 0xac)      // input_zp
    ld.w            l4, (sp, 0xb0)      // weight_zp
    ld.w            l7, (sp, 0xb4)      // output_zp
    ld.w            l5, (sp, 0xb8)      // out_mult
    ld.w            l6, (sp, 0xbc)      // out_shift

    vdupg.8         vr11, l3            // z1
    vdupg.8         vr12, l4            // z2
    vdupg.32        vr13, l5            // out_mult
    vdupg.32        vr14, l6            // out_shift

    // z1 * z2 * k  constraints: z1 * z2 * k < int32
    mult            t3, l3, l4
    mult            t3, l1, t3          // t3 = z1 * z2 * k
    vdupg.32        vr15, t3            // v10[0..3] = z1 * z2 * k


.L00:
    mov             l6, a2              // l6 = bias_data
    mov             l3, a1              // l3 = weight_data
    mov             t2, a1              // t2 = weight_data

    lsri            t0, l2, 2           // n >> 2
    bez             t0, .L5

.L0:
    vldu.32.4       vr10, (l6)

    vmovi.8         vr5, 0
    vmovi.8         vr6, 0
    vmovi.8         vr7, 0
    vmovi.8         vr8, 0
    vmovi.8         vr9, 0              // clear vr5-vr9

    mov             l4, a0              // l4 = input_data

    lsri            t1, l1, 4           // k >> 4
    bez             t1, .L2

.L1:    // 4x16

    mov             l5, l3
    vldmu.8         vr0-vr0, (l4)
    vldmru.8        vr1-vr4, (l5), l1   // load 4 lines weight_data

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

    addi            l3, l3, 16          // weight point + 16
    bnezad          t1, .L1

.L2:
    andi            t1, l1, 15          // k % 15
    bez             t1, .L4

.L3:    // 4xtail
    mov             l5, l3              // weight_data point bump to line_tail
    vldx.8          vr0, (l4), t1       // load tail input data
    addu            l4, l4, t1          // bump to input next line

    vldx.8          vr1, (l5), t1
    addu            l5, l5, l1
    vldx.8          vr2, (l5), t1
    addu            l5, l5, l1
    vldx.8          vr3, (l5), t1
    addu            l5, l5, l1
    vldx.8          vr4, (l5), t1       // load tail weight data

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

.L4:    // 4xrow_len
    vpadd.s32.s     vr6, vr6, vr7
    vpadd.s32.s     vr8, vr8, vr9
    vpadd.s32.s     vr6, vr6, vr8       // sum(q2*z1)

    vpadd.s32.s     vr5, vr5, vr5
    vpadd.s32.s     vr5, vr5, vr5       // sum(q1*z2)

    vadd.s32        vr5, vr5, vr6       // sum(q2*z1) + sum(q1*z2)
    vadd.s32        vr10, vr10, vr15    // sum(q1*q2) + bias + z1*z2*k
    vsub.s32        vr10, vr10, vr5     // sum(q1*q2) + bias + z1*z2*k - ( sum(q2*z1) + sum(q1*z2) )

    vrmulh.s32.rs   vr0, vr10, vr13
    vshr.s32        vr0, vr0, vr14      // mult scale

    vdupg.32        vr1, l7             // vr1[0..3] = z3
    vadd.s32        vr0, vr0, vr1       // add z3
    vclip.u32       vr0, vr0, 8

    vmov.u32.sl     vr0, vr0, vr0
    vmov.u16.sl     vr0, vr0, vr0
    vstu.8.4        vr0, (a3)

    lsli            t1, l1, 2           // t1 = 4 * k
    addu            t2, t2, t1
    mov             l3, t2
    bnezad          t0, .L0             // n4 --

.L5:
    andi            t0, l2, 3           // n_tail
    bez             t0, .L22

.L12:
    vldu.32.1       vr10, (l6)

    vmovi.8         vr5, 0              // accumulate temp for q1*z
    vmovi.8         vr6, 0              // accumulate temp for q2*z1
    vmovi.8         vr7, 0              // accumulate temp for q1*q2

    mov             l4, a0              // updata input_data point
    mov             l5, l3              // updata weight_data point

    lsri            t1, l1, 4           // k >> 4
    bez             t1, .L7

.L6:    // 1x16
    vldmu.8         vr0-vr0, (l4)
    vldmu.8         vr1-vr1, (l5)

    vmulacaa.u8     vr7, vr0, vr1       // acc(q1*q2)

    vmulacaa.u8     vr5, vr0, vr12      // acc(q1*z2)
    vmulacaa.u8     vr6, vr1, vr11      // acc(q2*z1)

    bnezad          t1, .L6

.L7:
    andi            t1, l1, 15          // k % 15
    bez             t1, .L9

.L8:    // 1xtail
    vldx.8          vr0, (l4), t1
    vldx.8          vr1, (l5), t1


    vmulacaa.u8     vr7, vr0, vr1       // acc(q1*q2)

    vmulacaa.u8     vr5, vr0, vr12      // acc(q1*z2)
    vmulacaa.u8     vr6, vr1, vr11      // acc(q2*z1)

.L9:    // 1xrow_len
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
    vshr.s32        vr0, vr0, vr14      // mult scale

    vdupg.32        vr1, l7
    vadd.s32        vr0, vr0, vr1       // add output_zp
    vclip.u32       vr0, vr0, 8

    vmov.u32.sl     vr0, vr0, vr0
    vmov.u16.sl     vr0, vr0, vr0
    vstu.8.1        vr0, (a3)

    addu            l3, l3, l1          // weight_data point bump to next line
    bnezad          t0, .L12            // n_tail - 1 > 0 ?

.L22:
    addu            a0, a0, l1          // input_data point bump to next line
    bnezad          l0, .L00            // m - 1 > 0 ?

.L10:
    vldmu.8         vr12-vr15, (sp)
    vldmu.8         vr8-vr11, (sp)
    pop             l0, l1, l2, l3, l4, l5, l6, l7
    .size           shl_i805_mat_mult_nt_t_opt_u8, .-shl_i805_mat_mult_nt_t_opt_u8
