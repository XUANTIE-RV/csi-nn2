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
 * @file     shl_i805_elementwise_mul_8.S
 * @brief    uint8 elementwise mul layer function.
 * @version  V1.0
 * @date     9. Jul 2021
 ******************************************************************************/


/*
    void shl_i805_elementwise_mul_opt_u8(uint8_t *input_0,
                                         uint8_t *input_1,
                                         uint8_t *output,
                                         int32_t size,
                                         int32_t input_0_zeroponit,
                                         int32_t input_1_zeropoint,
                                         int32_t output_zeropoint,
                                         int32_t output_mult,
                                         int32_t output_shift);


    Algorithm works as follows:
        (1) s3*(q3-z3) = s1*(q1-z1) * s2*(q2-z2)
        (2) q3 = [ (q1-z1) * (q2-z2) * (s1*s2/s3) ]  +  z3
        (3) params:output_mult means mult of s1*s2/s3 and params:output_shift represents the right shift(>0)

    register definition:
        a0:         input_0_data pointer
        a1:         input_1_data pointer
        a2:         output_data pointer
        a3:         element size to mul
        sp+0x14/l0: input0_zp
        sp+0x18/l1: input1_zp
 */

    .file           "shl_i805_elementwise_mul_8.S"
    .section        .text.shl_i805_elementwise_mul_opt_u8,"ax",@progbits
    .align          2
    .global         shl_i805_elementwise_mul_opt_u8
    .type           shl_i805_elementwise_mul_opt_u8, @function

shl_i805_elementwise_mul_opt_u8:
    push            l0, l1, l2, l3, l4
    ld.w            l0, (sp, 0x14)      // input_0_zeroponit
    ld.w            l1, (sp, 0x18)      // input_1_zeropoint

    ld.w            l2, (sp, 0x1c)      // output_zeropoint
    ld.w            l3, (sp, 0x20)      // output_mult
    ld.w            l4, (sp, 0x24)      // output_shift

    vdupg.32        vr3, l0             // z1
    vdupg.32        vr4, l1             // z2
    vdupg.32        vr5, l2             // z3
    vdupg.32        vr6, l3             // mult
    vdupg.32        vr7, l4             // shift

    lsri            t0, a3, 2           // size >> 2u
    bez             t0, .TAIL_4

.LOOP_4:
    // deal with 4 elements for this loop
    vldu.8.4        vr0, (a0)
    vmov.u8.e       vr0, vr0
    vmov.u16.e      vr0, vr0

    vldu.8.4        vr1, (a1)
    vmov.u8.e       vr1, vr1
    vmov.u16.e      vr1, vr1

    vadd.s32        vr0, vr0, vr3
    vadd.s32        vr1, vr1, vr4       // add zero_point
    vmul.s32        vr2, vr0, vr1       // vr2 = (q1-z1)(q2-z2)
    vrmulh.s32.rs   vr2, vr2, vr6       // doubling multiply high half

    vand.32         vr1, vr2, vr7
    vshri.s32       vr1, vr1, 31
    vadd.s32.S      vr1, vr1, vr2
    vshr.s32        vr1, vr1, vr7

    vadd.s32        vr1, vr1, vr5

    vmov.u32.sl     vr1, vr1, vr1
    vmov.u16.sl     vr1, vr1, vr1       // narrow to 8 bit

    vstu.8.4        vr1, (a2)
    bnezad          t0, .LOOP_4

.TAIL_4:
    // deal with tail(< 4) elements for this loop
    andi            t0, a3, 4
    bez             t0, .END

    vldx.8          vr0, (a0), t0
    vmov.u8.e       vr0, vr0
    vmov.u16.e      vr0, vr0

    vldx.8          vr1, (a1), t0
    vmov.u8.e       vr1, vr1
    vmov.u16.e      vr1, vr1

    vadd.s32        vr0, vr0, vr3
    vadd.s32        vr1, vr1, vr4       // add zero_point
    vmul.s32        vr2, vr0, vr1       // vr2 = (q1-z1)(q2-z2)
    vrmulh.s32.rs   vr2, vr2, vr6       // doubling multiply high half

    vand.32         vr1, vr2, vr7
    vshri.s32       vr1, vr1, 31
    vadd.s32.S      vr1, vr1, vr2
    vshr.s32        vr1, vr1, vr7

    vadd.s32        vr1, vr1, vr5       // add output_zero_point

    vmov.u32.sl     vr1, vr1, vr1
    vmov.u16.sl     vr1, vr1, vr1       // narrow to 8 bit

    vstx.8          vr1, (a2), t0

.END:
    pop             l0, l1, l2, l3, l4
    rts
    .size           shl_i805_elementwise_mul_opt_u8, .-shl_i805_elementwise_mul_opt_u8

