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
 * @file     csi_i805_elementwise_add_8.S
 * @brief    uint8 elementwise add layer function.
 * @version  V1.0
 * @date     9. Jul 2021
 ******************************************************************************/


/*
    void csi_i805_elementwise_add_opt_u8(uint8_t *input_0,
                                         uint8_t *input_1,
                                         uint8_t *output,
                                         int32_t size,
                                         int32_t input_0_zeroponit,
                                         int32_t input_0_mult,
                                         int32_t input_0_shift,
                                         int32_t input_1_zeropoint,
                                         int32_t input_1_mult,
                                         int32_t input_1_shift,
                                         int32_t output_zeropoint,
                                         int32_t output_mult,
                                         int32_t output_shift)

    Algorithm works as follows:
        (1): 先按照s3(q3-z3) = s1(q1-z1) + s2(q2-z2)
        (2): 上式其实可以 q3-z3 = s1/s3(q1-z1) + s2/s3(q2-z2)
        (3): 上式中 s1/s3*z1 和 s2/s3*z2 其实是常量，在函数刚进来就可以算出来，不用在循环体内计算

    register definition:
        a0:         input_0_data pointer
        a1:         input_1_data pointer
        a2:         output_data pointer
        a3:         element size to add
        t0:         loop count
        vr6:        input_0 zero_point
        vr7:        input_1 zero_point
        vr4:        input_0 mult
        vr5:        input_1 mult
        vr14:       input_0 shift
        vr15:       input_1 shift
        vr13:       output zero_point
        vr3:        output mult
        vr12:       output shift

 */


    .file           "csi_i805_elementwise_add_8.S"
    .section        .text.csi_i805_elementwise_add_opt_u8,"ax",@progbits
    .align          2
    .global         csi_i805_elementwise_add_opt_u8
    .type           csi_i805_elementwise_add_opt_u8, @function

csi_i805_elementwise_add_opt_u8:
    push            l0, l1, l2, l3, l4, l5, l6, l7
    subi            sp, sp, 64
    vstm.8          vr8-vr11, (sp)
    subi            sp, sp, 64
    vstm.8          vr12-vr15, (sp)

    ld.w            l0, (sp, 0xa0)      //  input_0 zero_point
    vdupg.32        vr6, l0
    ld.w            l3, (sp, 0xac)      //  input_1_zeropoint
    vdupg.32        vr7, l3

    ld.w            l1, (sp, 0xa4)      //  input_0_mult
    vdupg.32        vr4, l1
    ld.w            l4, (sp, 0xb0)      //  input_1_mult
    vdupg.32        vr5, l4

    ld.w            l2, (sp, 0xa8)      //  input_0_shift
    vdupg.32        vr14, l2
    ld.w            l5, (sp, 0xb4)      //  input_1_shift
    vdupg.32        vr15, l5

    ld.w            l6, (sp, 0xb8)      //  output_zeropoint
    vdupg.32        vr13, l6
    ld.w            l7, (sp, 0xbc)      //  output_mult
    vdupg.32        vr3, l7
    ld.w            l0, (sp, 0xc0)      //  output_shift
    vdupg.32        vr12, l0

    lsri            t0, a3, 2           // size >> 2
    bez             t0, .TAIL_4

.LOOP_4:
    // deal with 4 elements for this loop
    vldu.8.4        vr0, (a0)           // updata input_0_data pointer
    vmov.u8.e       vr0, vr0
    vmov.u16.e      vr0, vr0

    vldu.8.4        vr1, (a1)           // updata input_1_data pointer
    vmov.u8.e       vr1, vr1
    vmov.u16.e      vr1, vr1

    vsub.s32        vr0, vr0, vr6       // vr0 = q1-z1
    vsub.s32        vr1, vr1, vr7       // vr1 = q2-z2

    vrmulh.s32.rs   vr8, vr0, vr4
    vshr.s32        vr8, vr8, vr14

    vrmulh.s32.rs   vr9, vr1, vr5
    vshr.s32        vr9, vr9, vr15

    vadd.s32        vr8, vr8, vr9
    vadd.s32        vr8, vr8, vr13      // FIXME: fault when vr8 < 0
    vclip.u32       vr8, vr8, 8

    vmov.u32.sl     vr8, vr8, vr8
    vmov.u16.sl     vr8, vr8, vr8       // narrow to 8 bit

    vstu.8.4        vr8, (a2)
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

    vsub.s32        vr0, vr0, vr6           // vr0 = q1-z1
    vsub.s32        vr1, vr1, vr7           // vr1 = q2-z2

    vrmulh.s32.rs   vr8, vr0, vr4
    vshr.s32        vr8, vr8, vr14

    vrmulh.s32.rs   vr9, vr1, vr5
    vshr.s32        vr9, vr9, vr15

    vadd.s32        vr8, vr8, vr9
    vadd.s32        vr8, vr8, vr13

    vmov.u32.sl     vr8, vr8, vr8
    vmov.u16.sl     vr8, vr8, vr8           // narrow to 8 bit

    vstx.8          vr8, (a2), t0

.END:
    vldmu.8         vr12-vr15, (sp)
    vldmu.8         vr8-vr11, (sp)
    pop             l0, l1, l2, l3, l4, l5, l6, l7
    rts
    .size           csi_i805_elementwise_add_opt_u8, .-csi_i805_elementwise_add_opt_u8

