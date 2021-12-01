/*
 * Copyright (C) 2016-2021 C-SKY Limited. All rights reserved.
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

/* CSI-NN2 version 1.10.x */

/******************************************************************************
 * @file     csi_i805_fullyconnected_8.S
 * @brief    uint8 basic fully-connected layer function.
 * @version  V1.0
 * @date     9. Jul 2021
 ******************************************************************************/

/*
    void csi_i805_fullyconnected_opt_u8(uint8_t * input_data,
                                        uint8_t * weight_data,
                                        int32_t * bias_data,
                                        uint8_t * output_data,
                                        int32_t  in_nodes,
                                        int32_t  out_nodes,
                                        int32_t  input_zero_point,
                                        int32_t  weight_zero_point,
                                        int32_t  output_zero_point,
                                        int32_t  output_multi,
                                        int32_t  output_shift)

    Algorithm works as follows:
        (1) perform verctor(input) matrix(weight) multiplication
        (2) input[vector] * weight[matrix]:
                1 x 4lines: 1 x [4*loop_16 + 4*tail_16]  --> loop out_nodes / 4
                1 x 1lines: 1 x [1*loop_16 + 1*tail_16]  --> loop out_nodes % 4
        (3) input[vector]:  1 x in_nodes
            weight[matrix]: out_nodes x in_nodes

    register definition:
        a0: bias_data
        a1: output_data
        a2: input_data
        a3: weight_data
        vr0: bias
        vr1-vr4: weight
        vr6-vr9: accumulate temp for input * weight
        vr7: accumulate temp
        vr8: accumulate temp
        vr9: accumulate temp

*/

    .file           "csi_i805_fullyconnected_8.S"
    .section        .text.csi_i805_fullyconnected_opt_u8,"ax",@progbits
    .align          2
    .global         csi_i805_fullyconnected_opt_u8
    .type           csi_i805_fullyconnected_opt_u8, @function

csi_i805_fullyconnected_opt_u8:
    push            l0, l1, l2, l3, l4, l5, l6, l7
    subi            sp, sp, 64
    vstm.8          vr8-vr11, (sp)
    subi            sp, sp, 64
    vstm.8          vr12-vr15, (sp)

    ld.w            l0, (sp, 0xa0)      // in_nodes
    ld.w            l1, (sp, 0xa4)      // out_nodes
    ld.w            l2, (sp, 0xa8)      // input_zp
    ld.w            l3, (sp, 0xac)      // weight_zp
    ld.w            l4, (sp, 0xb0)      // output_zp
    ld.w            l5, (sp, 0xb4)      // out_mult
    ld.w            l6, (sp, 0xb8)      // out_shift
    // ld.w            l7, (sp, 0xbc)

    // z1 * z2 * k  constraints: z1 * z2 * k < int32
    mult            t3, l2, l3
    mult            t3, l0, t3          // t3 = z1 * z2 * k

    vdupg.32        vr3, t3             // v10[0..3] = t3   B3 = z1 * z2 * k
    vdupg.32        vr13, l4            // vr11[0..3] = z3

    vdupg.8         vr14, l2            // input_z1
    vdupg.8         vr15, l3            // input_z2

    vdupg.32        vr8, l5             // out_mult
    vdupg.32        vr9, l6             // out_shift

    mov             l7, a1              // save origin weight addr

    lsri            t0, l1, 1           // n >> 1
    bez             t0, .L5

.L0:
    vldu.32.2       vr5, (a2)           // load 2 int32 bias

    vmovi.8         vr6, 0
    vmovi.8         vr7, 0
    vmovi.8         vr10, 0
    vmovi.8         vr11, 0
    vmovi.8         vr12, 0             // clear vr6-vr9

    mov             l4, a0              // l4 = input_data

    lsri            t1, l0, 4           // k >> 4
    bez             t1, .L2

.L1:    // 4 x 16
    mov             l5, a1              // l5 = weight_data
    vldmu.8         vr0-vr0, (l4)       // vr0 = input_data
    vldmru.8        vr1-vr2, (l5), l0   // load 2 lines weight_data

    vmulacaa.u8     vr6, vr0, vr1       // vr6 hold 4 acc temp result, vr6[0] = vr6[0] + acc(vr0[0..3] * vr1[0..3]); vr6[1] = vr6[1] + acc(vr0[4..7] * vr1[4..7]) ...
    vmulacaa.u8     vr7, vr0, vr2

    vmulacaa.u8     vr10, vr0, vr15     // acc(q1*z2)
    vmulacaa.u8     vr11, vr1, vr14
    vmulacaa.u8     vr12, vr2, vr14     // acc(q2*z1)

    addi            a1, a1, 16          // weight_data += 16
    bnezad          t1, .L1

.L2:
    andi            t1, l0, 15          // t1 = col % 15
    bez             t1, .L4

.L3:    // 4 x tail
    mov             l5, a1              // weight pointer bump to the first line tail
    vldx.8          vr0, (l4), t1       // load tail input_data

    vldx.8          vr1, (l5), t1
    addu            l5, l5, l0          // weight pointer bump to next line
    vldx.8          vr2, (l5), t1

    vmulacaa.u8     vr6, vr0, vr1
    vmulacaa.u8     vr7, vr0, vr2

    vmulacaa.u8     vr10, vr0, vr15
    vmulacaa.u8     vr11, vr1, vr14
    vmulacaa.u8     vr12, vr2, vr14

.L4:
    vpadd.s32.s     vr10, vr10, vr10
    vpadd.s32.s     vr10, vr10, vr10    // vr10[0-3] = acc(q1*z2)

    vpadd.s32.s     vr1, vr11, vr12
    vpadd.s32.s     vr1, vr1, vr1       // vr1[0-1] = acc(q2*z1)

    vpadd.s32.s     vr0, vr6, vr7
    vpadd.s32.s     vr0, vr0, vr0       // vr0[0-1] = acc(q1*q2)

    vadd.s32.s      vr1, vr1, vr10
    vadd.s32.s      vr0, vr0, vr5       // +bias
    vadd.s32.s      vr0, vr0, vr3       // +z1*z2*k
    vsub.s32.s      vr0, vr0, vr1

    vrmulh.s32.rs   vr1, vr0, vr8
    vshr.s32        vr1, vr1, vr9
    vadd.s32        vr1, vr1, vr13
    vclip.u32       vr1, vr1, 8         // 上限可以不用管，只要保证下限 >=0 即可，所以也可以用 vmax

    vmov.u32.sl     vr1, vr1, vr1
    vmov.u16.sl     vr1, vr1, vr1
    vstu.8.2        vr1, (a3)           // store 2 outputs

    lsli            t1, l0, 1
    addu            l7, l7, t1
    mov             a1, l7              // weight_data addr + 4 lines
    bnezad          t0, .L0             // t0 - 1 > 0 ?

.L5:
    andi            t0, l1, 1           // tail lines of weight_data < 4
    bez             t0, .L10


.L12:
    vldu.32.1       vr5, (a2)           // load bias_data one by one

    mov             l4, a0
    mov             l5, a1
    vmovi.8         vr6, 0
    vmovi.8         vr10, 0             // clear vr6-vr9
    vmovi.8         vr11, 0             // clear vr6-vr9

    lsri            t1, l0, 4
    bez             t1, .L7

.L6:    //  1 x 16
    vldmu.8         vr0-vr0, (l4)       // load 16 input_data, and update addr
    vldmu.8         vr1-vr1, (l5)       // load 16 weight_data, and update addr

    vmulacaa.u8     vr6, vr0, vr1

    vmulacaa.u8     vr10, vr0, vr15
    vmulacaa.u8     vr11, vr1, vr14

    bnezad          t1, .L6

.L7:
    andi            t1, l0, 15          // k & 15u
    bez             t1, .L9

.L8:    // 1 x tail
    vldx.8          vr0, (l4), t1
    vldx.8          vr1, (l5), t1
    addu            l5, l5, t1

    vmulacaa.u8     vr6, vr0, vr1

    vmulacaa.u8     vr10, vr0, vr15
    vmulacaa.u8     vr11, vr1, vr14

.L9:

    vpadd.s32.s     vr10, vr10, vr10
    vpadd.s32.s     vr10, vr10, vr10    // vr10[0-3] = acc(q1*z2)

    vpadd.s32.s     vr1, vr11, vr11
    vpadd.s32.s     vr1, vr1, vr1       // vr1[0-3] = acc(q2*z1)

    vpadd.s32.s     vr0, vr6, vr6
    vpadd.s32.s     vr0, vr0, vr0       // vr0[0-1] = acc(q1*q2)

    vadd.s32.s      vr1, vr1, vr10
    vadd.s32.s      vr0, vr0, vr5       // +bias
    vadd.s32.s      vr0, vr0, vr3       // +z1*z2*k
    vsub.s32.s      vr0, vr0, vr1

    vrmulh.s32.rs   vr1, vr0, vr8
    vshr.s32        vr1, vr1, vr9
    vadd.s32        vr1, vr1, vr13
    vclip.u32       vr1, vr1, 8         // 其实上限可以不用管，只要保证下限 >=0 即可，所以也可以用vmax

    vmov.u32.sl     vr1, vr1, vr1
    vmov.u16.sl     vr1, vr1, vr1
    vstu.8.1        vr1, (a3)

    mov             a1, l5
    bnezad          t0, .L12            // tail_n --

.L10:
    vldmu.8         vr12-vr15, (sp)
    vldmu.8         vr8-vr11, (sp)
    pop             l0, l1, l2, l3, l4, l5, l6, l7
    .size           csi_i805_fullyconnected_opt_u8, .-csi_i805_fullyconnected_opt_u8
