/*
 * Copyright (C) 2016-2020 C-SKY Limited. All rights reserved.
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

/**
 *
 * void csi_dequantize_f32_c860(
 * uint8_t  *input,
 * float    *output,
 * int32_t  offset,
 * int32_t  multiplier,
 * int32_t  shift,
 * uint32_t length)
 *
**/

    .file           "utils.S"
    .section        .text.csi_dequantize_f32_c860,"ax",@progbits
    .align          2
    .global         csi_dequantize_f32_c860

csi_dequantize_f32_c860:
    ld.w            t0, (sp, 0x4)           // length
    ld.w            t3, (sp, 0x0)           // shift
    vdupg.32        vr0, a3
    addi            t1, t3, 96              // shift - 31 + 127
    lsli            t1, t1, 23              // the float value
    vitof.s32.f32   vr1, vr0
    vdupg.32        vr7, a2                 // offset
    vdupg.32        vr6, t1                 // scale
    vmul.f32        vr6, vr6, vr1
    lsri            t2, t0, 4               // length >> 4
    bez             t2, .L2

.L0:
    vldmu.8         vr0-vr0, (a0)           // input
    vmov.u8.e       vr2, vr0
    vmov.u16.e      vr4, vr2
    vmov.u16.e      vr0, vr3
    vadd.s32        vr4, vr4, vr7           // add offset
    vadd.s32        vr5, vr5, vr7
    vadd.s32        vr0, vr0, vr7
    vadd.s32        vr1, vr1, vr7
    vitof.s32.f32   vr2, vr4
    vitof.s32.f32   vr3, vr5
    vitof.s32.f32   vr4, vr0
    vitof.s32.f32   vr5, vr1
    vmul.f32        vr2, vr2, vr6
    vmul.f32        vr3, vr3, vr6
    vmul.f32        vr4, vr4, vr6
    vmul.f32        vr5, vr5, vr6
    vstmu.32        vr2-vr5, (a1)
    bnezad          t2, .L0

.L1:
    andi            t2, t0, 15              // length & 15
    lsri            t1, t2, 2
    bez             t1, .L3

.L2:
    vldu.8.4        vr0, (a0)               // input
    vmov.u8.e       vr2, vr0
    vmov.u16.e      vr4, vr2
    vadd.s32        vr4, vr4, vr7           // add offset
    vitof.s32.f32   vr2, vr4
    vmul.f32        vr2, vr2, vr6
    vstmu.32        vr2-vr2, (a1)

    bnezad          t1, .L2

.L3:
    andi            t1, t2, 3
    bez             t1, .L4

    vldx.8          vr0, (a0), t1           // input
    vmov.u8.e       vr2, vr0
    vmov.u16.e      vr4, vr2
    vadd.s32        vr4, vr4, vr7           // add offset
    vitof.s32.f32   vr2, vr4
    vmul.f32        vr2, vr2, vr6
    vstx.32         vr2, (a1), t1

.L4:
    rts
    .size           csi_dequantize_f32_c860, .-csi_dequantize_f32_c860

