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

/* SHL version 2.1.x */


/******************************************************************************
 * @file     shl_i805_clip_8.S
 * @brief    uint8 clip layer function.
 * @version  V1.0
 * @date     2. Aug 2021
 ******************************************************************************/

/*
    void shl_i805_clip_opt_u8(uint8_t *input_data,
                              uint8_t *output_data,
                              int32_t size,
                              int32_t clip_qmin,
                              int32_t clip_qmax,
                              int32_t input_zeropoint,
                              int32_t output_zeropoint,
                              int32_t out_multiplier,
                              int32_t out_shift)

    Algorithm works as follows:
        (1) f_out = clip(f_in)
        (2) s_out(q_out - zp_out) = clip{ s_in(q_in - zp_in) }
        (3) q_temp = min( max(q_in, clip_qmin), clip_qmax )
        (4) q_out = s_in/s_out * (q_temp - zp_in) + zp_out

    register definition:
        a0:     input_data
        a1:     output_data
        a2:     size
        a3:     clip_min
        vr2:    clip_qmin
        vr3:    clip_qmax
        vr4:    input_zp
        vr5:    output_zp
        vr6:    output_mult
        vr7:    output left shift
 */

    .file           "shl_i805_clip_8.S"
    .section        .text.shl_i805_clip_opt_u8,"ax",@progbits
    .align          2
    .global         shl_i805_clip_opt_u8
    .type           shl_i805_clip_opt_u8, @function

shl_i805_clip_opt_u8:

    ld.w            t7, (sp, 0x00)      // clip_qmax
    ld.w            t0, (sp, 0x04)      // input_zp
    ld.w            t1, (sp, 0x08)      // output_zp
    ld.w            t2, (sp, 0x0c)      // out_multiplier
    ld.w            t3, (sp, 0x10)      // out_shift: shift left > 0


    vdupg.32        vr4, t0             // vr4[0..3] = input_zero_point
    vdupg.32        vr5, t1             // vr5[0..3] = output_zero_point

    vdupg.32        vr6, t2             // vr6[0..3] = multiplier
    vdupg.32        vr7, t3             // shift left > 0

    vdupg.32        vr2, a3             // clip_min
    vdupg.32        vr3, t7             // clip_max

    lsri            t4, a2, 2           // size >> 2
    bez             t4, .TAIL_4

.LOOP_4:
    // deal with 4 elements for this loop
    vldu.8.4        vr0, (a0)
    vmov.u8.e       vr0, vr0
    vmov.u16.e      vr0, vr0            // widen to 32 bit
    vmax.u32        vr0, vr0, vr2       // max(q_in, clip_qmin)
    vmin.u32        vr0, vr0, vr3       // min(q_in, clip_qmax)
    vsub.u32        vr0, vr0, vr4       // q_ = q - zp

    vshl.s32.S      vr0, vr0, vr7       // shift left

    vrmulh.s32.rs   vr1, vr0, vr6       // doubling multiply high half

    vadd.u32        vr1, vr1, vr5       // + output_zp

    vmov.u32.sl     vr1, vr1, vr1
    vmov.u16.sl     vr1, vr1, vr1       // narrow to 8 bit

    vstu.8.4        vr1, (a1)
    bnezad          t4, .LOOP_4

.TAIL_4:
    andi            t4, a2, 4           // size & 3u
    bez             t4, .END

    vldx.8          vr0, (a0), t4
    vmov.u8.e       vr0, vr0
    vmov.u16.e      vr0, vr0            // 16bit -> 32bit
    vmax.u32        vr0, vr0, vr2       // max(q_in, clip_qmin)
    vmin.u32        vr0, vr0, vr3       // min(q_in, clip_qmax)
    vsub.u32        vr0, vr0, vr4       // q_ = q - zp

    vshl.s32.S      vr0, vr0, vr7       // shift left

    vrmulh.s32.rs   vr1, vr0, vr6       // doubling multiply high half
    vadd.u32        vr1, vr1, vr5

    vmov.u32.sl     vr1, vr1, vr1
    vmov.u16.sl     vr1, vr1, vr1       // narrow to 8 bit

    vstx.8          vr1, (a1), t4

.END:
    rts
    .size           shl_i805_clip_opt_u8, .-shl_i805_clip_opt_u8
